import argparse
import editdistance
import os
import re
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from checkpoint import default_checkpoint, load_checkpoint
from model import Encoder, Decoder
from dataset import CrohmeDataset, START, PAD, SPECIAL_TOKENS, collate_batch

input_size = (128, 128)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)

batch_size = 4
num_workers = 4
beam_width = 10

test_sets = {
    "train": {"groundtruth": "./data/groundtruth_train.tsv", "root": "./data/train/"},
    "2013": {"groundtruth": "./data/groundtruth_2013.tsv", "root": "./data/test/2013/"},
    "2014": {"groundtruth": "./data/groundtruth_2014.tsv", "root": "./data/test/2014/"},
    "2016": {"groundtruth": "./data/groundtruth_2016.tsv", "root": "./data/test/2016/"},
}

# These are not counted as symbol, because they are used for formatting / grouping, and
# they do not render anything on their own.
non_symbols = [
    "{",
    "}",
    "\\left",
    "\\right",
    "_",
    "^",
    "\\Big",
    "\\Bigg",
    "\\limits",
    "\\mbox",
]

tokensfile = "./data/tokens.tsv"
use_cuda = torch.cuda.is_available()

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]
)


# strip_only means that only special tokens on the sides are removed. Equivalent to
# String.strip()
def remove_special_tokens(tokens, special_tokens=SPECIAL_TOKENS, strip_only=False):
    if strip_only:
        num_left = 0
        num_right = 0
        for tok in tokens:
            if tok not in special_tokens:
                break
            num_left += 1
        for tok in reversed(tokens):
            if tok not in special_tokens:
                break
            num_right += 1
        return tokens[num_left:-num_right]
    else:
        return torch.tensor(
            [tok for tok in tokens if tok not in special_tokens], dtype=tokens.dtype
        )


def calc_distances(actual, expected):
    return [
        editdistance.eval(act.tolist(), exp.tolist())
        for act, exp in zip(actual, expected)
    ]


def create_statistics(hypothesis):
    sequence = hypothesis["sequence"]
    expected = hypothesis["expected"]
    num_tokens = hypothesis["num_tokens"]

    distances_full = calc_distances(sequence["full"], expected["full"])
    distances_removed = calc_distances(sequence["removed"], expected["removed"])
    distances_symbols = calc_distances(sequence["symbols"], expected["symbols"])
    distance = {
        "full": sum(distances_full),
        "removed": sum(distances_removed),
        "symbols": sum(distances_symbols),
    }
    correct = {
        "full": sum(
            [
                torch.equal(seq, exp)
                for seq, exp in zip(sequence["full"], expected["full"])
            ]
        ),
        "removed": sum(
            [
                torch.equal(seq, exp)
                for seq, exp in zip(sequence["removed"], expected["removed"])
            ]
        ),
        "symbols": sum(
            [
                torch.equal(seq, exp)
                for seq, exp in zip(sequence["symbols"], expected["symbols"])
            ]
        ),
        # This is the batch size, since each sequence is one expression.
        "total": len(sequence["full"]),
    }
    hypothesis["distance"] = distance
    hypothesis["error"] = {
        "full": distance["full"] / num_tokens["full"],
        "removed": distance["removed"] / num_tokens["removed"],
        "symbols": distance["symbols"] / num_tokens["symbols"],
    }
    hypothesis["correct"] = correct
    hypothesis["correct"]["percent"] = {
        "full": correct["full"] / correct["total"],
        "removed": correct["removed"] / correct["total"],
        "symbols": correct["symbols"] / correct["total"],
    }


def to_percent(decimal, precision=2):
    after_decimal_point = 10 ** precision
    shifted = decimal * 100 * after_decimal_point
    percent = round(shifted) / after_decimal_point
    return "{value:.{precision}f}%".format(value=percent, precision=precision)


def create_markdown_tables(results):
    model = "Model"
    model_pad = max(len(key) for key in results.keys())
    model_pad = max(len(model), model_pad)
    err_token = "Token error rate"
    err_no_special = "Token error rate (no special tokens)"
    err_symbol = "Symbol error rate"
    err_header = "| {model:>{model_pad}} | {token} | {no_special} | {symbol} |".format(
        model=model,
        model_pad=model_pad,
        token=err_token,
        no_special=err_no_special,
        symbol=err_symbol,
    )
    err_delimiter = re.sub("[^|]", "-", err_header)
    err_values = [
        (
            "| {model:>{model_pad}} "
            "| {token:>{token_pad}} "
            "| {no_special:>{no_special_pad}} "
            "| {symbol:>{symbol_pad}} |"
        ).format(
            model=name,
            token=to_percent(result["error"]["full"]),
            no_special=to_percent(result["error"]["removed"]),
            symbol=to_percent(result["error"]["symbols"]),
            model_pad=model_pad,
            token_pad=len(err_token),
            no_special_pad=len(err_no_special),
            symbol_pad=len(err_symbol),
        )
        for name, result in results.items()
    ]
    err_table = "\n".join([err_header, err_delimiter, *err_values])

    correct_token = "Correct expressions"
    correct_no_special = "Correct expressions (no special tokens)"
    correct_symbol = "Correct expressions (Symbols)"
    correct_header = (
        "| {model:>{model_pad}} | {token} | {no_special} | {symbol} |"
    ).format(
        model=model,
        model_pad=model_pad,
        token=correct_token,
        no_special=correct_no_special,
        symbol=correct_symbol,
    )
    correct_delimiter = re.sub("[^|]", "-", correct_header)
    correct_values = [
        (
            "| {model:>{model_pad}} "
            "| {token:>{token_pad}} "
            "| {no_special:>{no_special_pad}} "
            "| {symbol:>{symbol_pad}} |"
        ).format(
            model=name,
            token=to_percent(result["correct"]["percent"]["full"]),
            no_special=to_percent(result["correct"]["percent"]["removed"]),
            symbol=to_percent(result["correct"]["percent"]["symbols"]),
            model_pad=model_pad,
            token_pad=len(correct_token),
            no_special_pad=len(correct_no_special),
            symbol_pad=len(correct_symbol),
        )
        for name, result in results.items()
    ]
    correct_table = "\n".join([correct_header, correct_delimiter, *correct_values])

    return err_table, correct_table


# Convert hypothesis batches to hypothesis grouped by sequence.
def unbatch_hypotheses(hypotheses):
    if not hypotheses:
        return []
    hypotheses_by_seq = [[] for _ in hypotheses[0]["probability"]]
    for h in hypotheses:
        for i in range(len(h["probability"])):
            single_h = {
                "sequence": {"full": h["sequence"]["full"][i]},
                # The hidden weights have batch size in the second dimension, not first.
                "hidden": h["hidden"][:, i],
                "attn": {"low": h["attn"]["low"][i], "high": h["attn"]["high"][i]},
                "probability": h["probability"][i],
            }
            hypotheses_by_seq[i].append(single_h)
    return hypotheses_by_seq


def batch_single_hypotheses(single_hypotheses):
    # It might be possible that the different sequences have a different number of total
    # hypotheses, since there might be duplicates in one of them. To prevent that take
    # the lowest number that is available. It might be smaller than the beam width.
    # But there can only be batches where each sequence is present.
    min_len = min(len(hs) for hs in single_hypotheses)
    batched_hypotheses = []
    for i in range(min_len):
        batch_h = {
            "sequence": {
                "full": torch.stack(
                    [hs[i]["sequence"]["full"] for hs in single_hypotheses]
                )
            },
            # The hidden weights have batch size in the second dimension, not first.
            "hidden": torch.stack([hs[i]["hidden"] for hs in single_hypotheses], dim=1),
            "attn": {
                "low": torch.stack([hs[i]["attn"]["low"] for hs in single_hypotheses]),
                "high": torch.stack(
                    [hs[i]["attn"]["high"] for hs in single_hypotheses]
                ),
            },
            "probability": torch.stack(
                [hs[i]["probability"] for hs in single_hypotheses]
            ),
        }
        batched_hypotheses.append(batch_h)
    return batched_hypotheses


# Picks the k sequences with the best probabilities. Each sequence is inspected
# separately and at the end new hypotheses are created by stacking the k best ones of
# each sequence to create batches, that can be used for the next step.
def pick_top_k_unique(hypotheses, count):
    sorted_hypotheses = [
        sorted(hs, key=lambda h: h["probability"].item(), reverse=True)
        for hs in unbatch_hypotheses(hypotheses)
    ]
    unique_hypotheses = [[] for _ in sorted_hypotheses]

    for i, hs in enumerate(sorted_hypotheses):
        for h in hs:
            if len(unique_hypotheses[i]) >= count:
                break
            already_exists = False
            for h_uniq in unique_hypotheses[i]:
                already_exists = torch.equal(
                    h["sequence"]["full"], h_uniq["sequence"]["full"]
                )
                if already_exists:
                    break
            if not already_exists:
                unique_hypotheses[i].append(h)

    return batch_single_hypotheses(unique_hypotheses)


def evaluate(
    enc,
    dec,
    data_loader,
    device,
    checkpoint=default_checkpoint,
    beam_width=beam_width,
    prefix="",
):
    special_tokens = [data_loader.dataset.token_to_id[tok] for tok in SPECIAL_TOKENS]
    non_symbols_encoded = [data_loader.dataset.token_to_id[tok] for tok in non_symbols]
    best = {
        "num_tokens": {"full": 0, "removed": 0, "symbols": 0},
        "distance": {"full": 0, "removed": 0, "symbols": 0},
        "correct": {"full": 0, "removed": 0, "symbols": 0, "total": 0},
    }
    mean = {
        "num_tokens": {"full": 0, "removed": 0, "symbols": 0},
        "distance": {"full": 0, "removed": 0, "symbols": 0},
        "correct": {"full": 0, "removed": 0, "symbols": 0, "total": 0},
    }
    highest_prob = {
        "num_tokens": {"full": 0, "removed": 0, "symbols": 0},
        "distance": {"full": 0, "removed": 0, "symbols": 0},
        "correct": {"full": 0, "removed": 0, "symbols": 0, "total": 0},
    }

    for d in data_loader:
        input = d["image"].to(device)
        # The last batch may not be a full batch
        curr_batch_size = len(input)
        expected = d["truth"]["encoded"].to(device)
        batch_max_len = expected.size(1)
        # Replace -1 with the PAD token
        expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
        enc_low_res, enc_high_res = enc(input)
        # Decoder needs to be reset, because the coverage attention (alpha)
        # only applies to the current image.
        dec.reset(curr_batch_size)
        hidden = dec.init_hidden(curr_batch_size).to(device)
        # Starts with a START token
        sequence = torch.full(
            (curr_batch_size, 1),
            data_loader.dataset.token_to_id[START],
            dtype=torch.long,
            device=device,
        )
        hypotheses = [
            {
                "sequence": {"full": sequence},
                "hidden": hidden,
                "attn": {
                    "low": dec.coverage_attn_low.alpha,
                    "high": dec.coverage_attn_high.alpha,
                },
                # This will be a tensor of probabilities (one for each batch), but at
                # the beginning it can be 1.0 because it will be broadcast for the
                # multiplication and it means the first tensor of probabilities will be
                # kept as is.
                "probability": 1.0,
            }
        ]
        for i in range(batch_max_len - 1):
            step_hypotheses = []
            for hypothesis in hypotheses:
                curr_sequence = hypothesis["sequence"]["full"]
                previous = curr_sequence[:, -1].view(-1, 1)
                curr_hidden = hypothesis["hidden"]
                # Set the attention to the corresponding values, otherwise it would use
                # the attention from another hypothesis.
                dec.coverage_attn_low.alpha = hypothesis["attn"]["low"]
                dec.coverage_attn_high.alpha = hypothesis["attn"]["high"]
                out, next_hidden = dec(previous, curr_hidden, enc_low_res, enc_high_res)
                probabilities = torch.softmax(out, dim=1)
                topk_probs, topk_ids = torch.topk(probabilities, beam_width)
                # topks are transposed, because the columns are needed, not the rows.
                # One column is the top values for the batches, and there are k rows.
                for top_prob, top_id in zip(topk_probs.t(), topk_ids.t()):
                    next_sequence = torch.cat(
                        (curr_sequence, top_id.view(-1, 1)), dim=1
                    )
                    probability = hypothesis["probability"] * top_prob
                    next_hypothesis = {
                        "sequence": {"full": next_sequence},
                        "hidden": next_hidden,
                        "attn": {
                            "low": dec.coverage_attn_low.alpha,
                            "high": dec.coverage_attn_high.alpha,
                        },
                        "probability": probability,
                    }
                    step_hypotheses.append(next_hypothesis)
            # Only the beam_width number of hypotheses with the highest probabilities
            # are kept for the next iteration.
            hypotheses = pick_top_k_unique(step_hypotheses, beam_width)

        expected_removed = [
            remove_special_tokens(exp, special_tokens) for exp in expected
        ]
        expected_symbols = [
            remove_special_tokens(exp, non_symbols_encoded) for exp in expected_removed
        ]
        # Can't use .numel() for the removed versions, because they can't
        # be converted to a tensor (stacked), as they may not have the same length.
        # Instead it's a list of tensors.
        num_tokens = {
            "full": expected.numel(),
            "removed": sum([exp.numel() for exp in expected_removed]),
            "symbols": sum([exp.numel() for exp in expected_symbols]),
        }
        for hypothesis in hypotheses:
            sequence = hypothesis["sequence"]
            sequence["removed"] = [
                remove_special_tokens(seq, special_tokens) for seq in sequence["full"]
            ]
            sequence["symbols"] = [
                remove_special_tokens(seq, non_symbols_encoded)
                for seq in sequence["removed"]
            ]
            hypothesis["expected"] = {
                "full": expected,
                "removed": expected_removed,
                "symbols": expected_symbols,
            }
            hypothesis["num_tokens"] = num_tokens
            create_statistics(hypothesis)

        correct_totals = torch.tensor(
            [hypothesis["correct"]["total"] for hypothesis in hypotheses],
            dtype=torch.float,
        )
        best["correct"]["total"] += torch.max(correct_totals).item()
        # This should be constant, as every hypothesis should contain the same
        # number of expressions and therefore the mean should also be the same.
        mean["correct"]["total"] += torch.mean(correct_totals).item()
        highest_prob["correct"]["total"] += hypotheses[0]["correct"]["total"]
        for category in ["full", "removed", "symbols"]:
            category_distance = torch.tensor(
                [hypothesis["distance"][category] for hypothesis in hypotheses],
                dtype=torch.float,
            )
            category_correct = torch.tensor(
                [hypothesis["correct"][category] for hypothesis in hypotheses],
                dtype=torch.float,
            )
            category_num_tokens = torch.tensor(
                [hypothesis["num_tokens"][category] for hypothesis in hypotheses],
                dtype=torch.float,
            )
            best_distance = torch.argmin(category_distance)
            best_correct = torch.argmax(category_correct)
            best["distance"][category] += hypotheses[best_distance]["distance"][
                category
            ]
            best["num_tokens"][category] += hypotheses[best_distance]["num_tokens"][
                category
            ]
            best["correct"][category] += hypotheses[best_correct]["correct"][category]
            mean["distance"][category] += torch.mean(category_distance).item()
            mean["num_tokens"][category] += torch.mean(category_num_tokens).item()
            mean["correct"][category] += torch.mean(category_correct).item()
            # The highest probability is the first hypotheses, since it was sorted
            # when the top k were chosen.
            highest_prob["distance"][category] += hypotheses[0]["distance"][category]
            highest_prob["num_tokens"][category] += hypotheses[0]["num_tokens"][
                category
            ]
            highest_prob["correct"][category] += hypotheses[0]["correct"][category]

    best["error"] = {
        "full": best["distance"]["full"] / best["num_tokens"]["full"],
        "removed": best["distance"]["removed"] / best["num_tokens"]["removed"],
        "symbols": best["distance"]["symbols"] / best["num_tokens"]["symbols"],
    }
    best["correct"]["percent"] = {
        "full": best["correct"]["full"] / best["correct"]["total"],
        "removed": best["correct"]["removed"] / best["correct"]["total"],
        "symbols": best["correct"]["symbols"] / best["correct"]["total"],
    }
    mean["error"] = {
        "full": mean["distance"]["full"] / mean["num_tokens"]["full"],
        "removed": mean["distance"]["removed"] / mean["num_tokens"]["removed"],
        "symbols": mean["distance"]["symbols"] / mean["num_tokens"]["symbols"],
    }
    mean["correct"]["percent"] = {
        "full": mean["correct"]["full"] / mean["correct"]["total"],
        "removed": mean["correct"]["removed"] / mean["correct"]["total"],
        "symbols": mean["correct"]["symbols"] / mean["correct"]["total"],
    }
    highest_prob["error"] = {
        "full": highest_prob["distance"]["full"] / highest_prob["num_tokens"]["full"],
        "removed": highest_prob["distance"]["removed"]
        / highest_prob["num_tokens"]["removed"],
        "symbols": highest_prob["distance"]["symbols"]
        / highest_prob["num_tokens"]["symbols"],
    }
    highest_prob["correct"]["percent"] = {
        "full": highest_prob["correct"]["full"] / highest_prob["correct"]["total"],
        "removed": highest_prob["correct"]["removed"]
        / highest_prob["correct"]["total"],
        "symbols": highest_prob["correct"]["symbols"]
        / highest_prob["correct"]["total"],
    }
    return {"best": best, "mean": mean, "highest_prob": highest_prob}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        nargs="+",
        required=True,
        help="Path to the checkpoint to be used for the evaluation",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=batch_size,
        type=int,
        help="Size of data batches [default: {}]".format(batch_size),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        default=["2016"],
        type=str,
        choices=test_sets.keys(),
        nargs="+",
        help="Dataset used for evaluation [default: {}]".format("2016"),
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        default=num_workers,
        type=int,
        help="Number of workers for loading the data [default: {}]".format(num_workers),
    )
    parser.add_argument(
        "--beam-width",
        dest="beam_width",
        default=beam_width,
        type=int,
        help="Width of the beam [default: {}]".format(beam_width),
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        default="",
        type=str,
        help="Prefix of checkpoint names",
    )

    return parser.parse_args()


def main():
    options = parse_args()
    is_cuda = use_cuda and not options.no_cuda
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    for dataset_name in options.dataset:
        results = {"best": {}, "mean": {}, "highest_prob": {}}
        for checkpoint_path in options.checkpoint:
            checkpoint_name, _ = os.path.splitext(os.path.basename(checkpoint_path))
            checkpoint = (
                load_checkpoint(checkpoint_path, cuda=is_cuda)
                if checkpoint_path
                else default_checkpoint
            )
            encoder_checkpoint = checkpoint["model"].get("encoder")
            decoder_checkpoint = checkpoint["model"].get("decoder")

            test_set = test_sets[dataset_name]
            dataset = CrohmeDataset(
                test_set["groundtruth"],
                tokensfile,
                root=test_set["root"],
                transform=transformers,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                collate_fn=collate_batch,
            )

            enc = Encoder(img_channels=3, checkpoint=encoder_checkpoint).to(device)
            dec = Decoder(
                len(dataset.id_to_token),
                low_res_shape,
                high_res_shape,
                checkpoint=decoder_checkpoint,
                device=device,
            ).to(device)
            enc.eval()
            dec.eval()

            result = evaluate(
                enc,
                dec,
                data_loader=data_loader,
                device=device,
                checkpoint=checkpoint,
                beam_width=options.beam_width,
                prefix=options.prefix,
            )
            results["best"][checkpoint_name] = result["best"]
            results["mean"][checkpoint_name] = result["mean"]
            results["highest_prob"][checkpoint_name] = result["highest_prob"]

        highest_prob_err_table, highest_prob_correct_table = create_markdown_tables(
            results["highest_prob"]
        )
        best_err_table, best_correct_table = create_markdown_tables(results["best"])
        mean_err_table, mean_correct_table = create_markdown_tables(results["mean"])
        print(
            (
                "\n# Dataset {name}\n\n"
                "Beam width: {beam_width}\n\n"
                "## Highest Probability\n\n{highest_prob_err_table}\n\n"
                "{highest_prob_correct_table}\n\n"
                "## Best\n\n{best_err_table}\n\n{best_correct_table}\n\n"
                "## Mean\n\n{mean_err_table}\n\n{mean_correct_table}"
            ).format(
                name=dataset_name,
                beam_width=options.beam_width,
                highest_prob_err_table=highest_prob_err_table,
                highest_prob_correct_table=highest_prob_correct_table,
                best_err_table=best_err_table,
                best_correct_table=best_correct_table,
                mean_err_table=mean_err_table,
                mean_correct_table=mean_correct_table,
            )
        )


if __name__ == "__main__":
    main()
