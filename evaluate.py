import argparse
import editdistance
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


def to_percent(decimal, precision=2):
    after_decimal_point = 10 ** precision
    shifted = decimal * 100 * after_decimal_point
    percent = round(shifted) / after_decimal_point
    return "{value:.{precision}f}%".format(value=percent, precision=precision)


def create_markdown_tables(result):
    err_token = "Token error rate"
    err_no_special = "Token error rate (no speical tokens)"
    err_symbol = "Symbol error rate"
    err_header = "| {token} | {no_special} | {symbol} |".format(
        token=err_token, no_special=err_no_special, symbol=err_symbol
    )
    err_delimiter = re.sub("[^|]", "-", err_header)
    err_values = (
        "| {token:>{token_pad}} "
        "| {no_special:>{no_special_pad}} "
        "| {symbol:>{symbol_pad}} |"
    ).format(
        token=to_percent(result["error"]["full"]),
        no_special=to_percent(result["error"]["removed"]),
        symbol=to_percent(result["error"]["symbols"]),
        token_pad=len(err_token),
        no_special_pad=len(err_no_special),
        symbol_pad=len(err_symbol),
    )
    err_table = "\n".join([err_header, err_delimiter, err_values])

    correct_token = "Correct expressions"
    correct_no_special = "Correct expressions (no special tokens)"
    correct_symbol = "Correct expressions (Symbols)"
    correct_header = "| {token} | {no_special} | {symbol} |".format(
        token=correct_token, no_special=correct_no_special, symbol=correct_symbol
    )
    correct_delimiter = re.sub("[^|]", "-", correct_header)
    correct_values = (
        "| {token:>{token_pad}} "
        "| {no_special:>{no_special_pad}} "
        "| {symbol:>{symbol_pad}} |"
    ).format(
        token=to_percent(result["correct"]["full"]),
        no_special=to_percent(result["correct"]["removed"]),
        symbol=to_percent(result["correct"]["symbols"]),
        token_pad=len(correct_token),
        no_special_pad=len(correct_no_special),
        symbol_pad=len(correct_symbol),
    )
    correct_table = "\n".join([correct_header, correct_delimiter, correct_values])

    return err_table, correct_table


def evaluate(enc, dec, data_loader, device, checkpoint=default_checkpoint, prefix=""):
    special_tokens = [data_loader.dataset.token_to_id[tok] for tok in SPECIAL_TOKENS]
    non_symbols_encoded = [data_loader.dataset.token_to_id[tok] for tok in non_symbols]
    correct_tokens = 0
    distance = {"full": 0, "removed": 0, "symbols": 0}
    num_tokens = {"full": 0, "removed": 0, "symbols": 0}
    correct = {"full": 0, "removed": 0, "symbols": 0, "total": 0}

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
        decoded_values = []
        for i in range(batch_max_len - 1):
            previous = sequence[:, -1].view(-1, 1)
            out, hidden = dec(previous, hidden, enc_low_res, enc_high_res)
            _, top1_id = torch.topk(out, 1)
            sequence = torch.cat((sequence, top1_id), dim=1)
            decoded_values.append(out)
        decoded_values = torch.stack(decoded_values, dim=2).to(device)

        sequence_removed = [
            remove_special_tokens(seq, special_tokens) for seq in sequence
        ]
        sequence_symbols = [
            remove_special_tokens(seq, non_symbols_encoded) for seq in sequence_removed
        ]
        expected_removed = [
            remove_special_tokens(exp, special_tokens) for exp in expected
        ]
        expected_symbols = [
            remove_special_tokens(exp, non_symbols_encoded) for exp in expected_removed
        ]
        correct_tokens += torch.sum(sequence == expected, dim=(0, 1)).item()
        distances_full = calc_distances(sequence, expected)
        distances_removed = calc_distances(sequence_removed, expected_removed)
        distances_symbols = calc_distances(sequence_symbols, expected_symbols)
        distance["full"] += sum(distances_full)
        distance["removed"] += sum(distances_removed)
        distance["symbols"] += sum(distances_symbols)
        correct["full"] += sum(
            [torch.equal(seq, exp) for seq, exp in zip(sequence, expected)]
        )
        correct["removed"] += sum(
            [
                torch.equal(seq, exp)
                for seq, exp in zip(sequence_removed, expected_removed)
            ]
        )
        correct["symbols"] += sum(
            [
                torch.equal(seq, exp)
                for seq, exp in zip(sequence_symbols, expected_symbols)
            ]
        )
        correct["total"] += len(expected_symbols)

        # Can't use .numel() for the removed versions, because they can't
        # be converted to a tensor (stacked), as they may not have the same length.
        # Instead it's a list of tensors.
        num_tokens["full"] += expected.numel()
        num_tokens["removed"] += sum([exp.numel() for exp in expected_removed])
        num_tokens["symbols"] += sum([exp.numel() for exp in expected_symbols])

    return {
        "error": {
            "full": distance["full"] / num_tokens["full"],
            "removed": distance["removed"] / num_tokens["removed"],
            "symbols": distance["symbols"] / num_tokens["symbols"],
        },
        "correct": {
            "full": correct["full"] / correct["total"],
            "removed": correct["removed"] / correct["total"],
            "symbols": correct["symbols"] / correct["total"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint to be loaded to resume training",
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

    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint
        else default_checkpoint
    )
    encoder_checkpoint = checkpoint["model"].get("encoder")
    decoder_checkpoint = checkpoint["model"].get("decoder")

    for dataset_name in options.dataset:
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
            prefix=options.prefix,
        )
        err_table, correct_table = create_markdown_tables(result)
        print(
            "# Dataset {name}\n\n{err_table}\n\n{correct_table}".format(
                name=dataset_name, err_table=err_table, correct_table=correct_table
            )
        )


if __name__ == "__main__":
    main()
