import argparse
import editdistance
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from checkpoint import default_checkpoint, load_checkpoint
from model import Encoder, Decoder
from dataset import CrohmeDataset, START, SPECIAL_TOKENS

input_size = (256, 256)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)

batch_size = 4
num_workers = 4

test_sets = {
    "2013": {"groundtruth": "./data/groundtruth_2013.tsv", "root": "./data/test/2013/"},
    "2014": {"groundtruth": "./data/groundtruth_2014.tsv", "root": "./data/test/2014/"},
    "2016": {"groundtruth": "./data/groundtruth_2016.tsv", "root": "./data/test/2016/"},
}

tokensfile = "./data/tokens.tsv"
use_cuda = torch.cuda.is_available()

transformers = transforms.Compose(
    [
        # Resize to 256x256 so all images have the same size
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
        return torch.tensor([tok for tok in tokens if tok not in special_tokens])


def calc_distances(actual, expected):
    return [
        editdistance.eval(act.tolist(), exp.tolist())
        for act, exp in zip(actual, expected)
    ]


def evaluate(
    enc, dec, name, data_loader, device, checkpoint=default_checkpoint, prefix=""
):
    special_tokens = [data_loader.dataset.token_to_id[tok] for tok in SPECIAL_TOKENS]
    correct_tokens = 0
    distance = {"full": 0, "removed": 0, "stripped": 0}
    num_tokens = {"full": 0, "removed": 0, "stripped": 0}

    for d in data_loader:
        input = d["image"].to(device)
        # The last batch may not be a full batch
        curr_batch_size = len(input)
        expected = torch.stack(d["truth"]["encoded"], dim=1).to(device)
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
        for i in range(data_loader.dataset.max_len - 1):
            previous = sequence[:, -1].view(-1, 1)
            out, hidden = dec(previous, hidden, enc_low_res, enc_high_res)
            _, top1_id = torch.topk(out, 1)
            sequence = torch.cat((sequence, top1_id), dim=1)
            decoded_values.append(out)
        decoded_values = torch.stack(decoded_values, dim=2).to(device)

        sequence_removed = [
            remove_special_tokens(seq, special_tokens) for seq in sequence
        ]
        sequence_stripped = [
            remove_special_tokens(seq, special_tokens, strip_only=True)
            for seq in sequence
        ]
        expected_removed = [
            remove_special_tokens(exp, special_tokens) for exp in expected
        ]
        expected_stripped = [
            remove_special_tokens(exp, special_tokens, strip_only=True)
            for exp in expected
        ]
        correct_tokens += torch.sum(sequence == expected, dim=(0, 1)).item()
        distances_full = calc_distances(sequence, expected)
        distances_removed = calc_distances(sequence_removed, expected_removed)
        distances_stripped = calc_distances(sequence_stripped, expected_stripped)
        distance["full"] += sum(distances_full)
        distance["removed"] += sum(distances_removed)
        distance["stripped"] += sum(distances_stripped)
        # Can't use .numel() for the removed / stripped versions, because they can't
        # be converted to a tensor (stacked), as they may not have the same length.
        # Instead it's a list of tensors.
        num_tokens["full"] += expected.numel()
        num_tokens["removed"] += sum([exp.numel() for exp in expected_removed])
        num_tokens["stripped"] += sum([exp.numel() for exp in expected_stripped])

    print(
        "# Dataset {name}:\n"
        "\nToken - Full\n"
        "==============\n"
        "Accuracy = {full_accuracy}\n"
        "Error Rate = {full_error}\n"
        "\nToken - Removed special tokens\n"
        "================================\n"
        "Error Rate = {removed_error}\n"
        "\nToken - Stripped special tokens\n"
        "=================================\n"
        "Error Rate = {stripped_error}\n".format(
            name=name,
            full_accuracy=correct_tokens / num_tokens["full"],
            full_error=distance["full"] / num_tokens["full"],
            removed_error=distance["removed"] / num_tokens["removed"],
            stripped_error=distance["stripped"] / num_tokens["stripped"],
        )
    )


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
        default="2016",
        type=str,
        choices=test_sets.keys(),
        help="Dataset used for evaluation (year) [default: {}]".format("2016"),
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
    hardware = "cuda" if use_cuda and not options.no_cuda else "cpu"
    device = torch.device(hardware)

    checkpoint = (
        load_checkpoint(options.checkpoint)
        if options.checkpoint
        else default_checkpoint
    )
    encoder_checkpoint = checkpoint["model"].get("encoder")
    decoder_checkpoint = checkpoint["model"].get("decoder")

    test_set = test_sets[options.dataset]
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
    )

    enc = Encoder(checkpoint=encoder_checkpoint).to(device)
    dec = Decoder(
        len(dataset.id_to_token),
        low_res_shape,
        high_res_shape,
        checkpoint=decoder_checkpoint,
        device=device,
    ).to(device)
    enc.eval()
    dec.eval()

    evaluate(
        enc,
        dec,
        name=options.dataset,
        data_loader=data_loader,
        device=device,
        checkpoint=checkpoint,
        prefix=options.prefix,
    )


if __name__ == "__main__":
    main()
