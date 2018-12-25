import argparse
import csv
import os
import random

validation_percent = 0.2
output_dir = "gt-split"


# Split the ground truth into train, validation sets
def split_gt(groundtruth, validation_percent=0.2):
    with open(groundtruth, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
        random.shuffle(data)
        validation_len = round(len(data) * validation_percent)
        return data[validation_len:], data[:validation_len]


def write_tsv(data, path):
    with open(path, "w") as fd:
        writer = csv.writer(fd, delimiter="\t")
        writer.writerows(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--validation-percent",
        dest="validation_percent",
        default=validation_percent,
        type=float,
        help="Percent of data to use for validation [Default: {}]".format(
            validation_percent
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        help="Path to input ground truth file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="Directory to save the split ground truth files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    train_gt, validation_gt = split_gt(options.input, options.validation_percent)
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    write_tsv(train_gt, os.path.join(options.output_dir, "train.tsv"))
    write_tsv(validation_gt, os.path.join(options.output_dir, "validation.tsv"))
