import argparse
import csv


def parse_symbols(truth):
    unique_symbols = set()
    i = 0
    while i < len(truth):
        char = truth[i]
        i += 1
        if char.isspace():
            continue
        elif char == "\\":
            if truth[i] == "{" or truth[i] == "}":
                unique_symbols.add(char + truth[i])
                i += 1
                continue
            escape_seq = char
            while i < len(truth) and truth[i].isalpha():
                escape_seq += truth[i]
                i += 1
            unique_symbols.add(escape_seq)
        else:
            unique_symbols.add(char)
    return unique_symbols


def create_tokens(groundtruth, output="tokens.txt"):
    with open(groundtruth, "r") as fd:
        unique_symbols = set()
        reader = csv.reader(fd, delimiter="\t")
        for _, truth in reader:
            truth_symbols = parse_symbols(truth)
            unique_symbols = unique_symbols.union(truth_symbols)

        # This is somehow wrong, as it should be recognised as "less than N"
        # It would be two symbols, which are both already present.
        unique_symbols.remove("\\ltN")
        symbols = list(unique_symbols)
        symbols.sort()
        with open(output, "w") as output_fd:
            writer = csv.writer(output_fd, delimiter="\t")
            writer.writerow(symbols)


if __name__ == "__main__":
    """
    extract_tokens path/to/groundtruth.tsv [-o OUTPUT]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="tokens.txt",
        help="Output path of the tokens text file",
    )
    parser.add_argument("groundtruth", nargs=1, help="Ground truth TSV file")
    args = parser.parse_args()
    create_tokens(args.groundtruth[0], args.output)
