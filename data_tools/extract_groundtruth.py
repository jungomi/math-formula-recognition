import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET

doc_namespace = "{http://www.w3.org/2003/InkML}"


def extract_truth(file_path):
    root = ET.parse(file_path).getroot()
    annotations = root.findall(doc_namespace + "annotation")
    truths = [ann for ann in annotations if ann.get("type") == "truth"]
    if len(truths) != 1:
        raise Exception(
            "{} does not contain a ground truth annotation".format(file_path)
        )
    return truths[0].text


def create_tsv(path, output="groundtruth.tsv"):
    files = glob.glob(os.path.join(path, "*.inkml"))
    with open(output, "w") as fd:
        writer = csv.writer(fd, delimiter="\t")
        for f in files:
            rel_path = os.path.relpath(f, path)
            reference_name = os.path.splitext(rel_path)[0]
            truth = extract_truth(f)
            # Remove $ because the entire forumla is surrounded by it.
            truth = truth.replace("$", "")
            writer.writerow([reference_name, truth])


if __name__ == "__main__":
    """
    extract_groundtruth path/to/dataset [-o OUTPUT]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="groundtruth.tsv",
        help="Output path of the TSV file",
    )
    parser.add_argument(
        "directory", nargs=1, help="Directory to data with ground truth"
    )
    args = parser.parse_args()
    create_tsv(args.directory[0], args.output)
