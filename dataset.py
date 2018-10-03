import csv
import os
from PIL import Image
from torch.utils.data import Dataset

START = "<SOS>"
END = "<EOS>"
SPECIAL_TOKENS = [START, END]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    truth_tokens = []
    remaining_truth = truth.strip()
    while len(remaining_truth) > 0:
        try:
            index, tok_len = next([i, len(tok)]
                                  for tok, i in token_to_id.items()
                                  if remaining_truth.startswith(tok))
            truth_tokens.append(index)
            remaining_truth = remaining_truth[tok_len:].lstrip()
        except StopIteration:
            raise Exception("Truth contains unknown token")
    return truth_tokens


def load_vocab(tokens_file):
    with open(tokens_file, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        tokens = next(reader)
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token


class CrohmeDataset(Dataset):
    """Dataset CROHME's handwritten mathematical formulas"""

    def __init__(self, groundtruth, tokens_file,
                 root=None, ext=".png", transform=None):
        """
        Args:
            groundtruth (string): Path to ground truth TSV file
            tokens_file (string): Path to tokens text file
            root (string): Path of the root directory of the dataset
            ext (string): Extension of the input files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(CrohmeDataset, self).__init__()
        if root is None:
            root = os.path.dirname(groundtruth)
        self.transform = transform
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        with open(groundtruth, "r") as fd:
            reader = csv.reader(fd, delimiter="\t")
            self.data = [{"path": os.path.join(root, p + ext), "truth": {
                "text": truth,
                "encoded": encode_truth(truth, self.token_to_id)
            }}
                for p, truth in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        # Remove the alpha channel
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image}
