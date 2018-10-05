import os
import torch

use_cuda = torch.cuda.is_available()

default_checkpoint = {"epoch": 0, "losses": [], "accuracy": [], "model": {}}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    # Padded to 4 digits because of lexical sorting of numbers.
    # e.g. 0009.pth
    filename = "{prefix}{num:0>4}.pth".format(num=checkpoint["epoch"], prefix=prefix)
    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(checkpoint, os.path.join(dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)
