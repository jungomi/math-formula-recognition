import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_accuracy": [],
    "validation_losses": [],
    "validation_accuracy": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
}


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


def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(base_dir, name))


def write_tensorboard(
    writer,
    epoch,
    grad_norm,
    train_loss,
    train_accuracy,
    validation_loss,
    validation_accuracy,
    encoder,
    decoder,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy, epoch)
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_accuracy", validation_accuracy, epoch)
    writer.add_scalar("grad_norm", grad_norm, epoch)

    for name, param in encoder.named_parameters():
        writer.add_histogram(
            "encoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "encoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )

    for name, param in decoder.named_parameters():
        writer.add_histogram(
            "decoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "decoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )
