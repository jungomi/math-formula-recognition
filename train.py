import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from checkpoint import default_checkpoint, load_checkpoint, save_checkpoint
from model import Encoder, Decoder
from dataset import CrohmeDataset, START

input_size = (256, 256)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)

batch_size = 4
num_workers = 4
num_epochs = 10
print_epochs = 1
learning_rate = 1e-3
weight_decay = 1e-4

groundtruth = "./data/groundtruth.tsv"
tokensfile = "./data/tokens.txt"
root = "./data/png/"
use_cuda = torch.cuda.is_available()

transformers = transforms.Compose(
    [
        # Resize to 256x256 so all images have the same size
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]
)


def train(
    enc,
    dec,
    optimiser,
    criterion,
    data_loader,
    device,
    num_epochs=100,
    print_epochs=None,
    checkpoint=default_checkpoint,
):
    if print_epochs is None:
        print_epochs = num_epochs

    total_symbols = len(data_loader.dataset) * data_loader.dataset.max_len
    start_epoch = checkpoint["epoch"]
    accuracy = checkpoint["accuracy"]
    losses = checkpoint["losses"]

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_losses = []
        epoch_correct_symbols = 0

        for d in data_loader:
            input = d["image"].to(device)
            expected = torch.stack(d["truth"]["encoded"], dim=1).to(device)
            enc_low_res, enc_high_res = enc(input)
            # Decoder needs to be reset, because the coverage attention (alpha)
            # only applies to the current image.
            dec.reset(data_loader.batch_size)
            hidden = dec.init_hidden(data_loader.batch_size).to(device)
            # Starts with a START token
            sequence = torch.full(
                (data_loader.batch_size, 1),
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
            optimiser.zero_grad()
            # decoded_values does not contain the start symbol
            loss = criterion(decoded_values, expected[:, 1:])
            loss.backward()
            optimiser.step()

            epoch_losses.append(loss.item())
            epoch_correct_symbols += torch.sum(sequence == expected, dim=(0, 1)).item()

        mean_epoch_loss = np.mean(epoch_losses)
        losses.append(mean_epoch_loss)
        epoch_accuracy = epoch_correct_symbols / total_symbols
        accuracy.append(accuracy)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "losses": losses,
                "accuracy": accuracy,
                "model": {"encoder": enc.state_dict(), "decoder": dec.state_dict()},
                "optimiser": optimiser.state_dict(),
            }
        )

        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % print_epochs == 0 or epoch == num_epochs - 1:
            print(
                "[{current:>{pad}}/{end}] Epoch {epoch}: "
                "Accuracy = {accuracy:.5f}, "
                "Loss = {loss:.5f} "
                "(time elapsed {time})".format(
                    current=epoch + 1,
                    end=num_epochs,
                    epoch=start_epoch + epoch + 1,
                    pad=len(str(num_epochs)),
                    accuracy=epoch_accuracy,
                    loss=mean_epoch_loss,
                    time=elapsed_time,
                )
            )

    return np.array(losses), np.array(accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--learning-rate",
        dest="lr",
        default=learning_rate,
        type=float,
        help="Learning rate [default: {}]".format(learning_rate),
    )
    parser.add_argument(
        "-d",
        "--decay",
        dest="weight_decay",
        default=weight_decay,
        type=float,
        help="Weight decay [default: {}]".format(weight_decay),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint to be loaded to resume training",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        default=num_epochs,
        type=int,
        help="Number of epochs to train [default: {}]".format(num_epochs),
    )
    parser.add_argument(
        "-p",
        "--print-epochs",
        dest="print_epochs",
        default=print_epochs,
        type=int,
        help="Number of epochs to report [default: {}]".format(print_epochs),
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
    print("Running {} epochs on {}".format(options.num_epochs, hardware))
    encoder_checkpoint = checkpoint["model"].get("encoder")
    decoder_checkpoint = checkpoint["model"].get("decoder")
    if encoder_checkpoint is not None:
        print(
            "Resuming from - Epoch {}: "
            "Accuracy = {accuracy:.5f}, "
            "Loss = {loss:.5f} ".format(
                checkpoint["epoch"],
                accuracy=checkpoint["accuracy"][-1],
                loss=checkpoint["losses"][-1],
            )
        )

    dataset = CrohmeDataset(groundtruth, tokensfile, root=root, transform=transformers)
    data_loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    enc = Encoder(checkpoint=encoder_checkpoint).to(device)
    dec = Decoder(
        len(dataset.id_to_token),
        low_res_shape,
        high_res_shape,
        checkpoint=decoder_checkpoint,
        device=device,
    ).to(device)
    enc.train()
    dec.train()

    enc_params_to_optimise = [
        param for param in enc.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in dec.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    optimiser = optim.Adadelta(params_to_optimise, weight_decay=options.weight_decay)

    return train(
        enc,
        dec,
        optimiser,
        criterion,
        data_loader,
        print_epochs=options.print_epochs,
        device=device,
        num_epochs=options.num_epochs,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
