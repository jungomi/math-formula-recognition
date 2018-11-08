import argparse
import multiprocessing
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from model import Encoder, Decoder
from dataset import CrohmeDataset, START, PAD, collate_batch

input_size = (128, 128)
low_res_shape = (684, input_size[0] // 16, input_size[1] // 16)
high_res_shape = (792, input_size[0] // 8, input_size[1] // 8)

batch_size = 4
num_workers = multiprocessing.cpu_count()
num_epochs = 100
print_epochs = 1
learning_rate = 1e-3
lr_epochs = 20
lr_factor = 0.1
weight_decay = 1e-4
max_grad_norm = 5.0
dropout_rate = 0.2
teacher_forcing_ratio = 0.5
seed = 1234

groundtruth = "./data/groundtruth_train.tsv"
tokensfile = "./data/tokens.tsv"
root = "./data/train/"
use_cuda = torch.cuda.is_available()

transformers = transforms.Compose(
    [
        # Resize so all images have the same size
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
    teacher_forcing_ratio=teacher_forcing_ratio,
    lr_scheduler=None,
    num_epochs=100,
    print_epochs=None,
    checkpoint=default_checkpoint,
    prefix="",
    max_grad_norm=max_grad_norm,
):
    if print_epochs is None:
        print_epochs = num_epochs

    writer = init_tensorboard(name=prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    accuracy = checkpoint["accuracy"]
    losses = checkpoint["losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]
    optim_params = [
        p for param_group in optimiser.param_groups for p in param_group["params"]
    ]

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_losses = []
        epoch_grad_norms = []
        epoch_correct_symbols = 0
        total_symbols = 0

        if lr_scheduler:
            lr_scheduler.step()

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
            # The teacher forcing is done per batch, not symbol
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoded_values = []
            for i in range(batch_max_len - 1):
                previous = expected[:, i] if use_teacher_forcing else sequence[:, -1]
                previous = previous.view(-1, 1)
                out, hidden = dec(previous, hidden, enc_low_res, enc_high_res)
                hidden = hidden.detach()
                _, top1_id = torch.topk(out, 1)
                sequence = torch.cat((sequence, top1_id), dim=1)
                decoded_values.append(out)

            decoded_values = torch.stack(decoded_values, dim=2).to(device)
            optimiser.zero_grad()
            # decoded_values does not contain the start symbol
            loss = criterion(decoded_values, expected[:, 1:])
            loss.backward()
            # Clip gradients, it returns the total norm of all parameters
            grad_norm = nn.utils.clip_grad_norm_(optim_params, max_norm=max_grad_norm)
            optimiser.step()

            epoch_losses.append(loss.item())
            epoch_grad_norms.append(grad_norm)
            epoch_correct_symbols += torch.sum(sequence == expected, dim=(0, 1)).item()
            total_symbols += expected.numel()

        mean_epoch_loss = np.mean(epoch_losses)
        mean_epoch_grad_norm = np.mean(epoch_grad_norms)
        losses.append(mean_epoch_loss)
        grad_norms.append(mean_epoch_grad_norm)
        epoch_accuracy = epoch_correct_symbols / total_symbols
        accuracy.append(epoch_accuracy)
        epoch_lr = lr_scheduler.get_lr()[0]
        learning_rates.append(epoch_lr)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "losses": losses,
                "accuracy": accuracy,
                "lr": learning_rates,
                "grad_norm": grad_norms,
                "model": {"encoder": enc.state_dict(), "decoder": dec.state_dict()},
                "optimiser": optimiser.state_dict(),
            },
            prefix=prefix,
        )

        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % print_epochs == 0 or epoch == num_epochs - 1:
            print(
                "[{current:>{pad}}/{end}] Epoch {epoch}: "
                "Accuracy = {accuracy:.5f}, "
                "Loss = {loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})".format(
                    current=epoch + 1,
                    end=num_epochs,
                    epoch=start_epoch + epoch + 1,
                    pad=len(str(num_epochs)),
                    accuracy=epoch_accuracy,
                    loss=mean_epoch_loss,
                    lr=epoch_lr,
                    time=elapsed_time,
                )
            )
            write_tensorboard(
                writer,
                start_epoch + epoch + 1,
                mean_epoch_loss,
                epoch_accuracy,
                mean_epoch_grad_norm,
                enc,
                dec,
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
        "--lr-epochs",
        dest="lr_epochs",
        default=lr_epochs,
        type=float,
        help="Number of epochs until decay of learning rate [default: {}]".format(
            lr_epochs
        ),
    )
    parser.add_argument(
        "--lr-factor",
        dest="lr_factor",
        default=lr_factor,
        type=float,
        help="Decay factor of learning rate [default: {}]".format(lr_factor),
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
    parser.add_argument(
        "--prefix",
        dest="prefix",
        default="",
        type=str,
        help="Prefix of checkpoint names",
    )
    parser.add_argument(
        "--teacher-forcing",
        dest="teacher_forcing",
        default=teacher_forcing_ratio,
        type=float,
        help="Ratio to use the previous expected symbol [Default: {}]".format(
            teacher_forcing_ratio
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        dest="max_grad_norm",
        default=max_grad_norm,
        type=float,
        help="Maximum norm of gradients for gradient clipping [Default: {}]".format(
            max_grad_norm
        ),
    )
    parser.add_argument(
        "--dropout",
        dest="dropout_rate",
        default=dropout_rate,
        type=float,
        help="Probability of using dropout [Default: {}]".format(dropout_rate),
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(seed),
    )

    return parser.parse_args()


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
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
        collate_fn=collate_batch,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    enc = Encoder(
        img_channels=3, dropout_rate=options.dropout_rate, checkpoint=encoder_checkpoint
    ).to(device)
    dec = Decoder(
        len(dataset.id_to_token),
        low_res_shape,
        high_res_shape,
        dropout_rate=options.dropout_rate,
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
    optimiser = optim.Adadelta(
        params_to_optimise, lr=options.lr, weight_decay=options.weight_decay
    )
    optimiser_state = checkpoint.get("optimiser")
    if optimiser_state:
        optimiser.load_state_dict(optimiser_state)
    # Set the learning rate instead of using the previous state.
    # The scheduler somehow overwrites the LR to the initial LR after loading,
    # which would always reset it to the first used learning rate instead of
    # the one from the previous checkpoint. So might as well set it manually.
    for param_group in optimiser.param_groups:
        param_group["initial_lr"] = options.lr
    # Decay learning rate by a factor of lr_factor (default: 0.1)
    # every lr_epochs (default: 3)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimiser, step_size=options.lr_epochs, gamma=options.lr_factor
    )

    return train(
        enc,
        dec,
        optimiser,
        criterion,
        data_loader,
        teacher_forcing_ratio=options.teacher_forcing,
        lr_scheduler=lr_scheduler,
        print_epochs=options.print_epochs,
        device=device,
        num_epochs=options.num_epochs,
        checkpoint=checkpoint,
        prefix=options.prefix,
        max_grad_norm=options.max_grad_norm,
    )


if __name__ == "__main__":
    main()
