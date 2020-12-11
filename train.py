import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

import dataset as module_data
import model as module_arch
from parse_config import ConfigParser

torch.backends.cudnn.enabled = False

NUM_EPOCHS = 10

# Fix random seeds for reproducibility.
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # Setup the logger.
    logger = config.get_logger("train")

    # Setup the dataset.
    dataloader = config.init_obj("dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Prepare for GPU training.
    device = torch.device("cuda:{}".format(config["gpu_id"]))
    model = model.to(device)

    # Define our optimizer and criterion.
    optimizer = config.init_obj("optimizer", torch.optim, model.parameters())
    criterion = config.init_obj("criterion", nn)

    # Create some logger.
    losses = []

    # Train.
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (win_img_batch, lose_img_batch) in enumerate(dataloader):
            # Forward.
            win_img_batch, lose_img_batch = win_img_batch.to(device), lose_img_batch.to(device)
            label_batch = torch.Tensor([[1.0]] * win_img_batch.shape[0]).to(device)
            outputs = model(win_img_batch, lose_img_batch)
            loss = criterion(outputs, label_batch)

            # Backword.
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Record.
            losses.append(loss)
            if epoch % 10 == 0:
                print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, NUM_EPOCHS, loss.item()))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size")
    ]
    config = ConfigParser.from_args(args, options)

    # Let"s get started.
    main(config)