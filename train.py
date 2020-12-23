import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import dataset as module_data
import metric as module_metric
import model as module_arch
from parse_config import ConfigParser

# torch.backends.cudnn.enabled = False

NUM_EPOCHS = 10

# Fix random seeds for reproducibility.
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def display_valid_info(valid_metrics):
    print("############ Validation ############")
    for metric_name, metric_list in valid_metrics.items():
        print("{}: {:.4f}".format(metric_name, np.mean(metric_list)))
    print("############ Validation ############")

def main(config):
    # Setup the logger.
    logger = config.get_logger("train")

    # Setup the dataset.
    train_dataloader = config.init_obj("train_dataloader", module_data)
    valid_dataloader = config.init_obj("valid_dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Prepare for GPU training.
    device = torch.device("cuda:{}".format(config["gpu_id"]))
    model = model.to(device)

    # Define our optimizer, criterion and metrics.
    optimizer = config.init_obj("optimizer", torch.optim, model.parameters())
    criterion = config.init_obj("criterion", nn)
    metrics = dict((metric, getattr(module_metric, metric)) for metric in config["metrics"])

    # Create some logger.
    losses = []

    # Start train & validate.
    for epoch in range(NUM_EPOCHS):
        # Train part.
        model.train()
        for batch_idx, ((first_imgs_batch, second_imgs_batch), label_batch) in enumerate(train_dataloader):
            # Forward.
            first_imgs_batch, second_imgs_batch, label_batch = first_imgs_batch.to(device), second_imgs_batch.to(device), label_batch.to(device)
            outputs = model(first_imgs_batch, second_imgs_batch)
            loss = criterion(outputs, label_batch)

            # Backword.
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Record.
            losses.append(loss)
            if batch_idx % 10 == 0:
                print("Epoch [{}/{}]\tBatch: {}\tLoss: {:.4f}".format(epoch + 1, NUM_EPOCHS, batch_idx, loss.item()))

        # Validation part.
        model.eval()
        valid_metrics = collections.defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data_batch, label_batch) in enumerate(valid_dataloader):
                data_batch, label_batch = data_batch.to(device), label_batch.to(device)
                outputs = model.predict(data_batch)

                for metric_name, metric_fn in metrics.items():
                    metric = metric_fn(outputs, label_batch)
                    valid_metrics[metric_name].append(metric.item())

        display_valid_info(valid_metrics)

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