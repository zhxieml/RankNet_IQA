import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

import dataset as module_data
import metric as module_metric
import model as module_arch
from parse_config import ConfigParser

def main(config):
    # Setup the logger.
    logger = config.get_logger("train")

    # Setup the dataset.
    dataloader = config.init_obj("dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Define the criterion and metrics.
    criterion = config.init_obj("criterion", nn)
    metrics = [getattr(module_metric, metric) for metric in config["metrics"]]

    # Load the checkpoint.
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Prepare for GPU training.
    device = torch.device("cuda:{}".format(config["gpu_id"]))
    model = model.to(device)

    # Initialize records.
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for batch_idx, (win_img_batch, lose_img_batch, label_batch) in enumerate(dataloader):
            win_img_batch, lose_img_batch, label_batch = win_img_batch.to(device), lose_img_batch, label_batch.to(device)
            output = model(win_img_batch, lose_img_batch)

            # Compute loss, metrics on test set
            loss = criterion(output, label_batch)
            batch_size = win_img_batch.shape[0]
            total_loss += loss.item() * batch_size

            for batch_idx, metric in enumerate(metrics):
                total_metrics[batch_idx] += metric(output, label_batch) * batch_size

    n_samples = len(dataloader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        metric.__name__: total_metrics[i].item() / n_samples for i, metric in enumerate(metrics)
    })
    logger.info(log)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)

    # Let"s get started.
    main(config)
