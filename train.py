import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

import dataset as module_data
import metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.common import prepare_device

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
    print("It may take some time to prepare data (even longer if pin_memory is set).")
    train_dataloader = config.init_obj("train_dataloader", module_data)
    valid_dataloader = config.init_obj("valid_dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Prepare GPU(s).
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Define our optimizer, criterion and metrics.
    optimizer = config.init_obj("optimizer", torch.optim, model.parameters())
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
    criterion = config.init_obj("criterion", nn)
    metric_fns = dict((metric, getattr(module_metric, metric)) for metric in config["metrics"])

    trainer = Trainer(model, criterion, metric_fns, optimizer,
                      config=config, device=device,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="train_dataloader;args;batch_size")
    ]
    config = ConfigParser.from_args(args, options)

    # Let"s get started.
    main(config)