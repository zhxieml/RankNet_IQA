import argparse
import collections

import numpy as np
import torch
import torch.nn as nn

import dataset as module_data
import metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils.common import prepare_device

def main(config):
    # Setup the dataset.
    print("It may take some time to prepare data (even longer if pin_memory is set).")
    test_dataloader = config.init_obj("test_dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    print(model)

    # Load the checkpoint.
    print("Loading checkpoint: {}...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Prepare GPU(s).
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Set the eval mode.
    model.eval()

    with torch.no_grad():
        for _, (data_batch, filename_batch) in enumerate(test_dataloader):
            data_batch = data_batch.to(device)
            outputs = model.predict(data_batch)
            scores = outputs.cpu().numpy()[:, 0]
            rank = module_metric.get_rank(scores)

            print(dict(zip(filename_batch, rank)))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)

    # Let"s get started.
    main(config)
