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
    valid_dataloader = config.init_obj("valid_dataloader", module_data)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    print(model)

    # Define the metrics.
    metric_fns = dict((metric, getattr(module_metric, metric)) for metric in config["metrics"])

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
    valid_metrics = collections.defaultdict(list)

    with torch.no_grad():
        for _, (data_batch, label_batch) in enumerate(valid_dataloader):
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            outputs = model.predict(data_batch)

            for metric_name, metric_fn in metric_fns.items():
                metric = metric_fn(outputs, label_batch)
                valid_metrics[metric_name].append(metric.item())

    for metric_name, metric_list in valid_metrics.items():
        valid_metrics[metric_name] = np.mean(metric_list)

    print(valid_metrics)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)

    # Let"s get started.
    main(config)
