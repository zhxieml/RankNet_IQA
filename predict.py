import argparse
import itertools

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import dataset as module_data
import metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from utils.common import prepare_device

META_FILENAME = "/mnt/zhxie_hdd/dataset/ut-zap50k/ut-zap50k-data/meta-data-bin.csv"

def main(config):
    # Setup the dataset.
    print("It may take some time to prepare data (even longer if pin_memory is set).")
    df = pd.read_csv(META_FILENAME)
    feats = torch.Tensor(df.loc[:, "Category.Shoes": "ToeStyle.Medallion"].to_numpy())
    num_samples = len(feats)

    res = np.empty(num_samples)

    # Build the model architecture, then print it to console.
    model = config.init_obj("arch", module_arch)
    print(model)

    # Load the checkpoint.
    if config.resume:
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
        for sample_idx in tqdm(range(num_samples)):
            feat = feats[sample_idx].to(device)
            value = model.predict(feat)
            res[sample_idx] = value

    np.save("/mnt/zhxie_hdd/results/offlineCRS/relative_predict/predict/open", res)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Predict.")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args, enable_log=False)

    # Let"s get started.
    main(config)
