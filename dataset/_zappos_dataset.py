import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

SPLITS = ("train", "test")
CATEGORIES = ("open", "pointy", "sporty", "comfort")
FEAT_TYPES = ("meta", "color", "gist", "color+gist", "gist+color")
SPLIT_IDX = 0 # Choose 1 out of 10 train-test split.
LABEL_MAP = {1: 1.0, 2: 0.0, 3: 0.5}

class ZapposDataset(Dataset):
    def __init__(self, dirname, category, feat_type, is_include_equal=True, split="train"):
        # Check argument values.
        assert category.lower() in CATEGORIES
        assert feat_type.lower() in FEAT_TYPES
        assert split.lower() in SPLITS

        # Define paths to retrieve data.
        self._anno_dirname = os.path.join(dirname, "ut-zap50k-data")
        self._feat_dirname = os.path.join(dirname, "ut-zap50k-feats")
        self._meta_filename = os.path.join(self._anno_dirname, "meta-data-bin.csv")
        self._pair_filename = os.path.join(self._anno_dirname, "train-test-splits-pairs.mat")
        self._color_filename = os.path.join(self._feat_dirname, "zappos-color.mat")
        self._gist_filename = os.path.join(self._feat_dirname, "zappos-gist.mat")

        # Define other arguments.
        self._category_idx = CATEGORIES.index(category)
        self._is_include_equal = is_include_equal
        self._feat_type = feat_type
        self._split = split

        # Retrieve data.
        self._pairs = self._get_all_pairs()
        self._feats = self._get_feats()

    def __getitem__(self, index):
        first_data_idx, second_data_idx, label = self._pairs[index]
        return (self._feats[first_data_idx], self._feats[second_data_idx]), torch.Tensor([label])

    def __len__(self):
        return len(self._pairs)

    def _get_feats(self):
        if self._feat_type == "meta":
            return self._get_meta_feats()
        if self._feat_type == "gist":
            return self._get_gist_feats()
        if self._feat_type == "color":
            return self._get_color_feats()
        if self._feat_type == "gist+color" or "color+gist":
            return torch.cat((self._get_gist_feats(), self._get_color_feats()), 1)

    def _get_meta_feats(self):
        df = pd.read_csv(self._meta_filename)
        feats = df.loc[:, "Category.Shoes": "ToeStyle.Medallion"].to_numpy()

        return torch.Tensor(feats)

    def _get_gist_feats(self):
        gist_mat = sio.loadmat(self._gist_filename)["gistfeats"]
        return torch.Tensor(gist_mat)

    def _get_color_feats(self):
        color_mat = sio.loadmat(self._color_filename)["colorfeats"]
        return torch.Tensor(color_mat)

    def _get_all_pairs(self):
        pair_mat = sio.loadmat(self._pair_filename)
        ndarray = None

        if self._split == "train":
            ndarray = pair_mat["trainPairsAll"].flatten()[self._category_idx].flatten()[SPLIT_IDX]
        elif self._split == "test":
            ndarray = pair_mat["testPairsAll"].flatten()[self._category_idx].flatten()[SPLIT_IDX]

        ndarray = ndarray.astype(np.int32)
        if not self._is_include_equal:
            ids = np.where(ndarray[:, 3] != 3)[0]
            ndarray = ndarray[ids]

        # Notice that the indices need to be subtracted by one.
        all_pairs = [(ndarray[idx, 0] - 1, ndarray[idx, 1] - 1, LABEL_MAP[ndarray[idx, 3]]) for idx in range(len(ndarray))]

        return all_pairs

class ZapposDataloader(DataLoader):
    def __init__(self, dirname, category, feat_type, batch_size, shuffle, num_workers, validation_split=0.1, split="train"):
        self._dataset = ZapposDataset(dirname, category, feat_type, split=split)
        self._num_samples = len(self._dataset)
        self._shuffle = shuffle

        self._train_sampler, self._valid_sampler = self._split_sampler(validation_split)
        self._init_kwargs = {
            "dataset": self._dataset,
            "batch_size": batch_size,
            "shuffle": self._shuffle,
            "num_workers": num_workers
        }
        super().__init__(sampler=self._train_sampler, **self._init_kwargs)

    def _split_sampler(self, validation_split):
        if validation_split == 0.0:
            return None, None
        assert 0.0 < validation_split <= 1.0

        idxs = np.arange(self._num_samples)
        np.random.seed(0)
        np.random.shuffle(idxs)
        num_valid_idxs = int(self._num_samples * validation_split)

        valid_idx = idxs[0:num_valid_idxs]
        train_idx = np.delete(idxs, np.arange(0, num_valid_idxs))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Turn off the shuffle option which is mutually exclusive with sampler.
        self._shuffle = False
        self._num_samples = len(train_idx)

        return train_sampler, valid_sampler

    def get_valid_dataloader(self):
        if self._valid_sampler is None:
            return None

        return DataLoader(sampler=self._valid_sampler, **self._init_kwargs)

if __name__ == "__main__":
    test = ZapposDataset(
        dirname="/mnt/zhxie_hdd/dataset/ut-zap50k",
        category="open",
        feat_type="gist",
        pin=False,
        is_include_equal=True,
        split="test"
    )

    print(test[5])