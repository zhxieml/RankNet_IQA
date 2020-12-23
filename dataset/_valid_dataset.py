import csv
import itertools
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ValidDataset(Dataset):
    def __init__(self, dirname, type):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Validation")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Validation", "sort")
        self._type = type.lower()
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4312, 0.4158, 0.3733),
                std=(0.2344, 0.2295, 0.2448)
            )
        ])

        # Merge groups of data together.
        groups = os.listdir(self._data_dirname)
        self._ranks = self._get_all_ranks(groups)

    def __getitem__(self, index):
        img_filename, rank = self._ranks[index]
        img = Image.open(img_filename)

        # Apply the transformation if needed.
        if self._transform is not None:
            img = self._transform(img)

        return img, torch.Tensor([rank])

    def __len__(self):
        return len(self._ranks)

    def _get_ranks_by_group(self, group):
        img_dirname = os.path.join(self._data_dirname, group)
        label_filename = os.path.join(self._label_dirname, "{}.csv".format(group))
        img_filename_to_rank = {} # A map from image files to their corresponding rank.

        # Extract the rank.
        with open(label_filename, "r") as label_file:
            csv_reader = csv.reader(label_file)
            fields = [field.lower() for field in next(csv_reader)]
            type_index = fields.index(self._type)

            for record in csv_reader:
                img_filename = os.path.join(img_dirname, record[0])
                img_filename_to_rank[img_filename] = float(record[type_index])

        # The returned data are organized in a list of (img_filename, rank) pair.
        return list(img_filename_to_rank.items())

    def _get_all_ranks(self, groups):
        all_ranks = []

        for group in groups:
            ranks_by_group = self._get_ranks_by_group(group)
            all_ranks += ranks_by_group

        return all_ranks

class ValidDataloader(DataLoader):
    def __init__(self, dirname, type, num_workers, pin_memory):
        super().__init__(ValidDataset(dirname, type), batch_size=15, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

if __name__ == "__main__":
    dataset = ValidDataset("/mnt/zhxie_hdd/dataset/IQA", type="color")
    dataloader = ValidDataloader("/mnt/zhxie_hdd/dataset/IQA", type="color", num_workers=8)
    pass