import csv
import multiprocessing
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data import process_img

class ValidDataset(Dataset):
    def __init__(self, dirname, type, pin):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Validation")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Validation", "sort")
        self._type = type.lower()
        self._pin = pin

        # Merge groups of data together.
        groups = os.listdir(self._data_dirname)
        self._ranks, self._img_filenames = self._get_all_ranks(groups)
        self._img_filenames_to_tensor = {}

        # If set pin, prefetch them.
        if self._pin:
            pool = multiprocessing.Pool(processes=16)
            self._img_filenames_to_tensor = {img_filename: pool.apply_async(process_img, (img_filename,)) for img_filename in self._img_filenames}

            pool.close()
            pool.join()

    def __getitem__(self, index):
        img_filename, rank = self._ranks[index]

        if self._pin:
            img = self._img_filenames_to_tensor[img_filename].get()
        else:
            img = process_img(img_filename)

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
        return list(img_filename_to_rank.items()), list(img_filename_to_rank.keys())

    def _get_all_ranks(self, groups):
        all_ranks = []
        all_img_filenames = []

        for group in groups:
            ranks_by_group, img_filenames_by_group = self._get_ranks_by_group(group)
            all_ranks += ranks_by_group
            all_img_filenames += img_filenames_by_group

        return all_ranks, all_img_filenames

class ValidDataloader(DataLoader):
    def __init__(self, dirname, type, num_workers, pin_memory):
        super().__init__(ValidDataset(dirname, type, pin=pin_memory), batch_size=15, shuffle=False, num_workers=num_workers, pin_memory=not pin_memory)

if __name__ == "__main__":
    dataset = ValidDataset("/mnt/zhxie_hdd/dataset/IQA", type="color")
    dataloader = ValidDataloader("/mnt/zhxie_hdd/dataset/IQA", type="color", num_workers=8)
    pass