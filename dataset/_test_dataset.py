import csv
import multiprocessing
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils.data import process_img

class TestDataset(Dataset):
    def __init__(self, dirname, type, pin):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Test")
        self._type = type.lower()
        self._pin = pin

        # Merge groups of data together.
        groups = os.listdir(self._data_dirname)
        self._img_filenames = self._get_all_img_filenames(groups)
        self._img_filenames_to_tensor = {}

        # If set pin, prefetch them.
        if self._pin:
            pool = multiprocessing.Pool(processes=16)
            self._img_filenames_to_tensor = {img_filename: pool.apply_async(process_img, (img_filename,)) for img_filename in self._img_filenames}

            pool.close()
            pool.join()

    def __getitem__(self, index):
        img_filename = self._img_filenames[index]

        if self._pin:
            img = self._img_filenames_to_tensor[img_filename].get()
        else:
            img = process_img(img_filename)

        return img, img_filename

    def __len__(self):
        return len(self._img_filenames)

    def _get_img_filenames_by_group(self, group):
        img_dirname = os.path.join(self._data_dirname, group)
        img_filenames = [os.path.join(img_dirname, img_name) for img_name in os.listdir(img_dirname)]
        img_filenames = sorted(img_filenames)

        return img_filenames

    def _get_all_img_filenames(self, groups):
        all_img_filenames = []

        for group in groups:
            img_filenames_by_group = self._get_img_filenames_by_group(group)
            all_img_filenames += img_filenames_by_group

        return all_img_filenames

class TestDataloader(DataLoader):
    def __init__(self, dirname, type, num_workers, pin_memory):
        super().__init__(TestDataset(dirname, type, pin=pin_memory), batch_size=15, shuffle=False, num_workers=num_workers, pin_memory=not pin_memory)
