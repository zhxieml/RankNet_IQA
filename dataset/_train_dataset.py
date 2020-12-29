import csv
import itertools
import multiprocessing
import os

import torch
from torch.utils.data import Dataset, DataLoader

from utils.data import process_img

class TrainDataset(Dataset):
    def __init__(self, dirname, type, pin):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Training")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Training", "sort")
        self._type = type.lower()
        self._pin = pin

        # Merge groups of data together.
        groups = os.listdir(self._data_dirname)
        self._pairs, self._img_filenames = self._get_all_pairs(groups)
        self._img_filenames_to_tensor = {}

        # If set pin, prefetch them.
        if self._pin:
            pool = multiprocessing.Pool(processes=16)
            self._img_filenames_to_tensor = {img_filename: pool.apply_async(process_img, (img_filename,)) for img_filename in self._img_filenames}

            pool.close()
            pool.join()

    def __getitem__(self, index):
        first_img_filename, second_img_filename, prob = self._pairs[index]

        if self._pin:
            first_img, second_img = self._img_filenames_to_tensor[first_img_filename].get(), self._img_filenames_to_tensor[second_img_filename].get()
        else:
            first_img, second_img = process_img(first_img_filename), process_img(second_img_filename)

        return (first_img, second_img), torch.Tensor([prob])

    def __len__(self):
        return len(self._pairs)

    def _get_pairs_by_group(self, group):
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

        # The returned data are organized in a list of paired images.
        img_filenames_by_group = list(img_filename_to_rank.keys())
        pairs_by_group = []

        for first_img_filename, second_img_filename in itertools.permutations(img_filenames_by_group, 2):
            # Case 0: tie.
            if img_filename_to_rank[first_img_filename] == img_filename_to_rank[second_img_filename]:
                pairs_by_group.append((first_img_filename, second_img_filename, 0.5))
            # Case 1: win.
            elif img_filename_to_rank[first_img_filename] < img_filename_to_rank[second_img_filename]:
                pairs_by_group.append((first_img_filename, second_img_filename, 1.0))
            # Case 2: lose.
            else:
                pairs_by_group.append((first_img_filename, second_img_filename, 0.0))

        return pairs_by_group, img_filenames_by_group

    def _get_all_pairs(self, groups):
        all_pairs = []
        all_img_filenames = []

        for group in groups:
            pairs_by_group, img_filenames_by_group = self._get_pairs_by_group(group)
            all_pairs += pairs_by_group
            all_img_filenames += img_filenames_by_group

        return all_pairs, all_img_filenames

class TrainDataloader(DataLoader):
    def __init__(self, dirname, type, batch_size, shuffle, num_workers, pin_memory):
        super().__init__(TrainDataset(dirname, type, pin=pin_memory), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=not pin_memory)

if __name__ == "__main__":
    dataset = TrainDataset("/mnt/zhxie_hdd/dataset/IQA", type="color", pin=True)
    pass