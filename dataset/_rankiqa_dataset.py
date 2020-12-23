import csv
import itertools
import multiprocessing
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class RankIQADataset(Dataset):
    def __init__(self, dirname, type, pin):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Training")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Training", "sort")
        self._type = type.lower()
        self._pin = pin
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
        self._pairs = self._get_all_data(groups)

        # If set pin, prefetch them.
        if self._pin:
            pool = multiprocessing.Pool(processes=16)
            multiple_results = [pool.apply_async(self._process_data, (pair,)) for pair in tqdm(self._pairs)]

            pool.close()
            pool.join()

            for res in multiple_results:
                print(res.get())

    def __getitem__(self, index):
        if self._pin:
            return self._pairs[index]
        else:
            return self._process_data(self._pairs[index])

    def __len__(self):
        return len(self._pairs)

    def _get_data_by_group(self, group):
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
        img_filenames = list(img_filename_to_rank.keys())
        pairs_by_group = []

        for first_img_filename, second_img_filename in itertools.permutations(img_filenames, 2):
            # Case 0: tie.
            if img_filename_to_rank[first_img_filename] == img_filename_to_rank[second_img_filename]:
                pairs_by_group.append((first_img_filename, second_img_filename, 0.5))
            # Case 1: win.
            elif img_filename_to_rank[first_img_filename] < img_filename_to_rank[second_img_filename]:
                pairs_by_group.append((first_img_filename, second_img_filename, 1.0))
            # Case 2: lose.
            else:
                pairs_by_group.append((first_img_filename, second_img_filename, 0.0))

        return pairs_by_group

    def _get_all_data(self, groups):
        all_pairs = []

        for group in groups:
            pairs_by_group = self._get_data_by_group(group)
            all_pairs += pairs_by_group

        return all_pairs

    def _process_data(self, pair):
        first_img_filename, second_img_filename, prob = pair
        first_img, second_img = Image.open(first_img_filename), Image.open(second_img_filename)

        # Apply the transformation if needed.
        if self._transform is not None:
            first_img, second_img = self._transform(first_img), self._transform(second_img)

        print("Done")
        return (first_img, second_img), torch.Tensor([prob])

class RankIQADataloader(DataLoader):
    def __init__(self, dirname, type, batch_size, shuffle, num_workers, pin_memory):
        super().__init__(RankIQADataset(dirname, type, pin=pin_memory), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

if __name__ == "__main__":
    dataset = RankIQADataset("/mnt/zhxie_hdd/dataset/IQA", type="color", pin=True)
    pass