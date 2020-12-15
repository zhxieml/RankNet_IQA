import csv
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class IQADataset(Dataset):
    def __init__(self, dirname, type):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Training")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Training", "score")
        self._type = type.lower()
        self._transform = transforms.Compose([
            # transforms.Resize((3648, 2736)),
            transforms.Resize((400, 300)),
            # transforms.Resize((400, 300)),
            transforms.ToTensor()
        ])

        # Split the dataset by groups, and then merge together.
        self._groups = os.listdir(self._data_dirname)
        self._imgs = self._get_all_imgs()

    def __getitem__(self, index):
        img_filename, score = self._imgs[index]
        img = Image.open(img_filename)

        # Apply the transformation if needed.
        if self._transform is not None:
            img = self._transform(img)

        return img, torch.Tensor([score])

    def __len__(self):
        return len(self._imgs)

    def _get_imgs_by_group(self, group):
        img_dirname = os.path.join(self._data_dirname, group)
        label_filename = os.path.join(self._label_dirname, "{}_score.csv".format(group))
        imgs_by_group = []

        # Extract the rank.
        with open(label_filename, "r") as label_file:
            csv_reader = csv.reader(label_file)
            fields = [field.lower() for field in next(csv_reader)]
            type_index = fields.index(self._type)

            for record in csv_reader:
                img_filename = os.path.join(img_dirname, record[0])
                img_score = float(record[type_index])
                imgs_by_group.append((img_filename, img_score))

        return imgs_by_group

    def _get_all_imgs(self):
        all_imgs = []

        for group in self._groups:
            imgs_by_group = self._get_imgs_by_group(group)
            all_imgs += imgs_by_group

        return all_imgs

class IQADataloader(DataLoader):
    def __init__(self, dirname, type, batch_size, shuffle, num_workers):
        super().__init__(IQADataset(dirname, type), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class RankIQADataset(Dataset):
    def __init__(self, dirname, type):
        assert type.lower() in ["color", "exposure", "noise", "texture"]

        self._data_dirname = os.path.join(dirname, "Training")
        self._label_dirname = os.path.join(dirname, "score_and_sort", "Training", "sort")
        self._type = type.lower()
        self._transform = transforms.Compose([
            transforms.Resize((400, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4312, 0.4158, 0.3733),
                std=(0.2344, 0.2295, 0.2448)
            )
        ])

        # Split the dataset by groups, and then merge together.
        self._groups = os.listdir(self._data_dirname)
        self._pairs = self._get_all_pairs()

    def __getitem__(self, index):
        first_img_filename, second_img_filename, prob = self._pairs[index]
        win_img, lose_img = Image.open(first_img_filename), Image.open(second_img_filename)

        # Apply the transformation if needed.
        if self._transform is not None:
            win_img, lose_img = self._transform(win_img), self._transform(lose_img)

        return win_img, lose_img, torch.Tensor([prob])

    def __len__(self):
        return len(self._pairs)

    def _get_pairs_by_group(self, group):
        img_dirname = os.path.join(self._data_dirname, group)
        label_filename = os.path.join(self._label_dirname, "{}.csv".format(group))
        img_filename_to_rank = {} # A map from image files to their corresponding ranks.

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

        for first_img_filename in img_filenames:
            for second_img_filename in img_filenames:
                if first_img_filename == second_img_filename: pass

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

    def _get_all_pairs(self):
        all_pairs = []

        for group in self._groups:
            pairs_by_group = self._get_pairs_by_group(group)
            all_pairs += pairs_by_group

        return all_pairs

class RankIQADataloader(DataLoader):
    def __init__(self, dirname, type, batch_size, shuffle, num_workers):
        super().__init__(RankIQADataset(dirname, type), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)