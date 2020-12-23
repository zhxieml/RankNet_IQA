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
            transforms.Resize((3648, 2736)),
            # transforms.Resize((400, 300)),
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
