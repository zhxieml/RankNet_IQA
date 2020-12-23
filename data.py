from dataset import IQADataloader
from utils import cal_mean_and_std

INPUT_FOLDER = "/mnt/zhxie_hdd/dataset/IQA"

if __name__ == "__main__":
    dataloader = IQADataloader(INPUT_FOLDER, "color", 8, False, 8)
    mean, std = cal_mean_and_std(dataloader)
    print(mean, std)
    pass