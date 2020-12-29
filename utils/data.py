from PIL import Image
from torchvision import transforms

# Means & stds for (224, 224) training data.
MEAN = (0.4224, 0.4010, 0.3602)
STD = (0.2311, 0.2251, 0.2371)

def cal_mean_and_std(dataloader):
    mean, var = 0.0, 0.0
    num_sample = 0.0

    for imgs, _ in dataloader:
        num_sample_batch = imgs.size(0)
        imgs = imgs.view(num_sample_batch, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        var += imgs.var(2).sum(0)

        num_sample += num_sample_batch

    return mean / num_sample, (var / num_sample) ** 0.5

def process_img(img_filename):
    img = Image.open(img_filename)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    return transform(img)
