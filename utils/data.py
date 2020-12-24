from PIL import Image
from torchvision import transforms

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
            mean=(0.4312, 0.4158, 0.3733),
            std=(0.2344, 0.2295, 0.2448)
        )
    ])

    return transform(img)