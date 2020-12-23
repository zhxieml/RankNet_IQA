import torch

def srcc(output, target):
    num = output.shape[0]

    with torch.no_grad():
        output = torch.argsort(output, dim=0, descending=True).float() + 1.0
        diff = output - target
        res = 1 - 6 * torch.dot(diff[:, 0], diff[:, 0]) / (num ** 3 - num)

    return res

def accuracy(output, target):
    with torch.no_grad():
        pred = (output > 0.5).float()
        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)

if __name__ == "__main__":
    pass