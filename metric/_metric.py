import numpy as np

def get_rank(scores):
    num = len(scores)

    sorted_idxs = np.argsort(scores)[::-1]
    rank = np.empty(num)
    rank[sorted_idxs] = np.arange(num) + 1

    return rank

def srcc(scores, rank_gt):
    num_samples = len(scores)
    assert num_samples == len(rank_gt)

    rank = get_rank(scores)
    diff = rank - rank_gt
    res = 1 - 6 * np.dot(diff, diff) / (num_samples ** 3 - num_samples)

    return res

def accuracy(output, target):
    num_samples = len(output)
    assert num_samples == len(target)

    pred = (output > 0.5).astype(float)
    correct = np.sum(pred == target)

    return correct / num_samples

if __name__ == "__main__":
    pass