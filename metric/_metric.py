import numpy as np

def srcc(scores, rank_gt):
    scores, rank_gt = scores.cpu().numpy()[:, 0], rank_gt.cpu().numpy()[:, 0]
    assert len(scores) == len(rank_gt)
    num_img = len(scores)

    sorted_idxs = np.argsort(scores)[::-1]
    rank = np.empty(num_img)
    rank[sorted_idxs] = np.arange(num_img) + 1

    diff = rank - rank_gt
    res = 1 - 6 * np.dot(diff, diff) / (num_img ** 3 - num_img)

    return res

if __name__ == "__main__":
    pass