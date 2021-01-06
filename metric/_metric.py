import numpy as np

def get_rank(scores):
    num = len(scores)

    sorted_idxs = np.argsort(scores)[::-1]
    rank = np.empty(num)
    rank[sorted_idxs] = np.arange(num) + 1

    return rank

def srcc(scores, rank_gt):
    num_img = len(scores)
    assert num_img == len(rank_gt)

    rank = get_rank(scores)
    diff = rank - rank_gt
    res = 1 - 6 * np.dot(diff, diff) / (num_img ** 3 - num_img)

    return res

if __name__ == "__main__":
    pass