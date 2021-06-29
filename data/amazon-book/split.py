import random
from collections import defaultdict
import numpy as np


# 验证集在训练集中的比例
valset_ratio = 0.1

seed = 2021
random.seed(seed)
np.random.seed(seed)


def flatten_train_file(train_file):
    """
    把 u i1 i2 i5 转成 u i1 \\ u i2 \\ u i5
    :param train_file:
    """
    inter = []

    with open(train_file, 'r') as f:
        for l in f.readlines():
            line = [int(i) for i in l.strip().split()]
            u, items = line[0], line[1:]
            if items is []:
                continue
            for item in items:
                inter.append([u, item])

    return inter


def utilize_interactions(inters, filename):
    d = defaultdict(list)
    for i in range(len(inters)):
        u, item = inters[i][0], inters[i][1]
        d[u].append(item)

    with open(filename, 'w', encoding='utf-8') as f:
        for u in sorted(d.keys()):
            items = sorted(list(set(d[u])))
            if len(items) == 0:
                continue
            f.write(f"{u} " + ' '.join(map(str, items)) + '\n')


def split(train_file, train2_file, val_file):
    train = defaultdict(list)
    val = defaultdict(list)

    with open(train_file, 'r') as f:
        for l in f.readlines():
            line = list(map(int, l.strip().split()))
            u, items = line[0], list(set(line[1:]))
            if len(items) == 0:
                continue

            n_items = len(items)
            items = np.array(items)
            n_val = max(int(n_items * valset_ratio), 1)
            # TODO 选 n_val 个放入 val set

            indices = np.random.choice(list(range(n_items)), size=n_val, replace=False)
            left = list(set(range(n_items)) - set(indices))

            val_set = items[indices]
            train_set = items[left]

            val[u] = sorted(val_set.tolist())
            train[u] = sorted(train_set.tolist())

    with open(train2_file, 'w', encoding='utf-8') as f:
        for u in sorted(train.keys()):
            items = train[u]
            f.write(f"{u} " + ' '.join(map(str, items)) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for u in sorted(val.keys()):
            items = val[u]
            f.write(f"{u} " + ' '.join(map(str, items)) + '\n')


if __name__ == '__main__':
    split("train.txt", "train2.txt", "val.txt")
