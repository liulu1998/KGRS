import os
import numpy as np
import pandas as pd


def preprocess():
    train_file = "train.txt"
    # test_file = "test.txt"
    kg_file = "kg_final.txt"

    n_items = 0

    train_set = {}
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) == 0:
                continue
            line = l.strip()
            items = [int(i) for i in line.split(' ')]
            user, items = items[0], items[1:]

            n_items = max(n_items, max(items))

            for item in items:
                if item in train_set:
                    train_set[item].append(user)
                else:
                    train_set[item] = [user]
    # <<<
    # 从 0 开始
    n_items += 1

    print(n_items)

    # filter KG
    kg = pd.read_csv(kg_file, sep=' ', header=None)
    n_tuples = len(kg)

    heads = np.array(kg[0], dtype=np.int32)
    relations = np.array(kg[1], dtype=np.int32)
    tails = np.array(kg[2], dtype=np.int32)

    n_e, n_r = n_items, 0
    entity2id, relation2id = {}, {}

    n_preserved_tuples = 0

    # 如此处理的 kg_final3, 元组数少于 kg_final2, 但 kg_final2 是无向图
    with open("kg_final3.txt", 'w') as f:
        for i in range(n_tuples):
            h, r, t = heads[i], relations[i], tails[i]
            # train_set 中只有 items, 不含无法映射的其他 entity
            if h not in train_set:
                continue

            # t 不能映射为 item, ID 重排序, 从 n_items 开始排序
            if t in train_set:
                entity2id[t] = t
            elif t not in train_set and t not in entity2id:
                entity2id[t] = n_e
                n_e += 1

            if r not in relation2id:
                relation2id[r] = n_r
                n_r += 1

            n_preserved_tuples += 1
            f.write(f"{h} {relation2id[r]} {entity2id[t]}\n")

    print(f"n_e: {n_e}, n_r: {n_r}")
    print(f"n_raw_tuples: {n_tuples} n_preserved_tuples: {n_preserved_tuples}")


if __name__ == '__main__':
    preprocess()
