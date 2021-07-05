import os
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def load_data(args, data_dir, train_file: str = "train.txt"):
    train_file = os.path.join(data_dir, train_file)
    val_file = os.path.join(data_dir, 'val.txt')
    test_file = os.path.join(data_dir, 'test.txt')

    # number of users, number of items
    n_users, n_items = 0, 0
    # number of user-item interactions
    train_len = 0

    # count n_users, n_items, train_len
    with open(train_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                # current user id
                uid = int(l[0])
                n_items = max(n_items, max(items))
                n_users = max(n_users, uid)
                train_len += len(items)

    with open(val_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                n_items = max(n_items, max(items))

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')[1:]]
                except Exception:
                    continue
                n_items = max(n_items, max(items))
    n_items += 1
    n_users += 1

    kg_file = "kg_final3.txt"
    kg = pd.read_csv(os.path.join(data_dir, kg_file), sep=' ', header=None)
    n_facts = len(kg)

    # h, r, t
    heads = np.array(kg[0], dtype=np.int32)
    relations = np.array(kg[1], dtype=np.int32)
    tails = np.array(kg[2], dtype=np.int32)

    # item ID -> [user1, user2, ...]
    train_set = defaultdict(list)

    with open(train_file) as f_train:
        for l in f_train.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            # user ID, items
            uid, train_items = items[0], items[1:]
            for i in train_items:
                # i is item
                train_set[i].append(uid)

    # relation_set: head entity -> [r1, r2 ...]
    # tail_set: head entity -> [t1, t2 ...]
    # relation_set, tail_set = {}, {}
    relation_set, tail_set = defaultdict(list), defaultdict(list)

    for i in range(n_facts):
        h, r, t = heads[i], relations[i], tails[i]
        relation_set[h].append(r)
        tail_set[h].append(t)

    n_entities = max(np.max(heads), np.max(tails)) + 1
    n_relations = np.max(relations) + 1

    relation_len = n_facts

    """
    c0 and c1 determine the overall weight of non-observed instances in implicit feedback data.
    Specifically, c0 is for the recommendation task and c1 is for the knowledge embedding task.
    """
    negative_c, negative_ck = calculate_weight(
        args.c0, args.c1, args.p, train_set, train_len, relation_set, relation_len, n_items
    )

    # align
    train_set, relation_set, tail_set, max_user_pi, max_relation_pi = \
        build_train_set(train_set, n_users, relation_set, tail_set, n_relations, n_entities)

    return n_users, n_items, n_relations, n_entities, train_set, relation_set, tail_set, \
           max_user_pi, max_relation_pi, negative_c, negative_ck


def get_train_instances(train_set, relation_set, tail_set):
    """
    Args:
        train_set: dict, item -> [user1, user2 ...]
        tail_set: dict, item(entity) -> [t1, t2 ...]
        relation_set: dict, item(entity) -> [r1, r2 ...]
    """
    item_train, user_train, relation_train1, tail_train1 = [], [], [], []

    for e in relation_set:
        # i is item
        if e in train_set:
            item_train.append(e)
            # train_set[i] is a list
            user_train.append(train_set[e])
            relation_train1.append(relation_set[e])
            tail_train1.append(tail_set[e])
        # <<< if
    # <<< for

    item_train = np.array(item_train)
    item_train = item_train[:, np.newaxis]

    item_train = torch.from_numpy(item_train).long()
    user_train = torch.LongTensor(user_train)
    relation_train1 = torch.LongTensor(relation_train1)
    tail_train1 = torch.LongTensor(tail_train1)

    # they're all 2D Tensors
    return item_train, user_train, relation_train1, tail_train1


def calculate_weight(c0, c1, p, train_set, train_len, relation_train, relation_len, n_items):
    # m = torch.zeros(len(train_set.keys()))
    m = torch.zeros(n_items)

    for i in train_set.keys():
        m[i] = len(train_set[i]) * 1.0 / train_len

    c = torch.zeros(n_items)
    tem = 0
    for i in train_set.keys():
        tem += np.power(m[i], p)
    for i in train_set.keys():
        c[i] = c0 * np.power(m[i], p) / tem

    mk = torch.zeros(n_items)
    for i in relation_train.keys():
        mk[i] = len(relation_train[i]) * 1.0 / relation_len

    ck = torch.zeros(n_items)
    tem = 0
    for i in relation_train.keys():
        tem += np.power(mk[i], p)
    for i in relation_train.keys():
        ck[i] = c1 * np.power(mk[i], p) / tem

    # c = torch.tensor(c, dtype=torch.float)
    # ck = torch.tensor(ck, dtype=torch.float)
    return c, ck


def build_train_set(train_set, n_users, relation_set, tail_set, n_relations, n_entities):
    """
    :param train_set: item ID -> [user0, user3, ...]
    :param n_users:
    :param relation_set: entity ID -> [r0, r3, ...]
    :param tail_set: entity ID -> [t0, t3, ...]
    :param n_relations:
    :param n_entities:
    """
    # number of interactions of each user
    user_length = []

    # train_set : item -> [user1, user2 ...]
    for i in train_set:
        user_length.append(len(train_set[i]))
    user_length.sort()

    # Maximum number of interactions between item and user
    max_user_pi = user_length[int(len(user_length) * 0.9999)]

    # align
    # train_set :dict, item -> [user1, user2 ...]
    for i in train_set:
        if len(train_set[i]) > max_user_pi:
            train_set[i] = train_set[i][0: max_user_pi]
        # padding value is n_users
        while len(train_set[i]) < max_user_pi:
            train_set[i].append(n_users)

    relation_length = []
    for i in relation_set:
        relation_length.append(len(relation_set[i]))
    relation_length.sort()

    # Maximum number of connections between item (entity) and relations
    max_relation_pi = relation_length[int(len(relation_length) * 0.9999)]

    # align
    for i in relation_set:
        if len(relation_set[i]) > max_relation_pi:
            relation_set[i] = relation_set[i][0:max_relation_pi]
            tail_set[i] = tail_set[i][0:max_relation_pi]
        # padding value is n_relations and n_entities
        while len(relation_set[i]) < max_relation_pi:
            relation_set[i].append(n_relations)
            tail_set[i].append(n_entities)

    return train_set, relation_set, tail_set, max_user_pi, max_relation_pi


class DatasetKGRS(Dataset):
    def __init__(self, args, train_file: str):
        super().__init__()
        self.DATA_ROOT = args.data_root
        self.data_name = args.data_name
        self.data_dir = os.path.join(self.DATA_ROOT, self.data_name)

        self.n_users, self.n_items, self.n_relations, self.n_entities, \
        self.train_set, self.relation_set, self.tail_set, self.max_i_u, self.max_i_r, \
        self.negative_c, self.negative_ck = load_data(args, self.data_dir, train_file)

        self.item_train, self.user_train, self.relation_train1, self.tail_train1 = get_train_instances(
            self.train_set, self.relation_set, self.tail_set)

        self.length = len(self.item_train)

    def __getitem__(self, idx: int):
        return self.item_train[idx], self.user_train[idx], \
               self.relation_train1[idx], self.tail_train1[idx]

    def __len__(self) -> int:
        return self.length

    def stat(self):
        return self.n_users, self.n_items, self.n_relations, self.n_entities, \
               self.max_i_u, self.max_i_r, self.negative_c, self.negative_ck

    def prepare_test(self):
        """ build data for evaluation
        """
        item_test = range(self.n_items)

        relation_test, tail_test = [], []
        for item in item_test:
            if len(self.relation_set[item]) == 0:
                relation_test.append([self.n_relations] * self.max_i_r)
                tail_test.append([self.n_entities] * self.max_i_r)
            else:
                relation_test.append(self.relation_set[item])
                tail_test.append(self.tail_set[item])

        relation_test = torch.LongTensor(relation_test)
        tail_test = torch.LongTensor(tail_test)
        return relation_test, tail_test


class TestDataset(Dataset):
    def __init__(self, args, whether_test, item_ids, relation_test, tail_test):
        """
        :param whether_test:
        :param item_ids:
        :param relation_test:
        :param tail_test:
        """
        data_root = args.data_root
        data_name = args.data_name
        self.data_dir = os.path.join(data_root, data_name)
        self.batch_size = args.test_batch_size

        train_file = "train.txt" if whether_test else "train2.txt"
        train_file = os.path.join(self.data_dir, train_file)

        test_file = "test.txt" if whether_test else "val.txt"
        test_file = os.path.join(self.data_dir, test_file)

        self.train_user_dict = self.load_interactions(train_file)
        self.test_user_dict = self.load_interactions(test_file)

        self.relation_test = relation_test
        self.tail_test = tail_test

        self.item_ids = item_ids
        self.n_items = len(self.item_ids)

        # user ids in test set
        user_ids = torch.LongTensor(list(self.test_user_dict.keys()))
        self.user_ids = user_ids
        self.n_users = len(self.user_ids)
        # user_ids_batches = [user_ids[i: i + self.batch_size] for i in
        #                     range(0, len(user_ids), self.batch_size)]
        # user_ids_batches = [d.to(self.device) for d in user_ids_batches]
        #
        # self.user_ids_batches = user_ids_batches

    def __getitem__(self, index):
        return self.user_ids[index]

    def __len__(self):
        return self.n_users

    @staticmethod
    def load_interactions(filename):
        # user -> [item1, item2, ...]
        user_dict = dict()

        with open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tmp = line.strip()
                inter = [int(i) for i in tmp.split(' ')]

                if len(inter) > 1:
                    user_id, item_ids = inter[0], inter[1:]
                    item_ids = list(set(item_ids))
                    user_dict[user_id] = item_ids

        return user_dict
