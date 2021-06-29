import os
import heapq
import random
import logging
import multiprocessing

import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from models.KGRS import KGRS
from models.utils import metrics
from models.utils.parser import parse_args
from models.utils.dataset import DatasetKGRS, TestDataset
from models.utils.log_helper import create_log_id, logging_config

cores = multiprocessing.cpu_count() // 2

test_set = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def test_one_user(x):
    global test_set
    Ks = [10, 20, 40]
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    try:
        # user u's items in the training set
        training_items = test_set.train_user_dict[u]
    except:
        training_items = []

    # user u's items in the test set
    user_pos_test = test_set.test_user_dict[u]

    n_items = len(test_set.item_ids)
    all_items = set(range(n_items))

    test_items = list(all_items - set(training_items))

    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    # Top-K items for this user
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    recall_list, ndcg_list = [], []

    for K in Ks:
        recall_list.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg_list.append(metrics.ndcg_at_k(r, K))

    return {'recall': np.array(recall_list), 'ndcg': np.array(ndcg_list)}


def evaluate(model, test_set: TestDataset, test_loader: DataLoader, K=(10, 20, 40)):
    """
    :param model:
    :param test_set:
    :param test_loader:
    :param K: Top-K
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {'recall': np.zeros(len(K)), 'ndcg': np.zeros(len(K))}

    n_test_users = test_set.n_users

    pool = multiprocessing.Pool(cores)

    for user_batch in test_loader:
        user_batch = user_batch.to(device)

        rate_batch = model(
            "predict",
            r_test=test_set.relation_test,
            t_test=test_set.tail_test,
            pos_items=test_set.item_ids,
            users=user_batch
        ).squeeze_().cpu().numpy()

        user_batch = user_batch.cpu().numpy()
        user_batch_rating_uid = zip(rate_batch, user_batch)

        # [dict]
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        for re in batch_result:
            res['recall'] += re['recall'] / n_test_users
            res['ndcg'] += re['ndcg'] / n_test_users

    pool.close()
    return res


def train_and_test(args):
    global test_set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train data
    dataset = DatasetKGRS(args, train_file="train.txt")

    n_users, n_items, n_relations, n_entities, max_i_u, max_i_r, negative_c, negative_ck \
        = dataset.stat()

    negative_c = negative_c.to(device)
    negative_ck = None

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    n_heads = args.heads

    model = KGRS(
        n_users=n_users, n_items=n_items, n_relations=n_relations, n_entities=n_entities,
        max_i_u=max_i_u, max_i_r=max_i_r, negative_c=negative_c, negative_ck=negative_ck,
        emb_size=args.embed_size, attention_size=args.attention_size,
        dropout_kg=args.dropout_kg, dropout_cf=args.dropout_cf,
        weight_task_kg=args.weight_task_kg, weight_L2_kg=args.weight_L2_kg,
        n_heads=n_heads, aggregator_type=None
    ).to(device)

    optimizer = optim.Adagrad(
        model.parameters(),
        lr=args.lr,
        initial_accumulator_value=1e-8
    )

    # lr decay
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.6)

    # train
    epochs = args.epochs

    for epoch in range(epochs):
        tot_sum = 0.
        kg_sum = 0.
        cf_sum = 0.
        steps = 0

        for d in train_loader:
            # load data
            item_batch, user_batch, relation_batch, tail_batch = d

            item_batch = item_batch.to(device).long()
            user_batch = user_batch.to(device).long()
            relation_batch = relation_batch.to(device).long()
            tail_batch = tail_batch.to(device).long()

            optimizer.zero_grad()
            tot_loss, kg_loss, cf_loss = model(
                "cal_loss",
                input_i=item_batch,
                input_iu=user_batch,
                input_hr=relation_batch,
                input_ht=tail_batch
            )
            tot_loss.backward()
            optimizer.step()
            # record loss on this batch
            tot_sum += tot_loss
            kg_sum += kg_loss
            cf_sum += cf_loss
            steps += 1
        # <<< train_loader
        avg_train_loss, avg_kg_loss, avg_cf_loss = tot_sum / steps, kg_sum / steps, cf_sum / steps
        logging.info(f"epoch {epoch + 1} total loss={avg_train_loss: .2f} KG loss={avg_kg_loss :.2f} CF loss={avg_cf_loss :.2f}")
    # <<< train

    # test
    model.eval()
    relation_test, tail_test = dataset.prepare_test()
    relation_test = relation_test.to(device)
    tail_test = tail_test.to(device)

    item_ids = torch.arange(n_items, dtype=torch.long).to(device)
    test_set = TestDataset(args=args, whether_test=True,
                           item_ids=item_ids, relation_test=relation_test, tail_test=tail_test)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,
                             num_workers=1, pin_memory=True, shuffle=False)
    K = eval(args.Ks)

    with torch.no_grad():
        res = evaluate(
            model=model,
            test_set=test_set,
            test_loader=test_loader,
            K=K
        )

    logging.info(f"test results:")
    msg = "Recall"
    for k, r in zip(K, res["recall"]):
        msg += f" @{k}={r: .4f}"
    logging.info(msg)

    msg = "NDCG  "
    for k, n in zip(K, res["ndcg"]):
        msg += f" @{k}={n: .4f}"
    logging.info(msg)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    set_seed(args.seed)

    # create log
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    train_and_test(args)
