import os
import random
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from models.KGRS import KGRS
from models.utils.parser import parse_args
from models.utils.evaluate import evaluate
from models.utils.dataset import DatasetKGRS, TestDataset
from models.utils.log_helper import create_log_id, logging_config

cores = 32
# cores = multiprocessing.cpu_count() // 2

test_set = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        num_workers=4,
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

    # >> train
    epochs = args.epochs
    for epoch in range(epochs):
        tot_sum, kg_sum, cf_sum = 0., 0., 0.
        steps = 0
        model.train()
        for d in train_loader:
            # load data
            item_batch, user_batch, relation_batch, tail_batch = d

            item_batch = item_batch.to(device).long()
            user_batch = user_batch.to(device).long()
            relation_batch = relation_batch.to(device).long()
            tail_batch = tail_batch.to(device).long()

            optimizer.zero_grad()
            tot_loss, kg_loss, cf_loss = model(
                mode="cal_loss",
                input_i=item_batch,
                input_iu=user_batch,
                input_hr=relation_batch,
                input_ht=tail_batch
            )
            tot_loss.backward()
            optimizer.step()
            # record loss on this batch
            tot_sum += tot_loss.item()
            kg_sum += kg_loss
            cf_sum += cf_loss
            steps += 1
        # <<< train_loader
        avg_train_loss, avg_kg_loss, avg_cf_loss = tot_sum / steps, kg_sum / steps, cf_sum / steps
        logging.info(
            f"epoch {epoch + 1} total loss={avg_train_loss: .2f} KG loss={avg_kg_loss :.2f} CF loss={avg_cf_loss :.2f}")
    # << train

    # >> test
    # test data
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    relation_test, tail_test = dataset.prepare_test()
    relation_test = relation_test.to(device)
    tail_test = tail_test.to(device)

    test_set = TestDataset(args, whether_test=True, item_ids=item_ids, relation_test=relation_test, tail_test=tail_test)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size,
                             num_workers=4, pin_memory=True, shuffle=False)

    Ks = eval(args.Ks)
    model.eval()
    with torch.no_grad():
        logging.info(f"epoch {epoch + 1} evaluate..")
        # val_result = evaluate(
        #     model=model,
        #     users_to_test=np.array(list(test_set.test_user_dict.keys())),
        #     item_test=test_set.item_ids,
        #     r_test=relation_test,
        #     t_test=tail_test,
        #     K=[10, 20, 40]
        # )
        # logging.info(val_result)

        recall_ks, ndcg_ks = evaluate(
            model=model,
            test_set=test_set, test_loader=test_loader, K=Ks
        )
        for i, k in enumerate(Ks):
            logging.info(f"epoch {epoch + 1} Recall @{k}={recall_ks[i].item() :.4f}")
        for i, k in enumerate(Ks):
            logging.info(f"epoch {epoch + 1} NDCG   @{k}={ndcg_ks[i].item() :.4f}")
    # << test


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    set_seed(args.seed)

    # create log
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    train_and_test(args)
