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
from models.utils.helper import early_stopping
from models.utils.dataset import DatasetKGRS, TestDataset
from models.utils.log_helper import create_log_id, logging_config
from main import set_seed


def tune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train data
    dataset = DatasetKGRS(args, train_file="train2.txt")

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
    # >> test
    # test data
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    relation_test, tail_test = dataset.prepare_test()
    relation_test = relation_test.to(device)
    tail_test = tail_test.to(device)

    val_set = TestDataset(args, whether_test=False, item_ids=item_ids, relation_test=relation_test, tail_test=tail_test)
    val_loader = DataLoader(dataset=val_set, batch_size=args.test_batch_size,
                             num_workers=4, pin_memory=True, shuffle=False)

    Ks = eval(args.Ks)

    # >> train
    best_ndcg, best_val_epoch = 0., 0
    patience = args.patience
    epochs = args.epochs

    for epoch in range(epochs):
        tot_sum, kg_sum, cf_sum = 0., 0., 0.
        steps = 0
        model.train()
        for i, d in enumerate(train_loader):
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
            # 梯度归一化
            tot_loss = tot_loss / item_batch.shape[0]
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
        # << train_loader

        if (epoch + 1) % args.evaluate_every == 0:
            model.eval()
            with torch.no_grad():
                recall_ks, ndcg_ks = evaluate(
                    model=model,
                    test_set=val_set, test_loader=val_loader, Ks=Ks
                )
            ndcg = ndcg_ks[0].item()

            for i, k in enumerate(Ks):
                logging.info(f"epoch {epoch + 1} Recall @{k}={recall_ks[i].item() :.4f}")
            for i, k in enumerate(Ks):
                logging.info(f"epoch {epoch + 1} NDCG   @{k}={ndcg_ks[i].item() :.4f}")

            # early stop
            if ndcg > best_ndcg:
                patience = args.patience
                best_val_epoch = epoch + 1
                best_ndcg = ndcg
            else:
                patience -= 1

            if patience <= 0:
                logging.info(f"early stop at epoch {epoch + 1}")
                break
        # << eval
    # << trian and eval
    logging.info(f"best epoch: {best_val_epoch}, with NDCG={best_ndcg: .4f}")


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
    set_seed(args.seed)

    # create log
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    logging.info(args.m)

    tune(args)
