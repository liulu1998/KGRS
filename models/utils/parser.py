import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Run KGRS")

    parser.add_argument('--data_name', nargs='?', default='amazon-book',
                        help='Choose a dataset from {amazon-book, yelp2018}')
    # parser.add_argument('--pretrain', type=int, default=0, help="whether use pretrained model")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--heads', type=int, default=1, help="number of heads in GAT")
    parser.add_argument('--evaluate_every', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--attention_size', type=int, default=32,
                        help='hidden attention size')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Learning rate.')
    # for amazon-book, dropout_kg = 0.2
    # for yelp-2018, dropout_kg = 0.1
    parser.add_argument('--dropout_kg', type=float, default=0.2,
                        help='Dropout ratio of KG Embedding')
    parser.add_argument('--dropout_cf', type=float, default=0.3,
                        help='Dropout ratio of CF')
    """
    c0 and c1 determine the overall weight of non-observed instances in implicit feedback data.
    Specifically, c0 is for the recommendation task and c1 is for the knowledge embedding task.
    """
    # for amazon-book, c0=300, c1=600
    # for yelp-2018, c0=1000, c1=7000
    parser.add_argument('--c0', type=float, default=300,
                        help='initial weight of non-observed data')
    parser.add_argument('--c1', type=float, default=600,
                        help='initial weight of non-observed knowledge data')

    parser.add_argument('--p', type=float, default=0.5,
                        help='significance level of weight')

    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='batch size of evaluation')

    parser.add_argument('--Ks', nargs='?', default='[10, 20, 40]',
                        help='evaluation top K')

    parser.add_argument('--gpus', nargs='?', default='0', help='available GPUs')

    parser.add_argument('--weight_task_kg', type=float, default=0.5,
                        help="weight of KG Loss in multi-task learning")
    parser.add_argument('--weight_L2_kg', type=float, default=1e-5,
                        help="weight of L2 regularization of KG Embedding")

    args = parser.parse_args()

    # data_root
    args.data_root = "./data"

    # save training log
    cur_time = time.strftime("%Y-%m-%d", time.localtime())
    save_dir = f"./log/{cur_time}/{args.data_name}/"
    args.save_dir = save_dir
    return args
