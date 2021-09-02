import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    # dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    # return dcg
    return np.sum(rel / np.log2(np.arange(2, rel.size + 2)))


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hits_k = hits[:, :k]    # Top K

    # dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    denominator = torch.log2(torch.arange(2, hits.shape[1] + 2)).to(device)
    dcg = torch.sum(hits_k / denominator, dim=1)

    sorted_hits_k, _ = torch.sort(hits_k, dim=1, descending=True)
    # idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    # idcg = np.sum((sorted_hits_k / denominator), axis=1)
    idcg = torch.sum(sorted_hits_k / denominator, dim=1)

    epsilon = 1e-9
    res = dcg / (idcg + epsilon)
    return res


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, pos_nums, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    # res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    res = hits[:, :k].sum(axis=1) / pos_nums
    # res = torch.sum(hits, dim=1, keepdim=False) / pos_nums
    return res


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_ks(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = torch.zeros(size=[len(user_ids), len(item_ids)]).to(device)

    # 每个用户的正样本数
    pos_nums = torch.zeros(len(user_ids)).to(device)

    for i, u in enumerate(user_ids):
        uid = u.item()
        train_pos_items = train_user_dict[uid]
        # cf_scores 中, train set 出现过的 item, score 改为 负无穷
        cf_scores[i][train_pos_items] = float('-inf')

        # test set 中的 ground truth, 标记为 1
        test_pos_items = test_user_dict[uid]    # list
        label[i][test_pos_items] = 1.0

        pos_nums[i] = len(test_pos_items)

    # rank_indices 是 排序后的 items 各自在 cf_scores 里的下标（即 推荐的商品列表，按照喜好自大到小排序）
    # _, rank_indices = torch.sort(cf_scores, descending=True)
    max_k = max(Ks)
    _, rank_indices = torch.topk(cf_scores, dim=1, k=max_k, largest=True, sorted=True)  # (batch, max_K)

    # Q: binary_hit 怎么变成 二值的 ?   A: test_pos_item_binary : ground truth = 1, 其余没有交互的都为 0
    binary_hit = []
    # test_pos_item_binary[i] : 第 i 个用户的 y_true (test set 上所有物品的 ground truth)
    # rank_indices[i] : 第 i 个用户的 商品推荐列表
    for i in range(len(user_ids)):
        binary_hit.append(label[i][rank_indices[i]])
    # binary_hit 代表推荐列表中的每个商品 是否命中
    binary_hit = torch.stack(binary_hit)    # list of tensor, 用 torch.stack 直接转成新 tensor
    # (batch, max_K)

    recall_batch_ks = []
    ndcg_batch_ks = []

    for k in Ks:
        recall_batch = recall_at_k_batch(binary_hit, pos_nums, k)
        ndcg_batch = ndcg_at_k_batch(binary_hit, k)
        recall_batch_ks.append(recall_batch)
        ndcg_batch_ks.append(ndcg_batch)

    return torch.stack(recall_batch_ks), torch.stack(ndcg_batch_ks)


def evaluate(model, test_set, test_loader, Ks=(10, 20, 40)):
    """
    evaluate model using PyTorch purely, runs on GPU, faster that multiprocessing
    :return: recall_ks, ndcg_ks
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recall_ks = torch.zeros(len(Ks))
    ndcg_ks = torch.zeros(len(Ks))

    for i, user_batch in enumerate(test_loader):
        user_batch = user_batch.to(device)
        cf_scores_batch = model(
            "predict",
            pos_items=test_set.item_ids,
            users=user_batch,
            r_test=test_set.relation_test,
            t_test=test_set.tail_test
        ).squeeze_()    # [batch, n_items]

        recall_batch_ks, ndcg_batch_ks = calc_metrics_at_ks(
                cf_scores=cf_scores_batch, train_user_dict=test_set.train_user_dict,
                test_user_dict=test_set.test_user_dict, user_ids=user_batch,
                item_ids=test_set.item_ids, Ks=Ks
        )   # [ len(Ks), batch ],   [ len(Ks), batch ]

        recall_ks += torch.sum(recall_batch_ks, dim=1).cpu()
        ndcg_ks += torch.sum(ndcg_batch_ks, dim=1).cpu()

    recall_ks /= test_set.n_users   # [ len(Ks) ]
    ndcg_ks /= test_set.n_users
    return recall_ks, ndcg_ks
