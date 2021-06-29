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
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


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
    # top K 推荐列表
    hits_k = hits[:, :k]
    # dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    denominator = torch.log2(torch.arange(2, k + 2, dtype=torch.double)).to(device)
    dcg = torch.sum(hits_k / denominator, dim=1)

    # 升序排序, 再在 横向反转为 降序排序, 取 top K
    sorted_hits_k, _ = torch.sort(hits_k, dim=1, descending=True)
    # sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    # idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    idcg = torch.sum(sorted_hits_k / denominator, dim=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    # res = (torch.sum(hits[:, :k], dim=1, keepdim=False) / torch.sum(hits, dim=1))
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


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, K):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_pos_item_binary = torch.zeros(size=[len(user_ids), len(item_ids)], dtype=torch.float32).to(device)

    for i, u in enumerate(user_ids):
        uid = u.item()
        train_pos_item_list = train_user_dict[uid]
        # cf_scores 中, train set 出现过的 item, score 改为 负无穷
        cf_scores[i][train_pos_item_list] = -float('inf')
        # test set 中的 ground truth, 标记为 1
        test_pos_item_list = test_user_dict[uid]
        test_pos_item_binary[i][test_pos_item_list] = 1.0

    # rank_indices 是 排序后的 items 各自在 cf_scores 里的下标（即 推荐的商品列表，按照喜好自大到小排序）
    _, rank_indices = torch.sort(cf_scores, descending=True)
    # _, rank_indices = torch.topk(cf_scores, dim=1, k=K, largest=True, sorted=True)

    # Q: binary_hit 怎么变成 二值的 ?   A: test_pos_item_binary : ground truth = 1, 其余没有交互的都为 0
    binary_hit = []
    # test_pos_item_binary[i] : 第 i 个用户的 y_true (test set 上所有物品的 ground truth)
    # rank_indices[i] : 第 i 个用户的 商品推荐列表
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])

    # binary_hit 代表推荐列表中的每个商品 是否真的是 pos_item (y_pred)
    binary_hit = np.array([item.cpu().numpy() for item in binary_hit])

    # print(binary_hit.shape)

    # binary_hit = torch.tensor(binary_hit, dtype=torch.float32)

    recall = recall_at_k_batch(binary_hit, K)
    # ndcg = ndcg_at_k_batch(binary_hit, K)

    ndcg = ndcg_at_k(binary_hit, K)
    return recall, ndcg


def evaluate_torch_at_K(model, test_set, test_loader, K):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recall = []
    ndcg = []

    for i, user_batch in enumerate(test_loader):
        user_batch = user_batch.to(device)
        cf_scores_batch = model(
            "predict",
            pos_items=test_set.item_ids,
            users=user_batch,
            r_test=test_set.relation_test,
            t_test=test_set.tail_test
        ).squeeze_()

        # if i == 0:
        #     record = cf_scores_batch[0][:20]
        #     print(record)

        # cf_scores 存在负值
        # (n_batch_users, n_eval_items)
        recall_batch, ndcg_batch = calc_metrics_at_k(
            cf_scores_batch, test_set.train_user_dict,
            test_set.test_user_dict, user_batch,
            test_set.item_ids, K
        )
        recall.append(recall_batch)
        ndcg.append(ndcg_batch)

    # recall_k = torch.sum(torch.cat(recall, dim=0)) / test_set.n_users
    # ndcg_k = torch.sum(torch.cat(ndcg, dim=0)) / test_set.n_users
    recall_k = np.concatenate(recall, axis=0).sum() / test_set.n_users
    ndcg_k = np.concatenate(ndcg, axis=0).sum() / test_set.n_users
    return recall_k, ndcg_k
