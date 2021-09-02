import os
import time
import torch
import matplotlib.pyplot as plt


def early_stopping(result, threshold: float, evaluate_every: int, old_best_epoch: int):
    """ 返回是否有提升
    """
    # 在 六个指标上计算平均增益百分比, 若为正数, 则更新 best epoch
    ri = []
    for i in range(len(result["recall"])):
        old_value = result["recall"][i][old_best_epoch // evaluate_every]
        cur_v = result["recall"][i][-1]
        cur_ri = (cur_v - old_value) / old_value
        ri.append(cur_ri)

    for i in range(len(result["ndcg"])):
        old_value = result["recall"][i][old_best_epoch // evaluate_every]
        cur_v = result["recall"][i][-1]
        cur_ri = (cur_v - old_value) / old_value
        ri.append(cur_ri)

    avg_ri = sum(ri) / len(ri)
    if avg_ri > threshold:
        return True, avg_ri
    return False, avg_ri


def checkpoint(epoch, model, optimizer, val_result, save_dir, filename):
    """
    save checkpoint, including model weights and optimizer params
    """
    state = {
        'epoch': epoch,
        # only save parameters without structure
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_result': val_result
    }
    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)


def visualize_result(result: dict, show: bool = False):
    time_stamp = str(time.strftime("%Y-%m-%d_%H-%M", time.localtime()))
    save_dir = os.path.join('./log', 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Ks = result['Ks']
    # train loss
    epochs = range(1, result["epochs"] + 1)

    plt.figure(1)
    plt.title('Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, result["train_loss"], 'bo-', label='Train Loss')

    filename = f"train-loss_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    # validation Recall
    evaluate_every = result["evaluate_every"]
    epochs = range(evaluate_every, result["epochs"] + 1, evaluate_every)
    plt.figure(2, figsize=(8, 14))
    plt.title('Val Recall')
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    # for i, k in enumerate(result['Ks']):
    #     plt.plot(epochs, result['recall'][i], 'o-', label=f"Recall@{k}")
    #     plt.xticks(epochs)
    #     plt.yticks(torch.arange(0.1, 0.25, 0.002))

    # visualize Recall@20
    plt.plot(epochs, result['recall'][1], 'o-', label=f"Recall@{Ks[1]}")
    plt.xticks(epochs)
    plt.yticks(torch.arange(0.14, 0.17, 0.0005))
    plt.legend()
    filename = f"val-recall_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    # validation NDCG
    plt.figure(3, figsize=(8, 12))
    plt.title('Val NDCG')
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    # for i, k in enumerate(result['Ks']):
    #     plt.plot(epochs, result['ndcg'][i], 'o-', label=f"NDCG@{k}")
    #     plt.xticks(epochs)
    #     plt.yticks(torch.arange(0.08, 0.15, 0.002))
    plt.plot(epochs, result['ndcg'][1], 'o-', label=f"NDCG@{Ks[1]}")
    plt.xticks(epochs)
    plt.yticks(torch.arange(0.1, 0.13, 0.0005))
    plt.legend()
    filename = f"val-ndcg_epoch-{result['epochs']}_" + time_stamp + ".png"
    plt.savefig(os.path.join(save_dir, filename), dpi=400)

    if show:
        plt.show()


def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0.] = -float("inf")

    return torch.softmax(x_masked, **kwargs)