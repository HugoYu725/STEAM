import torch

import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import datetime as dt
import random

from utils import load_data, time_diff, generate_snapshots
import time
from Graph import AnomalyMotifSnapShot, AnomalyCircleSnapShot

import torch.nn.functional as F


def set_seed_everywhere(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def loss_func(adj, A_hat):
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    return structure_reconstruction_errors  # LOSS


# --------------------------------------------------------------------------------

def loss_func_hyper(H, Re_H, device):
    # structure reconstruction loss
    # H=H.to_dense().to("cuda")
    H = H.to(device)
    diff_structure = torch.pow(H - Re_H, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    # cloned_errors = structure_reconstruction_errors.clone()
    # max_error = torch.max(cloned_errors)
    # structure_reconstruction_errors /= max_error
    # structure_reconstruction_errors/=max(structure_reconstruction_errors)
    return structure_reconstruction_errors  # LOSS


# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
def loss_func_time(time_adj, time_rebuild, device):
    t = torch.stack(time_adj)
    t = t.to(device)
    diff_time = torch.pow(t - time_rebuild, 2)
    time_reconstruction_errors = torch.sqrt(torch.sum(diff_time, 1))
    return time_reconstruction_errors


# --------------------------------------------------------------------------------

def test(args):
    start_time = time.time()
    print(args.p)
    set_seed_everywhere(args.seed, cuda=True)
    data = args.dataset
    p = args.p
    start_t = dt.datetime.now()
    print('Start data building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                          start_t.day,
                                                          start_t.hour,
                                                          start_t.minute,
                                                          start_t.second))
    edges, times, n = load_data(data)
    snap_num = 10
    snapshots, times_list = generate_snapshots(edges, snap_num, times)
    if args.motif == 'triangle':
        SnapShots = [AnomalyMotifSnapShot(snapshots[i], p=p, device=args.device, time=times_list[i]) for i in
                     range(len(snapshots))]
    elif args.motif == 'circle':
        SnapShots = [AnomalyCircleSnapShot(snapshots[i], p=p, device=args.device, time=times_list[i]) for i in
                     range(len(snapshots))]


    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('dataset built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))




    model = torch.load('model-uci-0.1.pt')


    device = torch.device(args.device)
    for snap in SnapShots:  # 每个快照
        snap.norm_adj = snap.norm_adj.to(device)
        snap.adj = snap.adj.to(device)
        snap.label_adj = snap.label_adj.to(device)
        snap.motif_adj = snap.motif_adj.to(device)
        snap.motif_norm_adj = snap.motif_norm_adj.to(device)


    model = model.to(device)  # 模型

    scores = []
    labels = []

    model.eval()
    A_hat, Hyper_Rebuild, x_all, y_all, time_rebuild = model(SnapShots)
    y_all = [F.sigmoid(y_all[i]) for i in range(len(y_all))]
    result = []
    prek_result = []
    reck_result = []

    for i, snap in enumerate(SnapShots):

        if snap.motif_num == 0:
            continue

        loss = 0
        loss3 = loss_func_hyper(snap.hg.norm_MT, y_all[i], device)

        loss_time_rebuild = loss_func_time(snap.final_time, time_rebuild[i], device)
        score_time = loss_time_rebuild[len(snap.all_edges_times):].detach().cpu().numpy()

        loss = loss3

        score = loss.detach().cpu().numpy()
        motif_score = score[len(snap.new_edges):] + args.gamma * score_time[[i - snap.motif_id_list[0] for i in snap.motif_id_list]]

        scores.extend(motif_score)
        labels.extend(snap.motif_labels)

        auc = roc_auc_score(snap.motif_labels, motif_score)
        auc = round(auc, 4)
        result.append(auc)
        print(" Snap:", '%d' % (i + 1), 'Auc', auc)
        idx = motif_score.argsort()
        pred_label = snap.motif_labels[idx]
        topk = [50, 100, 200]
        for k in topk:
            predk = pred_label[-k:]
            precisionk = sum(predk) / len(predk)
            recallk = sum(predk) / sum(pred_label)
            prek_result.append(round(precisionk, 4))
            reck_result.append(round(recallk, 4))

    # --------------------------------------------------------------
    scores = np.array(scores)
    labels = np.array(labels)
    idx_all = scores.argsort()  # +++++++
    all_pred_label = labels[idx_all]
    tpk = [50, 100, 200]
    ans_prek = []
    ans_reck = []
    for k in tpk:
        predk = all_pred_label[-k:]
        precisionk = sum(predk) / len(predk)
        recallk = sum(predk) / sum(all_pred_label)
        ans_prek.append(round(precisionk, 4))
        ans_reck.append(round(recallk, 4))
    print("prek:", ans_prek)
    print("reck:", ans_reck)
    # --------------------------------------------------------------
    print("AUC:",np.mean(result))

    print("p=", p, result)
    pre50 = [prek_result[i] for i in range(0, len(prek_result), 3)]
    pre100 = [prek_result[i] for i in range(1, len(prek_result), 3)]
    pre200 = [prek_result[i] for i in range(2, len(prek_result), 3)]
    rec50 = [reck_result[i] for i in range(0, len(reck_result), 3)]
    rec100 = [reck_result[i] for i in range(1, len(reck_result), 3)]
    rec200 = [reck_result[i] for i in range(2, len(reck_result), 3)]
    print("Pre@50:", pre50, 'mean:', np.mean(pre50))
    print("Pre@100:", pre100, 'mean:', np.mean(pre100))
    print("Pre@200:", pre200, 'mean:', np.mean(pre200))
    print("Rec@50:", rec50, 'mean:', np.mean(rec50))
    print("Rec@100:", rec100, 'mean:', np.mean(rec100))
    print("Rec@200:", rec200, 'mean:', np.mean(rec200))

    print("time cost: {}".format(time.time() - start_time))

# CUDA_LAUNCH_BLOCKING=1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='uci', help='dataset name: uci/dnc/alpha/otc')
    parser.add_argument('--p', type=float, default=0.1, help='anomalous rate')
    parser.add_argument('--motif', default='triangle', help='triangle/circle')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda/cpu')
    parser.add_argument('--seed', default=66, type=int, help='seed')
    parser.add_argument('--sigma', default=0.25, type=float, help='time_loss')
    parser.add_argument('--gamma', default=0.25, type=float, help='time_score')
    args = parser.parse_args()

    test(args)
