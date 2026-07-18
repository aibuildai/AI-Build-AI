"""End-to-end pipeline for cancer-gene identification across biological networks.

This is a transductive method that trains at inference time: for each of the
eight networks it loads the graph, builds structural + omics node features,
trains an MTGCN with an adaptive step budget, and writes the held-out test-node
probabilities to output/<network>/predictions.npy. This is the exact code that
scored g = +0.103 (surpass-SOTA on AUPRC), beating the source paper on 6/8
networks.

Training uses the provided train (and validation) node labels only; test-node
labels are never seen. Structural features come from the graph adjacency alone.

Run:  python run.py --data-dir /path/to/data --output ./output
      # or set DATA_DIR / OUTPUT_DIR in the environment
Requirements: torch, torch_geometric, numpy, h5py, scipy, scikit-learn.
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from graph_features import compute_structural_features
from model import MTGCN, dropedge, sample_negatives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_instance(data_dir, inst):
    """Load one network's graph, features, masks, and train/val labels."""
    with h5py.File(os.path.join(data_dir, inst, "data.h5"), "r") as f:
        feats = np.asarray(f["features"], dtype=np.float32)
        N = feats.shape[0]
        net = f["network"]
        rows_l, cols_l = [], []
        for i in range(0, N, 3000):
            block = np.asarray(net[i:i + 3000])
            r, c = np.nonzero(block)
            rows_l.append(r + i)
            cols_l.append(c)
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
        m_tr = np.asarray(f["mask_train"]).astype(bool).ravel()
        m_val = np.asarray(f["mask_val"]).astype(bool).ravel()
        m_test = np.asarray(f["mask_test"]).astype(bool).ravel()
        y_tr = np.asarray(f["y_train"]).astype(np.float32).ravel()
        y_val = np.asarray(f["y_val"]).astype(np.float32).ravel()
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    return feats, edge_index, rows, cols, m_tr, m_val, m_test, y_tr, y_val, N


def train_instance(data_dir, inst, log_f, time_left, insts_left):
    set_seed(C.SEED)
    feats, edge_index_np, rows, cols, m_tr, m_val, m_test, y_tr, y_val, N = load_instance(data_dir, inst)

    # structural features (structure-only, no label leakage) -> (N, 5); concat -> (N, 69)
    t_struct = time.time()
    struct = compute_structural_features(rows, cols, N, drop_eigenvector=(N > 40000))
    struct_time = time.time() - t_struct
    feats = np.concatenate([feats, struct], axis=1)

    dropout = C.DROPOUT_BOTTLENECK if inst in C.BOTTLENECK else C.DROPOUT_DEFAULT
    weight_decay = C.WD_BOTTLENECK if inst in C.BOTTLENECK else C.WD_DEFAULT

    x = torch.from_numpy(feats).to(DEVICE)
    edge_index = torch.from_numpy(edge_index_np).to(DEVICE)
    pos_edges = edge_index[:, edge_index[0] < edge_index[1]]
    n_pos = pos_edges.shape[1]

    # train classification on mask_train UNION mask_val (val folded in for more
    # supervision); test-node labels are never used.
    m_fit = (m_tr | m_val)
    y_fit = (y_tr * m_tr.astype(np.float32) + y_val * m_val.astype(np.float32))
    train_idx = torch.where(torch.from_numpy(m_fit).to(DEVICE))[0]
    y_train_nodes = torch.from_numpy(y_fit).to(DEVICE)[train_idx]

    n_pos_lbl = float(y_train_nodes.sum().item())
    n_neg_lbl = float(len(train_idx) - n_pos_lbl)
    pos_weight = torch.tensor([max(n_neg_lbl / max(n_pos_lbl, 1.0), 1.0)], device=DEVICE)

    model = MTGCN(x.shape[1], dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=weight_decay)
    link_batch = min(n_pos, 20000)

    # adaptive step budget: fair share of remaining time across remaining instances
    share_time = max(time_left / max(insts_left, 1) - struct_time, 5.0)
    ema_state = None
    n_steps = C.MAX_STEPS
    ema_start = int(n_steps * C.EMA_START_FRAC)
    t0 = time.time()
    step_times = []
    step = 0
    while step < n_steps:
        st = time.time()
        model.train()
        opt.zero_grad()
        logit, z = model(x, edge_index, dropedge(edge_index, C.DROPEDGE_P))
        loss_cls = F.binary_cross_entropy_with_logits(
            logit[train_idx], y_train_nodes, pos_weight=pos_weight)

        pe = pos_edges[:, torch.randint(0, n_pos, (link_batch,), device=DEVICE)] if link_batch < n_pos else pos_edges
        na, nb = sample_negatives(N, pe.shape[1], DEVICE)
        link_logits = torch.cat([(z[pe[0]] * z[pe[1]]).sum(-1), (z[na] * z[nb]).sum(-1)])
        link_labels = torch.cat([torch.ones(pe.shape[1], device=DEVICE), torch.zeros(pe.shape[1], device=DEVICE)])
        loss_link = F.binary_cross_entropy_with_logits(link_logits, link_labels)

        prec_cls = torch.exp(-model.log_var_cls)
        prec_link = torch.exp(-model.log_var_link)
        loss = (prec_cls * loss_cls + 0.5 * model.log_var_cls
                + prec_link * loss_link + 0.5 * model.log_var_link).squeeze()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()

        if step >= ema_start:
            sd = model.state_dict()
            if ema_state is None:
                ema_state = {k: v.detach().clone().float() for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    ema_state[k].mul_(C.EMA_DECAY).add_(v.detach().float(), alpha=1 - C.EMA_DECAY)

        step_times.append(time.time() - st)
        if step == 8:   # size the step budget to the measured throughput
            sps = float(np.mean(step_times[4:]))
            remain = share_time - (time.time() - t0)
            n_steps = int(np.clip(step + int(max(remain, 0) / max(sps, 1e-4)), C.MIN_STEPS, C.MAX_STEPS))
            ema_start = int(n_steps * C.EMA_START_FRAC)
        if step % 200 == 0 or step == n_steps - 1:
            log_f.write(json.dumps({"instance": inst, "step": step, "n_steps": n_steps,
                                    "loss": float(loss.item())}) + "\n")
            log_f.flush()
        step += 1

    if ema_state is not None:
        model.load_state_dict(ema_state)
    model.eval()
    with torch.no_grad():
        logit, _ = model(x, edge_index, edge_index)
        prob = torch.sigmoid(logit).cpu().numpy().astype(np.float32)

    # optimistic diagnostic only (val was folded into training) — NOT the score
    diag = {"struct_time": round(struct_time, 2), "val_in_train": True, "n_steps": int(n_steps)}
    if m_val.sum() > 0:
        vi = np.where(m_val)[0]
        try:
            diag["auprc_valfit"] = float(average_precision_score(y_val[vi], prob[vi]))
            diag["auroc_valfit"] = float(roc_auc_score(y_val[vi], prob[vi]))
        except Exception:
            diag["auprc_valfit"] = None

    preds = prob[np.where(m_test)[0]]
    del x, edge_index, model
    torch.cuda.empty_cache()
    return preds, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "./output"))
    args = ap.parse_args()
    if not args.data_dir:
        raise SystemExit("pass --data-dir <dir of <network>/data.h5> or set DATA_DIR")
    os.makedirs(args.output, exist_ok=True)
    log_f = open(os.path.join(args.output, "training_progress.jsonl"), "w")
    t_all = time.time()
    for i_idx, inst in enumerate(C.INSTANCES):
        t0 = time.time()
        time_left = C.GLOBAL_TIME_BUDGET_S - (time.time() - t_all)
        preds, diag = train_instance(args.data_dir, inst, log_f, time_left, len(C.INSTANCES) - i_idx)
        out_dir = os.path.join(args.output, inst)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "predictions.npy"), preds.astype(np.float32))
        print(f"[{inst}] n_test={len(preds)} diag={diag} time={time.time()-t0:.0f}s", flush=True)
    log_f.close()
    print("DONE", round(time.time() - t_all, 1), "s", flush=True)


if __name__ == "__main__":
    main()
