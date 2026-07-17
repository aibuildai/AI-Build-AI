"""Train the anomalous-diffusion OOD detector, writing best_model.pth.

The training corpus is generated on the fly with `andi-datasets` (no external
download): SAMPLES_PER_MODEL trajectories per family, over a grid of anomalous
exponents and the SNR_CHOICES noise levels. The network is trained with a
family cross-entropy, an exponent regression, and a supervised-contrastive term.

Run:  python train.py            # writes ./best_model.pth
"""
import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import config as C
from model import Net, physics, channels, d4

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_chunk(args):
    """Simulate one chunk of trajectories for a single family (a worker task)."""
    model, n, seed = args
    from andi_datasets.models_theory import models_theory
    np.random.seed(seed)
    gen = getattr(models_theory(), model.lower())
    lo, hi = (1.05, 2.0) if model == "LW" else ((0.05, 1.0) if model in ("ATTM", "CTRW") else (0.05, 1.95))
    grid = np.round(np.arange(lo, hi + 0.001, 0.05), 2)
    out = np.empty((n, 2, C.TRAJ_LEN), np.float32)
    alpha = np.empty(n, np.float32)
    for i in range(n):
        a = float(grid[(seed + i) % len(grid)])
        snr = C.SNR_CHOICES[(seed // 7 + i) % len(C.SNR_CHOICES)]
        alpha[i] = a
        for dim in range(2):
            x = np.asarray(gen(C.TRAJ_LEN, a), dtype=np.float64)
            s = x.std()
            x += np.random.normal(0, s / snr, C.TRAJ_LEN)
            s = x.std()
            out[i, dim] = ((x - x.mean()) / (s + 1e-8)).astype(np.float32)
    return model, seed, out, alpha


def generate_corpus(cache: str) -> None:
    """Generate and cache the per-family training corpus (idempotent)."""
    paths = [os.path.join(cache, f"{m}.npz") for m in C.MODELS]
    if all(os.path.exists(p) for p in paths):
        return
    os.makedirs(cache, exist_ok=True)
    jobs = []
    for mi, model in enumerate(C.MODELS):
        for start in range(0, C.SAMPLES_PER_MODEL, C.CHUNK):
            jobs.append((model, min(C.CHUNK, C.SAMPLES_PER_MODEL - start),
                         C.CORPUS_SEED_BASE + mi * 100000 + start))
    gathered = {m: [] for m in C.MODELS}
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as ex:
        futures = [ex.submit(generate_chunk, j) for j in jobs]
        for f in as_completed(futures):
            model, seed, x, a = f.result()
            gathered[model].append((seed, x, a))
    for model, path in zip(C.MODELS, paths):
        parts = sorted(gathered[model])
        x = np.concatenate([p[1] for p in parts])
        a = np.concatenate([p[2] for p in parts])
        np.savez(path, x=x, alpha=a)


def load_train_arrays(cache: str):
    """Load the corpus and split into the training slice per family."""
    tx, ta, ty, vx = [], [], [], []
    for yi, m in enumerate(C.MODELS):
        z = np.load(os.path.join(cache, m + ".npz"))
        tx.append(z["x"][:C.TRAIN_PER_MODEL])
        ta.append(z["alpha"][:C.TRAIN_PER_MODEL])
        ty.append(np.full(C.TRAIN_PER_MODEL, yi, np.int64))
        vx.append(z["x"][C.TRAIN_PER_MODEL:])
    return tx, ta, ty, vx


def supcon(z: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Supervised-contrastive loss: same family AND same exponent bin are positives."""
    sim = z @ z.T / C.SUPCON_TEMP
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & \
          ((a[:, None] / .1).long() == (a[None, :] / .1).long()) & ~eye
    sim = sim.masked_fill(eye, -1e4)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -((logp * pos).sum(1) / pos.sum(1).clamp_min(1)).mean()


def train() -> str:
    torch.set_float32_matmul_precision("high")
    import random
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    cache = os.path.join(HERE, "_generated")
    generate_corpus(cache)
    tx, ta, ty, _ = load_train_arrays(cache)
    X = np.concatenate(tx)
    A = np.concatenate(ta)
    Y = np.concatenate(ty)
    P = physics(X)
    pm = P.mean(0)
    ps = P.std(0) + 1e-5

    model = Net(P.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for step in range(C.STEPS):
        idx = np.random.randint(0, len(X), C.BATCH_SIZE)
        xx = d4(X[idx], np.random.randint(8))
        ph = (physics(xx) - pm) / ps
        seq = torch.from_numpy(channels(xx)).cuda()
        phy = torch.from_numpy(ph).cuda()
        y = torch.from_numpy(Y[idx]).cuda()
        a = torch.from_numpy(A[idx]).cuda()
        if step < C.WARMUP_STEPS:
            lr = C.LR * (step + 1) / C.WARMUP_STEPS
        else:
            lr = C.MIN_LR + (C.LR - C.MIN_LR) * .5 * (
                1 + math.cos(math.pi * (step - C.WARMUP_STEPS) / (C.STEPS - C.WARMUP_STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            e, l, ap = model(seq, phy)
            loss = F.cross_entropy(l, y, label_smoothing=C.LABEL_SMOOTHING) \
                + C.ALPHA_LOSS_WEIGHT * F.smooth_l1_loss(ap, a, beta=.1) \
                + C.SUPCON_WEIGHT * supcon(e, y, a)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0:
            print("step", step, "loss", float(loss), flush=True)

    model_path = os.path.join(HERE, "best_model.pth")
    torch.save({"model": model.state_dict(), "mean": pm, "std": ps}, model_path)
    print("saved", model_path, flush=True)
    return model_path


if __name__ == "__main__":
    train()
