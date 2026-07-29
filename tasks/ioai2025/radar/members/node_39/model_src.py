"""
MultiBranch-ViewFusion-UNet for radar semantic segmentation.

Three parallel per-view encoder branches (range-azimuth, range-elevation,
range-velocity), each width [24,48,96], fused at every scale by an SE-gated
concat->1x1 projection, feeding a shared bilinear-upsample U-Net decoder to a
5-class 50x181 head. Weighted-CE + foreground-only soft-Dice loss. AdamW +
cosine + warmup + bf16 AMP + weight EMA. Horizontal-flip TTA at inference.

Entry point for the grading harness: reads DATA_DIR, writes
OUTPUT_DIR/radar/predictions.csv and a best_model.pkl sentinel under OUTPUT_DIR.
"""
import os
import csv
import glob
import time
import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/path/to/data",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_HERE, "output"))

INSTANCE = "radar"
H, W = 50, 181
NUM_CLASSES = 5  # remapped {-1..3} -> {0..4}
IN_CH = 6

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _list_files(split):
    d = os.path.join(DATA_DIR, INSTANCE, split)
    files = sorted(glob.glob(os.path.join(d, "*.pt")))
    return files


def _load_tensor(path):
    t = torch.load(path, map_location="cpu")
    return t.float()  # (7,H,W) train or (6,H,W) test


def compute_train_stats(train_files):
    """Per-channel mean/std over the 6 input channels, train only."""
    n = 0
    s = torch.zeros(IN_CH, dtype=torch.float64)
    ss = torch.zeros(IN_CH, dtype=torch.float64)
    for p in train_files:
        t = _load_tensor(p)[:IN_CH].double()  # (6,H,W)
        s += t.sum(dim=(1, 2))
        ss += (t * t).sum(dim=(1, 2))
        n += H * W
    mean = (s / n)
    var = (ss / n) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.float(), std.float()


class RadarDataset(Dataset):
    def __init__(self, files, mean, std, train=True):
        self.files = files
        self.mean = mean.view(IN_CH, 1, 1)
        self.std = std.view(IN_CH, 1, 1)
        self.train = train
        # preload into RAM (1800*(7)*50*181*4 bytes ~ 0.5 GB) -- fine under 32G
        self.cache = [None] * len(files)

    def __len__(self):
        return len(self.files)

    def _get_raw(self, idx):
        if self.cache[idx] is None:
            self.cache[idx] = _load_tensor(self.files[idx])
        return self.cache[idx]

    def __getitem__(self, idx):
        t = self._get_raw(idx)
        x = t[:IN_CH]  # (6,H,W)
        x = (x - self.mean) / self.std
        if self.train:
            y = t[IN_CH].long() + 1  # {-1..3} -> {0..4}
            x, y = self._augment(x, y)
            return x, y
        else:
            return x, os.path.basename(self.files[idx])

    def _augment(self, x, y):
        # joint horizontal (azimuth) flip p=0.5
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        # azimuth column shift +/-3 p=0.3
        if random.random() < 0.3:
            sh = random.randint(-3, 3)
            if sh != 0:
                x = torch.roll(x, shifts=sh, dims=2)
                y = torch.roll(y, shifts=sh, dims=1)
        # additive Gaussian noise sigma=0.01 p=0.3 (on standardized inputs)
        if random.random() < 0.3:
            x = x + torch.randn_like(x) * 0.01
        return x, y


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Branch(nn.Module):
    """One per-view encoder: 3 double-conv stages [24,48,96] with maxpool."""
    def __init__(self, cin=2, widths=(24, 48, 96)):
        super().__init__()
        self.enc1 = DoubleConv(cin, widths[0])
        self.enc2 = DoubleConv(widths[0], widths[1])
        self.enc3 = DoubleConv(widths[1], widths[2])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.enc1(x)              # full res
        f2 = self.enc2(self.pool(f1))  # /2
        f3 = self.enc3(self.pool(f2))  # /4
        return f1, f2, f3


class SEFusion(nn.Module):
    """Concat three branch maps (3*C), SE gate, project to C."""
    def __init__(self, c, reduction=4):
        super().__init__()
        cin = 3 * c
        self.se_fc1 = nn.Linear(cin, max(cin // reduction, 4))
        self.se_fc2 = nn.Linear(max(cin // reduction, 4), cin)
        self.proj = nn.Sequential(
            nn.Conv2d(cin, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, a, b, c):
        x = torch.cat([a, b, c], dim=1)  # (N,3C,h,w)
        s = x.mean(dim=(2, 3))           # (N,3C)
        s = F.relu(self.se_fc1(s), inplace=True)
        s = torch.sigmoid(self.se_fc2(s)).unsqueeze(-1).unsqueeze(-1)
        x = x * s
        return self.proj(x)


class Up(nn.Module):
    """Bilinear upsample + concat skip + double conv."""
    def __init__(self, cin, cskip, cout, dropout=0.1):
        super().__init__()
        self.conv = DoubleConv(cin + cskip, cout, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiBranchViewFusionUNet(nn.Module):
    def __init__(self, widths=(24, 48, 96)):
        super().__init__()
        self.branchA = Branch(2, widths)
        self.branchB = Branch(2, widths)
        self.branchC = Branch(2, widths)
        self.fuse1 = SEFusion(widths[0])
        self.fuse2 = SEFusion(widths[1])
        self.fuse3 = SEFusion(widths[2])
        # decoder: from fused f3 up to f2, then up to f1
        self.up2 = Up(widths[2], widths[1], widths[1])
        self.up1 = Up(widths[1], widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], NUM_CLASSES, 1)

    def forward(self, x):
        a = x[:, 0:2]
        b = x[:, 2:4]
        c = x[:, 4:6]
        a1, a2, a3 = self.branchA(a)
        b1, b2, b3 = self.branchB(b)
        c1, c2, c3 = self.branchC(c)
        f1 = self.fuse1(a1, b1, c1)
        f2 = self.fuse2(a2, b2, c2)
        f3 = self.fuse3(a3, b3, c3)
        d2 = self.up2(f3, f2)      # mid-scale (/2)
        d1 = self.up1(d2, f1)      # full res
        main = self.head(d1)       # (N,5,H,W)
        return main  # (N,5,H,W), single head in both train and eval


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
CLASS_WEIGHTS = torch.tensor([1.0, 15.0, 12.0, 25.0, 8.0])  # bg,suitcase,chair,human,wall


def foreground_dice_loss(logits, target, eps=1.0):
    """Soft Dice over the 4 foreground classes (indices 1..4)."""
    probs = F.softmax(logits, dim=1)
    dice = 0.0
    n_fg = NUM_CLASSES - 1
    for cls in range(1, NUM_CLASSES):
        p = probs[:, cls]
        g = (target == cls).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        dice += 1.0 - (2 * inter + eps) / (denom + eps)
    return (dice / n_fg).mean()


class CombinedLoss(nn.Module):
    """node_2's clean loss: weighted-CE + 1.0 * foreground-only soft-Dice."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

    def forward(self, logits, target):
        return (self.ce(logits, target)
                + 1.0 * foreground_dice_loss(logits, target))


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def make_scheduler(optimizer, max_steps, warmup_frac=0.05, base_lr=3e-3, min_lr=1e-5):
    warmup_steps = max(1, int(max_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _reseed(seed):
    """Re-seed all RNGs before building/training a fresh model copy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(train_files, mean, std, max_steps, batch_size=32, log_path=None,
          seed=SEED, log_append=False):
    _reseed(seed)
    ds = RadarDataset(train_files, mean, std, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=6,
                        drop_last=True, pin_memory=True, persistent_workers=True)

    model = MultiBranchViewFusionUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = make_scheduler(opt, max_steps)
    loss_fn = CombinedLoss()
    ema = EMA(model, decay=0.999)

    use_amp = DEVICE.type == "cuda"
    model.train()
    step = 0
    t0 = time.time()
    log_f = open(log_path, "a" if log_append else "w") if log_path else None

    def log_event(event_type, extra=None):
        if log_f is None:
            return
        import json
        m = dict(extra or {})
        m["seed"] = seed
        rec = {"step": step, "elapsed_seconds": round(time.time() - t0, 2),
               "event_type": event_type, "metrics": m}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    log_event("training_start", {"max_steps": max_steps})
    done = False
    while not done:
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                main = model(x)
                loss = loss_fn(main, y)  # single-head clean loss (node_2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            if step % 25 == 0 or step == 1:
                log_event("step", {"train_loss": float(loss.item()),
                                    "lr": sched.get_last_lr()[0]})
            if step >= max_steps:
                done = True
                break
    log_event("training_complete", {"train_loss": float(loss.item())})
    if log_f:
        log_f.close()

    # export EMA weights as the inference model
    ema_model = MultiBranchViewFusionUNet().to(DEVICE)
    ema.copy_to(ema_model)
    ema_model.eval()
    return ema_model


# ----------------------------------------------------------------------------
# Cost-sensitive decision rule
# ----------------------------------------------------------------------------
# The metric weights a correct non-bg cell 50x a correct bg cell, so the
# Bayes-optimal rule predicts foreground whenever 50*p_fg > 1*p_bg, i.e.
# p_fg > p_bg/50. We tune the threshold tau on a train-only hold-out slice.
TAU_GRID = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.007, 0.005, 0.003, 0.002]


def _cost_sensitive_pred(probs, tau):
    """probs: (...,C) with class 0 = background, 1..4 = foreground.
    Return predicted labels in {-1..3}. Predict fg class k=argmax_{1..4} p_k
    when p_k > tau * p_bg, else background (-1)."""
    p_bg = probs[..., 0]
    fg = probs[..., 1:]
    fg_best = fg.max(axis=-1)
    fg_cls = fg.argmax(axis=-1) + 1  # class index in 1..4
    take_fg = fg_best > tau * p_bg
    return np.where(take_fg, fg_cls - 1, -1)


def weighted_pixel_acc(pred, gt):
    fg_mask = gt != -1
    correct = pred == gt
    w = np.where(fg_mask, 50.0, 1.0)
    return (correct.astype(np.float64) * w).sum() / w.sum()


@torch.no_grad()
def _softmax_tta(models, x, use_amp):
    """flip-TTA-averaged softmax, averaged across an ensemble of models.
    `models` may be a single model or a list of models; returns (N,C,H,W)
    probabilities averaged over both the horizontal-flip TTA and the models."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    acc = None
    for m in models:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = m(x)
            logits_f = m(torch.flip(x, dims=[3]))
            logits_f = torch.flip(logits_f, dims=[3])
        p = (F.softmax(logits.float(), dim=1) + F.softmax(logits_f.float(), dim=1)) / 2
        acc = p if acc is None else acc + p
    return acc / len(models)


@torch.no_grad()
def select_tau(models, holdout_files, mean, std, batch_size=64, log_fn=None):
    """Sweep TAU_GRID on a TRAIN-only hold-out slice; return best tau by wpa.
    Standardization stats (mean/std) were fit WITHOUT these hold-out files.
    `models` is the ensemble whose flip-TTA softmax is averaged per cell FIRST,
    so tau adapts to the AVERAGED model's calibration."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    ds = RadarDataset(holdout_files, mean, std, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True)
    use_amp = DEVICE.type == "cuda"
    for m in models:
        m.eval()
    probs_list, gt_list = [], []
    # names map back to files to grab the ground-truth channel 6
    for x, names in loader:
        x = x.to(DEVICE, non_blocking=True)
        probs = _softmax_tta(models, x, use_amp)  # (N,C,H,W) ensemble+flip avg
        probs = probs.permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,C)
        for i, nm in enumerate(names):
            probs_list.append(probs[i])
        for nm in names:
            fp = os.path.join(DATA_DIR, INSTANCE, "train", nm)
            t = _load_tensor(fp)
            gt_list.append(t[IN_CH].long().numpy())  # {-1..3}
    P = np.stack(probs_list)  # (M,H,W,C)
    G = np.stack(gt_list)     # (M,H,W)

    argmax_pred = P.argmax(axis=-1) - 1
    base = weighted_pixel_acc(argmax_pred, G)
    results = {"argmax": base}
    best_tau, best_wpa = None, -1.0
    for tau in TAU_GRID:
        pred = _cost_sensitive_pred(P, tau)
        s = weighted_pixel_acc(pred, G)
        results[f"tau={tau}"] = s
        if s > best_wpa:
            best_wpa, best_tau = s, tau
    # If no tau beats argmax, fall back to argmax (tau=None handled by caller).
    if best_wpa <= base:
        best_tau = None
        best_wpa = base
    if log_fn is not None:
        log_fn(results, best_tau, best_wpa, base)
    return best_tau, best_wpa, base, results


# ----------------------------------------------------------------------------
# Inference (horizontal-flip TTA + cost-sensitive decision rule)
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(models, test_files, mean, std, out_dir, tau, batch_size=64):
    """tau=None -> plain argmax; else cost-sensitive rule. `models` is the
    ensemble whose flip-TTA softmax is averaged per cell before the rule; the
    written raw_test/*.npy is the 3-model-averaged softmax."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    ds = RadarDataset(test_files, mean, std, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6,
                        pin_memory=True)
    use_amp = DEVICE.type == "cuda"
    raw_dir = os.path.join(out_dir, "raw_test")
    os.makedirs(raw_dir, exist_ok=True)

    rows = []
    for m in models:
        m.eval()
    for x, names in loader:
        x = x.to(DEVICE, non_blocking=True)
        probs = _softmax_tta(models, x, use_amp)      # (N,C,H,W) ensemble+flip avg
        probs_hwc = probs.permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,C)
        probs_np = probs.cpu().numpy().astype(np.float32)    # (N,C,H,W) for raw_test
        if tau is None:
            pred_np = probs_hwc.argmax(axis=-1) - 1          # (N,H,W)
        else:
            pred_np = _cost_sensitive_pred(probs_hwc, tau)   # (N,H,W)
        pred_np = pred_np.astype(np.int64)
        for i, nm in enumerate(names):
            # keep raw_test as UNMODIFIED softmax so Aggregator can re-derive any rule
            np.save(os.path.join(raw_dir, nm.replace(".pt", "") + ".npy"), probs_np[i])
            rows.append((nm, pred_np[i].reshape(-1)))
    return rows


def write_predictions_csv(rows, csv_path):
    n_pix = H * W
    header = ["filename"] + [f"pixel_{k}" for k in range(n_pix)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for nm, flat in rows:
            w.writerow([nm] + flat.tolist())


# ----------------------------------------------------------------------------
# Micro-benchmark to size max_steps within budget
# ----------------------------------------------------------------------------
def benchmark_step_time(train_files, mean, std, batch_size=32, n_probe=12):
    ds = RadarDataset(train_files, mean, std, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=6,
                        drop_last=True, pin_memory=True)
    model = MultiBranchViewFusionUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = CombinedLoss()
    ema = EMA(model, decay=0.999)
    use_amp = DEVICE.type == "cuda"
    model.train()
    times = []
    it = iter(loader)
    for i in range(n_probe):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            main = model(x)
            loss = loss_fn(main, y)  # single-head clean loss (node_2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    # drop first 4 (warmup)
    steady = times[4:] if len(times) > 4 else times
    return float(np.mean(steady))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    out_path = os.path.join(OUTPUT_DIR, INSTANCE)
    os.makedirs(out_path, exist_ok=True)

    train_files = _list_files("train")
    test_files = _list_files("test")
    print(f"[data] train={len(train_files)} test={len(test_files)} device={DEVICE}")

    if smoke:
        train_files = train_files[:64]
        test_files = test_files[:16]

    t_start = time.time()
    # --- tau hold-out split: last N train files reserved for cost-rule tuning,
    #     EXCLUDED from the standardization-stats fit (no leakage; train-only). ---
    holdout_n = 8 if smoke else 200
    holdout_files = train_files[-holdout_n:]
    fit_files = train_files[:-holdout_n]  # stats fit set (excludes hold-out)
    print(f"[split] stats-fit={len(fit_files)} tau-holdout={len(holdout_files)}")

    print("[stats] computing train-only per-channel standardization (fit set) ...")
    mean, std = compute_train_stats(fit_files)
    print(f"[stats] mean={mean.tolist()}")
    print(f"[stats] std={std.tolist()}")

    # micro-benchmark
    step_t = benchmark_step_time(fit_files, mean, std, batch_size=32)
    print(f"[bench] steady-state step time = {step_t*1000:.1f} ms")

    # Ensemble of 3 identical models with different seeds; averaging decorrelated
    # seeds is prediction-space soft voting that reduces the FG/bg boundary
    # variance the 50:1 metric rewards (node_14 diagnosis: error is boundary
    # variance, not class-bias -- FG confusion is diagonal).
    SEEDS = [1234, 2025, 777] if not smoke else [1234, 2025]

    if smoke:
        max_steps = 20
    else:
        # The model is tiny and the A100 is fast (~27 ms/step). node_14 trained
        # ONE model for 8400 steps (~150 epochs) in ~220s. Three seeds fit
        # comfortably in budget; keep node_14's proven per-model step count so
        # each ensemble member matches the single-model baseline exactly.
        steps_per_epoch = max(1, len(train_files) // 32)
        max_steps = 150 * steps_per_epoch  # ~8400 steps PER SEED (node_14 parity)
        # Cap the TOTAL training wall (all seeds) at ~120 min regardless of step time.
        wall_cap_total = int(120 * 60 / step_t)
        max_steps = min(max_steps, wall_cap_total // max(1, len(SEEDS)))
        max_steps = max(max_steps, 40 * steps_per_epoch)
    est_min = max_steps * step_t * len(SEEDS) / 60
    print(f"[plan] max_steps={max_steps}/seed x {len(SEEDS)} seeds "
          f"(est total {est_min:.1f} min)")

    log_path = os.path.join(out_path, "training_progress.jsonl")
    # Train each seed on fit_files only (EXCLUDES the 200-file tau hold-out) so
    # the tau sweep below is on genuinely unseen data and its winner transfers.
    models = []
    for si, sd in enumerate(SEEDS):
        print(f"[train] === seed {sd} ({si+1}/{len(SEEDS)}) ===")
        m = train(fit_files, mean, std, max_steps, batch_size=32,
                  log_path=log_path, seed=sd, log_append=(si > 0))
        models.append(m)

    # --- tune the cost-sensitive tau on the ensemble-AVERAGED softmax over the
    #     TRAIN-only hold-out slice (average in prob space FIRST, then rule) ---
    def _log_tau(results, best_tau, best_wpa, base):
        print(f"[tau] (3-seed avg) argmax wpa={base:.5f}")
        for tau in TAU_GRID:
            print(f"[tau]   tau={tau:<5} wpa={results[f'tau={tau}']:.5f}")
        print(f"[tau] selected tau={best_tau} (holdout wpa={best_wpa:.5f}, "
              f"argmax={base:.5f}, delta={best_wpa-base:+.5f})")

    best_tau, best_wpa, base_wpa, tau_results = select_tau(
        models, holdout_files, mean, std, batch_size=64, log_fn=_log_tau)

    rows = predict(models, test_files, mean, std, out_path, tau=best_tau, batch_size=64)
    csv_path = os.path.join(out_path, "predictions.csv")
    write_predictions_csv(rows, csv_path)
    print(f"[out] wrote {csv_path} with {len(rows)} rows (tau={best_tau})")

    # save all 3 EMA model checkpoints + sentinel. best_model.pth = seed-0 model
    # (single-member fallback); the full ensemble weights are saved alongside.
    torch.save(models[0].state_dict(), os.path.join(out_path, "best_model.pth"))
    for si, (sd, m) in enumerate(zip(SEEDS, models)):
        torch.save(m.state_dict(),
                   os.path.join(out_path, f"ema_seed{sd}.pth"))
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({"arch": "MultiBranchViewFusionUNet",
                     "mean": mean.tolist(), "std": std.tolist(),
                     "max_steps": max_steps, "seeds": SEEDS,
                     "n_models": len(models),
                     "tau": best_tau, "holdout_wpa": best_wpa,
                     "argmax_wpa": base_wpa}, f)
    # also drop a sentinel inside the instance dir for convenience
    with open(os.path.join(out_path, "best_model.pkl"), "wb") as f:
        pickle.dump({"done": True}, f)

    print(f"[done] total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
