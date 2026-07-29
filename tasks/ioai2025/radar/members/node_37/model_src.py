import os
import csv
import glob
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/path/to/data",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(HERE, "output"))

INSTANCE = "radar"
H, W = 50, 181
N_CLASSES = 5  # remapped {0..4}; original {-1,0,1,2,3}

# Training hyperparameters (from design plan; do NOT tune)
LR = 6e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
BATCH_SIZE = 32
WARMUP_FRAC = 0.10
MIN_LR = 1e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.9995
DROP_PATH = 0.1
HEAD_DROP = 0.1
# Class weights for CE (bg 1, suitcase 15, chair 12, human 25, wall 8)
# remap: -1->0(bg), 0->1(suitcase), 1->2(chair), 2->3(human), 3->4(wall)
CE_WEIGHTS = [1.0, 15.0, 12.0, 25.0, 8.0]
DICE_WEIGHT = 1.0

# Runtime knobs
SMOKE = os.environ.get("RUN_SMOKE", "0") == "1"
BENCH = os.environ.get("RUN_BENCH", "0") == "1"
MAX_STEPS_ENV = os.environ.get("RUN_MAX_STEPS", "")
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def list_files(split):
    d = os.path.join(DATA_DIR, INSTANCE, split)
    return sorted(glob.glob(os.path.join(d, "*.pt")))


def load_train_tensors(files):
    X = np.empty((len(files), 6, H, W), dtype=np.float32)
    Y = np.empty((len(files), H, W), dtype=np.int64)
    for i, f in enumerate(files):
        t = torch.load(f, weights_only=False).float().numpy()
        X[i] = t[:6]
        lab = t[6].astype(np.int64)
        Y[i] = lab + 1  # remap -1..3 -> 0..4
    return X, Y


def load_test_tensors(files):
    X = np.empty((len(files), 6, H, W), dtype=np.float32)
    for i, f in enumerate(files):
        t = torch.load(f, weights_only=False).float().numpy()
        X[i] = t[:6]
    return X


# ----------------------------------------------------------------------------
# Model: small MiT-b0 (SegFormer) encoder + all-MLP decode head
# ----------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch, stride, pad):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=stride, padding=pad)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, Hh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, Hh, Ww


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, Hh, Ww):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, Hh, Ww)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
        kv = kv.reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x, Hh, Ww):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, Hh, Ww)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, sr_ratio, mlp_ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, Hh, Ww):
        x = x + self.drop_path(self.attn(self.norm1(x), Hh, Ww))
        x = x + self.drop_path(self.mlp(self.norm2(x), Hh, Ww))
        return x


class MiTb0(nn.Module):
    """4-stage MixVisionTransformer, gentle initial stride=2 for tiny input."""

    def __init__(self, in_ch=6, dims=(32, 64, 160, 256), depths=(2, 2, 2, 2),
                 heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1), mlp_ratio=4.0,
                 drop_path=0.1):
        super().__init__()
        self.dims = dims
        # stage1 gentle stride 2 (not 4); stages 2-4 stride 2 each
        self.patch1 = OverlapPatchEmbed(in_ch, dims[0], patch=7, stride=2, pad=3)
        self.patch2 = OverlapPatchEmbed(dims[0], dims[1], patch=3, stride=2, pad=1)
        self.patch3 = OverlapPatchEmbed(dims[1], dims[2], patch=3, stride=2, pad=1)
        self.patch4 = OverlapPatchEmbed(dims[2], dims[3], patch=3, stride=2, pad=1)

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            blocks = nn.ModuleList([
                Block(dims[i], heads[i], sr_ratios[i], mlp_ratio, dpr[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(dims[i]))
            cur += depths[i]
        self.patches = [self.patch1, self.patch2, self.patch3, self.patch4]

    def forward(self, x):
        feats = []
        for i in range(4):
            x, Hh, Ww = self.patches[i](x)
            for blk in self.stages[i]:
                x = blk(x, Hh, Ww)
            x = self.norms[i](x)
            x = x.reshape(x.shape[0], Hh, Ww, -1).permute(0, 3, 1, 2).contiguous()
            feats.append(x)
        return feats


class SegFormerHead(nn.Module):
    def __init__(self, dims, embed=256, n_classes=5, drop=0.1):
        super().__init__()
        self.linears = nn.ModuleList([nn.Conv2d(d, embed, 1) for d in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed * 4, embed, 1, bias=False),
            nn.BatchNorm2d(embed),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(drop)
        self.pred = nn.Conv2d(embed, n_classes, 1)

    def forward(self, feats, out_hw):
        target = feats[0].shape[2:]  # 1/4 resolution (relative to stage1)
        ups = []
        for f, lin in zip(feats, self.linears):
            y = lin(f)
            if y.shape[2:] != target:
                y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
            ups.append(y)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        x = self.drop(x)
        x = self.pred(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class SegFormer(nn.Module):
    def __init__(self, in_ch=6, n_classes=5, drop_path=0.1, head_drop=0.1):
        super().__init__()
        dims = (32, 64, 160, 256)
        self.encoder = MiTb0(in_ch=in_ch, dims=dims, drop_path=drop_path)
        self.head = SegFormerHead(dims, embed=256, n_classes=n_classes, drop=head_drop)

    def forward(self, x):
        out_hw = x.shape[2:]
        feats = self.encoder(x)
        return self.head(feats, out_hw)


# ----------------------------------------------------------------------------
# Loss: weighted CE + foreground-only soft Dice
# ----------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, ce_weights, dice_weight=1.0):
        super().__init__()
        self.register_buffer("w", torch.tensor(ce_weights, dtype=torch.float32))
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.w)
        probs = F.softmax(logits, dim=1)
        # foreground classes = 1..4 (bg is class 0)
        dice = 0.0
        n_fg = 0
        tgt_oh = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
        for c in range(1, N_CLASSES):
            p = probs[:, c]
            g = tgt_oh[:, c]
            inter = (p * g).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            d = 1 - (2 * inter + 1.0) / (denom + 1.0)
            dice = dice + d.mean()
            n_fg += 1
        dice = dice / max(n_fg, 1)
        return ce + self.dice_weight * dice, ce.detach(), dice.detach()


# ----------------------------------------------------------------------------
# Augmentation (geometry-safe): azimuth flip, small azimuth shift, gaussian noise
# ----------------------------------------------------------------------------
def augment_batch(x, y):
    # x: (B,6,H,W) tensor on device, y: (B,H,W)
    B = x.shape[0]
    # azimuth (last dim = W) horizontal flip p=0.5
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        idx = flip.nonzero(as_tuple=True)[0]
        x[idx] = torch.flip(x[idx], dims=[3])
        y[idx] = torch.flip(y[idx], dims=[2])
    # azimuth column shift +/-3 p=0.3 (per-sample roll along W)
    for b in range(B):
        if random.random() < 0.3:
            s = random.randint(-3, 3)
            if s != 0:
                x[b] = torch.roll(x[b], shifts=s, dims=2)
                y[b] = torch.roll(y[b], shifts=s, dims=1)
    # additive gaussian noise sigma=0.01 p=0.3
    noise_mask = torch.rand(B, device=x.device) < 0.3
    if noise_mask.any():
        idx = noise_mask.nonzero(as_tuple=True)[0]
        x[idx] = x[idx] + torch.randn_like(x[idx]) * 0.01
    return x, y


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ----------------------------------------------------------------------------
# LR schedule: linear warmup + cosine to MIN_LR
# ----------------------------------------------------------------------------
def lr_at(step, max_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    prog = min(1.0, prog)
    return MIN_LR + 0.5 * (base_lr - MIN_LR) * (1 + math.cos(math.pi * prog))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    out_path = os.path.join(OUTPUT_DIR, INSTANCE)
    os.makedirs(out_path, exist_ok=True)

    train_files = list_files("train")
    test_files = list_files("test")
    if SMOKE:
        train_files = train_files[:64]
        test_files = test_files[:16]
    print(f"[data] train={len(train_files)} test={len(test_files)}", flush=True)

    t0 = time.time()
    Xtr, Ytr = load_train_tensors(train_files)
    Xte = load_test_tensors(test_files)
    print(f"[data] loaded in {time.time()-t0:.1f}s Xtr={Xtr.shape} Xte={Xte.shape}", flush=True)

    # train-only per-channel standardization
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr)

    model = SegFormer(in_ch=6, n_classes=N_CLASSES, drop_path=DROP_PATH,
                      head_drop=HEAD_DROP).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M", flush=True)

    loss_fn = CombinedLoss(CE_WEIGHTS, DICE_WEIGHT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS,
                            weight_decay=WEIGHT_DECAY)

    # ---- determine max_steps ----
    if MAX_STEPS_ENV:
        max_steps = int(MAX_STEPS_ENV)
    elif SMOKE:
        max_steps = 8
    else:
        # Measured ~75 ms/step (steady) on this A100 at bs=32; the parent's
        # 0.15 s/step estimate was pessimistic. 9000 steps ~= 675 s training +
        # ~30 s first-step warmup, leaving headroom under the ~960 s cap while
        # using far more of the schedule than the 4800-step floor. The
        # wall-clock deadline below is the hard safety net that guarantees the
        # inference/CSV-write block always runs even if step-time is slower.
        max_steps = 7000
    warmup = max(1, int(WARMUP_FRAC * max_steps))
    # Wall-clock deadline: stop training in time to always reach the
    # inference + CSV-write block within the ~960 s per-node cap. Reserve
    # ~130 s for first-step warmup + inference + CSV write; deadline measured
    # from t0 (which already includes data-load time). Overridable for tests.
    TRAIN_DEADLINE_S = float(os.environ.get("RUN_TRAIN_DEADLINE_S", "1600"))
    print(f"[train] max_steps={max_steps} warmup={warmup} bs={BATCH_SIZE}", flush=True)

    ema = EMA(model, EMA_DECAY)
    n = Xtr_t.shape[0]

    def sample_batch():
        idx = torch.randint(0, n, (BATCH_SIZE,))
        xb = Xtr_t[idx].to(DEVICE, non_blocking=True)
        yb = Ytr_t[idx].to(DEVICE, non_blocking=True)
        return augment_batch(xb, yb)

    prog_path = os.path.join(out_path, "training_progress.jsonl")
    prog = open(prog_path, "w")
    step_times = []
    loss = torch.tensor(float("nan"))
    model.train()
    stop_step = max_steps
    for step in range(max_steps):
        if (time.time() - t0) > TRAIN_DEADLINE_S:
            print(f"[train] wall-clock deadline hit at step {step}; "
                  f"stopping training to reach inference/CSV-write block", flush=True)
            stop_step = step
            break
        st = time.time()
        lr = lr_at(step, max_steps, LR, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = sample_batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(xb)
            loss, ce, dice = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        ema.update(model)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - st
        step_times.append(dt)
        if step % 25 == 0 or step == max_steps - 1:
            import json
            prog.write(json.dumps({
                "step": step, "elapsed_seconds": round(time.time() - t0, 2),
                "event_type": "step",
                "metrics": {"train_loss": float(loss), "ce": float(ce),
                            "dice": float(dice), "lr": lr},
            }) + "\n")
            prog.flush()
            print(f"[step {step}] loss={float(loss):.4f} ce={float(ce):.4f} "
                  f"dice={float(dice):.4f} lr={lr:.2e} dt={dt*1000:.0f}ms", flush=True)

    steady = float(np.median(step_times[4:])) if len(step_times) > 4 else float(np.median(step_times))
    print(f"[train] done. steady step={steady*1000:.0f}ms", flush=True)

    if BENCH:
        with open(os.path.join(out_path, "bench.txt"), "w") as f:
            f.write(f"steady_step_s={steady}\n")
        import json
        prog.write(json.dumps({"step": max_steps, "event_type": "training_complete",
                               "elapsed_seconds": round(time.time() - t0, 2),
                               "metrics": {"steady_step_s": steady}}) + "\n")
        prog.close()
        return

    # ---- switch to EMA weights for inference ----
    infer_model = SegFormer(in_ch=6, n_classes=N_CLASSES, drop_path=0.0,
                            head_drop=0.0).to(DEVICE)
    ema.copy_to(infer_model)
    infer_model.eval()
    torch.save(infer_model.state_dict(), os.path.join(out_path, "best_model.pth"))

    # ---- inference with horizontal-flip TTA ----
    raw_dir = os.path.join(out_path, "raw_test")
    os.makedirs(raw_dir, exist_ok=True)
    rows = []
    Xte_t = torch.from_numpy(Xte)
    ib = 64
    all_preds = np.empty((len(test_files), H, W), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(test_files), ib):
            xb = Xte_t[s:s + ib].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                p1 = F.softmax(infer_model(xb).float(), dim=1)
                xf = torch.flip(xb, dims=[3])
                p2 = F.softmax(infer_model(xf).float(), dim=1)
                p2 = torch.flip(p2, dims=[3])
            probs = ((p1 + p2) / 2).cpu().numpy()  # (b,5,H,W)
            preds = probs.argmax(axis=1)  # (b,H,W) in {0..4}
            for j in range(probs.shape[0]):
                gi = s + j
                fn = os.path.basename(test_files[gi])
                # save per-sample softmax raw for ensembling
                np.save(os.path.join(raw_dir, fn.replace(".pt", "") + ".npy"),
                        probs[j].astype(np.float16))
                all_preds[gi] = preds[j]

    # remap {0..4} -> {-1,0,1,2,3}
    out_preds = all_preds - 1

    # ---- write predictions.csv ----
    csv_path = os.path.join(out_path, "predictions.csv")
    header = ["filename"] + [f"pixel_{k}" for k in range(H * W)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for gi, tf in enumerate(test_files):
            fn = os.path.basename(tf)
            flat = out_preds[gi].reshape(-1).tolist()
            writer.writerow([fn] + flat)
    print(f"[out] wrote {csv_path} rows={len(test_files)}", flush=True)

    # sentinel (write under both OUTPUT_DIR root and the instance subdir)
    for sent_dir in (OUTPUT_DIR, out_path):
        os.makedirs(sent_dir, exist_ok=True)
        with open(os.path.join(sent_dir, "best_model.pkl"), "wb") as f:
            f.write(b"sentinel")

    # results.json
    import json
    with open(os.path.join(out_path, "results.json"), "w") as f:
        json.dump({"status": "ok", "max_steps": max_steps,
                   "steps_run": int(stop_step), "steady_step_s": steady,
                   "n_test": len(test_files)}, f)
    prog.write(json.dumps({"step": max_steps, "event_type": "training_complete",
                           "elapsed_seconds": round(time.time() - t0, 2),
                           "metrics": {"final_loss": float(loss)}}) + "\n")
    prog.close()
    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
