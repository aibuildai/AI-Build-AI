"""
Node 3 -- Direct global count regression (metric-aligned scalar head).

Design: bypass per-pixel density; regress the scalar chicken count directly with a
ResNet34 (ImageNet-pretrained) backbone + GAP + MLP(512->256->1) + Softplus head.
The required (100,180,320) output is produced by multiplying the predicted count by a
fixed average spatial prior (mean of L1-normalized training densities), so each map
sums exactly to the regressed count.

Loss = mean|1 - pred/true| (the metric term) + 0.3 * smoothL1(log1p(pred), log1p(true)).
Augmentation: hflip, photometric jitter, area>=80% random-resized-crop with mass-based
count rescaling, Mixup(image,count) p=0.3. AdamW head 3e-4 / backbone 3e-5, cosine to
1e-6, fp16 autocast, grad-clip 1.0, EMA 0.999. Frozen stem+layer1.

Entry: reads DATA_DIR, writes predictions.npy + best_model.pkl sentinel under OUTPUT_DIR.
"""
import os
import sys
import time
import copy
import math
import glob
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms.functional as TF

# -------------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/path/to/data",
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/path/to/run",
)
INSTANCE = "chicken_counting"

# route torch hub cache inside the run tree (already populated there)
os.environ.setdefault(
    "TORCH_HOME",
    "/path/to/run",
)

# ---- design hyperparameters (fixed, do NOT tune) ----
IMG_H, IMG_W = 720, 1280         # native resolution (finest cluster detail for the high-count tail)
DENS_H, DENS_W = 180, 320        # output map size
BATCH = 8                        # halved for native res; grad-accum keeps effective batch 16
GRAD_ACCUM = 2                   # effective batch = BATCH * GRAD_ACCUM = 16
HEAD_LR = 3e-4
BACKBONE_LR = 5e-5               # raised from 3e-5 so layer2 adapts to dense-scene texture
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
LOGCOUNT_W = 0.3
MIXUP_P = 0.15
HFLIP_P = 0.5
JITTER_P = 0.5
CROP_MIN_AREA = 0.90
SEED = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Budget knob -- sized from smoke micro-benchmark at NATIVE 720x1280 + grad-accum 2:
# steady ~86 ms per optimizer step on A100 => 4000 steps ~= 6 min train + inference well
# under the 30-min cap. TIME_BUDGET_S caps wall-clock as a safety guard.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "4000"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "1800"))  # 30 min hard cap


def log(msg):
    print(f"[node3] {msg}", flush=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class CountDataset(Dataset):
    """Yields (image_tensor[3,H,W], scalar_count). Lazy loading, no preload."""

    def __init__(self, img_paths, dens_paths, train=True):
        self.img_paths = img_paths
        self.dens_paths = dens_paths
        self.train = train

    def __len__(self):
        return len(self.img_paths)

    def _load_img(self, path):
        with Image.open(path) as im:
            return im.convert("RGB")

    def __getitem__(self, idx):
        img = self._load_img(self.img_paths[idx])          # PIL, 1280x720
        dens = np.load(self.dens_paths[idx]).astype(np.float32)  # 180x320
        count = float(dens.sum())

        if self.train:
            # --- random-resized-crop keeping >=80% area, count rescaled by GT mass in crop ---
            if random.random() < 0.9:
                iw, ih = img.size  # 1280,720
                area_frac = random.uniform(CROP_MIN_AREA, 1.0)
                # aspect near-square-ish jitter but keep within image
                ar = random.uniform(0.85, 1.18)
                cw = min(iw, int(round(math.sqrt(area_frac * iw * ih * ar))))
                ch = min(ih, int(round(math.sqrt(area_frac * iw * ih / ar))))
                cw = max(1, cw); ch = max(1, ch)
                x0 = random.randint(0, iw - cw)
                y0 = random.randint(0, ih - ch)
                # map crop box to density grid (density is 4x downsample of ORIGINAL 720x1280)
                dx0 = int(round(x0 / iw * DENS_W)); dx1 = int(round((x0 + cw) / iw * DENS_W))
                dy0 = int(round(y0 / ih * DENS_H)); dy1 = int(round((y0 + ch) / ih * DENS_H))
                dx1 = max(dx1, dx0 + 1); dy1 = max(dy1, dy0 + 1)
                crop_count = float(dens[dy0:dy1, dx0:dx1].sum())
                img = img.crop((x0, y0, x0 + cw, y0 + ch))
                count = crop_count

            img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

            # horizontal flip (count-invariant)
            if random.random() < HFLIP_P:
                img = TF.hflip(img)

            # photometric brightness/contrast jitter +-0.2
            if random.random() < JITTER_P:
                b = 1.0 + random.uniform(-0.2, 0.2)
                c = 1.0 + random.uniform(-0.2, 0.2)
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)
        else:
            img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

        t = TF.to_tensor(img)  # [3,H,W] in [0,1]
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, torch.tensor(count, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with Image.open(self.img_paths[idx]) as im:
            img = im.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        t = TF.to_tensor(img)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class CountNet(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        try:
            backbone = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        except Exception as e:
            log(f"pretrained load failed ({e}); using random init")
            backbone = tvm.resnet34(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        # freeze stem + layer1
        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        out = self.head(x).squeeze(1)
        return F.softplus(out)  # non-negative scalar count

    def param_groups(self, head_lr, backbone_lr, wd):
        head_params, bb_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                head_params.append(p)
            else:
                bb_params.append(p)
        return [
            {"params": head_params, "lr": head_lr, "weight_decay": wd},
            {"params": bb_params, "lr": backbone_lr, "weight_decay": wd},
        ]


# -------------------------------------------------------------------------
# EMA
# -------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def rel_count_loss(pred, true, mean_train_count):
    """Count-proportional weighted relative-count loss + unweighted log-count smoothL1.

    Per-image relative term is weighted by w_i = clip(true_i / mean_train_count, 0.5, 2.0)
    then normalized so mean(w)=1 over the batch -- concentrates gradient on the underfit
    high-count tail while preserving the metric's per-image relative structure. The
    0.3*smoothL1(log1p) stability term stays UNWEIGHTED.
    """
    eps = 1e-6
    rel = torch.abs(1.0 - pred / (true + eps))
    w = torch.clamp(true / (mean_train_count + eps), 0.5, 2.0)
    w = w / (w.mean() + eps)  # normalize so mean(w)=1 over the batch
    rel_w = (w * rel).mean()
    log_l = F.smooth_l1_loss(torch.log1p(pred), torch.log1p(true))
    return rel_w + LOGCOUNT_W * log_l


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    t_start = time.time()
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prog_path = os.path.join(OUTPUT_DIR, "training_progress.jsonl")
    prog = open(prog_path, "w")

    def emit(step, event_type, metrics):
        import json
        prog.write(json.dumps({
            "step": step, "epoch": None,
            "elapsed_seconds": round(time.time() - t_start, 2),
            "event_type": event_type, "metrics": metrics,
        }) + "\n")
        prog.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inst_dir = os.path.join(DATA_DIR, INSTANCE)
    train_img_dir = os.path.join(inst_dir, "train", "images")
    train_dens_dir = os.path.join(inst_dir, "train", "densities")
    test_img_dir = os.path.join(inst_dir, "test", "images")

    img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    dens_paths = [
        os.path.join(train_dens_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")
        for p in img_paths
    ]
    test_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.png")))
    log(f"train n={len(img_paths)} test n={len(test_paths)} device={device}")

    # --- fixed spatial prior: mean of L1-normalized training densities, normalized to sum 1 ---
    acc = np.zeros((DENS_H, DENS_W), dtype=np.float64)
    for dp in dens_paths:
        d = np.load(dp).astype(np.float64)
        s = d.sum()
        if s > 0:
            acc += d / s
    prior = acc / max(len(dens_paths), 1)
    prior = prior / prior.sum()  # sums to 1
    prior = prior.astype(np.float32)
    log(f"spatial prior sum={prior.sum():.6f} max={prior.max():.6g}")

    # --- mean training count (train-only) for count-proportional loss weighting ---
    train_counts_all = np.array(
        [float(np.load(dp).astype(np.float32).sum()) for dp in dens_paths], dtype=np.float64)
    mean_train_count = float(train_counts_all.mean())
    log(f"mean_train_count={mean_train_count:.3f} "
        f"(min={train_counts_all.min():.1f} max={train_counts_all.max():.1f})")

    # --- data ---
    train_ds = CountDataset(img_paths, dens_paths, train=True)
    n_workers = 8 if not SMOKE else 2
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=n_workers,
        pin_memory=True, drop_last=True, persistent_workers=(n_workers > 0),
    )

    # --- model ---
    model = CountNet().to(device)
    ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.param_groups(HEAD_LR, BACKBONE_LR, WEIGHT_DECAY))

    global MAX_STEPS
    max_steps = MAX_STEPS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    it = infinite(train_loader)
    model.train()
    step = 0
    running = 0.0
    step_times = []
    emit(0, "training_start", {"max_steps": max_steps})

    while step < max_steps:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        # gradient accumulation: GRAD_ACCUM micro-batches per optimizer step (effective batch 16)
        loss_val = 0.0
        for _ in range(GRAD_ACCUM):
            imgs, counts = next(it)
            imgs = imgs.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)

            # Mixup on (image, count) pairs
            if random.random() < MIXUP_P:
                lam = np.random.beta(0.4, 0.4)
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]
                counts = lam * counts + (1 - lam) * counts[perm]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(imgs)
                loss = rel_count_loss(pred, counts, mean_train_count) / GRAD_ACCUM
            scaler.scale(loss).backward()
            loss_val += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        step += 1
        loss = torch.tensor(loss_val)  # combined loss over the accumulated micro-batches
        running += loss.item()
        dt = time.time() - t0
        if step > 4:
            step_times.append(dt)

        if step % 50 == 0 or step == 1:
            avg = running / (50 if step % 50 == 0 else 1)
            running = 0.0 if step % 50 == 0 else running
            lr = sched.get_last_lr()[0]
            emit(step, "train_step", {"train_loss": round(loss.item(), 5),
                                      "avg_loss": round(avg, 5), "learning_rate": lr})
            log(f"step {step}/{max_steps} loss={loss.item():.4f} avg={avg:.4f} "
                f"lr={lr:.2e} {dt*1000:.0f}ms/it")

        # time-budget guard (optional scheduling override)
        if TIME_BUDGET_S > 0 and (time.time() - t_start) > TIME_BUDGET_S:
            log(f"time budget {TIME_BUDGET_S}s reached at step {step}; stopping")
            break

    steady = float(np.median(step_times)) if step_times else 0.0
    log(f"training done: {step} steps, steady step={steady*1000:.0f}ms")
    emit(step, "training_complete", {"steps": step, "steady_step_s": round(steady, 4)})

    # --- inference with EMA weights (hflip TTA) ---
    net = ema.shadow.to(device).eval()

    def predict_counts(paths):
        ds = TestDataset(paths)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=8 if not SMOKE else 2, pin_memory=True)
        out = []
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    c0 = net(imgs)
                    c1 = net(torch.flip(imgs, dims=[3]))  # hflip is count-invariant
                c = 0.5 * (c0 + c1)
                out.append(c.float().cpu().numpy())
        return np.concatenate(out)

    # --- train-only global calibration: fit alpha minimizing the exact metric on TRAIN ---
    train_true = np.array([float(np.load(dp).astype(np.float32).sum()) for dp in dens_paths],
                          dtype=np.float64)
    train_pred = predict_counts(img_paths).astype(np.float64)
    alphas = np.linspace(0.80, 1.30, 501)
    errs = [np.mean(np.abs(1.0 - a * train_pred / np.maximum(train_true, 1e-6))) for a in alphas]
    alpha = float(alphas[int(np.argmin(errs))])
    log(f"train-fit calibration alpha={alpha:.4f} (train err {min(errs):.4f})")

    counts_pred = predict_counts(test_paths) * alpha  # (N_test,)
    log(f"pred counts: n={len(counts_pred)} min={counts_pred.min():.2f} "
        f"max={counts_pred.max():.2f} mean={counts_pred.mean():.2f} std={counts_pred.std():.2f}")

    # --- build (100,180,320) density maps = count * prior ---
    prior_t = prior[None, :, :]  # (1,180,320)
    maps = counts_pred[:, None, None] * prior_t  # (N,180,320)
    maps = np.clip(maps, 0.0, None).astype(np.float32)

    out_inst = os.path.join(OUTPUT_DIR, INSTANCE)
    os.makedirs(out_inst, exist_ok=True)
    np.save(os.path.join(out_inst, "predictions.npy"), maps)
    log(f"saved predictions.npy shape={maps.shape} sum-per-map "
        f"mean={maps.sum(axis=(1,2)).mean():.3f}")

    # raw_test artifact for the aggregator (scalar counts + reconstructable maps)
    np.save(os.path.join(OUTPUT_DIR, "raw_test_counts.npy"), counts_pred.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "spatial_prior.npy"), prior)

    # checkpoint (EMA weights)
    torch.save({"model": net.state_dict(), "prior": prior}, os.path.join(OUTPUT_DIR, "best_model.pth"))

    # sentinel required by grading
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        import pickle
        pickle.dump({"steps": step, "pred_count_mean": float(counts_pred.mean())}, f)

    # results.json
    import json
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({
            "steps": step, "steady_step_s": steady,
            "pred_count_mean": float(counts_pred.mean()),
            "pred_count_std": float(counts_pred.std()),
        }, f, indent=2)
    prog.close()
    log("DONE")


if __name__ == "__main__":
    main()
