"""
Chicken Counting -- DM-Count style distribution-matching / OT density decoder (node_4).

Design: CSRNet VGG16-BN backbone (vgg16_bn first-10-conv frontend, dilated 6-layer
backend, 1x1 head + Softplus, output 180x320 at stride 4), trained with a DM-Count-style
loss: L1 count loss + entropic-OT (Sinkhorn, fp32) between L1-normalized predicted and GT
densities + TV term. AdamW + cosine LR + fp16 autocast (Sinkhorn in fp32) + grad-clip + EMA.

Grading mode: standalone run.py. Reads DATA_DIR, writes predictions.npy of shape
(100, 180, 320) (non-negative) plus a best_model.pkl sentinel into OUTPUT_DIR.
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

# ----------------------------------------------------------------------------- paths
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/path/to/data",
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
INSTANCE = "chicken_counting"

# Environment-overridable knobs (used by smoke micro-benchmark; defaults = full run).
MAX_STEPS = int(os.environ.get("CC_MAX_STEPS", "6000"))
SMOKE = os.environ.get("CC_SMOKE", "0") == "1"
HOLDOUT_EVAL = os.environ.get("CC_HOLDOUT", "0") == "1"  # carve internal val for diagnostic

# ----------------------------------------------------------------------------- config
SEED = 1234
BATCH = 8
CROP_IMG = 512          # image crop
CROP_DEN = CROP_IMG // 4  # density crop (stride 4) = 128
IMG_H, IMG_W = 720, 1280
DEN_H, DEN_W = 180, 320
LR_BACKEND = 1e-4
LR_CONV4 = 1e-5
LR_MIN = 1e-6
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
LAMBDA_OT = 0.001
LAMBDA_TV = 0.001
LAMBDA_MSE = 1e4   # per-pixel MSE on raw density maps (values ~1e-3); primary loss
LAMBDA_COUNT = 0.01  # count-L1 demoted to a small calibration auxiliary
SINKHORN_ITERS = 10
SINKHORN_EPS = 0.1
NUM_WORKERS = 8

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- model
class CSRNetVGG16BN(nn.Module):
    """CSRNet with VGG16-BN frontend (first 10 conv layers, up to conv4_3 -> stride 8?)."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import vgg16_bn
        try:
            from torchvision.models import VGG16_BN_Weights
            weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = vgg16_bn(weights=weights)
        except Exception:
            vgg = vgg16_bn(pretrained=pretrained)

        # VGG16-BN features: we take layers up to and including the 3rd pooling
        # region so total downsample is stride 4 (frontend of CSRNet keeps stride 8
        # for standard CSRNet, but the density map here is stride 4 relative to input).
        # Layout of vgg16_bn.features with BN:
        #  conv-bn-relu blocks separated by MaxPool2d at indices 6,13,23,33,43.
        # Frontend "first 10 conv" -> up to conv4_3 would be index 33 (stride 8).
        # To output at stride 4 we keep frontend up to the 2nd pool (index 13 -> stride 4)?
        # CSRNet standard: frontend = first 10 conv layers of VGG16 = up to conv4_3,
        # giving stride 8. We then bilinearly upsample x2 to reach the stride-4 target.
        feats = list(vgg.features.children())
        # first 10 conv layers of VGG16(-BN) = conv1_1..conv4_3. Pools at idx 6,13,23,33.
        # feats[:33] stops just BEFORE the 4th pool (idx 33) -> ends at conv4_3 relu,
        # 512 channels, total downsample stride 8.
        self.frontend = nn.Sequential(*feats[:33])  # ends at conv4_3 relu, 512 ch, stride 8

        # dilated 6-layer backend (CSRNet backend), dilation=2
        self.backend = nn.Sequential(
            self._cbr(512, 512, 2),
            self._cbr(512, 512, 2),
            self._cbr(512, 512, 2),
            self._cbr(512, 256, 2),
            self._cbr(256, 128, 2),
            self._cbr(128, 64, 2),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.softplus = nn.Softplus()

        # init backend + head
        for m in list(self.backend.modules()) + [self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Bias the head so the initial softplus density is small. Target mean density
        # per stride-4 cell ~ 30 counts / 57600 cells ~= 5e-4. softplus(b)=5e-4 -> b~-7.6.
        # This starts the predicted count near the true count magnitude, so the L1 count
        # loss does not have to descend a huge constant offset over hundreds of steps.
        nn.init.constant_(self.output_layer.bias, -7.6)

    @staticmethod
    def _cbr(cin, cout, dil):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B,3,H,W). frontend -> stride 8. Upsample x2 -> stride 4 target.
        feat = self.frontend(x)
        feat = self.backend(feat)
        out = self.output_layer(feat)  # (B,1,H/8,W/8)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.softplus(out)  # non-negative density
        return out


def build_param_groups(model):
    """Discriminative LR: freeze conv1-conv3 (first ~3 conv blocks), conv4 at 0.1x."""
    frozen, conv4, backend = [], [], []
    # frontend indices: conv1_1..conv3_3 are the first blocks; conv4_* is the last block
    # of the frontend (channels 256->512). We identify by module position.
    feats = list(model.frontend.children())
    # Determine index boundaries: pools at 6,13,23,33 (relative). After 2nd pool (idx 13)
    # begins conv3; after 3rd pool (idx 23) begins conv4 block.
    for i, m in enumerate(feats):
        params = [p for p in m.parameters() if p.requires_grad]
        if not params:
            continue
        if i < 24:          # conv1..conv3 (up to 3rd pool at idx 23) -> freeze
            for p in params:
                p.requires_grad = False
        else:               # conv4 block (idx 24..32) -> low LR
            conv4 += params
    backend += [p for p in model.backend.parameters()]
    backend += [p for p in model.output_layer.parameters()]

    groups = [
        {"params": conv4, "lr": LR_CONV4},
        {"params": backend, "lr": LR_BACKEND},
    ]
    return groups


# ----------------------------------------------------------------------------- EMA
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


# ----------------------------------------------------------------------------- loss
def sinkhorn_loss(pred_p, gt_p, cost, eps, iters):
    """Entropic-regularized OT (Sinkhorn) between two prob vectors, fp32.
    pred_p, gt_p: (B, N) non-negative summing to 1. cost: (N, N). Returns mean transport cost.
    """
    pred_p = pred_p.float().clamp_min(1e-12)
    gt_p = gt_p.float().clamp_min(1e-12)
    K = torch.exp(-cost / eps)  # (N,N)
    B, N = pred_p.shape
    u = torch.ones(B, N, device=pred_p.device)
    v = torch.ones(B, N, device=pred_p.device)
    # a = pred_p (source), b = gt_p (target)
    for _ in range(iters):
        Kv = torch.einsum("nm,bm->bn", K, v)
        u = pred_p / (Kv + 1e-12)
        Ku = torch.einsum("mn,bm->bn", K, u)
        v = gt_p / (Ku + 1e-12)
    # transport plan P = diag(u) K diag(v); cost = <P, cost>
    # <P,C> = sum_ij u_i K_ij v_j C_ij = sum_i u_i * ( (K*C) @ v )_i
    KC = K * cost  # (N,N)
    Pv = torch.einsum("nm,bm->bn", KC, v)
    trans = (u * Pv).sum(dim=1)  # (B,)
    return trans.mean()


def build_cost_matrix(h, w, device):
    """Squared-euclidean cost over an h x w grid, normalized to [0,1]."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)  # (N,2)
    d2 = torch.cdist(coords, coords, p=2) ** 2  # (N,N)
    d2 = d2 / d2.max().clamp_min(1e-12)
    return d2


# ----------------------------------------------------------------------------- data
class ChickenDataset(Dataset):
    def __init__(self, img_dir, den_dir, ids, train=True):
        self.img_dir = img_dir
        self.den_dir = den_dir
        self.ids = ids
        self.train = train

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        i = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{i}.png")).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0  # (720,1280,3)
        den = np.load(os.path.join(self.den_dir, f"{i}.npy")).astype(np.float32)  # (180,320)
        return img, den

    def __getitem__(self, idx):
        img, den = self._load(idx)
        if self.train:
            img, den = self._augment(img, den)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        den = torch.from_numpy(den.copy()).unsqueeze(0)
        img = (img - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        return img, den

    def _augment(self, img, den):
        H, W = img.shape[:2]
        dH, dW = den.shape

        # mass-preserving isotropic scale [0.8, 1.2] via resizing image; density resized
        # and re-normalized so total count is preserved.
        s = random.uniform(0.8, 1.2)
        if abs(s - 1.0) > 1e-3:
            newW, newH = int(round(W * s)), int(round(H * s))
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((newW, newH), Image.BILINEAR)
            img = np.asarray(img_pil, dtype=np.float32) / 255.0
            old_sum = den.sum()
            ndW, ndH = max(1, newW // 4), max(1, newH // 4)
            den_pil = Image.fromarray(den)
            den = np.array(den_pil.resize((ndW, ndH), Image.BILINEAR), dtype=np.float32)
            new_sum = den.sum()
            if new_sum > 1e-8:
                den *= (old_sum / new_sum)
            H, W = img.shape[:2]
            dH, dW = den.shape

        # random joint crop 512x512 (image) with co-located 128x128 density crop
        ch = min(CROP_IMG, H)
        cw = min(CROP_IMG, W)
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        img = img[top:top + ch, left:left + cw]
        dtop, dleft = top // 4, left // 4
        dch, dcw = ch // 4, cw // 4
        den = den[dtop:dtop + dch, dleft:dleft + dcw]

        # pad to exact crop size if scaling made it smaller
        if img.shape[0] != CROP_IMG or img.shape[1] != CROP_IMG:
            pimg = np.zeros((CROP_IMG, CROP_IMG, 3), dtype=np.float32)
            pimg[:img.shape[0], :img.shape[1]] = img
            img = pimg
            pden = np.zeros((CROP_DEN, CROP_DEN), dtype=np.float32)
            pden[:den.shape[0], :den.shape[1]] = den
            den = pden

        # horizontal flip p=0.5
        if random.random() < 0.5:
            img = img[:, ::-1]
            den = den[:, ::-1]

        # photometric jitter (brightness/contrast) -- mild
        if random.random() < 0.8:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = (img - mean) * c + mean * b
            img = np.clip(img, 0.0, 1.0)

        return np.ascontiguousarray(img), np.ascontiguousarray(den)


# ----------------------------------------------------------------------------- train
def train_model(train_ids, img_dir, den_dir, max_steps, device, log_fn, smoke=False, seed=SEED):
    set_seed(seed)
    model = CSRNetVGG16BN(pretrained=True).to(device)
    groups = build_param_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=LR_MIN)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, EMA_DECAY)

    cost = build_cost_matrix(CROP_DEN, CROP_DEN, device)  # (N,N), N=128*128=16384

    ds = ChickenDataset(img_dir, den_dir, train_ids, train=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=True, persistent_workers=(NUM_WORKERS > 0))

    model.train()
    step = 0
    t_start = time.time()
    step_times = []
    data_iter = iter(loader)
    while step < max_steps:
        try:
            img, den = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            img, den = next(data_iter)
        t0 = time.time()
        img = img.to(device, non_blocking=True)
        den = den.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = model(img)  # (B,1,128,128)

        B = pred.shape[0]
        pred_f = pred.float().reshape(B, -1)  # (B,N)
        gt_f = den.float().reshape(B, -1)

        pred_sum = pred_f.sum(dim=1)
        gt_sum = gt_f.sum(dim=1)

        # (1) count loss (demoted to small auxiliary)
        count_loss = (pred_sum - gt_sum).abs().mean()

        # (0) PRIMARY: scaled per-pixel MSE on the raw density maps -- forces
        # image-dependent spatial structure and breaks constant-count collapse.
        mse_loss = F.mse_loss(pred_f, gt_f)

        # normalized distributions
        pred_p = pred_f / pred_sum.clamp_min(1e-8).unsqueeze(1)
        gt_p = gt_f / gt_sum.clamp_min(1e-8).unsqueeze(1)

        # (3) TV term
        tv_loss = (pred_p - gt_p).abs().sum(dim=1).mean()

        # (2) OT / Sinkhorn (fp32) -- guard against empty GT maps
        ot_loss = sinkhorn_loss(pred_p, gt_p, cost, SINKHORN_EPS, SINKHORN_ITERS)

        loss = LAMBDA_MSE * mse_loss + LAMBDA_COUNT * count_loss + LAMBDA_OT * ot_loss + LAMBDA_TV * tv_loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        step += 1

        if step % 50 == 0 or step <= 5 or step == max_steps:
            log_fn({
                "step": step, "elapsed_seconds": round(time.time() - t_start, 2),
                "event_type": "train_step", "seed": seed,
                "metrics": {
                    "loss": float(loss.item()),
                    "mse_loss": float(mse_loss.item()),
                    "count_loss": float(count_loss.item()),
                    "ot_loss": float(ot_loss.item()),
                    "tv_loss": float(tv_loss.item()),
                    "lr": float(opt.param_groups[-1]["lr"]),
                    "pred_sum_mean": float(pred_sum.mean().item()),
                    "gt_sum_mean": float(gt_sum.mean().item()),
                },
            })

    steady = float(np.median(step_times[4:])) if len(step_times) > 4 else float(np.median(step_times))
    return ema.shadow, model, steady


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(models, img_dir, ids, device):
    """Ensemble + hflip-TTA prediction.

    `models` is a single model or a list of models. For each image we average the
    density map over every (model x {identity, horizontal-flip}) view. Averaging in
    density space directly reduces the residual per-image count variance (the mean of
    the views' counts equals the count of the mean map, so the ensembled sum is the
    mean of the members' sums). The same averaging is applied to the calibration-fit
    train images and the test images, so the single global k stays matched to the
    ensembled output distribution.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        m.eval()
    n_views = len(models) * 2  # each model x {identity, hflip}
    preds = []
    for i in ids:
        img = Image.open(os.path.join(img_dir, f"{i}.png")).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        t = t.to(device)
        t_flip = torch.flip(t, dims=[-1])  # horizontal flip
        acc = None
        for m in models:
            for view, do_unflip in ((t, False), (t_flip, True)):
                with torch.cuda.amp.autocast():
                    out = m(view)  # (1,1,180,320)
                out = out.float().clamp_min(0.0)
                if out.shape[-2:] != (DEN_H, DEN_W):
                    out = F.interpolate(out, size=(DEN_H, DEN_W),
                                        mode="bilinear", align_corners=False)
                    out = out.clamp_min(0.0)
                if do_unflip:
                    out = torch.flip(out, dims=[-1])  # map flipped-view density back
                acc = out if acc is None else acc + out
        acc = acc / n_views
        preds.append(acc.squeeze().cpu().numpy().astype(np.float32))
    return np.stack(preds, axis=0)


def counting_score(pred_maps, gt_counts):
    pred_counts = pred_maps.reshape(pred_maps.shape[0], -1).sum(axis=1)
    rel = np.abs(1.0 - pred_counts / np.clip(gt_counts, 1e-8, None))
    return float(np.exp(-rel.mean()))


# ----------------------------------------------------------------------------- main
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_inst = os.path.join(OUTPUT_DIR, INSTANCE)
    os.makedirs(out_inst, exist_ok=True)

    data_path = os.path.join(DATA_DIR, INSTANCE)
    train_img = os.path.join(data_path, "train", "images")
    train_den = os.path.join(data_path, "train", "densities")
    test_img = os.path.join(data_path, "test", "images")

    all_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(train_img, "*.png")))
    test_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(test_img, "*.png")))

    log_path = os.path.join(OUTPUT_DIR, "training_progress.jsonl")
    logf = open(log_path, "w")
    import json

    def log_fn(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        m = rec.get("metrics", {})
        print(f"[step {rec['step']}] loss={m.get('loss'):.4f} "
              f"count={m.get('count_loss'):.4f} ot={m.get('ot_loss'):.4f} "
              f"tv={m.get('tv_loss'):.4f} pred_sum={m.get('pred_sum_mean'):.2f} "
              f"gt_sum={m.get('gt_sum_mean'):.2f} t={rec['elapsed_seconds']}s", flush=True)

    max_steps = 20 if SMOKE else MAX_STEPS

    # Optional internal holdout for diagnostic scoring only.
    if HOLDOUT_EVAL and not SMOKE:
        rng = random.Random(SEED)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        n_val = 20
        val_ids = sorted(shuffled[:n_val])
        tr_ids = sorted(shuffled[n_val:])
    else:
        tr_ids = all_ids
        val_ids = []

    log_fn({"step": 0, "elapsed_seconds": 0.0, "event_type": "start",
            "metrics": {"loss": 0.0, "count_loss": 0.0, "ot_loss": 0.0, "tv_loss": 0.0,
                        "lr": 0.0, "pred_sum_mean": 0.0, "gt_sum_mean": 0.0}})

    # --- Train TWO independently-seeded CSRNet models sequentially -------------
    # Different seeds -> de-correlated per-image count errors. Averaging their
    # (hflip-TTA'd) density maps reduces the residual per-image count variance that
    # is the diagnosed holdout<->train-fit gap, more reliably than a single-model
    # tweak. Each EMA model is ~65MB, so keeping both in memory is trivial.
    SEEDS = [1234, 2025]
    if SMOKE:
        SEEDS = [1234]  # smoke: single seed to keep the micro-benchmark cheap
    ema_models = []
    steady = None
    for si, sd in enumerate(SEEDS):
        print(f"[TRAIN] model {si + 1}/{len(SEEDS)} seed={sd}", flush=True)
        ema_model, raw_model, st = train_model(
            tr_ids, train_img, train_den, max_steps, device, log_fn,
            smoke=SMOKE, seed=sd)
        ema_models.append(ema_model)
        if steady is None:
            steady = st  # per-model steady step-time (benchmark uses model 1)
    ensemble = ema_models  # list of EMA models for ensembled prediction

    diag = None
    if val_ids:
        gt_counts = np.array([np.load(os.path.join(train_den, f"{i}.npy")).sum() for i in val_ids])
        vpred = predict(ensemble, train_img, val_ids, device)
        diag = counting_score(vpred, gt_counts)
        log_fn({"step": max_steps, "elapsed_seconds": round(steady * max_steps, 1),
                "event_type": "holdout_eval",
                "metrics": {"loss": 0.0, "count_loss": 0.0, "ot_loss": 0.0, "tv_loss": 0.0,
                            "lr": 0.0, "pred_sum_mean": float(vpred.reshape(len(val_ids), -1).sum(1).mean()),
                            "gt_sum_mean": float(gt_counts.mean())}})
        print(f"[HOLDOUT] counting_score(EMA) = {diag:.4f}", flush=True)

    # --- Train-derived global count calibration ---------------------------------
    # The per-pixel MSE objective localizes density well (predicted counts gain real
    # per-image spread) but systematically under-scales the TOTAL mass, so raw counts
    # are biased low. Since only the SUM is scored, estimate a single multiplicative
    # scale k on the TRAINING images (train-only, no leakage) that maximizes the
    # counting_score, then apply it to the test maps. k is found by a fine 1-D search
    # (the metric mean|1 - k*p/t| is piecewise-linear in k, so a dense grid is exact).
    cal_ids = all_ids  # calibrate on every available training image
    cal_pred = predict(ensemble, train_img, cal_ids, device)
    cal_pred_counts = cal_pred.reshape(len(cal_ids), -1).sum(axis=1)
    cal_gt_counts = np.array(
        [np.load(os.path.join(train_den, f"{i}.npy")).sum() for i in cal_ids],
        dtype=np.float64)
    valid = cal_pred_counts > 1e-6
    if valid.sum() >= 5:
        ratios = cal_gt_counts[valid] / cal_pred_counts[valid]
        lo, hi = float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))
        grid = np.linspace(max(0.1, lo * 0.5), hi * 1.5, 400)
        best_k, best_s = 1.0, -1.0
        for k in grid:
            s = counting_score(cal_pred * k, cal_gt_counts)
            if s > best_s:
                best_s, best_k = s, float(k)
        cal_k = best_k
    else:
        cal_k = 1.0
    print(f"[CALIBRATION] global scale k={cal_k:.4f} "
          f"(train raw mean={cal_pred_counts.mean():.2f}, "
          f"gt mean={cal_gt_counts.mean():.2f}, "
          f"train score raw={counting_score(cal_pred, cal_gt_counts):.4f} "
          f"-> cal={counting_score(cal_pred * cal_k, cal_gt_counts):.4f})", flush=True)

    # Re-diagnose the internal holdout with the calibrated scale, if available.
    if val_ids:
        gt_counts_v = np.array([np.load(os.path.join(train_den, f"{i}.npy")).sum()
                                for i in val_ids])
        vpred_c = predict(ensemble, train_img, val_ids, device) * cal_k
        diag = counting_score(vpred_c, gt_counts_v)
        print(f"[HOLDOUT] counting_score(ensemble, calibrated) = {diag:.4f}", flush=True)

    # Test predictions with the ensemble (hflip-TTA averaged), calibrated
    test_preds = predict(ensemble, test_img, test_ids, device)
    assert test_preds.shape == (len(test_ids), DEN_H, DEN_W), test_preds.shape
    test_preds = np.clip(test_preds * cal_k, 0.0, None).astype(np.float32)

    np.save(os.path.join(out_inst, "predictions.npy"), test_preds)

    # sentinel + checkpoints (save every ensemble member's EMA weights)
    torch.save(ensemble[0].state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
    for mi, m in enumerate(ensemble):
        torch.save(m.state_dict(), os.path.join(OUTPUT_DIR, f"ema_model_{mi}_seed{SEEDS[mi]}.pth"))
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        import pickle
        pickle.dump({"info": "2-seed EMA CSRNet-VGG16BN ensemble + hflip-TTA",
                     "seeds": SEEDS, "n_models": len(ensemble),
                     "steady_step_time_per_model": steady,
                     "max_steps": max_steps, "diag_score": diag, "cal_k": cal_k}, f)

    results = {
        "score": diag if diag is not None else None,
        "cal_k": cal_k,
        "steady_step_time_s": steady,
        "max_steps": max_steps,
        "n_models": len(ensemble),
        "seeds": SEEDS,
        "n_test": len(test_ids),
        "pred_count_mean": float(test_preds.reshape(len(test_ids), -1).sum(1).mean()),
        "pred_count_min": float(test_preds.reshape(len(test_ids), -1).sum(1).min()),
        "pred_count_max": float(test_preds.reshape(len(test_ids), -1).sum(1).max()),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log_fn({"step": max_steps, "elapsed_seconds": round(steady * max_steps, 1),
            "event_type": "training_complete",
            "metrics": {"loss": 0.0, "count_loss": 0.0, "ot_loss": 0.0, "tv_loss": 0.0,
                        "lr": 0.0, "pred_sum_mean": results["pred_count_mean"],
                        "gt_sum_mean": 0.0}})
    logf.close()
    print("DONE", results, flush=True)


if __name__ == "__main__":
    main()
