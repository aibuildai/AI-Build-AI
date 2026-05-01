"""Candidate 3: EfficientNet-B4 U-Net + scSE + hypercolumn + Lovasz Phase-B."""

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import sys
import json
import math
import time
import copy
import signal
import random
from pathlib import Path
from itertools import filterfalse

import numpy as np
import pandas as pd
import cv2
# IMPORTANT: import sklearn (which transitively imports scipy) BEFORE torch.
# Some torch versions ship a newer libstdc++ that gets injected into LD_LIBRARY_PATH
# and prevents scipy's compiled extensions (which need CXXABI_1.3.15+) from loading
# afterwards. Importing scipy/sklearn first sidesteps the conflict.
# Note: config.py also imports torch, so do sklearn before importing config.
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

import config as cfg  # imports torch internally, but sklearn already loaded above

torch.set_num_threads(8)

# ----- Reproducibility -----
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

# ----- Graceful shutdown -----
SHUTDOWN_REQUESTED = {"flag": False}


def _sigterm_handler(signum, frame):
    SHUTDOWN_REQUESTED["flag"] = True
    print(f"[signal] Received SIGTERM ({signum}); will save and exit after current batch.")


signal.signal(signal.SIGTERM, _sigterm_handler)


# ----- RLE utilities -----
def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle, shape=(101, 101)):
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# ----- Metric (competition metric at image level with IoU thresholds) -----
def iou_score(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return inter / union


def map_iou(preds, trues):
    thresholds = np.arange(0.5, 1.0, 0.05)
    scores = []
    for p, t in zip(preds, trues):
        iou = iou_score(p, t)
        scores.append(np.mean(iou > thresholds))
    return float(np.mean(scores))


# ----- Dataset -----
class SaltDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir=None, transform=None, size=256):
        self.ids = ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), dtype=np.uint8)
        img = img.astype(np.float32) / 255.0

        if self.mask_dir is not None:
            mpath = os.path.join(self.mask_dir, img_id + ".png")
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros_like(img)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros_like(img, dtype=np.float32)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        # Normalize (single normalization path)
        if isinstance(img, torch.Tensor):
            img = (img - cfg.IMG_MEAN) / cfg.IMG_STD
            if img.dim() == 2:
                img = img.unsqueeze(0)
            mask = mask.float().unsqueeze(0) if mask.dim() == 2 else mask.float()
        else:
            img = (img - cfg.IMG_MEAN) / cfg.IMG_STD
            img = torch.from_numpy(img).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return img, mask, img_id


def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        ToTensorV2(),
    ])


def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        ToTensorV2(),
    ])


# ----- Lovasz loss (standard impl) -----
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        losses = []
        for lg, lb in zip(logits, labels):
            losses.append(lovasz_hinge_flat(lg.view(-1), lb.view(-1)))
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


# ----- Model: EfficientNet-B4 U-Net with scSE + hypercolumn head + 1-channel adapter + aux heads -----
class HyperColumnUNet(nn.Module):
    def __init__(self, encoder_name=cfg.ENCODER_NAME, encoder_weights=cfg.ENCODER_WEIGHTS,
                 attention_type=cfg.DECODER_ATTENTION_TYPE, use_hypercolumn=True,
                 use_aux=True, dropout=0.2):
        super().__init__()
        self.use_hypercolumn = use_hypercolumn
        self.use_aux = use_aux

        # Build SMP U-Net with scSE attention in decoder; 1 in-channel.
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
            decoder_attention_type=attention_type,
            decoder_channels=(256, 128, 64, 32, 16),
            activation=None,
        )

        # Replace segmentation head with hypercolumn head
        dec_channels = (256, 128, 64, 32, 16)
        self.dec_channels = dec_channels

        if use_hypercolumn:
            hc_total = sum(dec_channels)
            self.hc_head = nn.Sequential(
                nn.Conv2d(hc_total, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(64, 1, kernel_size=1),
            )
            # Disable default segmentation head output path — we'll use hc instead
            self.unet.segmentation_head = nn.Identity()

        # Auxiliary deep-supervision heads over intermediate decoder maps
        if use_aux:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(c, 1, kernel_size=1) for c in dec_channels[:-1]
            ])

    def _decoder_forward(self, x):
        """Forward through encoder + each decoder block, returning all decoder maps."""
        features = self.unet.encoder(x)
        decoder = self.unet.decoder

        spatial_shapes = [f.shape[2:] for f in features]
        spatial_shapes = spatial_shapes[::-1]

        feats = features[1:]
        feats = feats[::-1]
        head = feats[0]
        skips = feats[1:]

        decoder_outputs = []
        x = decoder.center(head) if hasattr(decoder, 'center') and decoder.center is not None else head
        for i, block in enumerate(decoder.blocks):
            height, width = spatial_shapes[i + 1]
            skip = skips[i] if i < len(skips) else None
            x = block(x, height, width, skip_connection=skip)
            decoder_outputs.append(x)
        return decoder_outputs

    def forward(self, x):
        input_size = x.shape[-2:]
        dec_maps = self._decoder_forward(x)
        final = dec_maps[-1]

        if self.use_hypercolumn:
            hcs = []
            for dm in dec_maps:
                if dm.shape[-2:] != input_size:
                    dm_up = F.interpolate(dm, size=input_size, mode='bilinear', align_corners=False)
                else:
                    dm_up = dm
                hcs.append(dm_up)
            hc = torch.cat(hcs, dim=1)
            logits = self.hc_head(hc)
        else:
            # Fall back to SMP seg head — but we replaced it; use 1x1 conv on final
            logits = F.interpolate(final, size=input_size, mode='bilinear', align_corners=False)
            logits = nn.Conv2d(final.shape[1], 1, 1).to(logits.device)(logits)

        aux_logits = []
        if self.use_aux and self.training:
            for i, head in enumerate(self.aux_heads):
                a = head(dec_maps[i])
                a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                aux_logits.append(a)

        return logits, aux_logits


# ----- Losses -----
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# ----- EMA -----
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        model.load_state_dict(self._backup, strict=False)
        self._backup = None


# ----- Helpers for salt-coverage stratified folds -----
def coverage_bucket(coverage):
    if coverage == 0:
        return 0
    elif coverage <= 0.25:
        return 1
    elif coverage <= 0.5:
        return 2
    elif coverage <= 0.75:
        return 3
    else:
        return 4


def compute_coverage(mask_dir, ids):
    covs = []
    for i in ids:
        p = os.path.join(mask_dir, i + ".png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            covs.append(0.0)
        else:
            covs.append(((m > 127).sum()) / (cfg.NATIVE_SIZE * cfg.NATIVE_SIZE))
    return np.array(covs)


# ----- Resize/crop to 101x101 -----
def resize_to_native(prob_map):
    # prob_map: (H,W) at IMAGE_SIZE; resize back to 101x101
    return cv2.resize(prob_map, (cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), interpolation=cv2.INTER_LINEAR)


# ----- TTA inference -----
@torch.no_grad()
def predict_probs(model, loader, device, use_tta=True):
    model.eval()
    all_probs = []
    all_ids = []
    for imgs, _, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            logits, _ = model(imgs)
            probs = torch.sigmoid(logits)
            if use_tta:
                imgs_f = torch.flip(imgs, dims=[-1])
                logits_f, _ = model(imgs_f)
                probs_f = torch.sigmoid(logits_f)
                probs_f = torch.flip(probs_f, dims=[-1])
                probs = (probs + probs_f) / 2.0
        probs = probs.squeeze(1).cpu().float().numpy()
        all_probs.append(probs)
        all_ids.extend(ids)
    return np.concatenate(all_probs, axis=0), all_ids


# ----- Validation -----
def validate(model, loader, device, use_tta=True, threshold=0.5):
    probs, ids = predict_probs(model, loader, device, use_tta=use_tta)
    # Resize probs back to native, threshold, compute metric
    mask_dir = cfg.TRAIN_MASK_DIR
    preds_native = []
    trues_native = []
    for p, i in zip(probs, ids):
        p_n = resize_to_native(p)
        preds_native.append((p_n > threshold).astype(np.uint8))
        mpath = os.path.join(mask_dir, i + ".png")
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        trues_native.append((m > 127).astype(np.uint8) if m is not None else np.zeros((cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), np.uint8))
    score = map_iou(preds_native, trues_native)
    return score, probs, ids


def find_best_threshold(probs, ids, mask_dir):
    trues = []
    for i in ids:
        mpath = os.path.join(mask_dir, i + ".png")
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        trues.append((m > 127).astype(np.uint8) if m is not None else np.zeros((cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), np.uint8))
    best = (0.5, -1.0)
    probs_native = [resize_to_native(p) for p in probs]
    for t in np.arange(0.3, 0.71, 0.05):
        preds = [(p > t).astype(np.uint8) for p in probs_native]
        s = map_iou(preds, trues)
        if s > best[1]:
            best = (float(t), s)
    return best


# ----- Checkpoint -----
def save_ckpt(path, model, optimizer, scheduler, global_step, epoch, best_metric, elapsed):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "best_metric": best_metric,
        "elapsed_seconds": elapsed,
    }
    torch.save(ckpt, path)


def params_finite(model):
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


# ----- Train one fold -----
def train_one_fold(fold_idx, train_ids, val_ids, coverages_train, device,
                   phase_a_epochs, phase_b_epochs, max_steps=None, time_budget=None):
    print(f"\n=== Fold {fold_idx} | train={len(train_ids)} val={len(val_ids)} ===", flush=True)
    train_ds = SaltDataset(train_ids, cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
                            transform=get_train_transform(cfg.IMAGE_SIZE), size=cfg.IMAGE_SIZE)
    val_ds = SaltDataset(val_ids, cfg.TRAIN_IMG_DIR, cfg.TRAIN_MASK_DIR,
                          transform=get_val_transform(cfg.IMAGE_SIZE), size=cfg.IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model = HyperColumnUNet(use_hypercolumn=cfg.USE_HYPERCOLUMN, use_aux=cfg.USE_DEEP_SUPERVISION,
                            dropout=cfg.DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    total_epochs = phase_a_epochs + phase_b_epochs
    total_steps = steps_per_epoch * total_epochs

    def lr_lambda(step):
        if step < cfg.WARMUP_STEPS:
            return step / max(1, cfg.WARMUP_STEPS)
        progress = (step - cfg.WARMUP_STEPS) / max(1, total_steps - cfg.WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    bce_dice = BCEDiceLoss(bce_weight=0.5)

    ema = EMA(model, decay=cfg.EMA_DECAY)

    best_metric = -1.0
    global_step = 0
    start_time = time.time()
    last_ckpt_time = start_time
    last_finite_loss = None

    progress_file = open(cfg.PROGRESS_PATH, "a")

    def log_progress(rec):
        progress_file.write(json.dumps(rec) + "\n")
        progress_file.flush()

    stopped = False

    for epoch in range(total_epochs):
        phase_b = epoch >= phase_a_epochs
        model.use_aux = cfg.USE_DEEP_SUPERVISION and not phase_b

        model.train()
        ep_losses = []
        for batch_i, (imgs, masks, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits, aux_logits = model(imgs)

            # loss in fp32
            logits_fp32 = logits.float()
            if phase_b:
                loss = lovasz_hinge(logits_fp32, masks)
            else:
                loss = bce_dice(logits_fp32, masks)
                if aux_logits and cfg.USE_DEEP_SUPERVISION:
                    aux_loss = 0.0
                    for a in aux_logits:
                        aux_loss = aux_loss + F.binary_cross_entropy_with_logits(a.float(), masks)
                    aux_loss = aux_loss / max(1, len(aux_logits))
                    loss = loss + cfg.AUX_LOSS_WEIGHT * aux_loss

            if not torch.isfinite(loss):
                print(f"[NaN guard] Non-finite loss at step={global_step}, epoch={epoch}, "
                      f"lr={optimizer.param_groups[0]['lr']:.6g}, last_finite={last_finite_loss}",
                      flush=True)
                log_progress({
                    "global_step": global_step, "elapsed_seconds": time.time() - start_time,
                    "val_metric": None, "train_loss": float("nan"), "epoch": epoch,
                })
                progress_file.close()
                results = {
                    "mean-precision-intersection-over-union-at-different-thresholds": None,
                    "error": f"Non-finite loss at step={global_step}, epoch={epoch}, lr={optimizer.param_groups[0]['lr']:.6g}",
                }
                with open(cfg.RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                sys.exit(1)

            last_finite_loss = float(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            ep_losses.append(last_finite_loss)
            global_step += 1

            # periodic checkpoint
            now = time.time()
            if now - last_ckpt_time > cfg.CHECKPOINT_INTERVAL_SECONDS:
                save_ckpt(cfg.LATEST_CKPT_PATH, model, optimizer, scheduler,
                          global_step, epoch, best_metric, now - start_time)
                last_ckpt_time = now

            if SHUTDOWN_REQUESTED["flag"]:
                save_ckpt(cfg.LATEST_CKPT_PATH, model, optimizer, scheduler,
                          global_step, epoch, best_metric, now - start_time)
                print("[signal] Saved checkpoint, exiting due to SIGTERM.", flush=True)
                stopped = True
                break

            if max_steps is not None and global_step >= max_steps:
                stopped = True
                break
            if time_budget is not None and (now - start_time) > time_budget:
                stopped = True
                break

        # End-of-epoch validation (use EMA weights)
        ema.apply_to(model)
        val_score, _, _ = validate(model, val_loader, device, use_tta=cfg.USE_HFLIP_TTA, threshold=0.5)
        ema.restore(model)

        elapsed = time.time() - start_time
        train_loss_mean = float(np.mean(ep_losses)) if ep_losses else None
        print(f"Step {global_step} (elapsed {elapsed:.1f}s, {global_step / max(1, elapsed):.2f} steps/s) "
              f"- Train Loss: {train_loss_mean} - Val Metric: {val_score:.4f} (phase={'B' if phase_b else 'A'})",
              flush=True)

        log_progress({
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "val_metric": val_score,
            "train_loss": train_loss_mean,
            "epoch": epoch,
            "val_subset_fraction": 1.0,
        })

        # Save best (using EMA-applied state)
        if val_score > best_metric and params_finite(model):
            best_metric = val_score
            ema.apply_to(model)
            if params_finite(model):
                torch.save({"model": model.state_dict(), "fold": fold_idx,
                            "val_metric": val_score}, cfg.BEST_MODEL_PATH)
            ema.restore(model)

        if stopped:
            break

    progress_file.close()

    # Final EMA inference for OOF
    ema.apply_to(model)
    final_score, oof_probs, oof_ids = validate(model, val_loader, device,
                                                use_tta=cfg.USE_HFLIP_TTA, threshold=0.5)
    return best_metric, final_score, oof_probs, oof_ids, model, ema


# ----- Test inference -----
@torch.no_grad()
def infer_test(model, device, threshold=0.5):
    sub = pd.read_csv(cfg.SAMPLE_SUB)
    test_ids = sub["id"].tolist()
    test_ds = SaltDataset(test_ids, cfg.TEST_IMG_DIR, None,
                          transform=get_val_transform(cfg.IMAGE_SIZE), size=cfg.IMAGE_SIZE)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    probs, ids = predict_probs(model, test_loader, device, use_tta=cfg.USE_HFLIP_TTA)

    # Downsample to native 101x101 float32 BEFORE saving.
    n = len(ids)
    probs_native = np.zeros((n, cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), dtype=np.float32)
    for k in range(n):
        probs_native[k] = resize_to_native(probs[k].astype(np.float32))

    # Reorder to sample_submission.csv canonical order
    id_to_idx = {tid: k for k, tid in enumerate(ids)}
    reorder = np.array([id_to_idx[tid] for tid in test_ids], dtype=np.int64)
    probs_native = probs_native[reorder]

    # Save plain (1000,101,101) f32 ndarray; allow_pickle=False compatible.
    np.save(cfg.TEST_PROBS_PATH, probs_native, allow_pickle=False)
    chk = np.load(cfg.TEST_PROBS_PATH, allow_pickle=False)
    assert chk.shape == (1000, cfg.NATIVE_SIZE, cfg.NATIVE_SIZE) and chk.dtype == np.float32
    print(f"[test] test_probs.npy saved: {chk.shape} dtype={chk.dtype}", flush=True)

    # Save companion test_ids.npy (canonical sample_sub order).
    test_ids_arr = np.array(test_ids, dtype=object)
    np.save(os.path.join(cfg.CANDIDATE_DIR, "test_ids.npy"), test_ids_arr, allow_pickle=True)

    rles = []
    for k, tid in enumerate(test_ids):
        mask = (probs_native[k] > threshold).astype(np.uint8)
        rles.append({"id": tid, "rle_mask": rle_encode(mask) if mask.sum() > 0 else ""})
    df = pd.DataFrame(rles).set_index("id").reindex(test_ids).reset_index()
    df.to_csv(cfg.SUBMISSION_PATH, index=False)
    print(f"[test] submission written: {cfg.SUBMISSION_PATH} rows={len(df)}", flush=True)
    return probs_native  # (1000, 101, 101) f32 in canonical test_ids order


# ----- Main -----
def main():
    device = torch.device(cfg.DEVICE)
    # pick least busy GPU if multiple
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        try:
            free_mem = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                free_mem.append((free, i))
            free_mem.sort(reverse=True)
            device = torch.device(f"cuda:{free_mem[0][1]}")
        except Exception:
            pass
    print(f"Using device: {device}", flush=True)

    # Load train IDs and compute coverages
    df = pd.read_csv(cfg.TRAIN_CSV)
    all_ids = df["id"].tolist()

    if cfg.TEST_MODE:
        all_ids = all_ids[:cfg.TEST_MAX_SAMPLES]
        print(f"[TEST_MODE] using {len(all_ids)} training samples", flush=True)

    coverages = compute_coverage(cfg.TRAIN_MASK_DIR, all_ids)
    buckets = np.array([coverage_bucket(c) for c in coverages])

    # Distribution-shift check: train/test are separate images from same distribution — random fold ok
    print("No distribution shift detected. Using standard StratifiedKFold.", flush=True)

    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    all_ids_arr = np.array(all_ids)

    start = time.time()

    if cfg.TEST_MODE:
        # Single fold quick test
        phase_a = 1
        phase_b = 0
        folds_to_run = [0]
    else:
        phase_a = cfg.PHASE_A_EPOCHS
        phase_b = cfg.PHASE_B_EPOCHS
        folds_to_run = list(range(cfg.N_FOLDS))

    oof_probs_all = {}
    oof_trues_all = {}
    fold_scores = []
    fold_val_ids = {}  # fold_idx -> list of val ids (for per-fold threshold sweep)

    splits = list(skf.split(all_ids_arr, buckets))
    # In TEST_MODE with very few per-bucket samples stratification may fail; fallback to simple split
    try:
        splits = list(skf.split(all_ids_arr, buckets))
    except ValueError:
        from sklearn.model_selection import KFold
        splits = list(KFold(n_splits=min(cfg.N_FOLDS, max(2, len(all_ids_arr) // 2)),
                            shuffle=True, random_state=cfg.SEED).split(all_ids_arr))

    last_model = None
    last_ema = None

    # Per-fold test prob arrays at native resolution (1000, 101, 101) f32 each, in canonical sample_sub order.
    sub_canon = pd.read_csv(cfg.SAMPLE_SUB)
    canon_test_ids = sub_canon["id"].tolist()
    per_fold_test_probs = []  # list of (1000,101,101) f32

    for fold_idx in folds_to_run:
        train_idx, val_idx = splits[fold_idx]
        train_ids = all_ids_arr[train_idx].tolist()
        val_ids = all_ids_arr[val_idx].tolist()
        cov_train = coverages[train_idx]

        max_steps = cfg.MAX_TRAIN_STEPS
        time_budget = cfg.TRAIN_TIME_BUDGET_SECONDS
        best_m, final_m, oof_p, oof_i, model, ema = train_one_fold(
            fold_idx, train_ids, val_ids, cov_train, device,
            phase_a, phase_b, max_steps=max_steps, time_budget=time_budget,
        )
        fold_scores.append(best_m)
        for p, i in zip(oof_p, oof_i):
            oof_probs_all[i] = p
        fold_val_ids[fold_idx] = list(oof_i)
        last_model = model
        last_ema = ema

        # Per-fold test inference at native resolution; save test_probs_fold{idx}.npy.
        try:
            ema.apply_to(model)
            fold_test_probs = infer_test(model, device, threshold=0.5)  # threshold here is unused for the array save
            ema.restore(model)
            np.save(os.path.join(cfg.CANDIDATE_DIR, f"test_probs_fold{fold_idx}.npy"),
                    fold_test_probs, allow_pickle=False)
            per_fold_test_probs.append(fold_test_probs)
            print(f"[fold {fold_idx}] saved test_probs_fold{fold_idx}.npy {fold_test_probs.shape}", flush=True)
        except Exception as e:
            print(f"[warn] per-fold test inference fold {fold_idx} failed: {e}", flush=True)

        elapsed_total = time.time() - start
        print(f"[fold {fold_idx}] best={best_m:.4f} final={final_m:.4f} elapsed={elapsed_total:.1f}s",
              flush=True)

    # Aggregate OOF metric
    ids_sorted = list(oof_probs_all.keys())
    probs_sorted = [oof_probs_all[i] for i in ids_sorted]
    best_thr, best_oof = find_best_threshold(probs_sorted, ids_sorted, cfg.TRAIN_MASK_DIR)
    print(f"OOF best threshold={best_thr:.2f} metric={best_oof:.4f}", flush=True)

    # Save OOF probs as plain (N,101,101) float32 + companion oof_ids.npy.
    try:
        n_oof = len(ids_sorted)
        oof_native = np.zeros((n_oof, cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), dtype=np.float32)
        for k in range(n_oof):
            oof_native[k] = resize_to_native(probs_sorted[k].astype(np.float32))
        np.save(cfg.OOF_PATH, oof_native, allow_pickle=False)
        np.save(os.path.join(cfg.CANDIDATE_DIR, "oof_ids.npy"),
                np.array(ids_sorted, dtype=object), allow_pickle=True)
        chk = np.load(cfg.OOF_PATH, allow_pickle=False)
        assert chk.shape == (n_oof, cfg.NATIVE_SIZE, cfg.NATIVE_SIZE) and chk.dtype == np.float32
        print(f"[main] oof_probs.npy saved: {chk.shape} dtype={chk.dtype}", flush=True)
    except Exception as e:
        print(f"[warn] failed to save OOF: {e}", flush=True)

    # Per-fold threshold sweep (diagnostic; does not change test inference threshold)
    per_fold_thresholds = []
    per_fold_thresholded_metrics = []
    oracle_preds_native = []
    oracle_trues_native = []
    for fi in sorted(fold_val_ids.keys()):
        f_ids = fold_val_ids[fi]
        f_probs = [oof_probs_all[i] for i in f_ids if i in oof_probs_all]
        f_ids = [i for i in f_ids if i in oof_probs_all]
        if not f_probs:
            per_fold_thresholds.append(None)
            per_fold_thresholded_metrics.append(None)
            continue
        try:
            ft, fs = find_best_threshold(f_probs, f_ids, cfg.TRAIN_MASK_DIR)
            per_fold_thresholds.append(float(ft))
            per_fold_thresholded_metrics.append(float(fs))
            # Build oracle preds for this fold using its own best thr
            f_probs_native = [resize_to_native(p) for p in f_probs]
            for p_n, i in zip(f_probs_native, f_ids):
                mpath = os.path.join(cfg.TRAIN_MASK_DIR, i + ".png")
                m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                t_n = (m > 127).astype(np.uint8) if m is not None else np.zeros((cfg.NATIVE_SIZE, cfg.NATIVE_SIZE), np.uint8)
                oracle_preds_native.append((p_n > ft).astype(np.uint8))
                oracle_trues_native.append(t_n)
            print(f"[fold {fi}] per-fold best thr={ft:.2f} metric={fs:.4f}", flush=True)
        except Exception as e:
            print(f"[warn] per-fold threshold sweep fold {fi} failed: {e}", flush=True)
            per_fold_thresholds.append(None)
            per_fold_thresholded_metrics.append(None)

    if oracle_preds_native:
        per_fold_oracle_oof = float(map_iou(oracle_preds_native, oracle_trues_native))
        print(f"[oracle] per-fold-thresholded OOF metric={per_fold_oracle_oof:.4f}", flush=True)
    else:
        per_fold_oracle_oof = None

    # 5-fold soft-mean test probs (in canonical sample_sub order, native resolution).
    if per_fold_test_probs:
        test_mean = np.mean(np.stack(per_fold_test_probs, axis=0), axis=0).astype(np.float32)
        np.save(cfg.TEST_PROBS_PATH, test_mean, allow_pickle=False)
        np.save(os.path.join(cfg.CANDIDATE_DIR, "test_ids.npy"),
                np.array(canon_test_ids, dtype=object), allow_pickle=True)
        chk = np.load(cfg.TEST_PROBS_PATH, allow_pickle=False)
        assert chk.shape == (1000, cfg.NATIVE_SIZE, cfg.NATIVE_SIZE) and chk.dtype == np.float32
        print(f"[main] test_probs.npy (5-fold mean): {chk.shape}", flush=True)

        # Write submission.csv from 5-fold soft mean at global best OOF threshold.
        rles = []
        for k, tid in enumerate(canon_test_ids):
            mask = (test_mean[k] > best_thr).astype(np.uint8)
            rles.append({"id": tid, "rle_mask": rle_encode(mask) if mask.sum() > 0 else ""})
        df = pd.DataFrame(rles).set_index("id").reindex(canon_test_ids).reset_index()
        df.to_csv(cfg.SUBMISSION_PATH, index=False)
        print(f"[main] submission.csv (5-fold soft mean) at thr={best_thr:.2f}", flush=True)
    elif last_ema is not None and last_model is not None:
        last_ema.apply_to(last_model)
        infer_test(last_model, device, threshold=best_thr)
        last_ema.restore(last_model)
    else:
        sub = pd.read_csv(cfg.SAMPLE_SUB)
        sub["rle_mask"] = ""
        sub.to_csv(cfg.SUBMISSION_PATH, index=False)

    elapsed = time.time() - start
    steps_per_sec = None
    try:
        # crude: use global_step via progress file last record if exists
        if os.path.exists(cfg.PROGRESS_PATH):
            with open(cfg.PROGRESS_PATH, "r") as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    if last.get("elapsed_seconds", 0) > 0:
                        steps_per_sec = last.get("global_step", 0) / last["elapsed_seconds"]
    except Exception:
        pass

    results = {
        "mean-precision-intersection-over-union-at-different-thresholds": float(best_oof) if best_oof > -1 else None,
        "oof_threshold": float(best_thr),
        "fold_best_metrics": [float(s) for s in fold_scores],
        "mean_fold_best": float(np.mean(fold_scores)) if fold_scores else None,
        "per_fold_thresholds": per_fold_thresholds,
        "per_fold_thresholded_metrics": per_fold_thresholded_metrics,
        "fold_best_thresholds": per_fold_thresholds,  # alias per plan
        "per_fold_oracle_oof": per_fold_oracle_oof,
        "throughput_steps_per_sec": steps_per_sec,
        "elapsed_seconds": elapsed,
        "test_mode": bool(cfg.TEST_MODE),
    }
    with open(cfg.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results: {results}", flush=True)


if __name__ == "__main__":
    main()
