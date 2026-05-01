"""Segformer MiT-B2 @ 256, Stage 1 (real-train) + Stage 2 (4th-style cross-prediction
pseudo-labeling) per revised plan. Stratified 5-fold by salt coverage, per-fold threshold,
2-way HFlip TTA. Mandatory artifact export of oof_probs/oof_masks/test_probs (incl. per-fold)."""
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import sys
import json
import math
import time
import random
import signal
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
import albumentations as A

import config as C

torch.set_num_threads(8)

HERE = Path(__file__).resolve().parent
LATEST_CKPT = str(HERE / "checkpoint_latest.pth")
BEST_CKPT = str(HERE / "best_model.pth")
PROGRESS_FILE = str(HERE / "training_progress.jsonl")
RESULTS_FILE = str(HERE / "results.json")
SUBMISSION_FILE = str(HERE / "submission.csv")

# Canonical artifact names (post-revert)
OOF_PROBS_F = str(HERE / "oof_probs.npy")
OOF_IDS_F = str(HERE / "oof_ids.npy")
OOF_MASKS_F = str(HERE / "oof_masks.npy")
TEST_PROBS_F = str(HERE / "test_probs.npy")
TEST_IDS_F = str(HERE / "test_ids.npy")

# Stage 1 backup artifact names
OOF_PROBS_S1 = str(HERE / "oof_probs_stage1.npy")
OOF_IDS_S1 = str(HERE / "oof_ids_stage1.npy")
OOF_MASKS_S1 = str(HERE / "oof_masks_stage1.npy")
TEST_PROBS_S1 = str(HERE / "test_probs_stage1.npy")
TEST_IDS_S1 = str(HERE / "test_ids_stage1.npy")

# Pseudo-label audit artifacts
PSEUDO_MASKS_A_F = str(HERE / "pseudo_masks_A.npy")
PSEUDO_MASKS_B_F = str(HERE / "pseudo_masks_B.npy")
PSEUDO_IDS_A_F = str(HERE / "pseudo_ids_A.npy")
PSEUDO_IDS_B_F = str(HERE / "pseudo_ids_B.npy")

# Seed
random.seed(C.SEED)
np.random.seed(C.SEED)
torch.manual_seed(C.SEED)
torch.cuda.manual_seed_all(C.SEED)

# Graceful shutdown flag
_shutdown = {"flag": False}
def _sigterm_handler(signum, frame):
    _shutdown["flag"] = True
    print(f"[signal] SIGTERM received; will save and exit after current step", flush=True)
signal.signal(signal.SIGTERM, _sigterm_handler)


# -------------------- Metric --------------------
def calculate_map(y_true, y_pred):
    thresholds = np.arange(0.5, 1.0, 0.05)
    y_true = np.array(y_true) > 0.5
    y_pred = np.array(y_pred) > 0.5
    if y_true.ndim == 3:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
    scores = []
    for t_mask, p_mask in zip(y_true, y_pred):
        intersection = np.logical_and(t_mask, p_mask).sum()
        union = np.logical_or(t_mask, p_mask).sum()
        iou = 1.0 if union == 0 else intersection / union
        matches = iou > thresholds
        scores.append(np.mean(matches))
    return float(np.mean(scores))


# -------------------- RLE --------------------
def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    if not isinstance(rle_str, str) or rle_str.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = rle_str.split()
    starts = np.array(s[0::2], dtype=int) - 1
    lengths = np.array(s[1::2], dtype=int)
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        mask[st:st + ln] = 1
    return mask.reshape(shape[1], shape[0]).T


# -------------------- Dataset --------------------
class SaltDataset(Dataset):
    """Real-train/val/test dataset reading PNGs from disk."""
    def __init__(self, ids, img_dir, mask_dir=None, transform=None, image_size=256):
        self.ids = ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.ids)

    def _load_img(self, iid):
        p = os.path.join(self.img_dir, f"{iid}.png")
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read image {p}")
        return im

    def _load_mask(self, iid):
        p = os.path.join(self.mask_dir, f"{iid}.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask {p}")
        return (m > 127).astype(np.uint8)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        img = self._load_img(iid)
        if self.mask_dir is not None:
            mask = self._load_mask(iid)
        else:
            mask = np.zeros_like(img, dtype=np.uint8)

        img_r = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            out = self.transform(image=img_r, mask=mask_r)
            img_r = out["image"]
            mask_r = out["mask"]

        img_f = img_r.astype(np.float32) / 255.0
        img_f = (img_f - C.IMG_MEAN) / C.IMG_STD
        img_t = torch.from_numpy(img_f).unsqueeze(0)
        mask_t = torch.from_numpy(mask_r.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t, iid


class CombinedSaltDataset(Dataset):
    """Stage-2 dataset combining real-train PNGs with in-memory pseudo-labeled test images.

    Each item yields the same (image_tensor, mask_tensor, id) format as SaltDataset.
    The same augmentation pipeline is applied to BOTH real and pseudo samples.
    """
    def __init__(self, real_ids, real_img_dir, real_mask_dir,
                 pseudo_ids, pseudo_img_dir, pseudo_masks_dict,
                 transform=None, image_size=256):
        self.real_ids = list(real_ids)
        self.real_img_dir = real_img_dir
        self.real_mask_dir = real_mask_dir
        self.pseudo_ids = list(pseudo_ids)
        self.pseudo_img_dir = pseudo_img_dir
        self.pseudo_masks_dict = pseudo_masks_dict  # iid -> uint8 mask at ORIG_SIZE
        self.transform = transform
        self.image_size = image_size
        self.n_real = len(self.real_ids)
        self.n_pseudo = len(self.pseudo_ids)

    def __len__(self):
        return self.n_real + self.n_pseudo

    def __getitem__(self, idx):
        if idx < self.n_real:
            iid = self.real_ids[idx]
            p = os.path.join(self.real_img_dir, f"{iid}.png")
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            mp = os.path.join(self.real_mask_dir, f"{iid}.png")
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            mask = (m > 127).astype(np.uint8)
        else:
            iid = self.pseudo_ids[idx - self.n_real]
            p = os.path.join(self.pseudo_img_dir, f"{iid}.png")
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            mask = self.pseudo_masks_dict[iid].astype(np.uint8)

        if img is None:
            raise RuntimeError(f"Failed to read image for id {iid}")
        img_r = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            out = self.transform(image=img_r, mask=mask_r)
            img_r = out["image"]
            mask_r = out["mask"]
        img_f = img_r.astype(np.float32) / 255.0
        img_f = (img_f - C.IMG_MEAN) / C.IMG_STD
        img_t = torch.from_numpy(img_f).unsqueeze(0)
        mask_t = torch.from_numpy(mask_r.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t, iid


# -------------------- Augmentations --------------------
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=C.SHIFT_LIMIT, scale_limit=C.SCALE_LIMIT,
            rotate_limit=C.ROTATE_LIMIT, border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.CLAHE(p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])


# -------------------- Losses --------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        p = prob.view(prob.size(0), -1)
        t = target.view(target.size(0), -1)
        inter = (p * t).sum(1)
        den = p.sum(1) + t.sum(1)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        return 1 - dice.mean()


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels):
    losses = []
    for lg, lb in zip(logits, labels):
        losses.append(lovasz_hinge_flat(lg.view(-1), lb.view(-1)))
    return torch.stack(losses).mean()


class PhaseALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return C.BCE_WEIGHT * self.bce(logits, target) + C.DICE_WEIGHT * self.dice(logits, target)


# -------------------- Model --------------------
def build_model():
    return smp.Segformer(
        encoder_name=C.ENCODER_NAME,
        encoder_weights=C.ENCODER_WEIGHTS,
        in_channels=C.IN_CHANNELS,
        classes=C.NUM_CLASSES,
    )


# -------------------- Helpers --------------------
def salt_coverage_buckets(train_df):
    cov = []
    for iid in train_df["id"].values:
        m = cv2.imread(os.path.join(C.TRAIN_MASK_DIR, f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            cov.append(0.0)
        else:
            cov.append(float((m > 127).sum()) / (m.shape[0] * m.shape[1]))
    cov = np.array(cov)
    buckets = np.zeros(len(cov), dtype=int)
    buckets[(cov > 0) & (cov <= 0.25)] = 1
    buckets[(cov > 0.25) & (cov <= 0.5)] = 2
    buckets[(cov > 0.5) & (cov <= 0.75)] = 3
    buckets[cov > 0.75] = 4
    return cov, buckets


def resize_pred_to_orig(prob_np):
    return cv2.resize(prob_np, (C.ORIG_SIZE, C.ORIG_SIZE), interpolation=cv2.INTER_LINEAR)


def write_nan_error_and_exit(step, epoch, lr, last_finite):
    with open(PROGRESS_FILE, "a") as f:
        f.write(json.dumps({
            "global_step": step,
            "elapsed_seconds": 0,
            "val_metric": None,
            "train_loss": float("nan"),
        }) + "\n")
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "mean-precision-intersection-over-union-at-different-thresholds": None,
            "error": f"NaN/Inf loss at step {step}, epoch {epoch}, lr {lr}, last finite loss {last_finite}",
        }, f, indent=2)
    print(f"[ABORT] NaN/Inf loss at step {step}, epoch {epoch}, lr {lr}. Exiting.", flush=True)
    sys.exit(1)


def append_progress(row):
    with open(PROGRESS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


# -------------------- Training one fold (generic — accepts any datasets) --------------------
def train_one_fold(fold_idx, train_dataset_a, train_dataset_b, val_dataset,
                   sample_weights_a, phase_a_epochs, phase_b_epochs,
                   max_train_steps=None, is_quick=False, global_progress=None,
                   stage_label="stage1"):
    device = torch.device(C.DEVICE)
    print(f"\n===== [{stage_label}] Fold {fold_idx}: train_a={len(train_dataset_a)} val={len(val_dataset)} =====", flush=True)

    sampler_a = WeightedRandomSampler(sample_weights_a, num_samples=len(train_dataset_a), replacement=True)
    train_loader_a = DataLoader(train_dataset_a, batch_size=C.BATCH_SIZE, sampler=sampler_a,
                                num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    train_loader_b = DataLoader(train_dataset_b, batch_size=C.BATCH_SIZE, shuffle=True,
                                num_workers=C.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    phase_a_loss = PhaseALoss().to(device)

    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in n or "norm" in n.lower() or "bn" in n.lower():
            nodecay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": C.WEIGHT_DECAY},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=C.LEARNING_RATE,
    )

    steps_per_epoch = max(len(train_loader_a), 1)
    total_epochs = phase_a_epochs + phase_b_epochs
    total_steps = max(steps_per_epoch * total_epochs, 1)
    warmup = min(C.WARMUP_STEPS, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_metric = -1.0
    best_state = None
    global_step = 0
    fold_start = time.time()
    last_finite_loss = None

    for epoch in range(total_epochs):
        in_phase_b = epoch >= phase_a_epochs
        loader = train_loader_b if in_phase_b else train_loader_a
        phase_name = "B-lovasz" if in_phase_b else "A-bce+dice"

        model.train()
        ep_losses = []
        for batch in loader:
            if _shutdown["flag"]:
                break
            imgs, masks, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            logits_fp32 = logits.float()
            if in_phase_b:
                loss = lovasz_hinge(logits_fp32, masks)
            else:
                loss = phase_a_loss(logits_fp32, masks)

            if not torch.isfinite(loss):
                write_nan_error_and_exit(global_step, epoch, optimizer.param_groups[0]["lr"], last_finite_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            last_finite_loss = float(loss.item())
            ep_losses.append(last_finite_loss)
            global_step += 1
            if global_progress is not None:
                global_progress["step"] += 1

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        val_metric, _ = validate(model, val_loader, device, threshold=0.5, quick=is_quick)
        elapsed = time.time() - fold_start
        print(f"[{stage_label}] Fold {fold_idx} | Epoch {epoch+1}/{total_epochs} [{phase_name}] | step {global_step} | train_loss {train_loss:.4f} | val@0.5 {val_metric:.4f} | {elapsed:.1f}s", flush=True)

        append_progress({
            "stage": stage_label,
            "fold": fold_idx,
            "epoch": epoch + 1,
            "phase": phase_name,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "train_loss": train_loss,
            "val_metric": val_metric,
        })

        if val_metric > best_metric:
            all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
            if all_finite:
                best_metric = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if max_train_steps is not None and global_step >= max_train_steps:
            break
        if _shutdown["flag"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # OOF probs at original resolution with HFlip TTA
    oof_probs = predict_probs_tta(model, val_loader, device, hflip_tta=C.USE_HFLIP_TTA)
    oof_orig = np.zeros((len(val_dataset), C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.float32)
    for i in range(len(val_dataset)):
        oof_orig[i] = resize_pred_to_orig(oof_probs[i])

    return model, best_metric, oof_orig


@torch.no_grad()
def validate(model, loader, device, threshold=0.5, quick=False):
    model.eval()
    preds = []
    gts = []
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        preds.append(probs)
        gts.append(masks.numpy())
        if quick:
            break
    preds = np.concatenate(preds, axis=0)[:, 0]
    gts = np.concatenate(gts, axis=0)[:, 0]
    preds_orig = np.stack([resize_pred_to_orig(p) for p in preds])
    gts_orig = np.stack([cv2.resize(g, (C.ORIG_SIZE, C.ORIG_SIZE), interpolation=cv2.INTER_NEAREST) for g in gts])
    binary = (preds_orig > threshold).astype(np.uint8)
    metric = calculate_map(gts_orig, binary)
    return metric, preds_orig


@torch.no_grad()
def predict_probs_tta(model, loader, device, hflip_tta=True):
    model.eval()
    all_probs = []
    for batch in loader:
        imgs = batch[0]
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            if logits.shape[-2:] != imgs.shape[-2:]:
                logits = F.interpolate(logits, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits.float())
            if hflip_tta:
                imgs_f = torch.flip(imgs, dims=[-1])
                logits_f = model(imgs_f)
                if logits_f.shape[-2:] != imgs_f.shape[-2:]:
                    logits_f = F.interpolate(logits_f, size=imgs_f.shape[-2:], mode="bilinear", align_corners=False)
                probs_f = torch.flip(torch.sigmoid(logits_f.float()), dims=[-1])
                probs = (probs + probs_f) / 2.0
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)[:, 0]


# -------------------- Threshold search --------------------
def search_threshold(oof_probs, oof_masks):
    best_thr, best_m = 0.5, -1.0
    for thr in np.arange(0.30, 0.71, 0.02):
        binary = (oof_probs > thr).astype(np.uint8)
        m = calculate_map(oof_masks, binary)
        if m > best_m:
            best_m = m
            best_thr = float(thr)
    return best_thr, best_m


# -------------------- Stage runner --------------------
def run_stage(stage_label, all_ids, buckets, oof_masks_all, fold_iter,
              test_ids_infer, phase_a_epochs, phase_b_epochs,
              max_train_steps, is_quick, steps_done,
              pseudo_train_ids_per_fold=None, pseudo_masks_dict=None):
    """Runs one stage of 5-fold training and returns artifacts.

    pseudo_train_ids_per_fold: list of length N_FOLDS; each item is a list of pseudo
      test ids to ADD to that fold's training set. None for Stage 1.
    pseudo_masks_dict: dict iid -> uint8 mask (101x101) for pseudo samples. None for Stage 1.
    """
    n_train = len(all_ids)
    oof_probs_all = np.zeros((n_train, C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.float32)
    per_fold_metrics = []
    per_fold_thr = []
    per_fold_test_probs = []  # length N folds, each [n_test_infer, 101, 101]
    fold_models = []

    n_test = len(test_ids_infer)
    test_ds = SaltDataset(test_ids_infer, C.TEST_IMG_DIR, mask_dir=None, transform=None, image_size=C.IMAGE_SIZE)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=C.NUM_WORKERS, pin_memory=True)
    device = torch.device(C.DEVICE)

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_iter):
        tr_ids = list(all_ids[tr_idx])
        va_ids = list(all_ids[va_idx])

        # Build datasets
        if pseudo_train_ids_per_fold is None:
            # Stage 1: real-only train datasets
            train_ds_a = SaltDataset(tr_ids, C.TRAIN_IMG_DIR, C.TRAIN_MASK_DIR,
                                     transform=get_train_transform(), image_size=C.IMAGE_SIZE)
            train_ds_b = SaltDataset(tr_ids, C.TRAIN_IMG_DIR, C.TRAIN_MASK_DIR,
                                     transform=get_train_transform(), image_size=C.IMAGE_SIZE)
            # weighted sampler: salt vs non-salt over real ids
            has_salt = []
            for iid in tr_ids:
                m = cv2.imread(os.path.join(C.TRAIN_MASK_DIR, f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
                has_salt.append(1 if (m is not None and (m > 127).any()) else 0)
            has_salt = np.array(has_salt)
        else:
            # Stage 2: combine real + pseudo for this fold
            pseudo_ids_this_fold = pseudo_train_ids_per_fold[fold_idx]
            train_ds_a = CombinedSaltDataset(
                real_ids=tr_ids, real_img_dir=C.TRAIN_IMG_DIR, real_mask_dir=C.TRAIN_MASK_DIR,
                pseudo_ids=pseudo_ids_this_fold, pseudo_img_dir=C.TEST_IMG_DIR,
                pseudo_masks_dict=pseudo_masks_dict,
                transform=get_train_transform(), image_size=C.IMAGE_SIZE,
            )
            train_ds_b = CombinedSaltDataset(
                real_ids=tr_ids, real_img_dir=C.TRAIN_IMG_DIR, real_mask_dir=C.TRAIN_MASK_DIR,
                pseudo_ids=pseudo_ids_this_fold, pseudo_img_dir=C.TEST_IMG_DIR,
                pseudo_masks_dict=pseudo_masks_dict,
                transform=get_train_transform(), image_size=C.IMAGE_SIZE,
            )
            # has_salt over ALL items (real + pseudo)
            has_salt = []
            for iid in tr_ids:
                m = cv2.imread(os.path.join(C.TRAIN_MASK_DIR, f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
                has_salt.append(1 if (m is not None and (m > 127).any()) else 0)
            for iid in pseudo_ids_this_fold:
                m = pseudo_masks_dict[iid]
                has_salt.append(1 if m.any() else 0)
            has_salt = np.array(has_salt)

        n_pos = max(int(has_salt.sum()), 1)
        n_neg = max(len(has_salt) - n_pos, 1)
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        sample_weights = np.where(has_salt == 1, w_pos, w_neg)

        # val dataset is ALWAYS real-only (no pseudo in val)
        val_ds = SaltDataset(va_ids, C.TRAIN_IMG_DIR, C.TRAIN_MASK_DIR, transform=None, image_size=C.IMAGE_SIZE)

        model, best_m, oof_probs = train_one_fold(
            fold_idx, train_ds_a, train_ds_b, val_ds, sample_weights,
            phase_a_epochs=phase_a_epochs, phase_b_epochs=phase_b_epochs,
            max_train_steps=max_train_steps, is_quick=is_quick, global_progress=steps_done,
            stage_label=stage_label,
        )
        oof_probs_all[va_idx] = oof_probs

        thr, m_at_thr = search_threshold(oof_probs, oof_masks_all[va_idx])
        per_fold_metrics.append(m_at_thr)
        per_fold_thr.append(thr)
        print(f"[{stage_label}] Fold {fold_idx}: best_val={best_m:.4f} | per-fold thr={thr:.2f} -> {m_at_thr:.4f}", flush=True)
        fold_models.append(model)

        # Predict test probs for this fold (256 -> resize -> 101)
        probs_256 = predict_probs_tta(model, test_loader, device, hflip_tta=C.USE_HFLIP_TTA)
        test_probs_fold = np.zeros((n_test, C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.float32)
        for i in range(n_test):
            test_probs_fold[i] = resize_pred_to_orig(probs_256[i])
        per_fold_test_probs.append(test_probs_fold)

        # Save first fold model checkpoint as best_model.pth (overwritten each stage)
        if fold_idx == 0:
            torch.save({
                "model_state": model.state_dict(),
                "fold": 0,
                "threshold": thr,
                "metric": m_at_thr,
                "stage": stage_label,
            }, BEST_CKPT)
            torch.save({
                "model_state": model.state_dict(),
                "fold": 0,
                "threshold": thr,
                "metric": m_at_thr,
                "stage": stage_label,
                "global_step": steps_done["step"],
                "epoch": phase_a_epochs + phase_b_epochs,
            }, LATEST_CKPT)

    # Compute OOF metrics
    if len(per_fold_metrics) == C.N_FOLDS:
        oof_binary = np.zeros_like(oof_masks_all)
        for fold_idx, (_, va_idx) in enumerate(fold_iter):
            thr = per_fold_thr[fold_idx]
            oof_binary[va_idx] = (oof_probs_all[va_idx] > thr).astype(np.uint8)
        oof_score_perfold = calculate_map(oof_masks_all, oof_binary)
        global_thr, oof_score_global = search_threshold(oof_probs_all, oof_masks_all)
    else:
        # quick / partial (TEST_MODE): degrade gracefully
        oof_score_perfold = per_fold_metrics[0] if per_fold_metrics else 0.0
        oof_score_global = oof_score_perfold
        global_thr = per_fold_thr[0] if per_fold_thr else 0.5

    primary = max(oof_score_perfold, oof_score_global)
    test_probs_mean = np.mean(np.stack(per_fold_test_probs, axis=0), axis=0).astype(np.float32)

    return {
        "oof_probs": oof_probs_all,
        "primary": float(primary),
        "oof_score_perfold": float(oof_score_perfold),
        "oof_score_global": float(oof_score_global),
        "global_thr": float(global_thr),
        "per_fold_metrics": [float(x) for x in per_fold_metrics],
        "per_fold_thr": [float(x) for x in per_fold_thr],
        "per_fold_test_probs": per_fold_test_probs,
        "test_probs_mean": test_probs_mean,
    }


# -------------------- Pseudo-label split --------------------
def split_test_halves(test_ids):
    """md5-hash split — even -> half A, odd -> half B."""
    halves = {"A": [], "B": []}
    for iid in test_ids:
        h = int(hashlib.md5(iid.encode("utf-8")).hexdigest()[:8], 16)
        halves["A" if (h % 2 == 0) else "B"].append(iid)
    return halves["A"], halves["B"]


def assign_pseudo_to_folds(test_ids, n_folds):
    """Distribute pseudo-test ids across folds via hash-mod-N."""
    by_fold = [[] for _ in range(n_folds)]
    for iid in test_ids:
        h = int(hashlib.md5(iid.encode("utf-8")).hexdigest()[:8], 16)
        by_fold[h % n_folds].append(iid)
    return by_fold


# -------------------- Stage1 backup / canonical write helpers --------------------
def save_canonical(stage_artifacts, all_ids, oof_masks_all, test_ids_infer):
    np.save(OOF_PROBS_F, stage_artifacts["oof_probs"].astype(np.float32))
    np.save(OOF_IDS_F, np.array(list(all_ids), dtype=object))
    np.save(OOF_MASKS_F, oof_masks_all.astype(np.uint8))
    np.save(TEST_PROBS_F, stage_artifacts["test_probs_mean"].astype(np.float32))
    np.save(TEST_IDS_F, np.array(list(test_ids_infer), dtype=object))
    for i, tp in enumerate(stage_artifacts["per_fold_test_probs"]):
        np.save(str(HERE / f"test_probs_fold{i}.npy"), tp.astype(np.float32))


def save_stage1_backup(stage_artifacts, all_ids, oof_masks_all, test_ids_infer):
    np.save(OOF_PROBS_S1, stage_artifacts["oof_probs"].astype(np.float32))
    np.save(OOF_IDS_S1, np.array(list(all_ids), dtype=object))
    np.save(OOF_MASKS_S1, oof_masks_all.astype(np.uint8))
    np.save(TEST_PROBS_S1, stage_artifacts["test_probs_mean"].astype(np.float32))
    np.save(TEST_IDS_S1, np.array(list(test_ids_infer), dtype=object))
    for i, tp in enumerate(stage_artifacts["per_fold_test_probs"]):
        np.save(str(HERE / f"test_probs_fold{i}_stage1.npy"), tp.astype(np.float32))


def restore_stage1_to_canonical():
    shutil.copyfile(OOF_PROBS_S1, OOF_PROBS_F)
    shutil.copyfile(OOF_IDS_S1, OOF_IDS_F)
    shutil.copyfile(OOF_MASKS_S1, OOF_MASKS_F)
    shutil.copyfile(TEST_PROBS_S1, TEST_PROBS_F)
    shutil.copyfile(TEST_IDS_S1, TEST_IDS_F)
    for i in range(C.N_FOLDS):
        src = str(HERE / f"test_probs_fold{i}_stage1.npy")
        dst = str(HERE / f"test_probs_fold{i}.npy")
        if os.path.exists(src):
            shutil.copyfile(src, dst)


# -------------------- Final assertions --------------------
def assert_artifacts(n_train, n_test, n_folds, expect_submission_lines=True):
    assert os.path.exists(OOF_PROBS_F), "oof_probs.npy missing"
    a = np.load(OOF_PROBS_F)
    assert a.shape == (n_train, C.ORIG_SIZE, C.ORIG_SIZE), f"oof_probs shape={a.shape}"
    assert a.dtype == np.float32, f"oof_probs dtype={a.dtype}"
    assert os.path.exists(OOF_IDS_F), "oof_ids.npy missing"
    b = np.load(OOF_IDS_F, allow_pickle=True)
    assert b.shape == (n_train,), f"oof_ids shape={b.shape}"
    assert os.path.exists(OOF_MASKS_F), "oof_masks.npy missing"
    c = np.load(OOF_MASKS_F)
    assert c.shape == (n_train, C.ORIG_SIZE, C.ORIG_SIZE), f"oof_masks shape={c.shape}"
    assert c.dtype == np.uint8, f"oof_masks dtype={c.dtype}"
    assert os.path.exists(TEST_PROBS_F), "test_probs.npy missing"
    d = np.load(TEST_PROBS_F)
    assert d.shape == (n_test, C.ORIG_SIZE, C.ORIG_SIZE), f"test_probs shape={d.shape}"
    assert d.dtype == np.float32, f"test_probs dtype={d.dtype}"
    assert os.path.exists(TEST_IDS_F), "test_ids.npy missing"
    e = np.load(TEST_IDS_F, allow_pickle=True)
    assert e.shape == (n_test,), f"test_ids shape={e.shape}"
    for i in range(n_folds):
        f_ = str(HERE / f"test_probs_fold{i}.npy")
        assert os.path.exists(f_), f"{f_} missing"
        z = np.load(f_)
        assert z.shape == (n_test, C.ORIG_SIZE, C.ORIG_SIZE), f"{f_} shape={z.shape}"
        assert z.dtype == np.float32, f"{f_} dtype={z.dtype}"
    if expect_submission_lines:
        assert os.path.exists(SUBMISSION_FILE), "submission.csv missing"


# -------------------- Main --------------------
def main():
    t0 = time.time()
    with open(PROGRESS_FILE, "w") as f:
        pass

    train_df = pd.read_csv(C.TRAIN_CSV)
    ids_available = set(os.path.splitext(f)[0] for f in os.listdir(C.TRAIN_IMG_DIR))
    train_df = train_df[train_df["id"].isin(ids_available)].reset_index(drop=True)

    if C.TEST_MODE:
        train_df = train_df.iloc[:C.TEST_MAX_SAMPLES].reset_index(drop=True)
        print(f"[TEST_MODE] Capping training set to {len(train_df)} samples", flush=True)

    cov, buckets = salt_coverage_buckets(train_df)
    print(f"Coverage bucket distribution: {np.bincount(buckets, minlength=5).tolist()}", flush=True)

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    all_ids = train_df["id"].values
    n_train = len(all_ids)

    # GT masks (101x101) - load once
    oof_masks_all = np.zeros((n_train, C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.uint8)
    for i, iid in enumerate(all_ids):
        m = cv2.imread(os.path.join(C.TRAIN_MASK_DIR, f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        oof_masks_all[i] = (m > 127).astype(np.uint8)

    # Test ids
    sample_sub_df = pd.read_csv(C.SAMPLE_SUB)
    test_ids_full = sample_sub_df["id"].values.tolist()
    if C.TEST_MODE:
        test_ids_infer = test_ids_full[:50]
    else:
        test_ids_infer = test_ids_full
    n_test = len(test_ids_infer)

    # Decide regime
    if C.TEST_MODE:
        s1_phase_a = 1
        s1_phase_b = 0
        max_train_steps = C.TEST_MAX_STEPS
        is_quick = True
        n_folds_run = C.N_FOLDS  # we still want all folds so artifact shapes are valid
        run_stage2 = False
    else:
        s1_phase_a = C.PHASE_A_EPOCHS
        s1_phase_b = C.PHASE_B_EPOCHS
        max_train_steps = C.MAX_TRAIN_STEPS
        is_quick = False
        n_folds_run = C.N_FOLDS
        run_stage2 = bool(C.RUN_STAGE2)

    fold_iter = list(skf.split(all_ids, buckets))[:n_folds_run]
    steps_done = {"step": 0}

    # ============================ STAGE 1 ============================
    print(f"\n############ STAGE 1: real-train only (5-fold) ############\n", flush=True)
    s1 = run_stage(
        stage_label="stage1",
        all_ids=all_ids, buckets=buckets, oof_masks_all=oof_masks_all, fold_iter=fold_iter,
        test_ids_infer=test_ids_infer,
        phase_a_epochs=s1_phase_a, phase_b_epochs=s1_phase_b,
        max_train_steps=max_train_steps, is_quick=is_quick, steps_done=steps_done,
        pseudo_train_ids_per_fold=None, pseudo_masks_dict=None,
    )
    print(f"[stage1] OOF per-fold-thr={s1['oof_score_perfold']:.4f} | global-thr={s1['oof_score_global']:.4f} (thr={s1['global_thr']:.2f})", flush=True)

    # Save Stage 1 backup AND canonical (canonical may be overwritten by Stage 2)
    save_stage1_backup(s1, all_ids, oof_masks_all, test_ids_infer)
    save_canonical(s1, all_ids, oof_masks_all, test_ids_infer)

    stage1_oof = s1["primary"]
    pseudo_label_gate_passed = bool(stage1_oof >= C.PSEUDO_LABEL_GATE)

    # Decide whether to run Stage 2
    stage_used = "stage1"
    stage2_oof = None
    pseudo_regression = None
    s2 = None

    if run_stage2 and pseudo_label_gate_passed:
        # ============================ STAGE 2 ============================
        print(f"\n############ STAGE 2: 4th-style cross-prediction pseudo-label ############\n", flush=True)
        # Generate pseudo labels: binarize Stage 1 fold-mean test probs at Stage 1 global thr
        s1_thr = s1["global_thr"]
        test_probs_s1 = s1["test_probs_mean"]  # [n_test, 101, 101]
        pseudo_masks_full = (test_probs_s1 > s1_thr).astype(np.uint8)

        # Split halves
        half_A_ids, half_B_ids = split_test_halves(test_ids_infer)
        id_to_idx = {iid: i for i, iid in enumerate(test_ids_infer)}
        pseudo_masks_dict = {iid: pseudo_masks_full[id_to_idx[iid]] for iid in test_ids_infer}

        # Save audit artifacts
        np.save(PSEUDO_IDS_A_F, np.array(half_A_ids, dtype=object))
        np.save(PSEUDO_IDS_B_F, np.array(half_B_ids, dtype=object))
        np.save(PSEUDO_MASKS_A_F, np.stack([pseudo_masks_dict[i] for i in half_A_ids], axis=0).astype(np.uint8) if len(half_A_ids) else np.zeros((0, C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.uint8))
        np.save(PSEUDO_MASKS_B_F, np.stack([pseudo_masks_dict[i] for i in half_B_ids], axis=0).astype(np.uint8) if len(half_B_ids) else np.zeros((0, C.ORIG_SIZE, C.ORIG_SIZE), dtype=np.uint8))

        # Distribute pseudo across folds
        pseudo_per_fold = assign_pseudo_to_folds(test_ids_infer, C.N_FOLDS)
        for i, pf in enumerate(pseudo_per_fold):
            print(f"[stage2] fold {i}: pseudo-train ids = {len(pf)}", flush=True)

        s2 = run_stage(
            stage_label="stage2",
            all_ids=all_ids, buckets=buckets, oof_masks_all=oof_masks_all, fold_iter=fold_iter,
            test_ids_infer=test_ids_infer,
            phase_a_epochs=C.STAGE2_PHASE_A_EPOCHS, phase_b_epochs=C.STAGE2_PHASE_B_EPOCHS,
            max_train_steps=max_train_steps, is_quick=is_quick, steps_done=steps_done,
            pseudo_train_ids_per_fold=pseudo_per_fold, pseudo_masks_dict=pseudo_masks_dict,
        )
        stage2_oof = s2["primary"]
        print(f"[stage2] OOF per-fold-thr={s2['oof_score_perfold']:.4f} | global-thr={s2['oof_score_global']:.4f} (thr={s2['global_thr']:.2f})", flush=True)
        print(f"[stage2] stage1_oof={stage1_oof:.4f} stage2_oof={stage2_oof:.4f} delta={stage2_oof-stage1_oof:+.4f}", flush=True)

        if stage2_oof >= stage1_oof - C.PSEUDO_REGRESSION_TOLERANCE:
            # Keep Stage 2 as canonical
            save_canonical(s2, all_ids, oof_masks_all, test_ids_infer)
            stage_used = "stage2"
        else:
            # Revert
            pseudo_regression = stage1_oof - stage2_oof
            print(f"[stage2] REGRESSION -> reverting to Stage 1 artifacts", flush=True)
            restore_stage1_to_canonical()
            stage_used = "stage1_reverted"
    elif run_stage2 and not pseudo_label_gate_passed:
        print(f"[stage2] GATE FAILED: stage1_oof={stage1_oof:.4f} < {C.PSEUDO_LABEL_GATE}; skipping Stage 2", flush=True)
        stage_used = "stage1"

    # ============================ FINAL: build submission from canonical ============================
    canonical_oof = np.load(OOF_PROBS_F)
    canonical_test = np.load(TEST_PROBS_F)
    final_thr, _ = search_threshold(canonical_oof, oof_masks_all)
    print(f"[final] canonical global thr = {final_thr:.2f}", flush=True)

    test_binary = (canonical_test > final_thr).astype(np.uint8)
    sub_map = {}
    for i, iid in enumerate(test_ids_infer):
        sub_map[iid] = rle_encode(test_binary[i])
    rows = []
    for iid in test_ids_full:
        rows.append({"id": iid, "rle_mask": sub_map.get(iid, "")})
    pd.DataFrame(rows).to_csv(SUBMISSION_FILE, index=False)
    print(f"[final] Wrote submission.csv with {len(rows)} rows", flush=True)

    # Decide which stage's metric/details to report
    if stage_used == "stage2":
        primary = s2["primary"]
        report_perfold = s2["oof_score_perfold"]
        report_global = s2["oof_score_global"]
        report_global_thr = s2["global_thr"]
        report_pf_metrics = s2["per_fold_metrics"]
        report_pf_thr = s2["per_fold_thr"]
    else:
        primary = s1["primary"]
        report_perfold = s1["oof_score_perfold"]
        report_global = s1["oof_score_global"]
        report_global_thr = s1["global_thr"]
        report_pf_metrics = s1["per_fold_metrics"]
        report_pf_thr = s1["per_fold_thr"]

    total_elapsed = time.time() - t0
    steps_per_sec = steps_done["step"] / max(total_elapsed, 1e-6)

    artifacts_dict = {
        "oof_probs": "oof_probs.npy",
        "oof_ids": "oof_ids.npy",
        "oof_masks": "oof_masks.npy",
        "test_probs": "test_probs.npy",
        "test_ids": "test_ids.npy",
        "test_probs_per_fold": [f"test_probs_fold{i}.npy" for i in range(C.N_FOLDS)],
        "stage1_backup": {
            "oof_probs": "oof_probs_stage1.npy",
            "test_probs": "test_probs_stage1.npy",
            "test_probs_per_fold": [f"test_probs_fold{i}_stage1.npy" for i in range(C.N_FOLDS)],
        },
        "pseudo_audit": {
            "pseudo_masks_A": "pseudo_masks_A.npy",
            "pseudo_masks_B": "pseudo_masks_B.npy",
            "pseudo_ids_A": "pseudo_ids_A.npy",
            "pseudo_ids_B": "pseudo_ids_B.npy",
        },
    }

    results = {
        "mean-precision-intersection-over-union-at-different-thresholds": float(primary),
        "oof_per_fold_threshold": float(report_perfold),
        "oof_global_threshold": float(report_global),
        "global_threshold": float(report_global_thr),
        "per_fold_metrics": [float(x) for x in report_pf_metrics],
        "per_fold_thresholds": [float(x) for x in report_pf_thr],
        "folds_trained": int(C.N_FOLDS),
        "stage_used": stage_used,
        "stage1_oof": float(stage1_oof),
        "stage2_oof": (float(stage2_oof) if stage2_oof is not None else None),
        "pseudo_label_gate_passed": bool(pseudo_label_gate_passed),
        "pseudo_label_regression": (float(pseudo_regression) if pseudo_regression is not None else None),
        "artifacts": artifacts_dict,
        "throughput_steps_per_sec": float(steps_per_sec),
        "total_elapsed_seconds": float(total_elapsed),
        "test_mode": bool(C.TEST_MODE),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results.json: stage_used={stage_used} primary={primary:.4f}", flush=True)

    # ============================ FINAL ASSERTIONS ============================
    try:
        assert_artifacts(n_train=n_train, n_test=n_test, n_folds=C.N_FOLDS, expect_submission_lines=True)
        print(f"[assert] All artifact assertions passed", flush=True)
    except AssertionError as ae:
        # Report structured error
        print(f"[assert] FAILED: {ae}", flush=True)
        results["error"] = f"artifact assertion failed: {ae}"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        raise


if __name__ == "__main__":
    main()
