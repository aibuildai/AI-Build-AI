#!/usr/bin/env python
"""Standalone inference for TGS-Salt Identification Challenge.

Two-model probability-average ensemble:
  - SegFormer (MiT-B2)        — candidate_0/best_model.pth
  - EfficientNet-B4 U-Net +
    scSE + hypercolumn        — candidate_1/best_model.pth

Pipeline:
  1. Load test PNGs (101 x 101 grayscale) from <input>/images/
     (or <input>/test/images/ if that subpath exists).
  2. Resize to 256, normalize (x/255 - 0.449)/0.226.
  3. Forward through both models with horizontal-flip TTA.
  4. Sigmoid + downsample to 101 (bilinear).
  5. Equal-weight blend, threshold = 0.55 (loaded from ensemble_info.json).
  6. RLE-encode (Kaggle TGS-Salt: column-major, 1-indexed).
  7. Write submission.csv with columns id, rle_mask.

Usage:
    python inference.py --input /path/to/data_dir --output submission.csv

The ensemble_info.json and per-member weight files (candidate_0/best_model.pth,
candidate_1/best_model.pth) must sit next to this script.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp


SCRIPT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ utils ----
def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resolve_test_image_dir(input_dir: Path) -> Path:
    """Accept either <input>/images/ or <input>/test/images/."""
    for candidate in (input_dir / "images", input_dir / "test" / "images"):
        if candidate.is_dir() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(
        f"No test images found. Expected PNGs under {input_dir}/images/ "
        f"or {input_dir}/test/images/"
    )


# Kaggle TGS-Salt RLE: column-major flatten, 1-indexed (start, length) pairs.
def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.T.flatten().astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# ---------------------------------------------------------------- dataset ----
class TestDataset(Dataset):
    def __init__(self, img_dir: Path, image_size: int, mean: float, std: float):
        self.img_dir = img_dir
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.ids = sorted(p.stem for p in img_dir.glob("*.png"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        im = cv2.imread(str(self.img_dir / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read {iid}.png")
        im = cv2.resize(im, (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LINEAR)
        x = im.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(0), iid


# ----------------------------------------------------- per-model inference ----
@torch.no_grad()
def infer_model(model, loader, device, native_size, returns_tuple):
    model.eval()
    out = []
    ids = []
    n_done = 0
    t0 = time.time()
    for imgs, batch_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            if returns_tuple:
                logits = logits[0]
            if logits.shape[-2:] != imgs.shape[-2:]:
                logits = F.interpolate(logits, size=imgs.shape[-2:],
                                       mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits.float())

            imgs_f = torch.flip(imgs, dims=[-1])
            logits_f = model(imgs_f)
            if returns_tuple:
                logits_f = logits_f[0]
            if logits_f.shape[-2:] != imgs_f.shape[-2:]:
                logits_f = F.interpolate(logits_f, size=imgs_f.shape[-2:],
                                         mode="bilinear", align_corners=False)
            probs_f = torch.sigmoid(logits_f.float())
            probs_f = torch.flip(probs_f, dims=[-1])

            probs = (probs + probs_f) / 2.0  # (B, 1, 256, 256)

        probs_native = F.interpolate(probs, size=(native_size, native_size),
                                     mode="bilinear", align_corners=False)
        probs_native = probs_native.squeeze(1).cpu().numpy().astype(np.float32)
        out.append(probs_native)
        ids.extend(list(batch_ids))
        n_done += len(batch_ids)
        bs = loader.batch_size or 1
        if n_done % (bs * 50) == 0 or n_done == len(loader.dataset):
            print(f"  [{n_done:5d}/{len(loader.dataset)}] "
                  f"{time.time() - t0:.1f}s "
                  f"({n_done / max(time.time() - t0, 1e-6):.1f} img/s)",
                  flush=True)
    return np.concatenate(out, axis=0), ids


# ----------------------------------------------- model construction (each) ----
def build_segformer(device):
    """SegFormer MiT-B2, 1ch in, 1cls out."""
    m = smp.Segformer(encoder_name="mit_b2", encoder_weights=None,
                      in_channels=1, classes=1).to(device)
    return m


def build_unet_hypercolumn(device, member_dir: Path):
    """HyperColumnUNet defined in <member_dir>/train.py.

    The class is defined in train.py, which does `import config as cfg`
    from config.py in the same directory. We dynamically register that
    config under the name `config` so train.py imports correctly.
    """
    cfg_module = import_module_from_path("config", member_dir / "config.py")
    sys.modules["config"] = cfg_module
    train_module = import_module_from_path("unet_train", member_dir / "train.py")
    model = train_module.HyperColumnUNet(
        use_hypercolumn=cfg_module.USE_HYPERCOLUMN,
        use_aux=False,
        dropout=cfg_module.DROPOUT,
    ).to(device)
    return model


# ---------------------------------------------------------------- main -------
def main():
    p = argparse.ArgumentParser(description="TGS-Salt ensemble inference")
    p.add_argument("--input", type=str, required=True,
                   help="Path to data dir containing test images "
                        "(<input>/images/ or <input>/test/images/)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to write submission.csv")
    p.add_argument("--batch-segformer", type=int, default=64,
                   help="Batch size for SegFormer (default 64)")
    p.add_argument("--batch-unet", type=int, default=16,
                   help="Batch size for the U-Net member "
                        "(hypercolumn concatenation is memory-hungry, default 16)")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_dir = resolve_test_image_dir(input_dir)
    print(f"[INFER] test image dir: {img_dir}", flush=True)

    info_path = SCRIPT_DIR / "ensemble_info.json"
    if not info_path.exists():
        print(f"ERROR: {info_path} not found.", file=sys.stderr)
        sys.exit(1)
    with open(info_path) as f:
        info = json.load(f)
    pp = info["preprocessing"]
    threshold = float(info["threshold"])
    print(f"[INFER] threshold = {threshold}", flush=True)
    print(f"[INFER] members = {[m['name'] for m in info['members']]}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFER] device = {device}", flush=True)

    ds = TestDataset(img_dir, pp["image_size"], pp["mean"], pp["std"])
    print(f"[INFER] {len(ds)} test images", flush=True)

    loader_seg = DataLoader(ds, batch_size=args.batch_segformer, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    loader_unet = DataLoader(ds, batch_size=args.batch_unet, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- member 0: SegFormer ---
    m0 = build_segformer(device)
    m0_path = SCRIPT_DIR / info["members"][0]["checkpoint"]
    state = torch.load(m0_path, map_location="cpu", weights_only=False)
    miss, unex = m0.load_state_dict(state["model_state"], strict=True)
    assert not miss and not unex, f"SegFormer strict-load mismatch: {miss[:3]} / {unex[:3]}"
    print(f"[segformer] loaded {m0_path.relative_to(SCRIPT_DIR)} "
          f"(fold={state.get('fold')})", flush=True)
    p_seg, ids = infer_model(m0, loader_seg, device,
                             native_size=pp["native_size"], returns_tuple=False)
    del m0
    torch.cuda.empty_cache()

    # --- member 1: HyperColumnUNet ---
    m1_relpath = info["members"][1]["checkpoint"]
    m1_dir = (SCRIPT_DIR / m1_relpath).parent
    m1 = build_unet_hypercolumn(device, m1_dir)
    m1_path = SCRIPT_DIR / m1_relpath
    state = torch.load(m1_path, map_location="cpu", weights_only=False)
    miss, unex = m1.load_state_dict(state["model"], strict=False)
    bad_unex = [k for k in unex if not k.startswith("aux_heads.")]
    assert not miss and not bad_unex, (
        f"U-Net unexpected non-aux mismatch: miss={miss}, unex={bad_unex}"
    )
    print(f"[unet] loaded {m1_path.relative_to(SCRIPT_DIR)} "
          f"(fold={state.get('fold')}, dropped {len(unex)} aux_head keys)",
          flush=True)
    p_unet, ids2 = infer_model(m1, loader_unet, device,
                               native_size=pp["native_size"], returns_tuple=True)
    assert ids == ids2, "ID ordering mismatch between members"
    del m1
    torch.cuda.empty_cache()

    # --- blend + threshold + RLE ---
    blend = (p_seg + p_unet) / 2.0
    print(f"[blend] mean={blend.mean():.4f}  >thr frac="
          f"{(blend > threshold).mean():.4f}", flush=True)

    masks = (blend > threshold).astype(np.uint8)
    n_empty = 0
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "rle_mask"])
        for i, iid in enumerate(ids):
            mk = masks[i]
            rle = rle_encode(mk) if mk.any() else ""
            if not rle:
                n_empty += 1
            w.writerow([iid, rle])

    print(f"[INFER] wrote {output_path} ({len(ids)} rows, {n_empty} empty)",
          flush=True)


if __name__ == "__main__":
    main()
