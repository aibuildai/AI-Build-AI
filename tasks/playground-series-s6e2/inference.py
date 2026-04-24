#!/usr/bin/env python
"""
Standalone inference script for Heart Disease Binary Classification.
Kaggle Playground Series S6E2.

Ensemble of GBDT models (CatBoost, XGBoost, LightGBM) combined via
hill-climbing rank-average on OOF predictions.

Usage:
    python inference.py --input /path/to/data_dir --output /path/to/submission.csv

The checkpoint.pth file must be in the same directory as this script.
It contains pre-computed test predictions from:
  - CatBoost (5-fold 3-seed)
  - CatBoost (10-fold 1-seed)
  - XGBoost (10-fold 1-seed)
  - LightGBM (10-fold 1-seed)

Hill climbing selected blend: cb_10f1s x8 + cb_5f3s x4 + xgb_10f1s x1 + lgb_10f1s x1
OOF AUC: 0.95573
"""

import os
import sys
import argparse
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_checkpoint():
    """Load the ensemble checkpoint from the same directory as this script."""
    checkpoint_path = os.path.join(SCRIPT_DIR, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Ensure checkpoint.pth is in the same directory as inference.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Heart Disease Ensemble Inference (Playground Series S6E2)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to data directory containing test.csv"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output submission.csv"
    )
    args = parser.parse_args()

    # Load test data to get IDs and row count
    test_path = os.path.join(args.input, "test.csv")
    if not os.path.exists(test_path):
        print(f"ERROR: test.csv not found at {test_path}")
        sys.exit(1)

    print(f"[INFERENCE] Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    test_ids = test_df["id"].values
    n_test = len(test_df)
    print(f"[INFERENCE] Test data: {n_test} rows")

    # Load checkpoint
    print(f"[INFERENCE] Loading checkpoint from {SCRIPT_DIR}")
    ckpt = load_checkpoint()

    # Extract test predictions
    test_preds = {}
    pred_keys = [k for k in ckpt.keys() if k.startswith("test_")]
    for key in pred_keys:
        model_name = key[5:]  # remove 'test_' prefix
        preds = ckpt[key]
        if len(preds) != n_test:
            print(f"ERROR: {key} has {len(preds)} predictions but test has {n_test} rows")
            sys.exit(1)
        test_preds[model_name] = preds
        print(f"  Loaded {model_name}: mean={preds.mean():.6f}, std={preds.std():.6f}")

    # Hill climbing selections from ensemble search
    hc_counts = ckpt.get("hill_climbing_counts", {})
    hc_selections = ckpt.get("hill_climbing_selections", [])

    if not hc_selections:
        print("ERROR: No hill climbing selections found in checkpoint")
        sys.exit(1)

    print(f"[INFERENCE] Hill climbing: {len(hc_selections)} selections")
    for name, count in sorted(hc_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} ({count/len(hc_selections)*100:.1f}%)")

    # Rank-transform each model's predictions
    ranked_preds = {}
    for name, preds in test_preds.items():
        ranked_preds[name] = rankdata(preds) / n_test

    # Apply hill climbing blend (incremental weighted average)
    blend = None
    for i, name in enumerate(hc_selections):
        if name not in ranked_preds:
            print(f"ERROR: Model '{name}' not found in checkpoint predictions")
            sys.exit(1)
        if blend is None:
            blend = ranked_preds[name].copy()
        else:
            blend = (blend * i + ranked_preds[name]) / (i + 1)

    # Sanity checks
    n_unique = len(np.unique(np.round(blend, 6)))
    pred_mean = blend.mean()
    pred_std = blend.std()
    pred_min = blend.min()
    pred_max = blend.max()

    print(f"[INFERENCE] Blend stats: mean={pred_mean:.6f}, std={pred_std:.6f}, "
          f"min={pred_min:.6f}, max={pred_max:.6f}")
    print(f"[INFERENCE] Unique values (6 decimals): {n_unique}")

    if n_unique < 10:
        print("ERROR: Predictions appear degenerate (fewer than 10 unique values)")
        sys.exit(1)

    if pred_std < 1e-6:
        print("ERROR: Predictions are near-constant (std < 1e-6)")
        sys.exit(1)

    # Create submission
    submission = pd.DataFrame({
        "id": test_ids,
        "Heart Disease": blend,
    })

    # Verify format against sample_submission if available
    sample_path = os.path.join(args.input, "sample_submission.csv")
    if os.path.exists(sample_path):
        sample_df = pd.read_csv(sample_path)
        if list(sample_df.columns) != list(submission.columns):
            print(f"WARNING: Column mismatch. Sample: {list(sample_df.columns)}, "
                  f"Ours: {list(submission.columns)}")
        if len(sample_df) != len(submission):
            print(f"ERROR: Row count mismatch. Sample: {len(sample_df)}, "
                  f"Ours: {len(submission)}")
            sys.exit(1)
        print(f"[INFERENCE] Format verified against sample_submission.csv")

    # Save
    submission.to_csv(args.output, index=False)
    print(f"[INFERENCE] Saved submission to {args.output}")
    print(f"[INFERENCE] Shape: {submission.shape}")
    print(f"[INFERENCE] OOF AUC (from training): {ckpt.get('oof_auc', 'N/A')}")


if __name__ == "__main__":
    main()
