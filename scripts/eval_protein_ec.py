#!/usr/bin/env python3
"""Evaluate protein EC number predictions using macro F1-score.

Usage:
    python scripts/eval_protein_ec.py \
        --labels data/labels/protein-ec-prediction.csv \
        --submission /path/to/submission.csv
"""

import argparse
import csv
import sys


def f1_per_class(true_labels, pred_labels, cls):
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(true_labels, pred_labels):
    classes = sorted(set(true_labels) | set(pred_labels))
    f1s = [f1_per_class(true_labels, pred_labels, cls) for cls in classes]
    return sum(f1s) / len(f1s) if f1s else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein EC number predictions (macro F1)."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels/protein-ec-prediction.csv",
        help="Path to ground-truth labels CSV (columns: ID, label)",
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission CSV (columns: ID, label)",
    )
    args = parser.parse_args()

    # Read ground-truth labels
    gt = {}
    with open(args.labels, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["ID"]] = int(row["label"])

    # Read predictions
    pred = {}
    with open(args.submission, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred[row["ID"]] = int(row["label"])

    # Check ID alignment
    gt_ids = set(gt.keys())
    pred_ids = set(pred.keys())
    if gt_ids != pred_ids:
        missing = gt_ids - pred_ids
        extra = pred_ids - gt_ids
        if missing:
            print(f"Error: {len(missing)} IDs in labels but missing from submission")
        if extra:
            print(f"Error: {len(extra)} IDs in submission but not in labels")
        sys.exit(1)

    # Align and compute
    ids = sorted(gt.keys())
    true_labels = [gt[i] for i in ids]
    pred_labels = [pred[i] for i in ids]

    score = macro_f1(true_labels, pred_labels)

    # Per-class breakdown
    classes = sorted(set(true_labels) | set(pred_labels))
    print("Per-class F1:")
    for cls in classes:
        f1 = f1_per_class(true_labels, pred_labels, cls)
        count = sum(1 for t in true_labels if t == cls)
        print(f"  Class {cls}: F1={f1:.4f}  (n={count})")

    print(f"\nMacro F1: {score:.4f}")


if __name__ == "__main__":
    main()
