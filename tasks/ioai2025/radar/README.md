# Radar — Semantic Segmentation (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Radar**
(Task 1). Produced autonomously by AIBuildAI; graded with the organisers' scoring
code ([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Weighted Pixel Accuracy = 0.9815** on the organisers' held-out test set,
matched value-for-value against the official ground truth (500 / 500).

## Task

Turn six millimetre-wave radar heatmap views (static/dynamic × range-azimuth /
-elevation / -velocity, each `50 × 181`) into a dense per-cell semantic map, labelling
every cell as `{-1,0,1,2,3}` = {background, suitcase, chair, human, wall}. Foreground
is sparse, so the metric is **Weighted Pixel Accuracy** with non-background cells
weighted **50×** background.

## Method

A **7-model deep ensemble** of from-scratch segmentation networks:

1. **Members** — six multi-branch view-fusion U-Net variants (one encoder per radar
   view, fused, with a 5-class per-cell head) plus a SegFormer-MiT-b0, each trained
   from scratch (no ImageNet analogue for 6-channel radar) with a metric-aligned loss.
2. **Inference** — each member is run with horizontal-flip test-time augmentation; the
   7 softmaxes are soft-averaged in probability space.
3. **Cost-sensitive decision** — the Bayes-optimal rule for the 50:1 cost:
   `predict foreground k = argmax_{1..4} p_k iff p_k > TAU · p_bg, else background`,
   with `TAU = 0.025`.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Self-contained ensemble scorer: loads every member from `members/`, runs hflip-TTA, soft-averages, applies the cost-sensitive decision, writes `predictions.csv`. |
| `members/node_<id>/` | Each member's `model_src.py` (its model class + preprocessing), its checkpoint(s) (`best_model.pth`, seed/EMA variants), and `root.pkl` / `stats.npz` (normalisation state). This is the trained artifact (~76 MB total). |
| `predictions/radar/predictions.csv` | The exact winning submission — one row per test sample, `filename` + `pixel_0…pixel_9049` (the `50 × 181` label map, row-major). |

> There is no single `best_model.pkl`; the checkpoints live under `members/` because the
> solution is an ensemble. The members are the "old-code cannot combine with new-resource"
> snapshot — each carries a **copy** of its own training source (`model_src.py`) so
> inference has no dependency on the original run tree.

## Usage

```bash
python inference.py --input /path/to/data --output ./output
# writes ./output/radar/predictions.csv
```

`--input` is a directory containing `radar/test/` (the 500 `*.mat.pt` test tensors,
6 channels each). The test data (~245 MB) is **not** bundled — point `--input` at it.
Runs on GPU if available, else CPU; ~1–2 min for all 500 samples × 7 members × TTA.

## Reproducibility

Deterministic (frozen checkpoints + fixed TTA + fixed `TAU`).

Verified from this folder:
- **`inference.py` reproduces `predictions/radar/predictions.csv` byte-for-byte**
  (grading to **Weighted Pixel Accuracy = 0.9815**), in ~**56 s** on one GPU.

## Dependencies

`numpy`, `torch`, `transformers` (SegFormer member), `safetensors`. No internet needed —
all checkpoints are local under `members/`.
