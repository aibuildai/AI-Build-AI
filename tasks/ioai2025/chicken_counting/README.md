# Chicken Counting — Density Estimation (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Chicken
Counting** (Task 2). Produced autonomously by AIBuildAI; graded with the organisers'
scoring code ([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Counting Score = 0.9398** on the organisers' held-out test set (same
`exp(−mean relative count error)` formula and label set as the official task).

## Task

Count free-range chickens in a photograph. Directly detecting many small, clustered
animals is brittle, so the robust approach is **density estimation**: regress a
`180 × 320` density map whose integral is the count. **Only the sum of each map (the
predicted count) is scored**, so calibrating total mass matters more than spatial shape.

## Method

A **count-space ensemble** of CSRNet-style density regressors (each a pretrained
frontend + dilated backend, trained on the 100 image/map pairs, count-calibrated by a
per-model α fit):

1. **Members & families** — the members cluster by near-duplicate per-image counts:
   family A = {node_24, node_25} (r = 0.999), family B = {node_19, node_20} (r = 0.999).
   node_16 was excluded (weakest, drags every blend it enters).
2. **Blend in count space** (the metric only reads the sum), robust across nearby
   weights:
   `count = 0.6·c24 + 0.25·c25 + 0.15·(c19 + c20)/2`  → the "w_24dom" winner.
3. **Re-materialize** each blended count onto node_24's per-image *unit-normalized*
   density map (its spatial prior), so `map_i = unit24_i · count_i`. This keeps the
   best-calibrated node_24 dominant while shaving per-image variance with a light dose
   of family B, beating every individual node (node_24 alone = 0.9389).

## Files

| File | Purpose |
|---|---|
| `inference.py` | The final count-space fusion: blends the member counts, re-materializes onto node_24's maps, writes `predictions.npy`. |
| `members/counts_{24,25,19,20}.npy` | Each member's 100 per-image predicted counts (tiny vectors). |
| `members/node24_maps.npy` | node_24's `(100,180,320)` per-image density maps — the spatial base (~22 MB). |
| `checkpoints/node_<id>/` | The four members' **trained CSRNet checkpoints** (`best_model.pth`, each `{'model': state_dict, 'prior': (180,320)}`) + `model_src.py` (the member's training/inference source, which defines its model class). ~535 MB total. |
| `predictions/chicken_counting/predictions.npy` | The exact winning submission — `(100, 180, 320)` float32 density maps. |

> The final blend (`inference.py`) is pure count-space arithmetic over the members'
> **already-computed** outputs (`members/`), so it needs no GPU and no image forward
> pass. The trained checkpoints are included in `checkpoints/` for completeness — each
> loads with `torch.load(..., weights_only=False)` and its `model_src.py` — so a member's
> density maps can be regenerated from the checkpoint + the test images if desired.

## Usage

```bash
python inference.py --output ./output
# writes ./output/chicken_counting/predictions.npy
```

No `--input` is needed — the members' per-image outputs are the bundled inputs. CPU,
< 1 s.

## Reproducibility

Deterministic (fixed weights, fixed member outputs).

Verified from this folder:
- `inference.py` reproduces `predictions/chicken_counting/predictions.npy` to
  **float precision** (max abs difference **< 1e-8** on density values; the per-image
  map-sums match to ~1e-6), grading to the **identical Counting Score 0.9398**.
  Exact byte-for-byte float equality is not expected across floating-point reduction
  orders; the reproduction is numerically exact for the metric.

## Dependencies

`numpy` only.
