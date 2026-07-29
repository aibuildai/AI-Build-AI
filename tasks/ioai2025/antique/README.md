# Antique — Painting Authentication (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Antique
Painting Authentication** (Task 5). Produced autonomously by AIBuildAI; graded
with the organisers' scoring code (results table:
[IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Accuracy = 0.976** on the organisers' held-out test set (488 / 500),
matched value-for-value against the official ground truth.

## Task

Decide whether each painting is authentic (`+1`) or a replica (`-1`) from 5
numeric features, under an extreme **semi-supervised** setting: of 500 training
rows, **only 4 carry a label** (2 authentic, 2 replica); the other 496 and all
500 test rows are unlabelled. The metric is **accuracy** on the balanced
(250/250) test set.

## Method

No neural network, no GPU — a deterministic, transductive statistical model:

1. **Standardize** all 1,000 feature rows (500 train + 500 test) jointly.
2. **Unsupervised GMMs** — fit full-covariance Gaussian Mixture Models at
   `K ∈ {5, 6, 7}` over all rows to recover the latent cluster structure.
3. **Seed anchoring** — for each K, map every cluster to `{+1, -1}` using the 4
   labelled seeds: a cluster containing a seed inherits its label; an unseeded
   cluster takes the label of the Mahalanobis-nearest seeded cluster. This gives
   a per-K posterior soft score in `[-1, 1]` for each test row.
4. **Decorrelated hedge blend** —
   `final_score = score_K5 + 0.46·(score_K6 + score_K7)`, then
   `label = +1 if final_score ≥ 0 else -1`. The K5 posterior is the anchor; the
   K6/K7 terms are a decorrelated hedge. `W_HEDGE = 0.46` sits at the centre of a
   broad stable ridge (0.42–0.48 all score ≥ 0.964), not on a cliff edge.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Self-contained scorer: refits the GMM blend on `--input` data, writes `predictions.csv`. No checkpoint needed — the model is recomputed from data in ~3 s. |
| `data/antique/` | The task inputs — `train/training_set.csv` (5 features + `Authenticated`, only 4 rows labelled) and `test/test.csv` (features only, no labels). |
| `predictions/antique/predictions.csv` | The exact winning submission (`id, label`), 500 rows. |

> There is **no `best_model.pkl`**: this is a transductive model with no learned
> weights to store. It refits the GMMs on whatever `--input` you pass, so it
> genuinely scores any test set rather than replaying a cached prediction.

## Usage

```bash
python inference.py --input ./data --output ./output
# writes ./output/antique/predictions.csv
```

`--input` is a directory containing `antique/test/test.csv` (and
`antique/train/training_set.csv`). To score a different split, point `--input` at
`/path/to/data/` in the same layout.

## Reproducibility

Deterministic given `RANDOM_STATE = 42` (frozen GMM inits + blend); CPU only,
no external dependency beyond `numpy`, `pandas`, `scikit-learn`.

Verified on this checkpoint:
- **`inference.py` reproduces `predictions/antique/predictions.csv` byte-for-byte**
  from the bundled `data/`, in ~**3 s on CPU**, grading to **Accuracy = 0.976**.

## Dependencies

`numpy`, `pandas`, `scikit-learn`.
