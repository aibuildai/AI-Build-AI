# Protein Variant Effect Prediction

AI-generated solution for the NatureBench task **Protein Variant Effect Prediction**
(`s41592-025-02776-2`, *Nature Methods*, doi:10.1038/s41592-025-02776-2).

**Result: g = +0.1235** (mean over 11 assays of the normalized gap to SOTA),
beating the reference **METL-Local** on **8 of 11** deep-mutational-scanning
assays. This exceeds the task's "surpass SOTA" bar (g > 0.1).

## Task

Predict how amino-acid mutations change protein fitness. For each of 11 assays,
score every test variant; the metric is the **Spearman correlation** between
predicted and measured effects. Variants are single- or multi-mutation records
(`A23V`, `A23V,K45E`, ...).

## Method

A single per-assay pipeline (no ensemble of separate models):

1. **ESM-2 650M** (`facebook/esm2_t33_650M_UR50D`, frozen) — masked-marginal
   log-probability table over the wild-type sequence → per-mutation log-likelihood
   ratios and per-position conservation/entropy features.
2. **Structure** — per-position SASA, B-factor, contact number, and depth from the
   assay's PDB, aligned to the WT sequence.
3. **Biophysics** — hydropathy / volume / charge deltas, BLOSUM62, proline-glycine
   flags, plus a site-independent one-hot of `(position, mutant-AA)` substitutions.
4. **Head** — per assay, a **LightGBM (L2 + L1) + Ridge** blend; the blend
   (`mean(L2,L1)` vs `0.5·GBM + 0.5·Ridge`) is chosen by train-only-holdout Spearman.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Standalone scorer: loads `best_model.pkl`, predicts on `--input`, writes predictions. |
| `train.py` | Builds features, fits the per-assay heads, writes `best_model.pkl` + predictions. |
| `model.py` | Feature extraction (ESM / PDB / biophysics) and the LightGBM+Ridge model. |
| `config.py` | Hyperparameters (seed, LightGBM params, Ridge alphas, ESM id, instance list). |
| `best_model.pkl` | Trained checkpoint: per-assay fitted boosters + Ridge **and** the feature state (one-hot vocab, ESM log-prob table, PDB structural features, WT map). |
| `predictions/` | The exact submission — `predictions/<instance>/predictions.tsv` for all 11 assays. |

> `best_model.pkl` is a Python **pickle**, not a Torch `.pth`, because the head is
> a LightGBM+Ridge model (ESM-2 is used only as a frozen feature extractor and is
> not stored — it is downloaded/cached from HuggingFace by `train.py`).

## Usage

**Inference** (loads the trained checkpoint; no GPU, no ESM, no training data needed —
the feature state is baked into `best_model.pkl`):

```bash
python inference.py --input /path/to/instances --output ./output
```

`--input` has one subdirectory per instance, each with a `test.tsv`
(`variant<TAB>n_mut[<TAB>score]`). Writes `output/<instance>/predictions.tsv`.

**Training** (regenerates `best_model.pkl` and predictions from scratch):

```bash
python train.py --data-dir /path/to/data --output-dir ./output
```

Each instance dir needs `train.tsv`, `test.tsv`, and one `.pdb`. Training runs
ESM-2 (uses a GPU if available) over the 11 wild-type sequences and fits the heads.

## Reproducibility

Deterministic given the seed in `config.py` (frozen ESM-2 features + seeded
LightGBM/Ridge), up to LightGBM multithread float noise that does not move the
rank-based Spearman/g.

Verified on this checkpoint:
- **`inference.py` reproduces `predictions/` byte-for-byte** and grades to
  **g = +0.123479** — in ~**18 s on CPU** (no GPU).
- A from-scratch `train.py` run reproduces the same predictions and the identical
  `g = +0.123479` (LightGBM float noise stays below the 4th decimal of every
  per-assay Spearman).

## Per-assay results

Spearman correlation; SOTA is METL-Local (`~` = figure-read approximate value).

| assay | ours | SOTA | beat? |
|---|---|---|---|
| gb1 | 0.9711 | ~0.87 | ✓ |
| tem_1 | 0.9253 | ~0.78 | ✓ |
| pab1 | 0.9192 | ~0.76 | ✓ |
| grb2_binding | 0.9082 | ~0.72 | ✓ |
| dlg4_binding | 0.8933 | ~0.61 | ✓ |
| grb2_abundance | 0.8713 | ~0.82 | ✓ |
| dlg4_abundance | 0.8681 | ~0.82 | ✓ |
| gfp | 0.8619 | ~0.88 | ✗ |
| pten_abundance | 0.7360 | ~0.77 | ✗ |
| ube4b | 0.7099 | ~0.62 | ✓ |
| pten_activity | 0.6524 | ~0.71 | ✗ |

**8 / 11 beat SOTA; overall g = +0.1235.**

## Dependencies

- `inference.py`: `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `biopython`
  (no `torch`/`transformers` — the ESM features are already in the checkpoint).
- `train.py` additionally: `torch`, `transformers` (for the ESM-2 feature pass).
