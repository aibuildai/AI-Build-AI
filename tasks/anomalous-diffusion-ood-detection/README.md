# Anomalous Diffusion — Out-of-Distribution Dynamics Detection

AI-generated solution for the NatureBench [1] task *Anomalous Diffusion
Out-of-Distribution Dynamics Detection* (from a *Nature Computational Science*
study [2]), produced end-to-end by AIBuildAI. Given 2D particle trajectories, the
method must, per trajectory, flag whether the dynamics are out-of-distribution,
classify the in-distribution diffusion family, and estimate the anomalous
exponent. The headline metric is AUROC for OOD detection.

## Result

NatureBench reports the mean per-instance AUROC and a relative-improvement
score `g` against the published state-of-the-art method. Across the 10
evaluation instances:

| | Mean AUROC | `g` vs SOTA | Instances won |
|---|---|---|---|
| **AIBuildAI** | **0.820** | **+0.5838** (surpass-SOTA) | **8 / 10** |
| Published SOTA | 0.609 | 0.000 (reference) | — |

Per-instance AUROC versus the published SOTA — above it on 8 of the 10 instances:

| Instance | Our AUROC | Published SOTA | Above SOTA |
|---|---|---|---|
| acfls_dbm | 0.7743 | 0.6347 | yes |
| acfls_tsm | 0.7271 | 0.7652 | no |
| acfls_cbm | 0.6639 | 0.7199 | no |
| acfls_sinai | 0.7681 | 0.7067 | yes |
| acfls_ou | 0.9475 | 0.3600 | yes |
| cfls_attm | 0.7930 | 0.7035 | yes |
| afls_ctrw | 0.9276 | 0.2780 | yes |
| acls_fbm | 0.8070 | 0.7536 | yes |
| acfs_lw | 0.9160 | 0.3869 | yes |
| acfl_sbm | 0.8791 | 0.7817 | yes |

## Background

Anomalous diffusion describes particle transport that deviates from classical
Brownian motion, with mean-squared displacement following a power law
`MSD ~ t^alpha` (`alpha != 1`). Different theoretical families — fractional
Brownian motion, continuous-time random walks, Levy walks, scaled Brownian
motion, and annealed transient time motion — capture different mechanisms, and
identifying the family behind an observed trajectory matters for molecular
dynamics in biological and materials systems. Real trajectories, however, may
arise from dynamics outside the training distribution; a model that has only
seen in-distribution families will confidently misclassify such
out-of-distribution (OOD) trajectories. Detecting OOD dynamics while still
recognising in-distribution anomalous diffusion is what makes the analysis
reliable.

## Task

Given a set of 2D particle trajectories, the method must simultaneously: (1)
detect OOD dynamics — assign each trajectory a confidence score for being
in-distribution (higher = more ID); (2) classify the in-distribution diffusion
family; and (3) estimate the anomalous exponent `alpha`.

- **Input**: 2D trajectories of length 50 (100 floats: 50 x then 50 y),
  individually Z-score normalized. Each instance declares its ID family set.
- **Output** per trajectory: an OOD confidence score (higher = more ID), a
  predicted ID family index, and a predicted exponent `alpha`.
- **Metric**: AUROC for OOD detection (OOD = positive), averaged over 10
  instances. Secondary: FPR95, ID F1-score, exponent MAE.
- **Instances**: 5 use all five families as ID paired with one OOD type (DBM,
  TSM, CBM, SINAI, OU); 5 hold one family out as OOD and keep the other four.

## Final model

A single multi-task network scores all three sub-tasks from one embedding:

- **Backbone.** A multi-kernel 1D-CNN (kernels 3/5/9) over six per-timestep
  channels — position, velocity, speed, and time — followed by four dilated
  residual blocks, fused with a 44-dimensional block of hand-crafted physics
  features (log-MSD and its slopes, velocity autocorrelations, speed
  statistics, gyration-tensor anisotropy, early/late energy ratios).
- **Heads.** A normalized 128-d embedding (for density-based OOD scoring), a
  five-way family classifier, and a bounded exponent regressor in [0.05, 2.0].
- **Training.** Family cross-entropy + exponent Huber loss + supervised
  contrastive loss, on 30,000 simulated trajectories per family across a grid of
  exponents and SNR in {1, 2, 10}. Eight-fold D4 (axis-swap/sign-flip)
  test-time augmentation at inference.
- **OOD score.** Instance-specific, all higher = more in-distribution:
  probability mass on the ID families for the four-family instances; a
  Mahalanobis distance to the family Gaussians for `acfls_ou`; and a
  CDF-calibrated blend of embedding-density, centroid-alignment, and
  physics-typicality scores (with an exponent-conditional residual correction)
  for the remaining five-family instances.

## Files

| File | What it is |
|---|---|
| `inference.py` | Standalone inference — loads `best_model.pth`, writes one `output/<instance>/predictions.npz` per instance |
| `train.py` | Generates the `andi-datasets` corpus and trains the network, writing `best_model.pth` |
| `model.py` | The network, the physics features, the input channels, and the TTA forward pass (shared by train and inference) |
| `config.py` | Every hyperparameter, in one place |
| `best_model.pth` | Trained weights plus the physics-feature normalization stats |
| `predictions/` | The produced `predictions.npz` for all 10 instances — the submission that scored mean AUROC 0.820 |

## Reproduce inference

```bash
python inference.py --input /path/to/instances --output ./output
```

`--input` is a directory holding one sub-directory per instance, each with
`instance_info.json` and `x_test.npy` (also accepts `<input>/problem/data/`).
Building the density banks regenerates the training corpus with `andi-datasets`
on first run (cached under `./_generated`), so inference takes a few minutes on
a single GPU. Requirements: `torch`, `numpy`, `scikit-learn`, `andi-datasets`.

The `predictions/` files are the exact scored submission (mean AUROC 0.820).
Re-running `inference.py` reproduces that score to within GPU float16
nondeterminism — six of the ten instances bit-for-bit, and the four
density-calibrated instances (`acfls_dbm`, `acfls_tsm`, `acfls_cbm`,
`acfls_sinai`) within ±0.003 AUROC.

To retrain from scratch:

```bash
python train.py          # writes best_model.pth
```

## References

[1] Wang, Y. et al. NatureBench: Can Coding Agents Match the Published SOTA of
Nature-Family Papers? arXiv:2606.24530 (2026).

[2] Feng, X. et al. Reliable deep learning in anomalous diffusion against
out-of-distribution dynamics. *Nature Computational Science* **4**, 761–772
(2024). DOI: 10.1038/s43588-024-00703-7.
