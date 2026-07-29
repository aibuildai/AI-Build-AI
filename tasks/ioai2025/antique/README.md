# Antique — Painting Authentication (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 5. **Accuracy = 0.976** on the organisers'
held-out set.

**Task.** Classify each painting authentic (+1) / replica (−1) from 5 numeric
features, with only 4 of 500 training rows labelled (semi-supervised). Metric: accuracy.

**Method.** No neural net. Standardize all 1,000 rows (train+test), fit
full-covariance GMMs at K∈{5,6,7}, map each cluster to ±1 via the 4 labelled seeds
(Mahalanobis-nearest), then blend: `label = sign(score_K5 + 0.46·(score_K6+score_K7))`.

**Run.**
```bash
python inference.py --input ./data --output ./output   # -> output/antique/predictions.csv
```

**Files.** `inference.py` (self-contained scorer; refits the GMMs from data, so there
is no checkpoint) · `data/antique/` (inputs) · `predictions/antique/predictions.csv`.

**Reproduces** the submission byte-for-byte (0.976, ~3 s CPU).
Deps: `numpy`, `pandas`, `scikit-learn`.
