# Chicken Counting — Density Estimation (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 2. **Counting Score = 0.9398.**

**Task.** Count chickens by regressing a 180×320 density map whose sum is the count;
only the sum (per-image count) is scored.

**Method.** A count-space ensemble of four CSRNet members. Blend their per-image
counts `count = 0.6·c24 + 0.25·c25 + 0.15·(c19+c20)/2`, then re-materialize onto
node_24's unit-normalized density maps. The final blend is pure arithmetic — no image
forward pass.

**Run.**
```bash
python inference.py --output ./output   # -> output/chicken_counting/predictions.npy
```

**Files.** `inference.py` (the blend) · `members/` (member count vectors + node_24's
maps) · `checkpoints/node_<id>/` (the 4 trained CSRNet checkpoints + model source, via
Git LFS, ~535 MB) · `predictions/chicken_counting/predictions.npy`.

**Reproduces** the submission to float precision (identical Counting Score, <1 s CPU).
Deps: `numpy` (`torch` to load the checkpoints).
