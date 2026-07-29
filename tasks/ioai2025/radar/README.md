# Radar — Semantic Segmentation (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 1. **Weighted Pixel Accuracy = 0.9815**
(matched the organisers' ground truth, 500/500).

**Task.** Turn six radar heatmap views (each 50×181) into a per-cell class map
{background, suitcase, chair, human, wall}; non-background cells count 50×.

**Method.** A 7-model deep ensemble of from-scratch nets (six view-fusion U-Nets +
a SegFormer-MiT-b0) with horizontal-flip TTA. Softmaxes are soft-averaged, then a
cost-sensitive rule decides each cell: `fg k if p_k > 0.025·p_bg, else background`.

**Run.**
```bash
python inference.py --input /path/to/data --output ./output   # -> output/radar/predictions.csv
```
`--input` holds `radar/test/` (500 `.mat.pt` tensors, ~245 MB, not bundled). GPU ~1 min.

**Files.** `inference.py` · `members/node_<id>/` — each member's checkpoint(s) +
`model_src.py` (its model class/preprocessing, so inference is self-contained) +
normalisation state (~76 MB, the trained artifact) · `predictions/radar/predictions.csv`.

**Reproduces** the submission byte-for-byte (~56 s, one GPU).
Deps: `torch`, `transformers`, `numpy`, `safetensors`.
