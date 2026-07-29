# Pixel — Pixel-Efficiency Challenge (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 6. **Accuracy = 0.9370** (frozen CLIP correct on
654/698 crops).

**Task.** For each 224×224 image, output one crop box covering ≤ 6.25% of the pixels;
the rest is blacked out. Scored on whether frozen `openai/clip-vit-large-patch14`, shown
only the crop, still classifies it correctly (9 animal classes + `other`).

**Method.** No training. Pseudo-label each image with full-image CLIP, then brute-force
a dense grid of masked candidate crops and pick `argmax [ P(ŷ) − 0.5·P('other') ]`
through the same frozen CLIP.

**Run.**
```bash
python inference.py   # DATA_DIR=./data, OUTPUT_DIR=./output -> output/pixel/submission.jsonl
```
CLIP weights are public (fetched on first run, not bundled). GPU ~20 min (the search).

**Files.** `inference.py` · `data/pixel/` (698 images + `index.txt` + `classes.json`) ·
`predictions/pixel/submission.jsonl` (winning submission).

**Reproduces** the submission byte-for-byte (698/698 boxes).
Deps: `torch`, `transformers`, `numpy`, `Pillow`.
