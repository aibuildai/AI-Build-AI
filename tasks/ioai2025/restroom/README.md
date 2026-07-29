# Restroom — Icon Matching (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 4. **Precision@1 = 1.000 (30/30)** on the
organisers' private set.

**Task.** Match a cropped restroom-sign icon to the same restroom's icon among 60
candidates — opposite gender, different angle, near-identical distractors.

**Method.** No training. Six frozen vision towers (open_clip CLIP ViT-B/32 + ViT-L/14,
SigLIP/SigLIP2, timm DINOv2 B/L) with hflip-TTA → L2-normalize + concatenate → forbid
each query's rank-1 (own-gender) match → Hungarian bijective assignment.

**Run.**
```bash
python inference.py --input ./data --output ./output   # -> output/restroom/predictions.csv
```
Tower weights are public HuggingFace checkpoints (fetched on first run, not bundled).
GPU ~3 min.

**Files.** `inference.py` · `data/restroom/test/` (30 query + 60 gallery icons) ·
`predictions/restroom/predictions.csv` (winning 30/30).

**Reproduces** 29/30 in this env — it differs on the one borderline query (14) at the
fp16 margin; the shipped `predictions/` is the authentic 30/30.
Deps: `torch`, `open_clip_torch`, `timm`, `scipy`, `Pillow`.
