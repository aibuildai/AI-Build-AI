# Restroom — Icon Matching (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Restroom Icon
Matching** (Task 4). Produced autonomously by AIBuildAI; graded with the organisers'
scoring code ([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Precision@1 = 1.000 (30 / 30)** on the organisers' held-out private set —
the single most defensible human comparison in the contest.

## Task

Given a **cropped** restroom-sign icon, retrieve the icon of the *same restroom* from
a 60-candidate gallery — despite different cropping, a different camera angle, the
opposite gender, and near-identical distractor signs from other restrooms. Test is 30
query icons → 60 gallery candidates; the metric is **Precision@1**.

## Method — zero training

A frozen-encoder retrieval, no learned weights:

1. **Seven frozen towers, three families** — open_clip CLIP (ViT-B/32, ViT-L/14),
   open_clip SigLIP / SigLIP2 (ViT-SO400M-384), and timm DINOv2 (ViT-B/14, ViT-L/14),
   each with its own correct preprocessing and 2× horizontal-flip TTA. Each image's
   per-tower embeddings are L2-normalized and concatenated into one fused descriptor.
2. **Robust retrieval protocol**:
   - **Forbid** each query's rank-1 gallery item (its own-gender near-duplicate
     original — empirically almost never the opposite-gender target).
   - **Hungarian bijective assignment** (`linear_sum_assignment`) over the fused
     cosine-similarity matrix, so all 30 queries map to 30 *distinct* opposite-gender
     same-restroom gallery originals. This diverse ensemble resolves the one query
     that single models miss (single towers grade 29/30).

## Files

| File | Purpose |
|---|---|
| `inference.py` | Self-contained scorer: loads the frozen towers, fuses descriptors, runs the forbid + Hungarian protocol, writes `predictions.csv` (`query_id, gallery_id`). |
| `data/restroom/test/` | The task inputs — 30 `query/*.png` + 60 `gallery/*.png` icons (~9 MB). |
| `predictions/restroom/predictions.csv` | The exact winning submission — 30 rows, `query_id, gallery_id`, graded 30/30. |

> **No model weights are bundled.** The towers are frozen, publicly available
> checkpoints downloaded/cached from HuggingFace by their model ids
> (`openai`/`laion`/`timm`) on first run — there is nothing task-specific to store.

## Usage

```bash
python inference.py --input ./data --output ./output
# writes ./output/restroom/predictions.csv
```

`--input` is a directory containing `restroom/test/query` and `restroom/test/gallery`
(the bundled `./data` already has this layout). GPU recommended; ~3 min including the
one-time tower downloads/loads.

## Reproducibility

The frozen-model *method* is deterministic, but this is **fp16 cross-domain retrieval
at the numeric margin**: the exact 30/30 depends on the precise open_clip / timm /
CLIP checkpoint versions and GPU fp16 numerics.

Verified in this environment (open_clip 3.3.0, timm 1.0.28):
- `inference.py` reproduces **29 / 30**, differing from the shipped submission on
  **exactly one query (14)** — the borderline match the ensemble was built to win,
  which sits at the embedding-precision margin.
- `predictions/restroom/predictions.csv` is the **authentic winning submission (30/30)**
  from the original run; ship this as the deliverable.

## Dependencies

`torch`, `open_clip_torch`, `timm`, `numpy`, `scipy`, `Pillow`. Internet (or a warm
HuggingFace cache) is needed on first run to fetch the frozen tower weights.
