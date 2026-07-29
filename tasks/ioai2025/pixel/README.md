# Pixel — Pixel-Efficiency Challenge (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Pixel Parsimony**
(Task 6). Produced autonomously by AIBuildAI; graded with the organisers' scoring code
([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Accuracy = 0.9370** — a frozen CLIP, shown only the retained ≤ 6.25% crop,
still classifies **654 / 698** images correctly.

## Task

How little of an image does a vision model need? For each `224 × 224` image, output a
single axis-aligned crop box covering **at most 6.25% of the pixels (≤ 3,136 px)**;
everything outside is blacked out. The metric is the fraction of images that a **frozen
`openai/clip-vit-large-patch14`**, shown only the retained crop, still classifies
correctly (zero-shot over the 9 animal classes + `'other'`). Per-image labels are not
given; the class universe is.

## Method — zero training

The frozen grader CLIP is turned into the search objective (nothing is trained):

1. **Grader-exact text bank** — `text = sorted(9 class names) + ['other']`, encoded
   once with frozen CLIP (bare label names, matching the grader).
2. **Pseudo-label** — score the full image over the 10 labels; its top-1 over the 9
   animal classes is the per-image pseudo-label `ŷ`.
3. **Brute-force crop search** — a dense grid of candidate boxes (squares 56² plus
   elongated rectangles like 44×71, 71×44, …; all area ≤ 3,136; stride 8). Each box is
   masked (outside → black) and scored through frozen CLIP.
4. **Selection** — pick `box* = argmax [ P(ŷ) − λ·P('other') ]` (λ = 0.5), tie-broken by
   highest `P(ŷ)`, with a most-animal-confident fallback. Because `P(...)` uses the
   grader's exact text/logit pipeline, maximizing it directly maximizes the probability
   the grader's argmax lands on `ŷ`.

## Files

| File | Purpose |
|---|---|
| `inference.py` | Self-contained scorer: frozen-CLIP pseudo-label + brute-force masked-crop search, writes `submission.jsonl` (one crop box per image). |
| `data/pixel/` | The task inputs — `index.txt` (698 image ids, order), `classes.json` (the 9-class universe), and `images/*.png` (the 698 `224²` images). |
| `predictions/pixel/submission.jsonl` | The exact winning submission — 698 lines, `{idx, coordinates:[[top,left],[bottom,right]]}`. |

> **No model weights are bundled.** `openai/clip-vit-large-patch14` is a frozen, public
> checkpoint downloaded/cached from HuggingFace on first run — there is nothing
> task-specific to store. An `best_model.pkl` sentinel is written by the harness only to
> mark the attempt complete.

## Usage

```bash
python inference.py                       # DATA_DIR defaults to ./data, OUTPUT_DIR to ./output
# writes ./output/pixel/submission.jsonl
```

Override `DATA_DIR` / `OUTPUT_DIR` (env vars) to point elsewhere. GPU recommended
(fp16); the dense per-image crop search is the compute — ~1.8 s/image, ~20 min for all
698 on one GPU.

## Reproducibility

Deterministic given the frozen CLIP weights (the masked-crop `argmax` search is stable
at fp16).

Verified in this environment:
- **`inference.py` reproduces `predictions/pixel/submission.jsonl` byte-for-byte** —
  all **698 / 698** crop boxes identical — grading to **Accuracy = 0.9370 (654/698)**,
  in ~**20 min** on one GPU (the dense per-image crop search is the compute).

## Dependencies

`torch`, `transformers`, `numpy`, `Pillow`. Internet (or a warm HuggingFace cache) is
needed on first run to fetch the frozen CLIP-L/14 weights.
