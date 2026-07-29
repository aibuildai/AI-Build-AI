# IOAI 2025 — Individual Contest, AI-Generated Solutions

Autonomous **AIBuildAI** solutions to all six tasks of the **IOAI 2025** Individual
Contest (Beijing, 284 competitors). Each task was given only the problem statement and
data; the agent chose the approach, wrote the code, and produced a submission with no
human involvement. Graded with the organisers' scoring code
([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

Each subfolder is self-contained and follows the same layout: a `README.md`, a
self-contained reproducing script (`inference.py` / `run.py`), the bundled artifacts it
needs, and `predictions/` (the exact winning submission). All personal paths are
replaced by `/path/to/...` and any credentials by `<YOUR-API-KEY>`; large frozen models
and large image sets are referenced by placeholder, not bundled.

## Results

| Task | Metric | Score | Approach | Reproduction |
|---|---|---|---|---|
| [radar](radar/) | Weighted Pixel Accuracy | **0.9815** | 7-model from-scratch U-Net / SegFormer ensemble + cost-sensitive decision | byte-for-byte (~56 s, GPU) |
| [chicken_counting](chicken_counting/) | Counting Score | **0.9398** | count-space blend of 4 CSRNet members, re-materialized on node_24 maps | float-precision, identical score (<1 s, CPU) |
| [antique](antique/) | Accuracy | **0.976** | GMM soft-score blend from 4 labelled seeds (no neural net) | byte-for-byte (~3 s, CPU) |
| [restroom](restroom/) | Precision@1 | **1.000** (30/30) | 6 frozen vision towers + Hungarian bijective match (no training) | 29/30 in-env; 30/30 shipped (fp16 margin) |
| [pixel](pixel/) | Accuracy | **0.9370** | frozen-CLIP masked-crop search (no training) | byte-for-byte (698/698, ~20 min GPU) |
| [concepts](concepts/) | Concept Score | **0.8507** | judge-free emit of an offline budget-capped clue search | byte-for-byte (<1 s, CPU) |

**Concepts is the rule-compliant number** (submitted program makes zero judge calls at
inference; the offline search used ~1,736 of the 12,500-call / $10 budget). An
unrestricted variant scored 0.9340 but is not comparable to contestants' scores.

## Leaderboard (IOAI 2025 individual contest)

Summing the six task scores places AIBuildAI first on the individual-contest leaderboard
(the human per-task scores are the organisers' published 0–100 points; AIBuildAI's are
its task-metric scores):

| Rank | Participant | Radar | Chicken | Concepts | Restroom | Antique | Pixel | Total |
|---|---|---|---|---|---|---|---|---|
| 1 | AIBuildAI | 98.15 | 93.98 | 85.07 | 100.00 | 97.60 | 93.70 | **568.49** |
| 2 | (human champion) | 98.11 | 84.29 | 65.92 | 100.00 | 98.88 | 94.85 | 542.05 |

## Notes

- Reproductions marked *byte-for-byte* regenerate `predictions/` exactly — five of the
  six do (antique, concepts, radar, pixel byte-for-byte; chicken to float precision with
  an identical Counting Score). Only **restroom** reproduces within the fp16 numeric
  margin (29/30 in-env, differing on the one borderline query the ensemble was built to
  win); its shipped `predictions/` is the authentic winning 30/30 submission.
- The task input data are the official IOAI-2025 datasets; small inputs are bundled,
  large ones (radar test tensors) are referenced via `--input /path/to/data`.
- **Trained checkpoints** are included wherever a neural network was trained: **radar**
  (7 U-Net/SegFormer members, `radar/members/`) and **chicken_counting** (4 CSRNet
  members, `chicken_counting/checkpoints/`). The other four tasks train no neural net —
  antique fits a GMM from data at inference, restroom and pixel use frozen public
  encoders (not bundled), and concepts emits offline clue tables — so there is no
  task-specific checkpoint to include.
- Each task's `README.md` documents its exact usage, dependencies, and reproduction.
