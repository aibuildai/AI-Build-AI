# Concepts — Emergent Communication (IOAI 2025)

AI-generated solution for the **IOAI 2025** Individual-Contest task **Concepts**
(Task 3). Produced autonomously by AIBuildAI; graded with the organisers' judge
([IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)).

**Result: Concept Score = 0.8507**, rule-compliant. This is the constraint-respecting
number: the submitted program makes **zero judge calls at inference** (see below).
An unrestricted variant that queried the judge while generating its answer scored
0.9340 but is not comparable to contestants' scores.

## Task

Communicate a target keyword to an independent **judge LLM** using only a tiny fixed
vocabulary of 118 abstract "marker" ids (0–117). For each item you are given the
target keyword and 100 candidate options; you emit up to 4 sequences of up to 8
marker ids. The judge, shown only the chosen markers' descriptions plus the options,
returns 10 ranked guesses; the item scores `0.9·hits@10 + 0.1·ndcg@10`. **Concept
Score** is the mean over 50 validation + 100 test items.

## Method — and the competition constraint

The skill is the **encoding**, not knowing the word. The clues were found by a
budget-capped **offline** search (an LLM proposer proposes marker compositions, the
frozen judge scores each, beam search keeps the best per item — actual **~1,736**
judge calls, inside the contest's **12,500-call / $10** allowance). The lineage:
category heuristic → multi-sequence robust select → judge-feedback anti-synonym →
cross-node merge → a residual-zero rescue beam that fixed 7 hard items.

Crucially, the contest rule is that the *submitted* program runs where the judge is
**unreachable**, so clues must be pre-computed and baked in. `run.py` here honours
that exactly: it makes **zero judge/network calls** — it only reads the two
pre-computed, judge-validated clue tables and writes them out.

## Files

| File | Purpose |
|---|---|
| `run.py` | Judge-free emitter: reads the pre-computed clue tables + the per-item order, writes `clues_a.jsonl` / `clues_b.jsonl`. No network, no GPU. |
| `node14_merged_a.json`, `node14_merged_b.json` | The pre-computed, judge-validated clue tables (the offline search's output — this is the "trained artifact"). |
| `data/concepts/` | Task inputs — `validation.jsonl` (50) and `test.jsonl` (100), each `{idx, label, options}`. `label` is the target keyword you are told to communicate (an input, not a held-out answer). |
| `predictions/concepts/` | The exact winning submission — `clues_a.jsonl` (50) + `clues_b.jsonl` (100). |

> `run.py` writes an 31-byte `best_model.pkl` sentinel only to satisfy the harness's
> "attempt complete" contract; there is no learned checkpoint — the clue tables are
> the artifact.

## Usage

```bash
python run.py
# reads ./data/concepts/{validation,test}.jsonl + ./node14_merged_*.json
# writes ./output/concepts/{clues_a,clues_b}.jsonl
```

Override `DATA_DIR` / `OUTPUT_DIR` (env vars) to point elsewhere. Grading (running the
judge on the emitted clues) is done by the organisers' evaluator and needs an
OpenRouter key (`<YOUR-API-KEY>`) for the `google/gemini-2.5-flash-lite` judge — that
is the grader's job, not this program's.

## Reproducibility

Fully deterministic (a static table emit); CPU only, only the Python stdlib.

Verified:
- **`run.py` reproduces `predictions/concepts/clues_a.jsonl` and `clues_b.jsonl`
  byte-for-byte** from the bundled tables + data, in **< 1 s on CPU**.

## Dependencies

Python standard library only (`os`, `json`).
