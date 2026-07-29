# Concepts — Emergent Communication (IOAI 2025)

AIBuildAI's solution to IOAI 2025 Task 3. **Concept Score = 0.8507** (rule-compliant).

**Task.** Communicate a target keyword to a judge LLM using only marker ids (0–117).
Contestants had a 12,500 judge-call / $10 budget during development, and the submitted
program runs where the judge is unreachable — so clues must be pre-computed.

**Method.** Clues were found offline by a budget-capped proposer + judge beam search
(~1,736 calls). `run.py` makes **zero judge/network calls** — it just emits the
pre-computed, judge-validated clue tables.

**Run.**
```bash
python run.py   # -> output/concepts/{clues_a,clues_b}.jsonl
```

**Files.** `run.py` (judge-free emitter) · `node14_merged_{a,b}.json` (the pre-computed
clue tables — the artifact) · `data/concepts/` (items) ·
`predictions/concepts/{clues_a,clues_b}.jsonl` (winning submission).

**Reproduces** both clue files byte-for-byte (<1 s CPU). An unrestricted variant scored
0.9340 but queried the judge at inference, so it is not comparable. Deps: Python stdlib.
