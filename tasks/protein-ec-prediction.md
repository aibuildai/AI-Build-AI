# Protein EC Number Prediction

## Task

Multi-class classification: given a protein amino acid sequence, predict its EC (Enzyme Commission) class. The label is the top-level EC class (a single digit from 1–7):

- **1** — Oxidoreductases
- **2** — Transferases
- **3** — Hydrolases
- **4** — Lyases
- **5** — Isomerases
- **6** — Ligases
- **7** — Translocases

## Data

All files are tab-separated (TSV).

- `train.csv` — 6,444 labeled proteins. Columns: `Entry`, `EC class`, `Sequence`. The `EC class` column already contains the single-digit class label (1–7).
- `test.csv` — 1,313 unlabeled proteins. Columns: `ID`, `Sequences`
- `sample_submission.csv` — submission format. Columns: `ID`, `label`

Class distribution in train:

| Class | Count | Description      |
|-------|-------|------------------|
| 1     | 916   | Oxidoreductases  |
| 2     | 2,544 | Transferases     |
| 3     | 1,905 | Hydrolases       |
| 4     | 457   | Lyases           |
| 5     | 271   | Isomerases       |
| 6     | 203   | Ligases          |
| 7     | 148   | Translocases     |

## Output

Produce `submission.csv` with columns:
- **ID**: must match the `ID` column in `test.csv`
- **label**: predicted first-level EC class (integer in {1, 2, 3, 4, 5, 6, 7})

## Metric

**F1-score (macro)** across all 7 classes. Each unique first-level EC class is treated as a separate class. Higher is better.
