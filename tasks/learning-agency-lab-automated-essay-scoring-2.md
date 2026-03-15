# Learning Agency Lab - Automated Essay Scoring 2.0

**Source**: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview

## Task

Regression/ordinal classification: predict the holistic score of a student essay on an integer scale from 1 to 6.

Each sample is a full-text student essay written in response to a prompt. The goal is to approximate human rater scores as closely as possible.

## Data

- `train.csv` — columns: `essay_id`, `full_text`, `score` (integer 1–6)
- `test.csv` — columns: `essay_id`, `full_text`
- `sample_submission.csv` — columns: `essay_id`, `score`

The training set contains approximately 17,000 essays. Score distribution is imbalanced, with most essays scoring 3 or 4.

## Output

Produce a file called `submission.csv` with columns:
- **essay_id**: must match `test.csv`
- **score**: predicted score (integer 1–6)

## Metric

**Quadratic Weighted Kappa (QWK)**. Measures agreement between predicted and actual scores, adjusted for chance. **Higher is better** (range −1 to 1, with 1 being perfect agreement).

## Environment

Use conda. Create a new environment named `learning-agency-lab-automated-essay-scoring-2` with Python 3.11.

Install: transformers, torch, pandas, scikit-learn, numpy, datasets.

## Approach Hints

- **Baseline**: TF-IDF or count-based features + Ridge/LightGBM regression (round predictions to nearest integer, clamp to [1, 6]) can achieve a reasonable QWK (~0.70+).
- **Stronger**: Fine-tune a pretrained language model (DeBERTa-v3-base or DeBERTa-v3-large) as a regressor. Use mean pooling or [CLS] token + regression head. Round and clamp final predictions.
- **Training tips**: Use stratified k-fold cross-validation (5 folds). Optimize learning rate (~1e-5 to 2e-5) and max sequence length (512–1024 tokens). Gradient accumulation helps with large models on limited GPU memory.
- **Post-processing**: Rounding to the nearest integer and clamping to [1, 6] is essential since the metric expects discrete scores.

## Notes

- Essays vary in length; truncation strategy matters for transformer models.
- The score distribution is skewed — consider stratified splits and class-weighted loss.
- Ensemble of multiple folds and/or multiple model seeds improves stability.
- Save your best `submission.csv` in the code directory.
- Do not search for solutions to this specific Kaggle competition online.
