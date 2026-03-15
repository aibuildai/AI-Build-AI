# Aerial Cactus Identification

**Source**: https://www.kaggle.com/competitions/aerial-cactus-identification/overview

## Task

Binary image classification: determine whether an aerial image contains a columnar cactus (label=1) or not (label=0).

Each sample is a 32×32 RGB image taken from aerial photographs.

## Data

- `train/` — 17,500 images (JPEG, 32×32)
- `train.csv` — columns: `id` (filename), `has_cactus` (0 or 1)
- `test/` — 4,000 images (JPEG, 32×32)
- `sample_submission.csv` — columns: `id`, `has_cactus`

The training set is imbalanced (~75% positive).

## Output

Produce a file called `submission.csv` with columns:
- **id**: image filename (must match `sample_submission.csv`)
- **has_cactus**: predicted probability of containing a cactus (float in [0, 1])

## Metric

**AUC-ROC** (Area Under the Receiver Operating Characteristic curve). **Higher is better.**

## Environment

Use conda. Create a new environment named `aerial-cactus-identification` with Python 3.11.

Install: pytorch, torchvision, pandas, scikit-learn, numpy, pillow.

## Approach Hints

- **Baseline**: A simple CNN (few conv layers + FC) should easily achieve >0.95 AUC on this small dataset.
- **Stronger**: Use a pretrained model (ResNet-18, EfficientNet-B0) with fine-tuning. Data augmentation (flips, rotations) helps.
- **Note**: Images are very small (32×32), so lightweight architectures work well. Heavy models are unnecessary.

## Notes

- The images are small and the dataset fits easily in memory.
- Standard image augmentations (horizontal/vertical flip, random rotation) are effective.
- Save your best `submission.csv` in the code directory.
- Do not search for solutions to this specific Kaggle competition online.
