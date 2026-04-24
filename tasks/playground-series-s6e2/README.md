# Kaggle Playground Series S6E2 — Heart Disease Binary Classification

AI-generated solution for the [Kaggle Playground S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) competition, produced end-to-end by AIBuildAI.

## Result

| Board   | Score (AUC) | Rank       |
|---------|-------------|------------|
| Public  | 0.95386     | 527 / 4371 |
| Private | 0.95529     | 288 / 4371 |

## Task

Binary classification of heart disease from 13 tabular features. 630K train rows, 270K test rows, metric: ROC-AUC.

## What AIBuildAI produced

The agent designed 3 candidate approaches in parallel (GBDT trio, RealMLP with distillation, TabM neural ensemble), then selected the final blend via hill-climbing search over OOF predictions. All members of the final ensemble came from the GBDT candidate (`train.py`).

**Final blend** (OOF AUC 0.95573):

```
cb_10f1s × 8  +  cb_5f3s × 4  +  xgb_10f1s × 1  +  lgb_10f1s × 1
```

## Files

| File | What it is |
|---|---|
| `train.py` | Training script for the winning GBDT-trio candidate (CatBoost + XGBoost + LightGBM) |
| `ensemble_search.py` | Hill-climbing search over candidate OOF predictions |
| `build_checkpoint.py` | Bundles the selected test predictions into `checkpoint.pth` |
| `inference.py` | Standalone inference — loads `checkpoint.pth`, writes `submission.csv` |
| `checkpoint.pth` | Pre-computed test predictions from the selected ensemble members |
| `ensemble_info.json` | Final blend composition and OOF AUCs |
| `model_designs.json` | Designer agent's blueprints for the 3 candidate approaches |
| `progress.pdf` | Run progress report |

## Reproduce inference

```bash
python inference.py --input /path/to/data_dir --output submission.csv
```

`data_dir` should contain `test.csv` from the competition.
