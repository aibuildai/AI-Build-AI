# TGS Salt Identification Challenge — Image Segmentation

AI-generated solution for the [Kaggle TGS Salt Identification Challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge), produced end-to-end by AIBuildAI. The competition asks for binary salt-deposit segmentation of 101×101 grayscale seismic images; the metric is mean precision over IoU thresholds (mAP @ IoU 0.5–0.95).

## Result

| Board   | Rank        |
|---------|-------------|
| Public  | 185 / 3,219 |
| Private | 181 / 3,219 |

## Task

Binary segmentation: predict per-pixel salt mask for 4,000 train images and 18,000 test images (101×101 grayscale). Metric: mean precision averaged over IoU thresholds 0.5 to 0.95 in steps of 0.05.

## Final model

The final submission is an equal-weight probability-average ensemble of two segmentation models trained independently with 5-fold cross-validation:

- **SegFormer with MiT-B2 encoder.** ImageNet-pretrained MiT-B2 backbone, 256×256 input, 1-channel grayscale, BCE+Dice loss, horizontal-flip TTA.
- **U-Net with EfficientNet-B4 encoder.** ImageNet-pretrained EfficientNet-B4 backbone, scSE decoder attention, hypercolumn segmentation head, deep supervision in Phase A, Lovasz loss in Phase B, EMA decay 0.999, horizontal-flip TTA.

**Final blend** (OOF mAP@IoU 0.8550):

```
0.5 * sigmoid(SegFormer)  +  0.5 * sigmoid(EfficientNet-B4 U-Net)
threshold = 0.55
```

Both members run with horizontal-flip TTA, then their probability maps are downsampled from 256×256 to 101×101 (bilinear) and averaged. The final binary mask is RLE-encoded in the Kaggle TGS-Salt convention (column-major, 1-indexed).

## Files

| File | What it is |
|---|---|
| `inference.py` | Standalone inference — reads `ensemble_info.json` + the two backbone weights, writes `submission.csv` |
| `ensemble_info.json` | Final blend composition, per-member val/OOF mAP@IoU, preprocessing parameters |
| `submission.csv` | The Kaggle-format submission produced by the ensemble (18,000 rows + header). Byte-identical to the output of `python inference.py` against the same test images |
| `candidate_0/train.py` | Training script for SegFormer MiT-B2 |
| `candidate_0/config.py` | Hyperparameters for the SegFormer member |
| `candidate_0/best_model.pth` | Trained SegFormer weights — fold 0, val mAP@IoU 0.8545 |
| `candidate_1/train.py` | Training script for EfficientNet-B4 U-Net |
| `candidate_1/config.py` | Hyperparameters for the EfficientNet U-Net member |
| `candidate_1/best_model.pth` | Trained EfficientNet U-Net weights — fold 4, val mAP@IoU 0.8225 |

## Reproduce inference

```bash
python inference.py --input /path/to/data_dir --output submission.csv
```

`data_dir` should contain test PNGs at either `images/` or `test/images/` (both layouts supported). The 18,000 test images are available from the [competition page](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data) (`test.zip`).

Inference takes ~2 min on a single A100 (SegFormer ~31s + EfficientNet U-Net ~80s). Requirements: `torch`, `segmentation_models_pytorch`, `opencv-python`, `numpy`.
