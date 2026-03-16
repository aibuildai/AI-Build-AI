# Overview

## Description

### Goal of the Competition

Chronic wounds affect millions of patients worldwide and accurate wound measurement is critical for effective treatment planning and monitoring. The goal of this task is to automatically segment wound regions from natural images taken in clinical settings. Accurate segmentation enables objective wound area measurement, replacing subjective and inconsistent manual assessment.

### Context

Chronic wounds, including diabetic foot ulcers, venous leg ulcers, and pressure injuries, represent a significant healthcare burden. Clinicians currently rely on manual measurement techniques that are time-consuming, subjective, and prone to inter-observer variability. Automated wound segmentation using deep learning can provide consistent, reproducible, and rapid wound area quantification directly from photographs taken during clinical visits.

This dataset was created by the University of Wisconsin-Milwaukee Big Data Lab in collaboration with the Advancing the Zenith of Healthcare (AZH) Wound and Vascular Center. The images were fully annotated by wound care professionals and preprocessed with cropping and zero-padding to standardize the input. The dataset was also used as the basis for the MICCAI Foot Ulcer Segmentation Challenge (FUSC).

### Publication

C. Wang, D.M. Anisuzzaman, V. Williamson, M.K. Dhar, B. Rostami, J. Niezgoda, S. Gopalakrishnan, and Z. Yu, "Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks", Scientific Reports, 10:21897, 2020. https://doi.org/10.1038/s41598-020-78799-w

## Evaluation

This competition is evaluated on the **Dice coefficient** (also known as the Sorensen-Dice coefficient), which measures the pixel-wise agreement between a predicted segmentation mask and the corresponding ground truth mask. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where X is the set of predicted wound pixels and Y is the ground truth set of wound pixels.

The Dice coefficient ranges from 0 (no overlap) to 1 (perfect overlap). Higher scores indicate better segmentation performance.

### Submission File

The validation set is used as the test set for final evaluation. **Do not use the validation set during training in any way** (no training, no hyperparameter tuning, no early stopping based on validation performance).

For each image in the validation set, you must submit a binary segmentation mask as a PNG image with the same dimensions (512 x 512) as the input image. Wound pixels should have value 255 and non-wound pixels should have value 0.

The predicted masks should be saved in a directory with filenames matching the corresponding input images.

# Data

## Dataset Description

This dataset is derived from the [wound-segmentation GitHub repository](https://github.com/uwm-bigdata/wound-segmentation). Only the `train/` and `validation/` folders from the original `data/Foot Ulcer Segmentation Challenge/` directory were used; all other files and folders in the repository were excluded.

The images were captured in clinical settings at the AZH Wound and Vascular Center and annotated by wound care professionals. All images have been preprocessed with cropping and zero-padding to a standardized size of 512 x 512 pixels.

The task is binary semantic segmentation: for each pixel in a wound image, predict whether it belongs to the wound region or the background.

## Files

- **train/** - the training set containing 810 image-mask pairs:
    - **images/**: RGB wound photographs as PNG files (512 x 512 pixels). Each file is named with a zero-padded numeric ID (e.g., `0011.png`).
    - **labels/**: corresponding binary segmentation masks as PNG files (512 x 512 pixels). Wound pixels have value 255 and background pixels have value 0. Filenames match the corresponding images.
- **validation/** - the validation set containing 200 image-mask pairs, structured identically to the training set:
    - **images/**: RGB wound photographs as PNG files (512 x 512 pixels).
    - **labels/**: corresponding binary segmentation masks as PNG files (512 x 512 pixels).

## Citation

C. Wang, D.M. Anisuzzaman, V. Williamson, M.K. Dhar, B. Rostami, J. Niezgoda, S. Gopalakrishnan, and Z. Yu, "Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks", Scientific Reports, 10:21897, 2020. https://doi.org/10.1038/s41598-020-78799-w

GitHub Repository: https://github.com/uwm-bigdata/wound-segmentation
