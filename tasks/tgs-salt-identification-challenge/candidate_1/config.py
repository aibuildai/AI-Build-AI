"""Config for EfficientNet-B4 U-Net with scSE + hypercolumns, Lovasz Phase-B."""

import os
import torch

# --- Data ---
DATA_DIR = "/data1/peijia/projects/ML-agent/playground/data/tgs-salt-identification-challenge/prepared/public"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

# --- Image / model ---
NATIVE_SIZE = 101
IMAGE_SIZE = 256
NUM_CLASSES = 1
ENCODER_NAME = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
DECODER_ATTENTION_TYPE = "scse"
USE_HYPERCOLUMN = True
USE_DEEP_SUPERVISION = True  # only used in Phase A
AUX_LOSS_WEIGHT = 0.2

# --- Training ---
BATCH_SIZE = 24
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
DROPOUT = 0.2

# --- Phase A / B ---
PHASE_A_EPOCHS = 50
PHASE_B_EPOCHS = 40

# --- CV ---
N_FOLDS = 5
FOLD_TO_RUN = 0  # for sanity; will run fold 0 only during test

# --- TTA ---
USE_HFLIP_TTA = True

# --- Normalization (ImageNet adapted for 1-channel grayscale) ---
IMG_MEAN = 0.449
IMG_STD = 0.226

# --- Resume ---
RESUME_CHECKPOINT = None

# --- Training duration ---
TRAIN_TIME_BUDGET_SECONDS = 78000  # ~21.7h, generous cap for 5 folds at 90 epochs/fold
EPOCHS = 10  # fallback only
CHECKPOINT_INTERVAL_SECONDS = 600
VAL_INTERVAL_STEPS = None
VAL_SUBSET_FRACTION = 1.0
MIN_VAL_SAMPLES = 2000
EARLY_STOP_PATIENCE_SECONDS = None

# --- Test mode ---
TEST_MODE = False
TEST_MAX_SAMPLES = 100
MAX_TRAIN_STEPS = None

# --- Seeds ---
SEED = 42

# Output paths resolved relative to this file
CANDIDATE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(CANDIDATE_DIR, "best_model.pth")
LATEST_CKPT_PATH = os.path.join(CANDIDATE_DIR, "checkpoint_latest.pth")
RESULTS_PATH = os.path.join(CANDIDATE_DIR, "results.json")
SUBMISSION_PATH = os.path.join(CANDIDATE_DIR, "submission.csv")
PROGRESS_PATH = os.path.join(CANDIDATE_DIR, "training_progress.jsonl")
OOF_PATH = os.path.join(CANDIDATE_DIR, "oof_probs.npy")
TEST_PROBS_PATH = os.path.join(CANDIDATE_DIR, "test_probs.npy")
