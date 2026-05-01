"""Config for SegFormer MiT-B2 (Stage 1 + Stage 2 pseudo-label)."""
import os

# Data paths
DATA_DIR = "/data1/peijia/projects/ML-agent/playground/data/tgs-salt-identification-challenge/prepared/public"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

# Model / image
IMAGE_SIZE = 256
ORIG_SIZE = 101
NUM_CLASSES = 1
IN_CHANNELS = 1
ENCODER_NAME = "mit_b2"
ENCODER_WEIGHTS = "imagenet"

# Optim
BATCH_SIZE = 32
LEARNING_RATE = 7e-5
WEIGHT_DECAY = 3e-4
WARMUP_STEPS = 200

# Phases — Stage 1 (real-train only)
PHASE_A_EPOCHS = 25
PHASE_B_EPOCHS = 85
EPOCHS = PHASE_A_EPOCHS + PHASE_B_EPOCHS  # fallback

# Phases — Stage 2 (real + pseudo, ~6x larger per-epoch -> reduce epochs)
STAGE2_PHASE_A_EPOCHS = 10
STAGE2_PHASE_B_EPOCHS = 30
STAGE2_EPOCHS = STAGE2_PHASE_A_EPOCHS + STAGE2_PHASE_B_EPOCHS

# Pseudo-label config
PSEUDO_LABEL_GATE = 0.85
PSEUDO_REGRESSION_TOLERANCE = 0.002  # Stage2 must be >= Stage1 - this delta

# Loss
DICE_WEIGHT = 1.0
BCE_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0

# CV — 5-fold required by code path
N_FOLDS = 5
SEED = 42

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DataLoader
NUM_WORKERS = 4

# Resume
RESUME_CHECKPOINT = None

# Time-based training
MAX_TRAIN_STEPS = None  # full epochs
TRAIN_TIME_BUDGET_SECONDS = None
CHECKPOINT_INTERVAL_SECONDS = 1800
VAL_INTERVAL_STEPS = None
VAL_SUBSET_FRACTION = 1.0
MIN_VAL_SAMPLES = 2000
EARLY_STOP_PATIENCE_SECONDS = None

# Normalization (ImageNet-adapted for grayscale)
IMG_MEAN = 0.449
IMG_STD = 0.226

# Augmentation bounds
ROTATE_LIMIT = 10
SHIFT_LIMIT = 0.08
SCALE_LIMIT = 0.15

# TTA
USE_HFLIP_TTA = True

# Stage control
RUN_STAGE2 = True   # if False, only Stage 1 runs (used in TEST_MODE)

# Test mode: quick sanity run (disabled for production)
TEST_MODE = False
TEST_MAX_SAMPLES = 100
TEST_MAX_STEPS = 20
