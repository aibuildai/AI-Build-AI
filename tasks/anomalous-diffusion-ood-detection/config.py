"""Hyperparameters for the anomalous-diffusion OOD detector.

Every constant the training and inference scripts read lives here, so the
method is reproducible from one place. The values are exactly those the winning
run used; changing them changes the result.
"""

# The five in-distribution anomalous-diffusion families the detector knows.
MODELS = ["ATTM", "CTRW", "FBM", "LW", "SBM"]

# The ten evaluation instances. The first five keep all five families as
# in-distribution and add one out-of-distribution dynamics type (DBM, TSM, CBM,
# SINAI, OU); the last five hold one family out as OOD and keep the other four.
INSTANCES = [
    "acfls_dbm", "acfls_tsm", "acfls_cbm", "acfls_sinai", "acfls_ou",
    "cfls_attm", "afls_ctrw", "acls_fbm", "acfs_lw", "acfl_sbm",
]

# Trajectory shape (2D, length 50) and the training-corpus signal-to-noise set.
TRAJ_LEN = 50
TRAJ_DIM = 2
SNR_CHOICES = (1, 2, 10)

# Synthetic training corpus generated with `andi-datasets`.
SAMPLES_PER_MODEL = 33000   # total simulated per family
TRAIN_PER_MODEL = 30000     # first slice used for training; the rest is validation
CHUNK = 1100                # per-worker generation chunk
CORPUS_SEED_BASE = 1729

# Training.
SEED = 123
BATCH_SIZE = 768
STEPS = 900
LR = 8e-4
WARMUP_STEPS = 45
MIN_LR = 2e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0
LABEL_SMOOTHING = 0.03
ALPHA_LOSS_WEIGHT = 2.0     # weight on the anomalous-exponent regression term
SUPCON_WEIGHT = 0.12        # weight on the supervised-contrastive term
SUPCON_TEMP = 0.1

# Inference / OOD scoring.
TTA_VIEWS = 8               # D4 test-time-augmentation views averaged at inference
INFER_BATCH = 2048
BANK_PER_MODEL = 10000      # embeddings kept per family for the density bank
KNN_KS = (5, 20, 50)        # neighbourhood sizes for the density scores
ALPHA_BIN_WIDTH = 0.2       # exponent-bin width for the conditional CDF calibration
