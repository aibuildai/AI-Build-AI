"""Centralized hyperparameters for the Protein Variant Effect Prediction solution.

The method is an ESM-2 zero-shot + structure/biophysics feature set fed to a
per-instance LightGBM (L2+L1) + Ridge blend. All tunables live here; the model
and feature code is in model.py, orchestration in train.py, and standalone
scoring in inference.py.
"""

# Reproducibility: the LightGBM sub-split, Ridge fits, and blend selection are
# all keyed off this seed. Combined with the frozen ESM-2 encoder the pipeline is
# deterministic to within LightGBM multithread float noise (Spearman/g stable).
SEED = 42

# Frozen protein language model used for zero-shot masked-marginal log-probs.
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# The 11 deep-mutational-scanning assays this solution covers.
INSTANCES = [
    "gfp", "gb1", "dlg4_abundance", "dlg4_binding", "grb2_abundance",
    "grb2_binding", "pab1", "pten_abundance", "pten_activity", "tem_1", "ube4b",
]

# LightGBM booster hyperparameters (shared by the L2 and L1 objectives).
LGBM_PARAMS = dict(
    num_leaves=63, min_child_samples=50, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0, lambda_l2=1.0,
    max_depth=-1, learning_rate=0.03, verbose=-1, num_threads=0, max_bin=255,
)
LGBM_EARLY_STOPPING_ROUNDS = 80
LGBM_ROUND_CAP = 2000          # early-stop round cap for most instances
LGBM_ROUND_CAP_GB1 = 3000      # gb1 never plateaued at 2000; force_col_wise + higher cap

# Ridge member: alpha searched on the train-only holdout.
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
