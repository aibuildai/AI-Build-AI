#!/usr/bin/env python
"""
GBDT Trio Ensemble: XGBoost + LightGBM + CatBoost
Heart Disease Prediction (Kaggle Playground Series S6E2)

Full feature engineering pipeline:
- Digit extraction from numeric values
- Quantile and uniform binning
- Categorical copies of numeric features
- Target encoding within CV folds

Models are trained with GPU acceleration and ensembled via
rank-transformed ridge regression meta-learner.
"""

import os
import sys
import json
import time
import signal
import random
import argparse
import warnings
import gc

# CPU limits
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import yaml
import torch

torch.set_num_threads(4)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Globals for signal handling
# ---------------------------------------------------------------------------
CHECKPOINT_STATE = {}
OUTPUT_DIR = ""


def sigterm_handler(signum, frame):
    """Save checkpoint on SIGTERM."""
    if CHECKPOINT_STATE and OUTPUT_DIR:
        ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint_latest.pth")
        torch.save(CHECKPOINT_STATE, ckpt_path)
        print(f"[SIGTERM] Saved checkpoint to {ckpt_path}")
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
NUMERIC_COLS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
BINNING_COLS = ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]
ORIGINAL_CAT_COLS = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"
]
ALL_BASE_COLS = NUMERIC_COLS + ORIGINAL_CAT_COLS


def extract_digit_features(df, col):
    """Extract individual digits from numeric values."""
    new_cols = {}
    abs_vals = df[col].abs()
    int_part = abs_vals.astype(int)

    # Integer digits (ones, tens, hundreds, ...)
    for pos in range(1, 4):
        digit_col_name = f"{col}_digit_pos_{pos}"
        new_cols[digit_col_name] = ((int_part // (10 ** (pos - 1))) % 10).astype(np.int8)

    # Decimal digits for float columns
    if df[col].dtype in [np.float64, np.float32, float]:
        frac_part = abs_vals - int_part
        dec1 = (frac_part * 10).astype(int) % 10
        dec2 = (frac_part * 100).astype(int) % 10
        new_cols[f"{col}_digit_dec_1"] = dec1.astype(np.int8)
        new_cols[f"{col}_digit_dec_2"] = dec2.astype(np.int8)

    return pd.DataFrame(new_cols, index=df.index)


def create_binning_features(train_df, test_df, col, quantile_bins, uniform_bins):
    """Create quantile and uniform binning features."""
    train_new = {}
    test_new = {}

    for q in quantile_bins:
        col_name = f"{col}_qbin_{q}"
        try:
            train_new[col_name], bin_edges = pd.qcut(
                train_df[col], q=q, labels=False, retbins=True, duplicates="drop"
            )
            test_new[col_name] = pd.cut(
                test_df[col], bins=bin_edges, labels=False, include_lowest=True
            )
        except Exception:
            train_new[col_name] = 0
            test_new[col_name] = 0

    for n_bins in uniform_bins:
        col_name = f"{col}_ubin_{n_bins}"
        try:
            train_new[col_name], bin_edges = pd.cut(
                train_df[col], bins=n_bins, labels=False, retbins=True, include_lowest=True
            )
            test_new[col_name] = pd.cut(
                test_df[col], bins=bin_edges, labels=False, include_lowest=True
            )
        except Exception:
            train_new[col_name] = 0
            test_new[col_name] = 0

    return pd.DataFrame(train_new, index=train_df.index), pd.DataFrame(test_new, index=test_df.index)


def engineer_features(train_df, test_df, config):
    """Apply full feature engineering pipeline to train and test."""
    print("[FE] Starting feature engineering...")
    t0 = time.time()

    train_feats = []
    test_feats = []

    # 1. Digit features
    for col in NUMERIC_COLS:
        train_feats.append(extract_digit_features(train_df, col))
        test_feats.append(extract_digit_features(test_df, col))

    # 2. Binning features
    quantile_bins = config.get("quantile_bins", [4, 5, 10, 20])
    uniform_bins = config.get("uniform_bins", [5, 10])

    for col in BINNING_COLS:
        train_bin, test_bin = create_binning_features(
            train_df, test_df, col, quantile_bins, uniform_bins
        )
        train_feats.append(train_bin)
        test_feats.append(test_bin)

    # Age rounded to nearest 5
    train_feats.append(pd.DataFrame(
        {"Age_round5": (train_df["Age"] / 5).round() * 5},
        index=train_df.index
    ))
    test_feats.append(pd.DataFrame(
        {"Age_round5": (test_df["Age"] / 5).round() * 5},
        index=test_df.index
    ))

    # 3. Categorical copies of numeric features (as string)
    cat_copy_cols = []
    for col in NUMERIC_COLS:
        cat_col_name = f"{col}_cat"
        train_feats.append(pd.DataFrame(
            {cat_col_name: train_df[col].astype(str)}, index=train_df.index
        ))
        test_feats.append(pd.DataFrame(
            {cat_col_name: test_df[col].astype(str)}, index=test_df.index
        ))
        cat_copy_cols.append(cat_col_name)

    # Combine original + engineered
    train_all = pd.concat([train_df[ALL_BASE_COLS]] + train_feats, axis=1)
    test_all = pd.concat([test_df[ALL_BASE_COLS]] + test_feats, axis=1)

    # Fill NaN in binning columns
    for c in train_all.columns:
        if "bin" in c:
            train_all[c] = train_all[c].fillna(-1).astype(np.int32)
            test_all[c] = test_all[c].fillna(-1).astype(np.int32)

    print(f"[FE] Feature engineering done in {time.time()-t0:.1f}s. "
          f"Train shape: {train_all.shape}, Test shape: {test_all.shape}")

    # Identify all categorical columns for target encoding
    cat_cols_for_te = list(ORIGINAL_CAT_COLS) + cat_copy_cols
    # Also add binning columns as categorical
    bin_cols = [c for c in train_all.columns if "bin" in c or c == "Age_round5"]
    cat_cols_for_te += bin_cols

    return train_all, test_all, cat_cols_for_te


def apply_target_encoding(X_train, y_train, X_val, X_test, cat_cols, seed=42):
    """Apply target encoding within a CV fold."""
    te = TargetEncoder(
        categories="auto",
        target_type="binary",
        smooth="auto",
        random_state=seed
    )

    # Convert cat cols to string for TargetEncoder
    X_train_cat = X_train[cat_cols].astype(str)
    X_val_cat = X_val[cat_cols].astype(str)
    X_test_cat = X_test[cat_cols].astype(str)

    te.fit(X_train_cat, y_train)

    te_train = pd.DataFrame(
        te.transform(X_train_cat),
        columns=[f"{c}_te" for c in cat_cols],
        index=X_train.index
    )
    te_val = pd.DataFrame(
        te.transform(X_val_cat),
        columns=[f"{c}_te" for c in cat_cols],
        index=X_val.index
    )
    te_test = pd.DataFrame(
        te.transform(X_test_cat),
        columns=[f"{c}_te" for c in cat_cols],
        index=X_test.index
    )

    return te_train, te_val, te_test


def get_numeric_features(X, cat_cols):
    """Get columns that are numeric (not in cat_cols and not string type)."""
    numeric_cols = []
    for c in X.columns:
        if c not in cat_cols and X[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]:
            numeric_cols.append(c)
    return numeric_cols


def prepare_fold_data(X_train_fold, y_train_fold, X_val_fold, X_test_all, cat_cols_for_te, seed=42):
    """Prepare fold data: target encoding + combine numeric features."""
    # Get numeric columns (non-categorical)
    numeric_cols = get_numeric_features(X_train_fold, cat_cols_for_te)

    # Apply target encoding
    te_train, te_val, te_test = apply_target_encoding(
        X_train_fold, y_train_fold, X_val_fold, X_test_all, cat_cols_for_te, seed
    )

    # Combine numeric + target-encoded features
    X_train_num = pd.concat([X_train_fold[numeric_cols].reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
    X_val_num = pd.concat([X_val_fold[numeric_cols].reset_index(drop=True), te_val.reset_index(drop=True)], axis=1)
    X_test_num = pd.concat([X_test_all[numeric_cols].reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)

    # Ensure all numeric
    for c in X_train_num.columns:
        X_train_num[c] = pd.to_numeric(X_train_num[c], errors="coerce").fillna(0)
        X_val_num[c] = pd.to_numeric(X_val_num[c], errors="coerce").fillna(0)
        X_test_num[c] = pd.to_numeric(X_test_num[c], errors="coerce").fillna(0)

    return X_train_num, X_val_num, X_test_num


# ---------------------------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val, config, seed=42):
    """Train XGBoost with GPU."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        # Use cuda:0 since CUDA_VISIBLE_DEVICES handles GPU selection
        "device": "cuda:0",
        "max_depth": config.get("xgb_max_depth", 6),
        "min_child_weight": config.get("xgb_min_child_weight", 5),
        "subsample": config.get("xgb_subsample", 0.8),
        "colsample_bytree": config.get("xgb_colsample_bytree", 0.8),
        "reg_alpha": config.get("xgb_reg_alpha", 0.1),
        "reg_lambda": config.get("xgb_reg_lambda", 1.0),
        "gamma": config.get("xgb_gamma", 0.0),
        "learning_rate": config.get("xgb_learning_rate", 0.05),
        "seed": seed,
        "verbosity": 0,
    }

    n_rounds = config.get("xgb_n_rounds", 2000)
    early_stopping = config.get("xgb_early_stopping", 50)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )

    val_pred = model.predict(dval)

    return model, val_pred


def train_lightgbm(X_train, y_train, X_val, y_val, config, seed=42):
    """Train LightGBM (CPU or GPU based on config)."""
    lgb_device = config.get("lgb_device", "cpu")
    params = {
        "objective": "binary",
        "metric": "auc",
        "device": lgb_device,
        "num_leaves": config.get("lgb_num_leaves", 63),
        "min_data_in_leaf": config.get("lgb_min_data_in_leaf", 50),
        "feature_fraction": config.get("lgb_feature_fraction", 0.8),
        "bagging_fraction": config.get("lgb_bagging_fraction", 0.8),
        "bagging_freq": config.get("lgb_bagging_freq", 5),
        "lambda_l1": config.get("lgb_lambda_l1", 0.1),
        "lambda_l2": config.get("lgb_lambda_l2", 1.0),
        "path_smooth": config.get("lgb_path_smooth", 0.0),
        "learning_rate": config.get("lgb_learning_rate", 0.05),
        "seed": seed,
        "verbose": -1,
        "num_threads": 4,
    }
    if lgb_device == "gpu":
        params["gpu_device_id"] = 0

    n_rounds = config.get("lgb_n_rounds", 2000)
    early_stopping = config.get("lgb_early_stopping", 50)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)

    return model, val_pred


def train_catboost(X_train, y_train, X_val, y_val, cat_features, config, seed=42):
    """Train CatBoost with GPU."""
    # Set train_dir inside output_dir to contain catboost_info
    cb_train_dir = os.path.join(config.get("output_dir", "."), "catboost_info")

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        # Always use device 0 since CUDA_VISIBLE_DEVICES handles GPU selection
        "devices": "0",
        "depth": config.get("cb_depth", 6),
        "l2_leaf_reg": config.get("cb_l2_leaf_reg", 3.0),
        "random_strength": config.get("cb_random_strength", 1.0),
        "bagging_temperature": config.get("cb_bagging_temperature", 1.0),
        "learning_rate": config.get("cb_learning_rate", 0.05),
        "iterations": config.get("cb_iterations", 2000),
        "random_seed": seed,
        "verbose": 0,
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": config.get("cb_early_stopping", 50),
        "train_dir": cb_train_dir,
    }

    # Ensure cat features are strings
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    for c in cat_features:
        if c in X_train_cb.columns:
            X_train_cb[c] = X_train_cb[c].astype(str)
            X_val_cb[c] = X_val_cb[c].astype(str)

    valid_cat_features = [c for c in cat_features if c in X_train_cb.columns]

    pool_train = cb.Pool(X_train_cb, y_train, cat_features=valid_cat_features)
    pool_val = cb.Pool(X_val_cb, y_val, cat_features=valid_cat_features)

    model = cb.CatBoostClassifier(**params)
    model.fit(pool_train, eval_set=pool_val, verbose=0)

    val_pred = model.predict_proba(pool_val)[:, 1]

    return model, val_pred


# ---------------------------------------------------------------------------
# CatBoost data preparation (different from XGB/LGB -- uses raw cats)
# ---------------------------------------------------------------------------
def prepare_catboost_fold_data(X_train_fold, y_train_fold, X_val_fold, X_test_all, cat_cols_for_te, seed=42):
    """Prepare fold data for CatBoost: numeric + raw categorical features."""
    numeric_cols = get_numeric_features(X_train_fold, cat_cols_for_te)

    # Target encode only non-original categoricals (copies, bins)
    non_native_cats = [c for c in cat_cols_for_te if c not in ORIGINAL_CAT_COLS]
    te_train, te_val, te_test = apply_target_encoding(
        X_train_fold, y_train_fold, X_val_fold, X_test_all, non_native_cats, seed
    )

    # Keep original categoricals as-is for CatBoost
    cat_feature_names = [c for c in ORIGINAL_CAT_COLS if c in X_train_fold.columns]

    X_train_cb = pd.concat([
        X_train_fold[numeric_cols + cat_feature_names].reset_index(drop=True),
        te_train.reset_index(drop=True)
    ], axis=1)
    X_val_cb = pd.concat([
        X_val_fold[numeric_cols + cat_feature_names].reset_index(drop=True),
        te_val.reset_index(drop=True)
    ], axis=1)
    X_test_cb = pd.concat([
        X_test_all[numeric_cols + cat_feature_names].reset_index(drop=True),
        te_test.reset_index(drop=True)
    ], axis=1)

    # Ensure numeric columns are numeric
    for c in X_train_cb.columns:
        if c not in cat_feature_names:
            X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce").fillna(0)
            X_val_cb[c] = pd.to_numeric(X_val_cb[c], errors="coerce").fillna(0)
            X_test_cb[c] = pd.to_numeric(X_test_cb[c], errors="coerce").fillna(0)

    return X_train_cb, X_val_cb, X_test_cb, cat_feature_names


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_progress(output_dir, entry):
    """Append a JSON line to training_progress.jsonl."""
    path = os.path.join(output_dir, "training_progress.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    global OUTPUT_DIR
    OUTPUT_DIR = config["output_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_dir = config["data_dir"]
    test_mode = config.get("test_mode", False)
    num_folds = config.get("num_folds", 5)
    seeds = config.get("seeds", [42, 123, 456])
    gpu_id = config.get("gpu_id", 0)

    set_seed(42)

    print(f"[CONFIG] output_dir={OUTPUT_DIR}")
    print(f"[CONFIG] test_mode={test_mode}, num_folds={num_folds}, seeds={seeds}")
    print(f"[CONFIG] gpu_id={gpu_id}")

    # Resume checkpoint support
    resume_path = config.get("resume_checkpoint")
    start_state = None
    if resume_path and os.path.exists(resume_path):
        print(f"[RESUME] Loading checkpoint from {resume_path}")
        start_state = torch.load(resume_path, map_location="cpu")
        print(f"[RESUME] Resuming from step={start_state.get('step', 0)}, "
              f"epoch={start_state.get('epoch', 0)}")

    # -----------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------
    t_start = time.time()
    print("[DATA] Loading data...")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Map target
    target_map = {"Absence": 0, "Presence": 1}
    train_df["Heart Disease"] = train_df["Heart Disease"].map(target_map)

    test_ids = test_df["id"].values

    # Drop id
    train_df = train_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    y = train_df["Heart Disease"].values
    train_df = train_df.drop(columns=["Heart Disease"])

    if test_mode:
        max_samples = config.get("test_max_samples", 100)
        print(f"[TEST MODE] Limiting to {max_samples} train samples")
        train_df = train_df.iloc[:max_samples].reset_index(drop=True)
        y = y[:max_samples]
        test_df = test_df.iloc[:max_samples].reset_index(drop=True)
        test_ids = test_ids[:max_samples]
        # Override for speed
        num_folds = min(num_folds, 2)
        seeds = seeds[:1]

    print(f"[DATA] Train: {train_df.shape}, Test: {test_df.shape}, "
          f"Target mean: {y.mean():.4f}")

    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------
    X_train_all, X_test_all, cat_cols_for_te = engineer_features(
        train_df, test_df, config
    )

    del train_df
    gc.collect()

    print(f"[FE] Total features: {X_train_all.shape[1]}")
    print(f"[FE] Categorical cols for TE: {len(cat_cols_for_te)}")

    # -----------------------------------------------------------------------
    # CV Training
    # -----------------------------------------------------------------------
    n_train = len(y)
    n_test = len(X_test_all)

    # OOF and test predictions storage
    oof_xgb = np.zeros((n_train, len(seeds)))
    oof_lgb = np.zeros((n_train, len(seeds)))
    oof_cb = np.zeros((n_train, len(seeds)))

    test_xgb = np.zeros((n_test, num_folds, len(seeds)))
    test_lgb = np.zeros((n_test, num_folds, len(seeds)))
    test_cb = np.zeros((n_test, num_folds, len(seeds)))

    fold_aucs = {"xgb": [], "lgb": [], "cb": []}

    # Handle num_folds=1 as a special single train/val split (80/20)
    if num_folds == 1:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        fold_splits = list(sss.split(X_train_all, y))
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_splits = list(skf.split(X_train_all, y))

    step = 0
    best_metric = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"[FOLD {fold_idx+1}/{num_folds}]")
        print(f"{'='*60}")
        fold_start = time.time()

        X_train_fold = X_train_all.iloc[train_idx]
        X_val_fold = X_train_all.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        for seed_idx, seed in enumerate(seeds):
            print(f"\n  [Seed {seed}]")

            # --- Prepare data for XGBoost/LightGBM ---
            X_tr_num, X_va_num, X_te_num = prepare_fold_data(
                X_train_fold, y_train_fold, X_val_fold, X_test_all,
                cat_cols_for_te, seed
            )

            # --- XGBoost ---
            print(f"    Training XGBoost...")
            t0 = time.time()
            xgb_model, xgb_val_pred = train_xgboost(
                X_tr_num, y_train_fold, X_va_num, y_val_fold, config, seed
            )
            xgb_auc = roc_auc_score(y_val_fold, xgb_val_pred)
            print(f"    XGBoost AUC: {xgb_auc:.6f} ({time.time()-t0:.1f}s)")

            oof_xgb[val_idx, seed_idx] = xgb_val_pred
            dtest = xgb.DMatrix(X_te_num)
            test_xgb[:, fold_idx, seed_idx] = xgb_model.predict(dtest)

            del xgb_model, dtest
            gc.collect()

            # --- LightGBM ---
            print(f"    Training LightGBM...")
            t0 = time.time()
            lgb_model, lgb_val_pred = train_lightgbm(
                X_tr_num, y_train_fold, X_va_num, y_val_fold, config, seed
            )
            lgb_auc = roc_auc_score(y_val_fold, lgb_val_pred)
            print(f"    LightGBM AUC: {lgb_auc:.6f} ({time.time()-t0:.1f}s)")

            oof_lgb[val_idx, seed_idx] = lgb_val_pred
            test_lgb[:, fold_idx, seed_idx] = lgb_model.predict(X_te_num)

            del lgb_model
            gc.collect()

            # --- CatBoost ---
            print(f"    Training CatBoost...")
            t0 = time.time()
            X_tr_cb, X_va_cb, X_te_cb, cb_cat_features = prepare_catboost_fold_data(
                X_train_fold, y_train_fold, X_val_fold, X_test_all,
                cat_cols_for_te, seed
            )
            cb_model, cb_val_pred = train_catboost(
                X_tr_cb, y_train_fold, X_va_cb, y_val_fold,
                cb_cat_features, config, seed
            )
            cb_auc = roc_auc_score(y_val_fold, cb_val_pred)
            print(f"    CatBoost AUC: {cb_auc:.6f} ({time.time()-t0:.1f}s)")

            oof_cb[val_idx, seed_idx] = cb_val_pred
            # CatBoost test predictions
            X_te_cb_copy = X_te_cb.copy()
            for c in cb_cat_features:
                if c in X_te_cb_copy.columns:
                    X_te_cb_copy[c] = X_te_cb_copy[c].astype(str)
            pool_test = cb.Pool(X_te_cb_copy, cat_features=cb_cat_features)
            test_cb[:, fold_idx, seed_idx] = cb_model.predict_proba(pool_test)[:, 1]

            del cb_model, pool_test
            gc.collect()

            del X_tr_num, X_va_num, X_te_num, X_tr_cb, X_va_cb, X_te_cb
            gc.collect()

            step += 1

            # Log progress
            log_progress(OUTPUT_DIR, {
                "step": step,
                "fold": fold_idx + 1,
                "seed": seed,
                "elapsed_seconds": time.time() - t_start,
                "xgb_auc": float(xgb_auc),
                "lgb_auc": float(lgb_auc),
                "cb_auc": float(cb_auc),
            })

        # Per-fold AUC (averaged across seeds)
        fold_xgb_auc = roc_auc_score(y_val_fold, oof_xgb[val_idx].mean(axis=1))
        fold_lgb_auc = roc_auc_score(y_val_fold, oof_lgb[val_idx].mean(axis=1))
        fold_cb_auc = roc_auc_score(y_val_fold, oof_cb[val_idx].mean(axis=1))

        fold_aucs["xgb"].append(fold_xgb_auc)
        fold_aucs["lgb"].append(fold_lgb_auc)
        fold_aucs["cb"].append(fold_cb_auc)

        print(f"\n  Fold {fold_idx+1} (seed-averaged) - "
              f"XGB: {fold_xgb_auc:.6f}, LGB: {fold_lgb_auc:.6f}, CB: {fold_cb_auc:.6f}")
        print(f"  Fold time: {time.time()-fold_start:.1f}s")

        # Save checkpoint
        CHECKPOINT_STATE.update({
            "step": step,
            "fold": fold_idx + 1,
            "best_metric": best_metric,
            "oof_xgb": oof_xgb,
            "oof_lgb": oof_lgb,
            "oof_cb": oof_cb,
            "test_xgb": test_xgb,
            "test_lgb": test_lgb,
            "test_cb": test_cb,
        })

        if step % config.get("checkpoint_interval_steps", 500) == 0 or fold_idx == num_folds - 1:
            ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint_latest.pth")
            torch.save(CHECKPOINT_STATE, ckpt_path)
            print(f"  [CKPT] Saved checkpoint at step {step}")

    # -----------------------------------------------------------------------
    # Aggregate OOF predictions (average across seeds)
    # -----------------------------------------------------------------------
    oof_xgb_avg = oof_xgb.mean(axis=1)
    oof_lgb_avg = oof_lgb.mean(axis=1)
    oof_cb_avg = oof_cb.mean(axis=1)

    # Compute per-model CV AUC
    cv_xgb = roc_auc_score(y, oof_xgb_avg)
    cv_lgb = roc_auc_score(y, oof_lgb_avg)
    cv_cb = roc_auc_score(y, oof_cb_avg)

    print(f"\n{'='*60}")
    print(f"[CV RESULTS]")
    print(f"  XGBoost  CV AUC: {cv_xgb:.6f} (per-fold: {np.mean(fold_aucs['xgb']):.6f} +/- {np.std(fold_aucs['xgb']):.6f})")
    print(f"  LightGBM CV AUC: {cv_lgb:.6f} (per-fold: {np.mean(fold_aucs['lgb']):.6f} +/- {np.std(fold_aucs['lgb']):.6f})")
    print(f"  CatBoost CV AUC: {cv_cb:.6f} (per-fold: {np.mean(fold_aucs['cb']):.6f} +/- {np.std(fold_aucs['cb']):.6f})")

    # -----------------------------------------------------------------------
    # Meta-learner: Ridge on rank-transformed OOFs
    # -----------------------------------------------------------------------
    print(f"\n[ENSEMBLE] Fitting ridge meta-learner...")

    # Rank-transform OOFs
    oof_xgb_rank = rankdata(oof_xgb_avg) / len(oof_xgb_avg)
    oof_lgb_rank = rankdata(oof_lgb_avg) / len(oof_lgb_avg)
    oof_cb_rank = rankdata(oof_cb_avg) / len(oof_cb_avg)

    meta_X = np.column_stack([oof_xgb_rank, oof_lgb_rank, oof_cb_rank])

    ridge = Ridge(alpha=config.get("ridge_alpha", 1.0), random_state=42)
    ridge.fit(meta_X, y)

    meta_oof_pred = ridge.predict(meta_X)
    ensemble_auc = roc_auc_score(y, meta_oof_pred)
    print(f"  Ensemble CV AUC: {ensemble_auc:.6f}")
    print(f"  Ridge coefficients: {ridge.coef_}")

    # -----------------------------------------------------------------------
    # Generate test predictions
    # -----------------------------------------------------------------------
    test_xgb_avg = test_xgb.mean(axis=(1, 2))
    test_lgb_avg = test_lgb.mean(axis=(1, 2))
    test_cb_avg = test_cb.mean(axis=(1, 2))

    test_xgb_rank = rankdata(test_xgb_avg) / len(test_xgb_avg)
    test_lgb_rank = rankdata(test_lgb_avg) / len(test_lgb_avg)
    test_cb_rank = rankdata(test_cb_avg) / len(test_cb_avg)

    meta_test_X = np.column_stack([test_xgb_rank, test_lgb_rank, test_cb_rank])
    test_pred = ridge.predict(meta_test_X)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Submission
    submission = pd.DataFrame({
        "id": test_ids,
        "Heart Disease": test_pred
    })
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\n[OUTPUT] Saved submission to {submission_path}")
    print(f"  Submission shape: {submission.shape}")
    print(f"  Prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, "
          f"min={test_pred.min():.4f}, max={test_pred.max():.4f}")

    # Save OOF predictions for cross-candidate ensembling
    oof_data = {
        "oof_xgb": oof_xgb_avg.tolist(),
        "oof_lgb": oof_lgb_avg.tolist(),
        "oof_cb": oof_cb_avg.tolist(),
        "oof_ensemble": meta_oof_pred.tolist(),
        "y_true": y.tolist(),
    }
    oof_path = os.path.join(OUTPUT_DIR, "oof_predictions.json")
    with open(oof_path, "w") as f:
        json.dump(oof_data, f)
    print(f"[OUTPUT] Saved OOF predictions to {oof_path}")

    # Save numpy arrays too for easier loading
    np.save(os.path.join(OUTPUT_DIR, "oof_xgb.npy"), oof_xgb_avg)
    np.save(os.path.join(OUTPUT_DIR, "oof_lgb.npy"), oof_lgb_avg)
    np.save(os.path.join(OUTPUT_DIR, "oof_cb.npy"), oof_cb_avg)
    np.save(os.path.join(OUTPUT_DIR, "oof_ensemble.npy"), meta_oof_pred)
    np.save(os.path.join(OUTPUT_DIR, "test_xgb.npy"), test_xgb_avg)
    np.save(os.path.join(OUTPUT_DIR, "test_lgb.npy"), test_lgb_avg)
    np.save(os.path.join(OUTPUT_DIR, "test_cb.npy"), test_cb_avg)

    # Save ridge model coefficients
    ridge_data = {
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "alpha": config.get("ridge_alpha", 1.0),
    }
    ridge_path = os.path.join(OUTPUT_DIR, "ridge_meta.json")
    with open(ridge_path, "w") as f:
        json.dump(ridge_data, f)

    # Best model checkpoint (final state)
    best_ckpt = {
        "step": step,
        "best_metric": float(ensemble_auc),
        "oof_xgb": oof_xgb_avg,
        "oof_lgb": oof_lgb_avg,
        "oof_cb": oof_cb_avg,
        "test_xgb": test_xgb_avg,
        "test_lgb": test_lgb_avg,
        "test_cb": test_cb_avg,
        "ridge_coef": ridge.coef_,
        "ridge_intercept": ridge.intercept_,
        "config": config,
    }
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    torch.save(best_ckpt, best_path)
    print(f"[OUTPUT] Saved best model to {best_path}")

    # Results JSON
    results = {
        "score": float(ensemble_auc),
        "xgb_cv_auc": float(cv_xgb),
        "lgb_cv_auc": float(cv_lgb),
        "cb_cv_auc": float(cv_cb),
        "ensemble_cv_auc": float(ensemble_auc),
        "num_folds": num_folds,
        "seeds": seeds,
        "total_time_seconds": time.time() - t_start,
        "ridge_coef": ridge.coef_.tolist(),
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[RESULTS] {json.dumps(results, indent=2)}")

    print(f"\n[DONE] Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
