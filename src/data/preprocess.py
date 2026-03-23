"""
Data loading and cleaning for the demand forecasting pipeline.
Config-driven: all parameters come from config/config.yaml.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Load ─────────────────────────────────────────────────────
def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs based on config paths."""
    train_path = cfg["data"]["raw_train"]
    test_path = cfg["data"]["raw_test"]
    date_col = cfg["data"]["date_column"]
    low_mem = cfg["data"].get("low_memory", True)

    logger.info("Loading raw data from %s and %s", train_path, test_path)
    train = pd.read_csv(train_path, parse_dates=[date_col], low_memory=low_mem)
    test = pd.read_csv(test_path, parse_dates=[date_col], low_memory=low_mem)

    logger.info("Train shape: %s | Test shape: %s", train.shape, test.shape)
    return train, test


# ── Clean ────────────────────────────────────────────────────
def clean(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean raw data:
      1. Interpolate zero-sales records per store×item.
      2. Optimize dtypes.
      3. Save interim parquet files.
    """
    target = cfg["data"]["target"]
    group_cols = cfg["data"]["group_columns"]
    cleaning = cfg["cleaning"]

    # ── 1. Interpolate zeros ──
    if cleaning.get("interpolate_zeros", False):
        zeros_mask = train[target] == 0
        n_zeros = int(zeros_mask.sum())
        if n_zeros > 0:
            logger.warning("Detected %d records with sales=0. Interpolating...", n_zeros)
            train.loc[zeros_mask, target] = np.nan
            method = cleaning.get("interpolate_method", "linear")
            train[target] = (
                train.groupby(group_cols)[target]
                .transform(lambda x: x.interpolate(method=method))
            )
            # Fill any remaining NaNs at edges with forward/backward fill
            train[target] = (
                train.groupby(group_cols)[target]
                .transform(lambda x: x.ffill().bfill())
            )
            logger.info("[FIXED] Interpolated %d zero-sales records.", n_zeros)

    # ── 2. Optimize dtypes ──
    dtype_map = cleaning.get("dtypes", {})
    for df in [train, test]:
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    # ── 3. Save interim ──
    interim_dir = Path(cfg["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(interim_dir / "train_clean.parquet", index=False)
    test.to_parquet(interim_dir / "test_clean.parquet", index=False)
    logger.info("[SUCCESS] Interim data saved to %s", interim_dir)

    return train, test


# ── Convenience wrapper ──────────────────────────────────────
def run_preprocess(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing: load → clean → return."""
    train, test = load_raw(cfg)
    train, test = clean(train, test, cfg)
    return train, test
