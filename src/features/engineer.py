"""
Feature engineering for demand forecasting.
All feature definitions are config-driven.
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Calendar Features ────────────────────────────────────────
def _calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create calendar/time-based features."""
    date_col = cfg["data"]["date_column"]
    dt = df[date_col].dt

    df["day_of_week"] = dt.dayofweek.astype("int8")
    df["day_of_month"] = dt.day.astype("int8")
    df["month"] = dt.month.astype("int8")
    df["year"] = dt.year.astype("int16")
    df["week_of_year"] = dt.isocalendar().week.astype("int8")
    df["quarter"] = dt.quarter.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = dt.is_month_start.astype("int8")
    df["is_month_end"] = dt.is_month_end.astype("int8")

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

    logger.info("Calendar features created: %d columns", 14)
    return df


# ── Lag Features ─────────────────────────────────────────────
def _lag_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create lag features per store×item group."""
    group_cols = cfg["data"]["group_columns"]
    target = cfg["data"]["target"]
    lags = cfg["features"]["lags"]

    g = df.groupby(group_cols)[target]
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = g.shift(lag).astype("float32")

    logger.info("Lag features created: %d lags", len(lags))
    return df


# ── Rolling Features ─────────────────────────────────────────
def _rolling_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create rolling window statistics per store×item group."""
    group_cols = cfg["data"]["group_columns"]
    target = cfg["data"]["target"]
    rolling_cfg = cfg["features"]["rolling_windows"]

    g = df.groupby(group_cols)[target]
    count = 0

    for spec in rolling_cfg:
        shift = spec["shift"]
        for window in spec["windows"]:
            for agg in spec["aggs"]:
                col_name = f"roll_{agg}_{shift}_{window}"
                df[col_name] = (
                    g.transform(lambda x, s=shift, w=window, a=agg: getattr(x.shift(s).rolling(w), a)())
                    .astype("float32")
                )
                count += 1

    logger.info("Rolling features created: %d columns", count)
    return df


# ── Expanding Features ───────────────────────────────────────
def _expanding_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create expanding window statistics per store×item group."""
    group_cols = cfg["data"]["group_columns"]
    target = cfg["data"]["target"]
    expanding_cfg = cfg["features"].get("expanding")

    if not expanding_cfg:
        return df

    shift = expanding_cfg["shift"]
    g = df.groupby(group_cols)[target]
    count = 0

    for agg in expanding_cfg["aggs"]:
        col_name = f"expanding_{agg}_{shift}"
        df[col_name] = (
            g.transform(lambda x, s=shift, a=agg: getattr(x.shift(s).expanding(min_periods=1), a)())
            .astype("float32")
        )
        count += 1

    logger.info("Expanding features created: %d columns", count)
    return df


# ── Interaction Features ─────────────────────────────────────
def _interaction_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create ratio/difference features between existing columns."""
    interactions = cfg["features"].get("interactions", [])
    count = 0

    for spec in interactions:
        name = spec["name"]
        formula = spec["formula"]

        # Parse simple A / B or A - B formulas
        if "/" in formula:
            parts = [p.strip() for p in formula.split("/")]
            if parts[0] in df.columns and parts[1] in df.columns:
                denominator = df[parts[1]].replace(0, np.nan)
                df[name] = (df[parts[0]] / denominator).astype("float32")
                count += 1
        elif "-" in formula:
            parts = [p.strip() for p in formula.split("-")]
            if parts[0] in df.columns and parts[1] in df.columns:
                df[name] = (df[parts[0]] - df[parts[1]]).astype("float32")
                count += 1

    logger.info("Interaction features created: %d columns", count)
    return df


# ── Select Features ──────────────────────────────────────────
def select_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Separate features (X) from target (y).
    Returns: X, y, feature_names, categorical_features
    """
    target = cfg["data"]["target"]
    date_col = cfg["data"]["date_column"]
    cat_features = cfg["features"]["categorical"]

    exclude = [date_col, target, "id", "set"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df[target]

    # Set categorical dtype for LightGBM native handling
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    logger.info(
        "Feature selection: %d features (%d categorical)",
        len(feature_cols), len(cat_features),
    )
    return X, y, feature_cols, cat_features


# ── Main Pipeline ────────────────────────────────────────────
def build_features(
    train: pd.DataFrame, test: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline:
      1. Combine train+test (for correct lag computation).
      2. Create all features.
      3. Split back and save processed parquet.
    """
    train = train.copy()
    test = test.copy()

    train["_set"] = "train"
    test["_set"] = "test"
    target = cfg["data"]["target"]
    if target not in test.columns:
        test[target] = np.nan

    full = pd.concat([train, test], axis=0, ignore_index=True)
    full = full.sort_values(cfg["data"]["group_columns"] + [cfg["data"]["date_column"]])

    # Apply feature engineering steps
    full = _calendar_features(full, cfg)
    full = _lag_features(full, cfg)
    full = _rolling_features(full, cfg)
    full = _expanding_features(full, cfg)
    full = _interaction_features(full, cfg)

    # Split back
    train_feat = full[full["_set"] == "train"].drop("_set", axis=1).copy()
    test_feat = full[full["_set"] == "test"].drop("_set", axis=1).copy()

    # Filter by min_date (need enough history for lags)
    min_date = cfg["cleaning"].get("min_date")
    if min_date:
        date_col = cfg["data"]["date_column"]
        before = len(train_feat)
        train_feat = train_feat[train_feat[date_col] >= min_date]
        logger.info("Filtered train from %d to %d rows (min_date=%s)", before, len(train_feat), min_date)

    # Drop rows with NaN in lag columns (artifacts of shifting)
    lag_cols = [c for c in train_feat.columns if c.startswith("lag_") or c.startswith("roll_") or c.startswith("expanding_")]
    before = len(train_feat)
    train_feat = train_feat.dropna(subset=lag_cols)
    logger.info("Dropped %d rows with NaN lags. Final train: %d rows", before - len(train_feat), len(train_feat))

    # Save processed
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_feat.to_parquet(proc_dir / "train_features.parquet", index=False)
    test_feat.to_parquet(proc_dir / "test_features.parquet", index=False)
    logger.info("[SUCCESS] Processed features saved to %s", proc_dir)

    return train_feat, test_feat
