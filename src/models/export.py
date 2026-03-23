"""
Export trained model artifact to disk.
Packages model + metadata into a single joblib file.
"""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_artifact(
    model,
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
    feature_names: list[str],
    cat_features: list[str],
    holdout_metrics: dict,
    cv_metrics: dict,
    cfg: dict,
) -> Path:
    """
    Optionally retrain on ALL data, then export:
      - model (LightGBM Booster)
      - features list
      - categorical features list
      - metadata (version, metrics, config)
    """
    target = cfg["data"]["target"]
    date_col = cfg["data"]["date_column"]
    exclude = [date_col, target, "id", "set"]
    feat_cols = [c for c in train_full.columns if c not in exclude]

    # ── Retrain on full dataset if configured ──
    if cfg["export"].get("retrain_full", True):
        logger.info("Retraining model on FULL training data (%d rows)...", len(train_full))
        X_full = train_full[feat_cols].copy()
        y_full = train_full[target]

        for col in cat_features:
            if col in X_full.columns:
                X_full[col] = X_full[col].astype("category")

        dtrain = lgb.Dataset(X_full, y_full, categorical_feature=cat_features)
        final_model = lgb.train(
            params, dtrain,
            num_boost_round=model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration > 0 else cfg["training"]["num_boost_round"],
        )
        logger.info("Full retrain complete. best_iteration=%d", final_model.best_iteration)
    else:
        final_model = model

    # ── Generate submission predictions ──
    logger.info("Generating submission predictions...")
    X_test = test[feat_cols].copy()
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    preds = final_model.predict(X_test)
    preds = np.maximum(preds, 0)  # sales can't be negative

    sub = test[["id", "date", "store", "item"]].copy()
    sub["sales"] = preds
    if "id" in sub.columns:
        sub["id"] = sub["id"].astype(int)

    sub_path = Path(cfg["export"]["artifact_dir"]) / "submission.csv"
    sub[["id", "sales"]].to_csv(sub_path, index=False)
    logger.info("Submission saved: %s (%d predictions)", sub_path, len(preds))

    # ── Package artifact ──
    artifact = {
        "model": final_model,
        "features": feature_names,
        "cat_features": cat_features,
        "metadata": {
            "version": cfg["project"]["version"],
            "model_type": cfg["training"]["model"],
            "best_iteration": final_model.best_iteration,
            "holdout_metrics": holdout_metrics,
            "cv_metrics": cv_metrics,
            "params": params,
            "config": cfg,
        },
    }

    artifact_dir = Path(cfg["export"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{cfg['export']['artifact_name']}.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    logger.info("[SUCCESS] Artifact exported: %s (%.1f MB)", artifact_path, artifact_path.stat().st_size / 1e6)

    return artifact_path
