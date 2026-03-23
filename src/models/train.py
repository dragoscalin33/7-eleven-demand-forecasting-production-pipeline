"""
Training pipeline: Optuna tuning → Cross-Validation → Final model.
Orchestrates the full training flow from config.
"""

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import yaml

from src.data.preprocess import run_preprocess
from src.data.validate import validate
from src.features.engineer import build_features, select_features
from src.models.evaluate import run_cv, evaluate_holdout, smape, plot_all
from src.models.export import export_artifact

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helpers ──────────────────────────────────────────────────
def _lgbm_smape(preds, train_data):
    """Custom SMAPE metric for LightGBM callback."""
    labels = train_data.get_label()
    score = smape(preds, labels)
    return "SMAPE", score, False


def _temporal_split(df: pd.DataFrame, cfg: dict):
    """Split train data temporally: everything before val_start → train, rest → val."""
    date_col = cfg["data"]["date_column"]
    val_start = cfg["training"]["split"]["val_start"]
    train_set = df[df[date_col] < val_start].copy()
    val_set = df[df[date_col] >= val_start].copy()
    logger.info("Temporal split: train=%d, val=%d (val_start=%s)", len(train_set), len(val_set), val_start)
    return train_set, val_set


# ── Optuna ───────────────────────────────────────────────────
def _optuna_tune(X_train, y_train, X_val, y_val, cfg: dict, cat_features: list) -> dict:
    """Run Optuna hyperparameter search, return best params."""
    optuna_cfg = cfg["optuna"]
    search_space = optuna_cfg["search_space"]
    base_params = cfg["training"]["params"].copy()

    def objective(trial):
        params = base_params.copy()
        params["num_leaves"] = trial.suggest_int("num_leaves", search_space["num_leaves"]["low"], search_space["num_leaves"]["high"])
        params["learning_rate"] = trial.suggest_float("learning_rate", search_space["learning_rate"]["low"], search_space["learning_rate"]["high"], log=search_space["learning_rate"].get("log", False))
        params["feature_fraction"] = trial.suggest_float("feature_fraction", search_space["feature_fraction"]["low"], search_space["feature_fraction"]["high"])
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", search_space["bagging_fraction"]["low"], search_space["bagging_fraction"]["high"])
        params["min_child_samples"] = trial.suggest_int("min_child_samples", search_space["min_child_samples"]["low"], search_space["min_child_samples"]["high"])
        params["reg_alpha"] = trial.suggest_float("reg_alpha", search_space["reg_alpha"]["low"], search_space["reg_alpha"]["high"], log=search_space["reg_alpha"].get("log", False))
        params["reg_lambda"] = trial.suggest_float("reg_lambda", search_space["reg_lambda"]["low"], search_space["reg_lambda"]["high"], log=search_space["reg_lambda"].get("log", False))
        params["max_depth"] = trial.suggest_int("max_depth", search_space["max_depth"]["low"], search_space["max_depth"]["high"])

        dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature=cat_features)

        model = lgb.train(
            params, dtrain,
            num_boost_round=cfg["training"]["num_boost_round"],
            valid_sets=[dval],
            valid_names=["val"],
            feval=_lgbm_smape,
            callbacks=[lgb.early_stopping(cfg["training"]["early_stopping_rounds"], verbose=False)],
        )

        preds = model.predict(X_val)
        return smape(preds, y_val.values)

    study = optuna.create_study(
        direction=optuna_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=cfg["project"]["random_state"]),
    )
    study.optimize(
        objective,
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg.get("timeout"),
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info("Optuna best SMAPE: %.4f | best params: %s", study.best_value, best)
    return best


# ── Train Final Model ────────────────────────────────────────
def _train_lgbm(X_train, y_train, X_val, y_val, params: dict, cfg: dict, cat_features: list):
    """Train a single LightGBM model with early stopping."""
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature=cat_features)

    model = lgb.train(
        params, dtrain,
        num_boost_round=cfg["training"]["num_boost_round"],
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        feval=_lgbm_smape,
        callbacks=[
            lgb.early_stopping(cfg["training"]["early_stopping_rounds"]),
            lgb.log_evaluation(cfg["training"]["log_evaluation"]),
        ],
    )
    return model


# ── Main ─────────────────────────────────────────────────────
def run_training(cfg: dict) -> None:
    """
    Full training pipeline:
      1. Preprocess → Feature Engineering → Validate
      2. Temporal split
      3. Optuna tuning (optional)
      4. Cross-validation
      5. Train final model
      6. Evaluate on holdout
      7. Export artifact
    """
    # ── MLflow setup ──
    if cfg["mlflow"]["enabled"]:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run() if cfg["mlflow"]["enabled"] else _nullcontext():
        # Step 1: Preprocess
        logger.info("=" * 60)
        logger.info("STEP 1: PREPROCESS")
        logger.info("=" * 60)
        train_raw, test_raw = run_preprocess(cfg)

        # Step 2: Feature Engineering
        logger.info("=" * 60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 60)
        train_feat, test_feat = build_features(train_raw, test_raw, cfg)

        # Step 3: Validate
        logger.info("=" * 60)
        logger.info("STEP 3: VALIDATION (Pandera)")
        logger.info("=" * 60)
        validate(train_feat, cfg, is_train=True)
        validate(test_feat, cfg, is_train=False)

        # Step 4: Temporal split
        logger.info("=" * 60)
        logger.info("STEP 4: TEMPORAL SPLIT")
        logger.info("=" * 60)
        train_set, val_set = _temporal_split(train_feat, cfg)

        X_train, y_train, feature_names, cat_features = select_features(train_set, cfg)
        X_val, y_val, _, _ = select_features(val_set, cfg)

        # Step 5: Optuna (optional)
        params = cfg["training"]["params"].copy()
        if cfg["optuna"]["enabled"]:
            logger.info("=" * 60)
            logger.info("STEP 5: OPTUNA HYPERPARAMETER TUNING")
            logger.info("=" * 60)
            best_params = _optuna_tune(X_train, y_train, X_val, y_val, cfg, cat_features)
            params.update(best_params)
            if cfg["mlflow"]["enabled"]:
                for k, v in best_params.items():
                    mlflow.log_param(f"optuna_{k}", v)
        else:
            logger.info("STEP 5: OPTUNA SKIPPED (disabled in config)")

        # Step 6: Cross-Validation
        if cfg["cross_validation"]["enabled"]:
            logger.info("=" * 60)
            logger.info("STEP 6: CROSS-VALIDATION")
            logger.info("=" * 60)
            cv_metrics = run_cv(train_feat, params, cfg, cat_features, feature_names)
            if cfg["mlflow"]["enabled"]:
                for k, v in cv_metrics.items():
                    mlflow.log_metric(f"cv_{k}", v)
        else:
            logger.info("STEP 6: CROSS-VALIDATION SKIPPED")

        # Step 7: Train final model on train split
        logger.info("=" * 60)
        logger.info("STEP 7: TRAIN FINAL MODEL")
        logger.info("=" * 60)
        model = _train_lgbm(X_train, y_train, X_val, y_val, params, cfg, cat_features)

        # Step 8: Evaluate on holdout (val set)
        logger.info("=" * 60)
        logger.info("STEP 8: EVALUATE ON HOLDOUT")
        logger.info("=" * 60)
        holdout_metrics = evaluate_holdout(model, X_val, y_val, cfg)
        if cfg["mlflow"]["enabled"]:
            for k, v in holdout_metrics.items():
                mlflow.log_metric(f"holdout_{k}", v)

        # Step 9: Generate plots
        logger.info("=" * 60)
        logger.info("STEP 9: GENERATE PLOTS")
        logger.info("=" * 60)
        plot_paths = plot_all(model, X_val, y_val, feature_names, cfg)
        if cfg["mlflow"]["enabled"]:
            for p in plot_paths:
                mlflow.log_artifact(str(p))

        # Step 10: Export
        logger.info("=" * 60)
        logger.info("STEP 10: EXPORT ARTIFACT")
        logger.info("=" * 60)
        export_artifact(
            model=model,
            train_full=train_feat,
            test=test_feat,
            params=params,
            feature_names=feature_names,
            cat_features=cat_features,
            holdout_metrics=holdout_metrics,
            cv_metrics=cv_metrics if cfg["cross_validation"]["enabled"] else {},
            cfg=cfg,
        )

        if cfg["mlflow"]["enabled"]:
            mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
            mlflow.lightgbm.log_model(model, "model")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


class _nullcontext:
    """Minimal context manager for when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ── CLI Entry Point ──────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_training(cfg)


if __name__ == "__main__":
    main()
