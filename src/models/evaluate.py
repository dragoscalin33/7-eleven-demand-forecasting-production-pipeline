"""
Evaluation: metrics, cross-validation, and plotting.
"""

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def smape(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    preds = np.asarray(preds, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    mask = ~((preds == 0) & (actuals == 0))
    preds, actuals = preds[mask], actuals[mask]
    denom = np.abs(preds) + np.abs(actuals)
    denom = np.where(denom == 0, 1, denom)  # avoid division by zero
    return float(100.0 * np.mean(2.0 * np.abs(preds - actuals) / denom))


def mape(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    preds = np.asarray(preds, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    mask = actuals != 0
    return float(100.0 * np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])))


def rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def mae(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - actuals)))


def r2(preds: np.ndarray, actuals: np.ndarray) -> float:
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


METRIC_FNS = {
    "smape": smape,
    "mape": mape,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}


def compute_metrics(preds: np.ndarray, actuals: np.ndarray, metric_names: list[str]) -> dict:
    """Compute a dictionary of metrics."""
    results = {}
    for name in metric_names:
        fn = METRIC_FNS.get(name)
        if fn:
            results[name] = fn(preds, actuals)
    return results


# ══════════════════════════════════════════════════════════════
# CROSS-VALIDATION (Time Series)
# ══════════════════════════════════════════════════════════════

def _lgbm_smape_eval(preds, train_data):
    labels = train_data.get_label()
    score = smape(preds, labels)
    return "SMAPE", score, False


def run_cv(
    train_feat: pd.DataFrame, params: dict, cfg: dict,
    cat_features: list, feature_names: list,
) -> dict:
    """
    Time-series cross-validation with expanding window.
    Returns dict with mean and std of each metric.
    """
    cv_cfg = cfg["cross_validation"]
    date_col = cfg["data"]["date_column"]
    target = cfg["data"]["target"]
    n_splits = cv_cfg["n_splits"]
    gap_days = cv_cfg.get("gap", 90)
    val_size_days = cv_cfg.get("val_size_days", 90)
    metric_names = cfg["evaluation"]["metrics"]

    # Sort by date
    df = train_feat.sort_values(date_col).copy()
    dates = df[date_col]
    max_date = dates.max()
    min_date = dates.min()

    # Create temporal folds working backwards from max_date
    folds = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=i * (val_size_days + gap_days))
        val_start = val_end - pd.Timedelta(days=val_size_days)
        train_end = val_start - pd.Timedelta(days=gap_days)

        if train_end <= min_date:
            logger.warning("Fold %d: not enough data, skipping", i)
            continue

        fold_train = df[dates <= train_end]
        fold_val = df[(dates > val_start) & (dates <= val_end)]

        if len(fold_val) == 0:
            continue

        folds.append((fold_train, fold_val))

    logger.info("Created %d temporal CV folds", len(folds))

    # Run each fold
    all_metrics = {name: [] for name in metric_names}

    for i, (fold_train, fold_val) in enumerate(folds):
        exclude = [date_col, target, "id", "set"]
        feat_cols = [c for c in fold_train.columns if c not in exclude]

        X_tr = fold_train[feat_cols].copy()
        y_tr = fold_train[target]
        X_va = fold_val[feat_cols].copy()
        y_va = fold_val[target]

        for col in cat_features:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype("category")
                X_va[col] = X_va[col].astype("category")

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features)
        dval = lgb.Dataset(X_va, y_va, reference=dtrain, categorical_feature=cat_features)

        model = lgb.train(
            params, dtrain,
            num_boost_round=cfg["training"]["num_boost_round"],
            valid_sets=[dval],
            valid_names=["val"],
            feval=_lgbm_smape_eval,
            callbacks=[
                lgb.early_stopping(cfg["training"]["early_stopping_rounds"], verbose=False),
            ],
        )

        preds = model.predict(X_va)
        preds = np.maximum(preds, 0)  # sales can't be negative
        fold_metrics = compute_metrics(preds, y_va.values, metric_names)

        for name, val in fold_metrics.items():
            all_metrics[name].append(val)
            logger.info("  Fold %d | %s = %.4f", i, name, val)

    # Aggregate
    results = {}
    for name, values in all_metrics.items():
        if values:
            results[f"{name}_mean"] = float(np.mean(values))
            results[f"{name}_std"] = float(np.std(values))
            logger.info("CV %s: %.4f ± %.4f", name, np.mean(values), np.std(values))

    return results


# ══════════════════════════════════════════════════════════════
# HOLDOUT EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_holdout(model, X_val, y_val, cfg: dict) -> dict:
    """Evaluate model on holdout validation set."""
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    metric_names = cfg["evaluation"]["metrics"]
    metrics = compute_metrics(preds, y_val.values, metric_names)
    for k, v in metrics.items():
        logger.info("Holdout %s: %.4f", k, v)
    return metrics


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════

def plot_all(model, X_val, y_val, feature_names: list, cfg: dict) -> list[Path]:
    """Generate all evaluation plots. Returns list of saved file paths."""
    plots_dir = Path(cfg["evaluation"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = cfg["evaluation"]["plots"]
    paths = []

    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    actuals = y_val.values

    # ── Feature Importance ──
    if plot_cfg.get("feature_importance"):
        fig, ax = plt.subplots(figsize=(10, 8))
        imp = pd.DataFrame({
            "Feature": feature_names,
            "Gain": model.feature_importance("gain"),
        }).sort_values("Gain", ascending=False).head(25)
        sns.barplot(x="Gain", y="Feature", data=imp, palette="viridis", ax=ax)
        ax.set_title("Top 25 Feature Importance (Gain)")
        fig.tight_layout()
        p = plots_dir / "feature_importance.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        logger.info("Saved: %s", p)

    # ── Residuals ──
    if plot_cfg.get("residuals"):
        residuals = actuals - preds
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(preds, residuals, alpha=0.1, s=5)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")

        axes[1].scatter(preds, actuals, alpha=0.1, s=5)
        lims = [0, max(actuals.max(), preds.max())]
        axes[1].plot(lims, lims, "r--", alpha=0.7)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_title("Actual vs Predicted")
        fig.tight_layout()
        p = plots_dir / "residuals.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        logger.info("Saved: %s", p)

    # ── Forecast vs Actual ──
    if plot_cfg.get("forecast_vs_actual"):
        fig, ax = plt.subplots(figsize=(14, 5))
        sample_size = min(500, len(actuals))
        idx = np.random.RandomState(42).choice(len(actuals), sample_size, replace=False)
        idx = np.sort(idx)
        ax.plot(range(len(idx)), actuals[idx], label="Actual", alpha=0.7)
        ax.plot(range(len(idx)), preds[idx], label="Predicted", alpha=0.7, linestyle="--")
        ax.set_title("Forecast vs Actual (sample)")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "forecast_vs_actual.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        logger.info("Saved: %s", p)

    # ── Error Distribution ──
    if plot_cfg.get("error_distribution"):
        ape = np.abs(actuals - preds) / np.where(actuals == 0, 1, actuals) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(ape[ape < 100], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(np.median(ape), color="red", linestyle="--", label=f"Median: {np.median(ape):.1f}%")
        ax.set_xlabel("Absolute Percentage Error (%)")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "error_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        logger.info("Saved: %s", p)

    return paths
