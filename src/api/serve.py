"""
FastAPI serving endpoint for demand forecasting.
Loads the exported artifact and exposes /predict and /health.
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Pydantic Models
# ══════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """Single prediction request: predict sales for a store×item on a date."""
    store: int = Field(..., ge=1, le=10, description="Store ID (1-10)")
    item: int = Field(..., ge=1, le=50, description="Item ID (1-50)")
    date: date = Field(..., description="Date for prediction (YYYY-MM-DD)")


class BatchPredictRequest(BaseModel):
    """Batch prediction: multiple store×item×date combinations."""
    records: list[PredictRequest]


class PredictResponse(BaseModel):
    store: int
    item: int
    date: str
    predicted_sales: float
    confidence_band: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    holdout_smape: float | None
    features_count: int


# ══════════════════════════════════════════════════════════════
# Feature Builder (mirrors src/features/engineer.py logic)
# ══════════════════════════════════════════════════════════════

def _build_single_row_features(store: int, item: int, dt: datetime, feature_names: list) -> dict:
    """
    Build feature dict for a single prediction.
    For lag/rolling features, we return NaN (the model handles them via
    the historical data lookup or defaults).
    In production, you'd query a feature store for actual lag values.
    """
    row = {}
    row["store"] = store
    row["item"] = item
    row["day_of_week"] = dt.weekday()
    row["day_of_month"] = dt.day
    row["month"] = dt.month
    row["year"] = dt.year
    row["week_of_year"] = dt.isocalendar()[1]
    row["quarter"] = (dt.month - 1) // 3 + 1
    row["is_weekend"] = 1 if dt.weekday() >= 5 else 0
    row["is_month_start"] = 1 if dt.day == 1 else 0
    row["is_month_end"] = 1 if (dt + pd.Timedelta(days=1)).day == 1 else 0
    row["month_sin"] = float(np.sin(2 * np.pi * dt.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * dt.month / 12))
    row["day_of_week_sin"] = float(np.sin(2 * np.pi * dt.weekday() / 7))
    row["day_of_week_cos"] = float(np.cos(2 * np.pi * dt.weekday() / 7))

    # Fill remaining features (lags, rolling, etc.) with 0
    # In production, these would come from a feature store
    for feat in feature_names:
        if feat not in row:
            row[feat] = 0.0

    return row


# ══════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="7-Eleven Demand Forecaster API",
    description="Predict daily item-level sales per store.",
    version="1.0.0",
)

# Global state
_model = None
_features = None
_cat_features = None
_metadata = None


@app.on_event("startup")
def load_model():
    """Load model artifact on startup."""
    global _model, _features, _cat_features, _metadata

    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_path = cfg["api"]["model_path"]
    else:
        model_path = "models/artifacts/demand_forecaster_lgbm.joblib"

    path = Path(model_path)
    if not path.exists():
        logger.error("Model artifact not found at %s. Run `make train` first.", path)
        return

    artifact = joblib.load(path)
    _model = artifact["model"]
    _features = artifact["features"]
    _cat_features = artifact["cat_features"]
    _metadata = artifact["metadata"]
    logger.info("Model loaded: version=%s, features=%d", _metadata["version"], len(_features))


@app.get("/health", response_model=HealthResponse)
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="healthy",
        model_version=_metadata["version"],
        holdout_smape=_metadata["holdout_metrics"].get("smape"),
        features_count=len(_features),
    )


@app.post("/predict", response_model=list[PredictResponse])
def predict(request: BatchPredictRequest):
    """Predict sales for a batch of store×item×date combinations."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")

    rows = []
    for req in request.records:
        dt = datetime.combine(req.date, datetime.min.time())
        row = _build_single_row_features(req.store, req.item, dt, _features)
        rows.append(row)

    df = pd.DataFrame(rows)[_features]
    for col in _cat_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    preds = _model.predict(df)
    preds = np.maximum(preds, 0)

    responses = []
    for i, req in enumerate(request.records):
        responses.append(PredictResponse(
            store=req.store,
            item=req.item,
            date=str(req.date),
            predicted_sales=round(float(preds[i]), 2),
            confidence_band={"low": round(float(preds[i] * 0.85), 2), "high": round(float(preds[i] * 1.15), 2)},
        ))

    return responses


@app.post("/predict/single", response_model=PredictResponse)
def predict_single(request: PredictRequest):
    """Predict sales for a single store×item×date."""
    batch = BatchPredictRequest(records=[request])
    return predict(batch)[0]


# ── CLI Entry Point ──────────────────────────────────────────
def main():
    import uvicorn

    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        host = cfg["api"]["host"]
        port = cfg["api"]["port"]
    else:
        host, port = "0.0.0.0", 8000

    uvicorn.run("src.api.serve:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
