# 7-Eleven Demand Forecasting — Production ML Pipeline (v1.0)

> **SMAPE 12.04%** | LightGBM + Optuna | FastAPI + MLflow | Config-Driven

A production-grade demand forecasting system that predicts 90 days of daily item-level sales across 10 stores and 50 products for the CP All / 7-Eleven network. Built with a modular architecture, automated hyperparameter tuning, temporal cross-validation, experiment tracking, and a real-time prediction API.

---

## Pipeline Overview

<p align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/dragoscalin33/7-eleven-demand-forecasting-production-pipeline/blob/main/docs/pipeline_overview.html">
    <img src="https://img.shields.io/badge/%F0%9F%94%8D_View_Interactive-Pipeline_Diagram-00b4d8?style=for-the-badge" alt="View Pipeline Diagram">
  </a>
</p>

---

## Architectural Evolution

| Aspect | v0 (Notebook) | v1 (Production) |
|:---|:---|:---|
| **Configuration** | Hardcoded in cells | YAML config (zero magic numbers) |
| **Features** | 5 lags + 3 rolling + calendar | 8 lags + 10 rolling + 2 expanding + 3 interactions + 14 calendar = **38** |
| **Validation** | None | Pandera schemas (store, item, sales ranges) |
| **Cross-Validation** | Single time-based split | 3-fold temporal CV with 90-day gap |
| **Tuning** | Manual hyperparameters | Optuna (15 trials, TPE sampler) |
| **Experiments** | `print()` to stdout | MLflow tracking + Model Registry |
| **Evaluation** | SMAPE only | SMAPE + MAPE + RMSE + MAE + R² + 4 diagnostic plots |
| **Serving** | None | FastAPI with `/predict`, `/predict/single`, `/health` |
| **Reproducibility** | Difficult | Full config + MLflow + seed management |

---

## Project Structure

```
demand-forecasting-prod/
├── config/                        # YAML config (single source of truth)
│   └── config.yaml               # All parameters: data, features, model, optuna, api
├── src/                           # Production Python modules
│   ├── data/                     # preprocess.py, validate.py
│   ├── features/                 # engineer.py
│   ├── models/                   # train.py, evaluate.py, export.py
│   └── api/                      # serve.py (FastAPI)
├── data/                          # Data directory (git-ignored)
│   ├── raw/                      # train.csv, test.csv
│   ├── interim/                  # Cleaned parquet
│   └── processed/                # Feature-engineered parquet
├── models/                        # Model artifacts (git-ignored)
│   ├── artifacts/                # Joblib export + submission.csv
│   └── plots/                    # Feature importance, residuals, error dist.
├── docs/                          # Interactive pipeline diagram
├── Makefile                       # make train | make serve | make clean
├── requirements.txt               # Dependencies
└── .gitignore
```

---

## Training Pipeline (10 Steps)

```
 1. Load raw CSVs                         (src/data/preprocess.py)
 2. Clean & interpolate zeros             (src/data/preprocess.py)
 3. Feature engineering (38 features)     (src/features/engineer.py)
 4. Schema validation (Pandera)           (src/data/validate.py)
 5. Temporal train/val split              (src/models/train.py)
 6. [Optional] Optuna HP search           (src/models/train.py)
 7. 3-Fold temporal cross-validation      (src/models/evaluate.py)
 8. Train final model + holdout eval      (src/models/train.py)
 9. Generate diagnostic plots             (src/models/evaluate.py)
10. Export artifact + MLflow registry     (src/models/export.py)
```

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Train full pipeline (preprocess → features → validate → optuna → cv → train → export)
make train

# 3. Serve prediction API
make serve    # → http://localhost:8000

# 4. Test prediction
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "store": 3,
    "item": 12,
    "date": "2018-02-15"
  }'

# 5. Batch prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"store": 1, "item": 5, "date": "2018-01-15"},
      {"store": 7, "item": 33, "date": "2018-03-01"}
    ]
  }'

# 6. Clean generated files
make clean
```

---

## Key Technical Decisions

### Why 90+ day lags (not lag_1, lag_7)?

We're predicting 90 days into the future. If we used lag_1 (yesterday's sales), it would be available for the first prediction day, but by day 2 we'd need to use our own prediction as input — errors compound recursively. Using lag_90+ means every feature comes from real observed data, avoiding error accumulation entirely.

### Why temporal split (not random)?

Random shuffling would let the model train on November 2017 data and "predict" March 2016 — seeing the future during training. Our temporal split trains on Jan 2014 – Sep 2017 and validates on Oct – Dec 2017, simulating exactly what happens in production.

### Why 3-fold temporal CV with 90-day gap?

Each fold trains on the past and validates on a 90-day window with a 90-day gap between training end and validation start. This matches the actual forecast horizon. The gap ensures the model can't "cheat" using recent data that would be unavailable in production.

### Why Optuna (not grid search)?

Grid search is exhaustive but slow: testing 5 values for each of 8 hyperparameters = 5⁸ = 390,625 combinations. Optuna uses Bayesian optimisation (TPE) to explore the space intelligently — 15 trials found a SMAPE of 12.33%, likely matching what grid search would find in thousands of trials.

### Why LightGBM with native categoricals?

Store and item are categorical features. One-hot encoding would create 60 sparse columns. LightGBM's native categorical handling finds optimal splits over subsets (e.g., stores {2, 5, 8} vs {1, 3, 4, 6, 7, 9, 10}), which is more efficient and captures group-level patterns that one-hot encoding misses.

### Why retrain on all data before export?

The validation set (Oct–Dec 2017) already served its purpose: honest evaluation. The production model should learn from every available datapoint. Those 46,000 rows of Q4 2017 contain seasonal patterns crucial for predicting Q1 2018.

---

## API Reference

### `GET /health`

Returns model status, version, SMAPE, and feature count.

### `POST /predict/single`

Predict sales for a single store × item × date.

**Response:**
```json
{
  "store": 3,
  "item": 12,
  "date": "2018-02-15",
  "predicted_sales": 48.72,
  "confidence_band": {
    "low": 41.41,
    "high": 56.03
  }
}
```

### `POST /predict`

Batch prediction for multiple store × item × date combinations.

---

## Technology Stack

| Category | Tools |
|:---|:---|
| **ML** | LightGBM (native categorical support) |
| **Tuning** | Optuna (Bayesian, TPE sampler) |
| **Config** | YAML (single config file) |
| **Tracking** | MLflow (experiments + Model Registry) |
| **Validation** | Pandera (schema from config) |
| **API** | FastAPI + Uvicorn + Pydantic |
| **Serialisation** | Joblib (compressed artifact) |
| **Visualisation** | Matplotlib, Seaborn |
| **Data** | Pandas, PyArrow (parquet) |

---

## Dataset

- **Source:** CP All / 7-Eleven store-item daily sales
- **Size:** 913,000 training records (5 years: 2013–2017)
- **Target:** Daily sales per store × item (regression)
- **Scope:** 10 stores × 50 items = 500 unique combinations
- **Forecast horizon:** 90 days (Q1 2018)
- **Features:** 4 raw columns → 38 engineered features (lags, rolling, expanding, calendar, interactions)

---

## Results

| Metric | Holdout (Oct–Dec 2017) | CV (3-fold mean ± std) |
|:---|:---|:---|
| **SMAPE** | 12.33% | 12.04% ± 0.76% |
| **MAPE** | 12.89% | 12.59% ± 0.80% |
| **RMSE** | 7.59 | 7.77 ± 0.40 |
| **MAE** | 5.85 | 6.00 ± 0.33 |
| **R²** | 0.929 | 0.931 ± 0.005 |

---

**Author:** [Dragos Calin](https://www.linkedin.com/in/dragos-calin33/)
