#  7-Eleven Retail Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **90-day store-item sales forecasting for the CP All / 7-Eleven network using a global LightGBM model — achieving 12.46% SMAPE on held-out data.**

---

## Overview

This project forecasts daily sales across **500 unique store-item combinations** (10 stores × 50 items) for a 90-day horizon, using 5 years of historical data (2013–2017).

The core idea is to treat this as a **single supervised regression problem** rather than fitting 500 separate time series models. A global LightGBM model learns shared seasonality patterns across all series simultaneously, which is both more accurate and more scalable.

The full pipeline — from raw CSV ingestion to final forecast — runs end-to-end in a single notebook.

---

## Results

| Evaluation Set | SMAPE |
|---|---|
| Validation (Oct–Dec 2017, held out) | **12.46%** |

The model correctly captures the seasonal drop from the December peak into January — a key test for any retail forecasting solution.

---

## How It Works

### Feature Engineering

The most important step. Rather than using raw sales directly, the model learns from carefully constructed lag and window features that respect the forecast horizon:

- **Lag features** at 90, 91, 98, 120, and 365 days — all ≥ 90 days to avoid data leakage into the test period
- **Rolling means** (7-day and 28-day windows) shifted by 90 and 365 days, capturing recent trend and year-ago seasonality
- **Cyclical calendar encoding** — month encoded as `sin/cos` so December and January are treated as adjacent, not opposite ends of a scale
- **Standard calendar features** — day of week, week of year, is_weekend

### Model

**LightGBM** trained with a custom SMAPE evaluation metric. Validation uses a strict **time-based split** (no shuffling) to simulate real deployment conditions. Early stopping prevents overfitting. The final production model retrains on the full dataset using the optimal number of rounds found during validation.

### Data Cleaning

A small number of records had `sales = 0`, likely data entry gaps rather than true stockouts. These are corrected via **linear interpolation** within each store-item group before feature engineering.

---

## Repository Structure

```
7-eleven-demand-forecasting-production-pipeline/
├── 7eleven_demand_forecasting.ipynb   # Full pipeline: ingestion → features → model → forecast
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes data files and outputs
└── README.md
```

The `data/` folder is excluded from version control (see `.gitignore`). To reproduce results, add your `train.csv` and `test.csv` to the project root and run the notebook.

---

## Quickstart

```bash
git clone https://github.com/dragoscalin33/7-eleven-demand-forecasting-production-pipeline.git
cd 7-eleven-demand-forecasting-production-pipeline
pip install -r requirements.txt
jupyter notebook 7eleven_demand_forecasting.ipynb
```

**Expected input files** (place in project root):

| File | Columns |
|------|---------|
| `train.csv` | `date`, `store`, `item`, `sales` |
| `test.csv` | `id`, `date`, `store`, `item` |

---

## Key Design Decisions

**Why a global model instead of per-series models?**
500 individual ARIMA or ETS models would be brittle, slow to train, and unable to share information across series. A single LightGBM model generalizes better by learning that "weekends sell more" or "January always dips" applies across all stores and items.

**Why minimum 90-day lags?**
The test period starts exactly 90 days after the training cutoff. Using shorter lags (e.g., lag_7 or lag_30) would require recursive multi-step prediction, where each step's error compounds into the next. Constraining all lags to ≥ 90 days means every feature is computed from known historical data — no error accumulation.

**Why SMAPE over MAE or RMSE?**
Items with 5 daily sales and items with 500 daily sales should be weighted equally in a retail context. SMAPE normalizes errors by the scale of each series, treating a 10-unit error on a low-volume item the same as a 100-unit error on a high-volume item.

---

## Dependencies

```
lightgbm >= 4.1
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
pyarrow >= 13.0
```

---

## Author

**Dragos Estefan Calin** — Data Scientist

[LinkedIn](https://www.linkedin.com/in/dragos-calin) · [GitHub](https://github.com/dragoscalin33)
