#  7-Eleven Retail Demand Forecasting Production Pipeline

> **Achieved an Elite SMAPE of 12.46% on the CP All / 7-Eleven benchmark using a Global LightGBM Gradient Boosting architecture.**

##  Overview

This project addresses the **Multi-step Time Series Forecasting** challenge for 7-Eleven (CP All). The objective is to forecast 90 days of daily sales across 500 unique store-item combinations.

Instead of traditional per-series modeling, this solution implements a **Global Model approach**. This allows the algorithm to learn complex cross-series seasonalities and trends, providing a more robust and scalable solution for large-scale retail environments.

##  Business Problem

Inventory is a primary driver of retail margins. Overstocking traps capital, while understocking leads to lost revenue and customer churn.

### Success Criteria

* 
**Inventory Optimization**: Accurate 90-day forecasts enable precise stock adjustments, specifically targeting the "January Drop" following the Q4 holiday peak.


* 
**Operational Velocity**: The modular pipeline is designed to reduce the time needed to onboard and forecast new store-item data.


* 
**Financial Reliability**: By achieving low validation variance, the model provides a dependable baseline for quarterly financial planning.



##  Data Description

* 
**Source**: 5 years of daily sales data (2013-2017).


* 
**Scope**: 10 stores across 50 items (500 unique time series).


* 
**Target**: Daily unit sales for Q1 2018.



### Data Quality & Sanitation

* 
**Zero-Sales Correction**: Handled records with zero sales via linear interpolation to maintain signal continuity and prevent noise distortion.


* 
**Memory Optimization**: Implemented specific data types (`int8`, `float32`) to ensure the pipeline remains efficient even as the dataset scales.



##  Methodology

### 1. Advanced Feature Engineering

* 
**Cyclical Encoding**: Applied sine and cosine transformations to month and day features to preserve temporal proximity (e.g., ensuring December and January are treated as adjacent).


* 
**Lag Analysis**: Utilized a strategic mix of 90-day to 365-day lags to anchor the model in both long-term seasonality and the current forecast horizon.


* 
**Rolling Windows**: Implemented 7-day and 28-day moving averages to capture short-term trend volatility.



### 2. Modeling Strategy

* 
**Algorithm**: LightGBM (Gradient Boosting Decision Trees).


* 
**Validation**: Time-based holdout strategy using the final three months of 2017 (Oct–Dec) to prevent data leakage and simulate real-world deployment.


* 
**Metric**: SMAPE (Symmetric Mean Absolute Percentage Error).



##  Results

* 
**Validation Performance**: **12.46% SMAPE**.


* 
**Seasonality Continuity**: The model correctly identified the sharp transition from the December peak to the January lull, with January 2018 predictions aligning closely with January 2017 historical averages.



##  Project Structure

The repository follows a production-ready modular structure:

```text
├── configs/                     # Externalized model & pipeline parameters
├── data/                        # raw and processed datasets (Parquet format)
├── notebooks/                   # Exploratory data analysis and experimentation
├── scripts/                     # Modular execution scripts
├── src/                         # Core importable source modules
├── utils_ds_elite.py            # Reusable production-grade helper functions
└── 7eleven_demand_forecasting.ipynb # End-to-end technical execution

```

##  Installation & Usage

1. **Clone the repo**:
```bash
git clone https://github.com/[your-username]/7-eleven-demand-forecasting-production-pipeline.git
cd 7-eleven-demand-forecasting-production-pipeline

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Run the pipeline**: Execute the `7eleven_demand_forecasting.ipynb` notebook to reproduce the results.

---

*##  Author

**Dragos Estefan Calin**  
Data Scientist  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

*Built as part of the Gosoft (Thailand) Data Scientist hiring challenge — January 2026.*
