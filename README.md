# CommissionLens

**Commission-Adjusted Alpha Prediction in Indian Mutual Funds**

> Are regular-plan commissions (0.5–1.5% annually) justified by fund performance? This project uses ML to find out.

---

## Problem

India has 90M+ demat accounts, yet most retail investors don't realize that **regular plans** silently deduct 0.5–1.5% annually as distributor commission compared to direct plans. A fund charging 0.6% more but returning only 1.1% above its benchmark means the investor **loses 0.5% to commission** with almost nothing to show for it.

**CommissionLens** builds an ML pipeline that:
1. Computes **commission-adjusted net alpha** for 200+ Indian equity mutual funds
2. Predicts whether a fund will generate **positive net alpha next quarter**
3. Identifies which fund characteristics **most justify** paying the commission
4. Back-validates via **SIP simulation** over 2018–2023

## Repository Structure

```
CommissionLens/
├── config.py                    # All configurable parameters
├── requirements.txt             # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py       # Fetch NAV data (mfapi.in), benchmark, macro
│   ├── feature_engineering.py   # Rolling alpha/beta/Sharpe, expense gap, XIRR
│   ├── target_builder.py        # Net alpha computation & binary labels
│   ├── model_training.py        # Dual-head Deep Neural Network training
│   ├── shap_analysis.py         # SHAP explainability & top feature report
│   ├── sip_simulation.py        # SIP back-validation (model vs naive)
│   └── utils.py                 # Helpers (XIRR, rolling windows, etc.)
│
├── notebooks/
│   └── CommissionLens_Pipeline.ipynb   # End-to-end reproducible notebook
│
├── app.py                       # Streamlit dashboard (optional deliverable)
├── run_pipeline.py              # Single-command full pipeline execution
│
├── data/                        # Auto-populated by data_collection.py
├── models/                      # Saved trained models
└── reports/                     # SHAP plots, metrics, SIP comparison charts
```

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/psabhay2003/CommissionLens.git
cd CommissionLens
pip install -r requirements.txt

# 2. Run the full pipeline
python run_pipeline.py

# 3. Launch the dashboard (optional)
streamlit run app.py
```

## Deliverables Checklist

| # | Deliverable | File(s) |
|---|------------|---------|
| 1 | Cleaned dataset of 200+ funds × 5 years | `data/fund_features.csv` |
| 2 | Trained ML models with RMSE, AUC-ROC, F1 | `models/`, `reports/metrics.json` |
| 3 | SHAP explainability report (top 3–5 features) | `reports/shap_summary.png` |
| 4 | SIP back-validation (model vs naive XIRR) | `reports/sip_comparison.png` |
| 5 | Jupyter notebook pipeline | `notebooks/CommissionLens_Pipeline.ipynb` |
| 6 | Streamlit dashboard (optional) | `app.py` |

## Tech Stack

Python · Pandas · NumPy · scikit-learn · PyTorch · SHAP · SciPy · Matplotlib · Seaborn · Streamlit

## Data Sources

- **mfapi.in** — Historical NAV for every Indian mutual fund
- **AMFI** — Fund metadata and expense ratios
- **RBI DBIE** — Repo rate, CPI, yield curve
- **NSE/Yahoo Finance** — Nifty 50 benchmark returns
