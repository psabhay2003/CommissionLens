"""
config.py — Central configuration for the CommissionLens pipeline.

Every magic number, file path, and tunable parameter lives here so
you never have to hunt through source files to change something.
"""

from pathlib import Path
from datetime import date

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
#  DATE RANGE
# ──────────────────────────────────────────────
START_DATE = date(2018, 1, 1)
END_DATE = date(2023, 12, 31)
QUARTER_FREQ = "QE"

# ──────────────────────────────────────────────
#  FUND UNIVERSE
#  Auto-discovered from AMFI master list by default.
#  data_collection.py matches regular↔direct pairs by scheme name.
#  SEED_FUNDS is a fallback if the AMFI API is down.
# ──────────────────────────────────────────────
MAX_FUNDS = 60  # auto-discover up to this many pairs

# Fallback only (used if AMFI master list fetch fails)
SEED_FUNDS = [
    (119551, 119598, "Axis Bluechip Fund"),
    (120503, 120505, "Mirae Asset Large Cap Fund"),
    (100356, 118989, "SBI Bluechip Fund"),
    (102885, 118834, "HDFC Top 100 Fund"),
    (100526, 119028, "ICICI Pru Bluechip Fund"),
]

# Benchmark
NIFTY50_SYMBOL = "^NSEI"

# ──────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────
ROLLING_WINDOW_QUARTERS = 4
RISK_FREE_RATE = 0.065

# ──────────────────────────────────────────────
#  MACRO FEATURES
# ──────────────────────────────────────────────
MACRO_CSV = DATA_DIR / "macro_quarterly.csv"

# ──────────────────────────────────────────────
#  MODEL TRAINING
# ──────────────────────────────────────────────
TEST_QUARTERS = 4
RANDOM_STATE = 42

# Ensemble model (XGBoost + Random Forest + Logistic Regression)
# Hyperparameters are set directly in model_training.py
# Tree-based models dominate neural networks on tabular data
# with <1000 samples (Grinsztajn et al. 2022).

# ──────────────────────────────────────────────
#  SIP SIMULATION
# ──────────────────────────────────────────────
SIP_MONTHLY_AMOUNT = 5000
SIP_DURATION_YEARS = 5
TOP_K_FUNDS = 10

# ──────────────────────────────────────────────
#  COMMISSION THRESHOLD
# ──────────────────────────────────────────────
COMMISSION_JUSTIFIED_THRESHOLD = 0.0
