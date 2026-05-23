"""
config.py — Central configuration for the CommissionLens pipeline.

"""

from pathlib import Path
from datetime import date

#PATHS
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

#Creating directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)

#DATE RANGE
START_DATE = date(2018, 1, 1)
END_DATE = date(2023, 12, 31)
QUARTER_FREQ = "QE"  # Pandas quarter-end frequency

#FUND UNIVERSE
SEED_FUNDS = [
    (119551, 119598, "Axis Bluechip Fund"),
    (120503, 120505, "Mirae Asset Large Cap Fund"),
    (100356, 118989, "SBI Bluechip Fund"),
    (102885, 118834, "HDFC Top 100 Fund"),
    (100526, 119028, "ICICI Pru Bluechip Fund"),
    (105758, 119770, "Kotak Bluechip Fund"),
    (109437, 119775, "Aditya Birla SL Frontline Equity"),
    (100470, 119032, "UTI Mastershare"),
    (100473, 119568, "Franklin India Bluechip"),
    (103504, 118825, "Nippon India Large Cap Fund"),
]

#BENCHMARK
NIFTY50_SYMBOL = "^NSEI"  #Yahoo Finance ticker for Nifty 50

#FEATURE ENGINEERING
ROLLING_WINDOW_QUARTERS = 4   #1-year rolling window for alpha/beta/Sharpe
RISK_FREE_RATE = 0.065         #Approximate Indian 10Y govt bond yield

#MACRO FEATURES (manually updated or fetched from RBI DBIE)
#These are quarterly averages — the pipeline will interpolate missing values
MACRO_CSV = DATA_DIR / "macro_quarterly.csv"

#MODEL TRAINING
TEST_QUARTERS = 4             #Last 4 quarters held out (temporal split)
RANDOM_STATE = 42

#Deep Neural Network hyperparameters (PyTorch)
DNN_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 150,
    "batch_size": 64,
    "patience": 15,  #early stopping
}

#SIP SIMULATION
SIP_MONTHLY_AMOUNT = 5000     #5,000 per month
SIP_DURATION_YEARS = 5
TOP_K_FUNDS = 10              #Model picks top K funds each quarter

#COMMISSION THRESHOLD
#If net_alpha > 0, the fund's commission is "justified"
COMMISSION_JUSTIFIED_THRESHOLD = 0.0  #net alpha > 0%
