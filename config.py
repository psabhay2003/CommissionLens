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
#  ~40 equity mutual funds across Large/Mid/Small/Flexi/Value/Focused
#  categories.  Gives us ~40 × 20 quarters ≈ 800 rows before cleanup.
#  Format: (regular_plan_code, direct_plan_code, fund_name)
# ──────────────────────────────────────────────
SEED_FUNDS = [
    # ── Large Cap ──
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

    # ── Mid Cap ──
    (101762, 118837, "HDFC Mid-Cap Opportunities Fund"),
    (105760, 119772, "Kotak Emerging Equity Fund"),
    (119535, 119540, "Axis Midcap Fund"),
    (101203, 119019, "DSP Midcap Fund"),
    (103505, 118827, "Nippon India Growth Fund"),

    # ── Small Cap ──
    (104263, 125497, "SBI Small Cap Fund"),
    (119531, 119533, "Axis Small Cap Fund"),
    (114166, 118836, "HDFC Small Cap Fund"),
    (103508, 118826, "Nippon India Small Cap Fund"),

    # ── Flexi Cap / Multi Cap ──
    (122639, 122640, "Parag Parikh Flexi Cap Fund"),
    (101174, 118835, "HDFC Flexi Cap Fund"),
    (100471, 119033, "UTI Flexi Cap Fund"),
    (100354, 119061, "SBI Flexi Cap Fund"),

    # ── Value / Contra ──
    (100530, 119026, "ICICI Pru Value Discovery Fund"),
    (100092, 118831, "HDFC Capital Builder Value Fund"),
    (100355, 119062, "SBI Contra Fund"),

    # ── Focused ──
    (119541, 119544, "Axis Focused 25 Fund"),
    (100353, 119060, "SBI Focused Equity Fund"),

    # ── Large & Mid Cap ──
    (101161, 118833, "HDFC Large and Mid Cap Fund"),
    (100529, 119029, "ICICI Pru Large & Mid Cap Fund"),
    (120587, 120586, "Mirae Asset Emerging Bluechip Fund"),
    (103507, 118828, "Nippon India Vision Fund"),

    # ── Tax Saving (ELSS) — same dynamics, commission embedded ──
    (119530, 119532, "Axis Long Term Equity Fund"),
    (100516, 119024, "ICICI Pru Long Term Equity Fund"),
    (102188, 118832, "HDFC Tax Saver Fund"),
    (105756, 119769, "Kotak Tax Saver Fund"),

    # ── Aggressive Hybrid (equity-heavy) ──
    (102894, 118838, "HDFC Balanced Advantage Fund"),
    (100525, 119034, "ICICI Pru Balanced Advantage Fund"),
    (100527, 119027, "ICICI Pru Equity & Debt Fund"),
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

# DNN tuned for tabular financial data with advanced techniques
DNN_PARAMS = {
    # Architecture
    "hidden_layers": [64, 32, 16],  # residual blocks
    "dropout": 0.15,

    # Optimisation
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,           # AdamW L2 regularisation
    "epochs": 300,
    "batch_size": 32,
    "patience": 40,

    # Mixup augmentation (Zhang et al. 2018)
    "mixup_alpha": 0.4,             # Beta distribution parameter

    # Focal loss (Lin et al. 2017)
    "focal_gamma": 2.0,             # focusing parameter
    "focal_alpha": 0.25,            # class balance factor

    # Label smoothing
    "label_smoothing": 0.1,         # 0→0.05, 1→0.95

    # Cosine annealing with warm restarts
    "cosine_T0": 30,                # first restart period

    # Stochastic Weight Averaging (Izmailov et al. 2018)
    "swa_start": 120,               # start averaging from this epoch
}

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
