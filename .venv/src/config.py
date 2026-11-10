"""
config.py
=========
Single place for every project-wide choice. Changing a value here changes the
whole pipeline, which keeps the analysis reproducible and honest: the train/test
split and the cost assumption are *decisions made before looking at results*, so
they live here, not scattered through the code.

Nothing in this file looks at the data. These are the knobs the workbook's
ReadMe / Checklist / Pitfalls tabs tell you to fix up front.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ROOT = the repository root (one level up from this src/ folder).
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

MERGED_CSV = DATA_PROCESSED / "merged.csv"
FEATURES_CSV = DATA_PROCESSED / "features.csv"
LABELS_CSV = DATA_PROCESSED / "regime_labels.csv"
RESULTS_CSV = REPORTS / "results_log.csv"
RESULTS_MD = REPORTS / "results_summary.md"

# --------------------------------------------------------------------------- #
# Data sources  (Data tab of the workbook)
# --------------------------------------------------------------------------- #
# Gold: pick ONE instrument. Spot avoids futures roll gaps (Pitfall #6).
GOLD_TICKER = "XAUUSD=X"          # yfinance spot gold; alt: "GC=F" futures
GOLD_START = "2006-01-01"          # broad-dollar index (DTWEXBGS) starts ~2006

# FRED series ids. Free CSV download, one file per id. (Data tab)
FRED_SERIES = {
    "real_yield": "DFII10",        # 10y TIPS real yield
    "dollar":     "DTWEXBGS",      # broad trade-weighted USD index
    "vix":        "VIXCLS",        # equity-vol / fear gauge
}

# --------------------------------------------------------------------------- #
# The decisions you lock BEFORE seeing results  (ReadMe + Pitfalls tabs)
# --------------------------------------------------------------------------- #
# Train/test split by DATE, never shuffled (Pitfall #3). Train is used to fit
# the scaler, the clustering model AND to decide which regimes are "long"
# regimes. The test period is touched once, at the end.
SPLIT_DATE = "2019-01-01"          # train: start..2018-12-31 ; test: 2019-01-01..latest

# Every macro series is lagged by one day until release timing is verified, so a
# value only known after the close is not used on the same day (Pitfall #1).
MACRO_LAG_DAYS = 1

# Transaction cost charged on every position change, as a fraction of notional
# (Pitfall #5). ~5 bps is a deliberately conservative stand-in for a retail
# XAUUSD spread on a long<->flat switch. Vary it in the sensitivity check.
COST_PER_CHANGE = 0.0005

TRADING_DAYS = 252                 # for annualising vol and returns
RISK_FREE = 0.0                    # Sharpe uses excess return; 0 keeps it simple

# --------------------------------------------------------------------------- #
# Model choices (chosen on TRAIN only — Pitfall #4)
# --------------------------------------------------------------------------- #
K_REGIMES = 3                      # headline number of k-means regimes
K_GRID = (2, 3, 4)                 # compared on training data only
RANDOM_STATE = 42                  # reproducible clustering

# Feature windows (all backward-looking — Pitfall #9)
VOL_WINDOWS = (20, 60)             # rolling realised-vol windows (days)
CHANGE_WINDOWS = (5, 20)           # macro change / momentum windows (days)
CORR_WINDOW = 60                   # rolling correlation window (days)

# Which engineered columns feed the clustering model. Kept small and
# interpretable so each regime can be explained in the README.
CLUSTER_FEATURES = [
    "vol_20",            # realised volatility (stress)
    "ry_chg_20",         # 20d change in real yield
    "usd_chg_20",        # 20d change in the dollar
    "vix_level",         # level of VIX (fear)
]

# Robustness
BOOTSTRAP_N = 1000
SENSITIVITY_K = (2, 3, 4)
SENSITIVITY_VOL_WINDOW = (10, 20, 40)
SENSITIVITY_COST = (0.0, 0.0005, 0.0015)
