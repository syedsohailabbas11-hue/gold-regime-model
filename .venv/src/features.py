"""
features.py
===========
Turn the merged price/macro table into model-ready features (Checklist Day 2).

Every window is BACKWARD-LOOKING. No centred windows, no future data
(Pitfall #9). Returns are log returns so they add across days (Knowledge tab).

Features produced:
  * ret            : daily log return of gold
  * vol_20, vol_60 : annualised rolling realised volatility
  * ry_chg_5/20    : 5d / 20d change in the real yield (level difference)
  * usd_chg_5/20   : 5d / 20d % change in the dollar index
  * mom_20         : 20d gold momentum (sum of log returns)
  * vix_level      : VIX (already lagged in build_dataset)
  * corr_ry_60     : rolling 60d corr(gold return, daily real-yield change)
  * corr_usd_60    : rolling 60d corr(gold return, daily dollar return)

    python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(merged: pd.DataFrame | None = None) -> pd.DataFrame:
    if merged is None:
        merged = pd.read_csv(config.MERGED_CSV, parse_dates=["date"]).set_index("date")

    df = merged.copy()

    # --- gold log return ---------------------------------------------------- #
    df["ret"] = np.log(df["gold"]).diff()

    # --- rolling realised volatility (annualised) --------------------------- #
    for w in config.VOL_WINDOWS:
        df[f"vol_{w}"] = df["ret"].rolling(w).std() * np.sqrt(config.TRADING_DAYS)

    # --- macro changes (backward differences) ------------------------------- #
    for w in config.CHANGE_WINDOWS:
        # real yield is already a rate in %, so use a level difference
        df[f"ry_chg_{w}"] = df["real_yield"].diff(w)
        # dollar is an index level, so use a percentage change
        df[f"usd_chg_{w}"] = df["dollar"].pct_change(w)

    # --- gold momentum ------------------------------------------------------ #
    df["mom_20"] = df["ret"].rolling(20).sum()

    # --- fear level --------------------------------------------------------- #
    df["vix_level"] = df["vix"]

    # --- rolling correlations (context, Checklist Day 2) -------------------- #
    ry_daily = df["real_yield"].diff()
    usd_daily = df["dollar"].pct_change()
    df["corr_ry_60"] = df["ret"].rolling(config.CORR_WINDOW).corr(ry_daily)
    df["corr_usd_60"] = df["ret"].rolling(config.CORR_WINDOW).corr(usd_daily)

    # Drop warm-up rows where the longest window is still NaN.
    df = df.dropna(subset=config.CLUSTER_FEATURES + ["ret"])
    df.to_csv(config.FEATURES_CSV)
    print(f"Saved {config.FEATURES_CSV}  ({len(df):,} rows, "
          f"{df.index.min().date()} .. {df.index.max().date()})")
    return df


def train_test_masks(df: pd.DataFrame):
    """Boolean masks for the locked date split (Pitfall #3)."""
    split = pd.Timestamp(config.SPLIT_DATE)
    train = df.index < split
    test = df.index >= split
    return train, test


if __name__ == "__main__":
    build_features()
