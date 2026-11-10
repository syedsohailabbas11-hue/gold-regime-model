"""
plots.py
========
Every figure the project needs, saved to reports/figures/ (Checklist Days 1-6).
Charts are the main evidence, so they are kept clean and self-explanatory.

matplotlib only (no seaborn dependency). Colours are chosen to be distinct and
colour-blind friendly.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                     # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .features import train_test_masks

# A small, consistent palette keyed by regime name.
REGIME_COLORS = {
    "calm":     "#4C78A8",
    "normal":   "#72B7B2",
    "elevated": "#F58518",
    "stressed": "#E45756",
}
DEMO_BANNER = None   # set to a string to stamp "SYNTHETIC DEMO" on every figure


def _save(fig, name: str) -> str:
    if DEMO_BANNER:
        fig.text(0.5, 0.5, DEMO_BANNER, fontsize=40, color="grey",
                 alpha=0.12, ha="center", va="center", rotation=30,
                 weight="bold")
    path = config.FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path.name}")
    return str(path)


def plot_price(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["gold"], color="#B8860B", lw=1)
    ax.set_title("Gold price (XAUUSD)")
    ax.set_ylabel("USD / oz")
    ax.axvline(pd.Timestamp(config.SPLIT_DATE), color="k", ls="--", lw=0.8)
    ax.text(pd.Timestamp(config.SPLIT_DATE), ax.get_ylim()[1],
            "  train | test", va="top", fontsize=8)
    return _save(fig, "01_gold_price.png")


def plot_exploration(df: pd.DataFrame) -> list[str]:
    paths = []
    # volatility
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for w in config.VOL_WINDOWS:
        ax.plot(df.index, df[f"vol_{w}"], lw=0.8, label=f"{w}-day vol")
    ax.set_title("Rolling annualised volatility of gold")
    ax.legend(); ax.set_ylabel("ann. vol")
    paths.append(_save(fig, "02_volatility.png"))

    # rolling correlations
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(df.index, df["corr_ry_60"], lw=0.8, label="corr(gold, real-yield chg)")
    ax.plot(df.index, df["corr_usd_60"], lw=0.8, label="corr(gold, dollar ret)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Rolling 60-day correlations"); ax.legend()
    paths.append(_save(fig, "03_correlations.png"))

    # feature histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["vol_20"].dropna(), bins=50, color="#4C78A8")
    ax.set_title("Distribution of 20-day volatility")
    paths.append(_save(fig, "04_feature_hist.png"))
    return paths


def plot_regimes(df: pd.DataFrame, regime_col: str = "regime") -> str:
    """Gold price with background shaded by regime (the project's main figure)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["gold"], color="black", lw=0.9, zorder=3)

    regimes = list(df[regime_col].unique())
    # shade contiguous blocks
    block = (df[regime_col] != df[regime_col].shift()).cumsum()
    for _, seg in df.groupby(block):
        name = seg[regime_col].iloc[0]
        ax.axvspan(seg.index[0], seg.index[-1],
                   color=REGIME_COLORS.get(name, "#CCCCCC"), alpha=0.35, lw=0)
    # legend proxies
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS.get(r, "#CCCCCC"),
               alpha=0.5) for r in regimes]
    ax.legend(handles, regimes, loc="upper left", ncol=len(regimes), fontsize=8)
    ax.axvline(pd.Timestamp(config.SPLIT_DATE), color="k", ls="--", lw=0.8)
    ax.set_title("Gold price shaded by macro regime")
    ax.set_ylabel("USD / oz")
    return _save(fig, "05_regimes.png")


def plot_equity(strat: pd.DataFrame, period: str = "test") -> str:
    """Equity curve: strategy (net of costs) vs buy & hold, over one period."""
    train, test = train_test_masks(strat)
    mask = test if period == "test" else train
    s = strat.loc[mask]
    eq_strat = np.exp(s["net"].cumsum())
    eq_bh = np.exp(s["ret"].cumsum())
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, eq_bh, label="Buy & hold gold", color="#B8860B", lw=1.2)
    ax.plot(s.index, eq_strat, label="Regime strategy (net of costs)",
            color="#4C78A8", lw=1.2)
    ax.set_title(f"Equity curve — {period} period (start = 1.0)")
    ax.set_ylabel("growth of 1 unit"); ax.legend()
    return _save(fig, f"06_equity_{period}.png")


def plot_bootstrap(boot: pd.DataFrame, bh_sharpe: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(boot["sharpe"].dropna(), bins=40, color="#4C78A8")
    axes[0].axvline(bh_sharpe, color="#B8860B", lw=1.5, label="buy&hold Sharpe")
    axes[0].set_title("Bootstrap: test Sharpe"); axes[0].legend(fontsize=8)
    axes[1].hist(boot["max_drawdown"].dropna(), bins=40, color="#E45756")
    axes[1].set_title("Bootstrap: test max drawdown")
    return _save(fig, "07_bootstrap.png")
