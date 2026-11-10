"""
robustness.py
=============
Does the result survive small changes, and how uncertain is it? (Checklist Day 6)

  * sensitivity_table : re-run the whole regime+backtest for different k,
    volatility windows and cost levels. If the verdict flips every time a knob
    moves a little, there is no real edge (Pitfall #4 / #12).
  * bootstrap : resample the TEST-period strategy returns with replacement
    1,000 times to get a distribution of Sharpe and max drawdown. A Sharpe whose
    bootstrap interval straddles the buy&hold Sharpe is not convincing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import evaluate, performance
from .features import build_features, train_test_masks
from .regimes import fit_kmeans, label_by_stats, rule_based_regime


def _regime_frame(feats: pd.DataFrame, k: int, vol_window: int) -> pd.DataFrame:
    """Rebuild features' volatility at a given window, refit k-means, relabel."""
    df = feats.copy()
    # recompute the clustering vol feature at the requested window
    df["vol_20"] = df["ret"].rolling(vol_window).std() * np.sqrt(config.TRADING_DAYS)
    df = df.dropna(subset=config.CLUSTER_FEATURES + ["ret"])
    _, labels = fit_kmeans(df, k)
    df["regime"] = label_by_stats(df, labels)
    return df


def sensitivity_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in config.SENSITIVITY_K:
        for vw in config.SENSITIVITY_VOL_WINDOW:
            df = _regime_frame(feats, k, vw)
            for cost in config.SENSITIVITY_COST:
                res = evaluate(df, cost=cost)
                t = res["test"]
                rows.append({
                    "k": k,
                    "vol_window": vw,
                    "cost": cost,
                    "test_sharpe": t["strategy"]["sharpe"],
                    "bh_sharpe": t["buy_hold"]["sharpe"],
                    "sharpe_minus_bh": t["strategy"]["sharpe"] - t["buy_hold"]["sharpe"],
                    "test_cagr": t["strategy"]["cagr"],
                    "max_dd": t["strategy"]["max_drawdown"],
                    "pos_changes": t["n_position_changes"],
                })
    return pd.DataFrame(rows)


def bootstrap_test(df: pd.DataFrame, n: int = config.BOOTSTRAP_N,
                   seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Bootstrap the test-period net returns (block-free, i.i.d. resample)."""
    res = evaluate(df)
    strat = res["_strat_frame"]
    _, test = train_test_masks(df)
    net = strat.loc[test, "net"].dropna().to_numpy()
    rng = np.random.default_rng(seed)

    sharpes, dds = np.empty(n), np.empty(n)
    for i in range(n):
        sample = rng.choice(net, size=len(net), replace=True)
        s = pd.Series(sample)
        sharpes[i] = s.mean() / s.std() * np.sqrt(config.TRADING_DAYS) if s.std() > 0 else np.nan
        eq = np.exp(s.cumsum())
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return pd.DataFrame({"sharpe": sharpes, "max_drawdown": dds})


def bootstrap_summary(boot: pd.DataFrame) -> dict:
    return {
        "sharpe_mean": float(boot["sharpe"].mean()),
        "sharpe_p05": float(boot["sharpe"].quantile(0.05)),
        "sharpe_p95": float(boot["sharpe"].quantile(0.95)),
        "dd_mean": float(boot["max_drawdown"].mean()),
        "dd_p05": float(boot["max_drawdown"].quantile(0.05)),
        "dd_p95": float(boot["max_drawdown"].quantile(0.95)),
    }
