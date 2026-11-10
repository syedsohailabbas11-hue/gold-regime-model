"""
backtest.py
===========
Turn regimes into a testable long/flat rule and measure it honestly
(Checklist Days 4-6).

Signal rule (Strategy v1, ReadMe tab):
    Go LONG gold when today's regime is one whose *training-period* mean daily
    return is positive; otherwise stay FLAT. No shorting, no leverage.

Timing (Pitfall #1):
    The regime is known at the close of day t. The position is held on day t+1.
    So realised strategy return on a day uses the position decided the day
    before:  strat_ret_t = position_held_t * gold_ret_t, with
    position_held = target_position.shift(1).

Costs (Pitfall #5):
    A cost of COST_PER_CHANGE (fraction of notional) is charged every time the
    held position changes (flat->long or long->flat).

The headline you would defend in an interview is the TEST-period result only
(Checklist Day 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import train_test_masks


# --------------------------------------------------------------------------- #
# Per-regime statistics (Checklist Day 4)
# --------------------------------------------------------------------------- #
def forward_return_table(df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """Average forward gold return per regime, for TRAIN and TEST separately."""
    train, test = train_test_masks(df)
    out = {}
    for horizon in (1, 5, 20):
        fwd = df["ret"].rolling(horizon).sum().shift(-horizon)  # forward sum
        for name, mask in (("train", train), ("test", test)):
            g = fwd[mask].groupby(df.loc[mask, regime_col]).mean()
            out[f"{name}_fwd{horizon}"] = g
    return pd.DataFrame(out)


def regime_stats_table(df: pd.DataFrame, regime_col: str = "regime") -> pd.DataFrame:
    """Days, mean return, volatility, worst drawdown, number of episodes."""
    def episodes(s: pd.Series) -> int:
        # count contiguous blocks of each label change -> number of switches+1
        return int((s != s.shift()).sum())

    rows = []
    for name, grp in df.groupby(regime_col):
        ret = grp["ret"]
        eq = np.exp(ret.cumsum())
        dd = (eq / eq.cummax() - 1).min()
        rows.append({
            "regime": name,
            "days": len(grp),
            "share": len(grp) / len(df),
            "mean_daily_ret": ret.mean(),
            "ann_vol": ret.std() * np.sqrt(config.TRADING_DAYS),
            "worst_drawdown": dd,
        })
    stats = pd.DataFrame(rows).set_index("regime")
    # number of separate episodes across the whole sample (Pitfall #10)
    full = df[regime_col]
    switch = (full != full.shift())
    ep = df.assign(_sw=switch).groupby(regime_col)["_sw"].sum()
    stats["episodes"] = ep
    return stats.sort_values("ann_vol")


# --------------------------------------------------------------------------- #
# Signal + strategy returns
# --------------------------------------------------------------------------- #
def long_regimes_from_train(df: pd.DataFrame, regime_col: str = "regime") -> list:
    """Regimes whose TRAIN mean daily return is positive (Pitfall #4: train only)."""
    train, _ = train_test_masks(df)
    mean_ret = df.loc[train].groupby(regime_col)["ret"].mean()
    return list(mean_ret[mean_ret > 0].index)


def run_strategy(df: pd.DataFrame, regime_col: str = "regime",
                 cost: float | None = None) -> pd.DataFrame:
    """Build positions, apply costs, return a frame with daily series."""
    cost = config.COST_PER_CHANGE if cost is None else cost
    long_set = long_regimes_from_train(df, regime_col)

    target = df[regime_col].isin(long_set).astype(int)     # position decided at t
    held = target.shift(1).fillna(0)                        # held on t (Pitfall #1)
    gross = held * df["ret"]
    turnover = held.diff().abs().fillna(held.abs())        # 1 on each switch
    costs = cost * turnover
    net = gross - costs

    out = pd.DataFrame({
        "ret": df["ret"],            # gold (buy & hold)
        "position": held,
        "gross": gross,
        "cost": costs,
        "net": net,
        "turnover": turnover,
    }, index=df.index)
    out.attrs["long_regimes"] = long_set
    return out


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def performance(returns: pd.Series, position: pd.Series | None = None) -> dict:
    """CAGR, Sharpe, max drawdown, time in market, from a daily (log) return
    series. Returns are treated as log returns; equity = exp(cumsum)."""
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ("cagr", "sharpe", "max_drawdown", "time_in_market", "n_days")}
    eq = np.exp(r.cumsum())
    years = len(r) / config.TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = (r.mean() - config.RISK_FREE / config.TRADING_DAYS) / r.std() \
        * np.sqrt(config.TRADING_DAYS) if r.std() > 0 else np.nan
    max_dd = (eq / eq.cummax() - 1).min()
    tim = float(position.reindex(r.index).mean()) if position is not None else 1.0
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "time_in_market": tim,
        "n_days": int(len(r)),
    }


def evaluate(df: pd.DataFrame, regime_col: str = "regime",
             cost: float | None = None) -> dict:
    """Full evaluation: strategy vs buy&hold on TRAIN and TEST, plus counts."""
    strat = run_strategy(df, regime_col, cost)
    train, test = train_test_masks(df)

    result = {"long_regimes": strat.attrs["long_regimes"]}
    for name, mask in (("train", train), ("test", test)):
        s = strat.loc[mask]
        result[name] = {
            "strategy": performance(s["net"], s["position"]),
            "buy_hold": performance(s["ret"]),
            "n_position_changes": int(s["turnover"].sum()),
            "total_cost_drag": float(s["cost"].sum()),
        }
    result["_strat_frame"] = strat
    return result


if __name__ == "__main__":
    import pandas as pd
    feats = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"]).set_index("date")
    labels = pd.read_csv(config.LABELS_CSV, parse_dates=["date"]).set_index("date")
    df = feats.join(labels[["regime", "regime_id", "rule_regime"]])
    res = evaluate(df)
    print("Long regimes:", res["long_regimes"])
    print("TEST strategy:", res["test"]["strategy"])
    print("TEST buy&hold:", res["test"]["buy_hold"])
