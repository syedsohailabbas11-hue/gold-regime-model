"""
run_all.py
==========
Run the whole project end to end and write every output:
  data -> features -> regimes -> backtest -> robustness -> figures -> report.

Usage
-----
    # 1a. Real data (needs internet):
    python -m src.data_download
    # 1b.  OR offline demo data (FAKE numbers, for testing the pipeline):
    python -m src.make_synthetic

    # 2. Everything else:
    python run_all.py              # uses whatever raw data is present
    python run_all.py --demo       # stamps figures/report as SYNTHETIC

The script prints a summary and writes reports/results_summary.md and
reports/results_log.csv (the Results-tab style log).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config, plots
from src.build_dataset import build
from src.features import build_features, train_test_masks
from src.regimes import build_regimes, choose_k_by_bic
from src.backtest import (evaluate, forward_return_table, regime_stats_table,
                          run_strategy)
from src.robustness import (bootstrap_summary, bootstrap_test,
                            sensitivity_table)


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.2%}" if pct else f"{x:+.2f}"


def main(demo: bool = False) -> None:
    is_synth = (config.DATA_RAW / "SYNTHETIC.txt").exists()
    if demo or is_synth:
        plots.DEMO_BANNER = "SYNTHETIC DEMO"
        banner = ("\n" + "#" * 68 +
                  "\n#  RUNNING ON SYNTHETIC DATA — all numbers below are FAKE.\n"
                  "#  Replace with real data before drawing any conclusion.\n" +
                  "#" * 68 + "\n")
        print(banner)

    # --- Day 1: merge ------------------------------------------------------- #
    print("\n[1/6] Building dataset ...")
    merged = build()
    plots.plot_price(merged)

    # --- Day 2: features ---------------------------------------------------- #
    print("\n[2/6] Engineering features ...")
    feats = build_features(merged)
    plots.plot_exploration(feats)

    # --- model-selection diagnostic (train only) --------------------------- #
    bic = choose_k_by_bic(feats)
    print("\nk selection (TRAIN only):")
    print(bic.to_string())

    # --- Day 3: regimes ----------------------------------------------------- #
    print("\n[3/6] Finding regimes (k-means, train-only fit) ...")
    _, df = build_regimes(feats, k=config.K_REGIMES)
    plots.plot_regimes(df)
    print(df["regime"].value_counts().to_string())

    # --- Day 4: regime statistics ------------------------------------------ #
    print("\n[4/6] Regime statistics ...")
    stats = regime_stats_table(df)
    fwd = forward_return_table(df)
    print(stats.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- Day 5-6: backtest + headline -------------------------------------- #
    print("\n[5/6] Backtesting (net of costs) ...")
    res = evaluate(df)
    strat = res["_strat_frame"]
    plots.plot_equity(strat, "test")
    plots.plot_equity(strat, "train")

    # --- Day 6: robustness -------------------------------------------------- #
    print("\n[6/6] Robustness: sensitivity + bootstrap ...")
    sens = sensitivity_table(feats)
    boot = bootstrap_test(df)
    bsum = bootstrap_summary(boot)
    plots.plot_bootstrap(boot, res["test"]["buy_hold"]["sharpe"])

    # --- verdict logic ------------------------------------------------------ #
    t_s = res["test"]["strategy"]["sharpe"]
    t_bh = res["test"]["buy_hold"]["sharpe"]
    beats = (sens["sharpe_minus_bh"] > 0).mean()
    if bsum["sharpe_p05"] > t_bh:
        verdict = "Edge (test Sharpe beats buy & hold across the bootstrap 5th pct)"
    elif t_s > t_bh and beats > 0.5:
        verdict = "Inconclusive (edge in the base case, not robust enough to lean on)"
    else:
        verdict = "No edge after costs (buy & hold is as good or better)"

    # --- write results log (Results-tab style) ----------------------------- #
    log_row = {
        "Run": 1,
        "Description": f"{config.K_REGIMES} regimes, k-means, long when train mean>0",
        "Regimes_k": config.K_REGIMES,
        "Features": ",".join(config.CLUSTER_FEATURES),
        "Train": f"{feats.index.min().date()}..{pd.Timestamp(config.SPLIT_DATE).date()}",
        "Test": f"{pd.Timestamp(config.SPLIT_DATE).date()}..{feats.index.max().date()}",
        "Cost": config.COST_PER_CHANGE,
        "CAGR": t_s if False else res["test"]["strategy"]["cagr"],
        "Sharpe": t_s,
        "MaxDD": res["test"]["strategy"]["max_drawdown"],
        "TimeInMkt": res["test"]["strategy"]["time_in_market"],
        "Sharpe_vs_BH": t_s - t_bh,
        "Verdict": verdict.split(" (")[0],
    }
    bh_row = {
        "Run": "Buy & hold", "Description": "hold gold over TEST period",
        "Regimes_k": 1, "CAGR": res["test"]["buy_hold"]["cagr"],
        "Sharpe": t_bh, "MaxDD": res["test"]["buy_hold"]["max_drawdown"],
        "TimeInMkt": 1.0, "Verdict": "benchmark",
    }
    pd.DataFrame([bh_row, log_row]).to_csv(config.RESULTS_CSV, index=False)

    # --- write markdown summary -------------------------------------------- #
    _write_summary(res, stats, fwd, sens, bsum, verdict, demo or is_synth)

    # --- console headline --------------------------------------------------- #
    print("\n" + "=" * 68)
    print("HEADLINE (TEST period, net of costs)")
    print("=" * 68)
    print(f"  Long regimes            : {res['long_regimes']}")
    print(f"  Strategy  Sharpe {_fmt(t_s)}   CAGR {_fmt(res['test']['strategy']['cagr'], True)}"
          f"   maxDD {_fmt(res['test']['strategy']['max_drawdown'], True)}"
          f"   in-mkt {res['test']['strategy']['time_in_market']:.0%}")
    print(f"  Buy&hold  Sharpe {_fmt(t_bh)}   CAGR {_fmt(res['test']['buy_hold']['cagr'], True)}"
          f"   maxDD {_fmt(res['test']['buy_hold']['max_drawdown'], True)}")
    print(f"  Position changes (test) : {res['test']['n_position_changes']}")
    print(f"  Bootstrap Sharpe 90% CI : [{bsum['sharpe_p05']:+.2f}, {bsum['sharpe_p95']:+.2f}]")
    print(f"  VERDICT                 : {verdict}")
    if demo or is_synth:
        print("\n  (SYNTHETIC — not a real finding.)")
    print("=" * 68)
    print(f"\nWrote {config.RESULTS_MD}\nWrote {config.RESULTS_CSV}\nFigures in {config.FIGURES}")


def _write_summary(res, stats, fwd, sens, bsum, verdict, is_demo):
    t = res["test"]; tr = res["train"]
    lines = []
    if is_demo:
        lines += ["> ⚠️ **SYNTHETIC DEMO RUN — every number here is fake.** "
                  "Re-run on real data (`python -m src.data_download`) before "
                  "citing any result.\n"]
    lines += [
        "# Results summary\n",
        f"- **Verdict:** {verdict}",
        f"- **Long regimes (chosen on train):** {', '.join(res['long_regimes']) or 'none'}",
        f"- **Cost per position change:** {config.COST_PER_CHANGE:.2%}",
        "",
        "## Headline — TEST period, net of costs",
        "",
        "| Metric | Regime strategy | Buy & hold |",
        "|---|---|---|",
        f"| CAGR | {t['strategy']['cagr']:+.2%} | {t['buy_hold']['cagr']:+.2%} |",
        f"| Sharpe | {t['strategy']['sharpe']:+.2f} | {t['buy_hold']['sharpe']:+.2f} |",
        f"| Max drawdown | {t['strategy']['max_drawdown']:+.2%} | {t['buy_hold']['max_drawdown']:+.2%} |",
        f"| Time in market | {t['strategy']['time_in_market']:.0%} | 100% |",
        f"| Position changes | {t['n_position_changes']} | 1 |",
        "",
        "## Train period (for context only — not evidence)",
        "",
        f"- Strategy Sharpe {tr['strategy']['sharpe']:+.2f} vs buy&hold {tr['buy_hold']['sharpe']:+.2f}",
        "",
        "## Bootstrap (1,000 resamples of test net returns)",
        "",
        f"- Sharpe: mean {bsum['sharpe_mean']:+.2f}, 90% interval "
        f"[{bsum['sharpe_p05']:+.2f}, {bsum['sharpe_p95']:+.2f}]",
        f"- Max drawdown: mean {bsum['dd_mean']:+.2%}, 90% interval "
        f"[{bsum['dd_p05']:+.2%}, {bsum['dd_p95']:+.2%}]",
        "",
        "## Sensitivity (fraction of settings where strategy Sharpe > buy&hold)",
        "",
        f"- {(sens['sharpe_minus_bh'] > 0).mean():.0%} of "
        f"{len(sens)} (k × vol-window × cost) combinations beat buy & hold on test Sharpe.",
        "",
        "## Regime statistics (whole sample)",
        "",
        stats.to_markdown(floatfmt=".4f"),
        "",
        "## Forward-return table (mean gold return per regime)",
        "",
        fwd.to_markdown(floatfmt=".4f"),
        "",
    ]
    config.RESULTS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="stamp outputs as SYNTHETIC even if real data is present")
    args = ap.parse_args()
    try:
        main(demo=args.demo)
    except FileNotFoundError as e:
        print("\nERROR:", e)
        sys.exit(1)
