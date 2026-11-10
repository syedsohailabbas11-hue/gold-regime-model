# Gold (XAUUSD) macro-regime model

**Does gold behave differently across macro regimes (volatility, real yields, the
dollar, fear) — and can a simple rule built on those regimes beat buy-and-hold
after costs?**

This is a learning / portfolio project on historical daily data. It is **not a
trading system and not financial advice.** The honest expectation, stated before
any analysis, is a small or no edge after costs; a result that looks amazing
almost always means a bug or look-ahead bias.

---

## TL;DR (results)

> The numbers in `reports/results_summary.md` and the figures in
> `reports/figures/` currently come from a **synthetic demo run** (fake data,
> used to prove the pipeline works). Run the two commands below on real data,
> then paste your real headline numbers here.

| Metric (TEST period, net of costs) | Regime strategy | Buy & hold |
|---|---|---|
| CAGR | _run it_ | _run it_ |
| Sharpe | _run it_ | _run it_ |
| Max drawdown | _run it_ | _run it_ |
| Time in market | _run it_ | 100% |

**Verdict:** _Edge / No edge / Inconclusive_ — filled in automatically by `run_all.py`.

---

## How to run it

```bash
# 1. Environment
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 2a. Real data (needs internet)
python -m src.data_download

# 2b. OR offline demo data (FAKE numbers — just to see the pipeline run)
python -m src.make_synthetic

# 3. Run the whole analysis: data -> features -> regimes -> backtest -> report
python run_all.py
```

Outputs:
- `data/processed/merged.csv`, `features.csv`, `regime_labels.csv`
- `reports/figures/*.png` — price, volatility, correlations, **regime chart**, equity curves, bootstrap
- `reports/results_summary.md` and `reports/results_log.csv`

## Data

| Series | Measures | Source | ID |
|---|---|---|---|
| Gold XAUUSD (spot) | USD / troy oz | Yahoo via `yfinance` | `XAUUSD=X` |
| 10y real yield | TIPS real rate | FRED | `DFII10` |
| Broad USD index | trade-weighted USD | FRED | `DTWEXBGS` |
| VIX | equity-vol / fear | FRED | `VIXCLS` |

Window ≈ 2006–present (the broad-dollar index starts around 2006). Data files are
**not committed** (`.gitignore`); download them yourself. `yfinance` is an
unofficial Yahoo client and FRED series have their own terms — check them for
your use.

## Method

1. **Merge & clean** on the gold trading calendar; **every macro series is
   lagged one day** so a value is only used after it could have been known.
2. **Features** (all backward-looking): log returns, rolling annualised
   volatility (20/60d), 5/20d real-yield & dollar changes, 20d momentum, VIX
   level, rolling 60d correlations.
3. **Regimes**: a transparent rule-based baseline (vol × real-yield direction),
   plus **k-means / Gaussian mixture fit on the training period only** (scaler
   too). Clusters are named by their statistics (`calm`, `normal`, `stressed`),
   not by number, so labels stay stable.
4. **Backtest (Strategy v1)**: long gold when the regime's **training-period**
   mean return is positive, else flat. Signal is **shifted one day**; a cost is
   charged on **every position change**. No shorting, no leverage.
5. **Honesty checks**: headline is the **test period only**; plus sensitivity
   (k, vol window, cost) and a 1,000× bootstrap of test returns.

## Train / test split

Train ≈ 2006 → 2018, test 2019 → latest. **Decided before looking at results and
not changed.** See `src/config.py` (`SPLIT_DATE`).

## Guardrails against fooling yourself

Each is enforced in code and listed in the workbook's *Pitfalls* tab: look-ahead
bias (1-day lag + `shift(1)`), fitting regimes on all data (train-only scaler &
model), random splits (split by date), tuning on the test set (choices locked on
train), ignoring costs (charged per switch), too few regime episodes (counted),
unstable labels (named by stats), and overclaiming (limits section below).

## Limitations

One asset, one historical period, no live trading, costs are assumed not
measured, and regimes are a simplification of a messy world. Treat any apparent
edge with suspicion until it survives the robustness checks.

## Repository layout

```
gold-regime-model/
├── run_all.py            # one command runs the whole project
├── requirements.txt
├── src/
│   ├── config.py         # all locked decisions (split, costs, series ids)
│   ├── data_download.py  # real data (yfinance + FRED)
│   ├── make_synthetic.py # offline FAKE data for testing
│   ├── build_dataset.py  # merge + 1-day macro lag + quality note
│   ├── features.py       # feature engineering
│   ├── regimes.py        # rule baseline + k-means/GMM (train-only) + naming
│   ├── backtest.py       # signal, costs, CAGR/Sharpe/drawdown
│   ├── robustness.py     # sensitivity + bootstrap
│   └── plots.py          # all figures
└── reports/              # generated summary, log, figures
```

## Not financial advice

Historical backtest for learning only. It does not predict the future and must
not be traded live.
