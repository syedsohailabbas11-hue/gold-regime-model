"""
data_download.py
================
Download the REAL series into data/raw/ (Checklist Day 1).

  * Gold (XAUUSD spot) from Yahoo via yfinance.
  * Real yield / dollar / VIX from FRED as plain CSV (no API key needed).

Run this on a machine with internet access:

    python -m src.data_download

Each series is written as a two-column CSV (date, value) so the rest of the
pipeline never depends on a live connection. Nothing here computes features —
it only fetches and saves, which keeps downloads separate from analysis.

Note on licences (ReadMe / Data tabs): yfinance is an unofficial Yahoo client,
and FRED series have their own terms. Check them for your use and never commit
the downloaded files (they are git-ignored).
"""
from __future__ import annotations

import sys
import urllib.request

import pandas as pd

from . import config


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def download_fred(series_id: str, out_name: str) -> pd.DataFrame:
    """Fetch one FRED series as CSV and save a tidy (date, value) file."""
    url = FRED_CSV_URL.format(series_id=series_id)
    print(f"  FRED  {series_id:10s} -> {out_name}")
    # FRED serves a 2-column CSV: observation_date, <SERIES_ID>
    raw = pd.read_csv(url)
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"])
    # FRED marks missing observations with ".". Coerce to NaN, keep the gaps
    # visible (we count and handle them in build_dataset, not here).
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    path = config.DATA_RAW / out_name
    raw.to_csv(path, index=False)
    print(f"         {len(raw):>6,} rows  {raw['date'].min().date()} .. {raw['date'].max().date()}")
    return raw


def download_gold(ticker: str = config.GOLD_TICKER) -> pd.DataFrame:
    """Fetch daily gold close from Yahoo via yfinance and save (date, value)."""
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        sys.exit("yfinance is not installed. Run: pip install yfinance")

    print(f"  Yahoo {ticker:10s} -> gold.csv")
    df = yf.download(
        ticker,
        start=config.GOLD_START,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        sys.exit(
            f"No data returned for {ticker}. Check the ticker or your connection. "
            "Alternatives: 'GC=F' (futures) or a broker CSV export."
        )
    # yfinance may return a single- or multi-index column frame; take Close.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.rename("value").reset_index()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    path = config.DATA_RAW / "gold.csv"
    out.to_csv(path, index=False)
    print(f"         {len(out):>6,} rows  {out['date'].min().date()} .. {out['date'].max().date()}")
    return out


def main() -> None:
    print("Downloading raw data to", config.DATA_RAW)
    download_gold()
    for name, series_id in config.FRED_SERIES.items():
        download_fred(series_id, f"{name}.csv")
    print("Done. Next: python -m src.build_dataset")


if __name__ == "__main__":
    main()
