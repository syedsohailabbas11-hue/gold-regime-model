"""
build_dataset.py
================
Merge the four raw series into one clean daily table (Checklist Day 1, last step).

Key decisions, each tied to a Pitfall:
  * Align on date using the GOLD calendar as the spine (gold is the asset we
    trade). (Pitfall #7 — date misalignment)
  * Lag every macro series by MACRO_LAG_DAYS before merging, so a macro value is
    only used on the day *after* it could have been known. (Pitfall #1)
  * Forward-fill macro values over SHORT gaps only, and count how many values
    were filled. (Pitfall #8 — forward-filling too much)

Output: data/processed/merged.csv  with columns
    date, gold, real_yield, dollar, vix
and a printed data-quality note.

    python -m src.build_dataset
"""
from __future__ import annotations

import pandas as pd

from . import config

MAX_FFILL = 5   # forward-fill a macro gap of at most this many trading days


def _load_raw(name: str) -> pd.Series:
    path = config.DATA_RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_download` (real data) "
            "or `python -m src.make_synthetic` (offline demo) first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["value"].sort_index()
    # Collapse any accidental duplicate dates (keep last observation).
    s = s[~s.index.duplicated(keep="last")]
    return s


def build() -> pd.DataFrame:
    gold = _load_raw("gold").rename("gold")
    real_yield = _load_raw("real_yield").rename("real_yield")
    dollar = _load_raw("dollar").rename("dollar")
    vix = _load_raw("vix").rename("vix")

    # Spine = gold trading days. Reindex each macro series onto it.
    spine = gold.index

    def align(macro: pd.Series) -> pd.Series:
        # Lag first (shift by business positions on the macro's own index would
        # be ambiguous across calendars; we lag on the aligned daily spine).
        aligned = macro.reindex(spine.union(macro.index)).sort_index()
        aligned = aligned.ffill(limit=MAX_FFILL)          # short gaps only
        aligned = aligned.reindex(spine)                   # back onto gold days
        aligned = aligned.shift(config.MACRO_LAG_DAYS)     # Pitfall #1: use t-1
        return aligned

    merged = pd.DataFrame({"gold": gold})
    merged["real_yield"] = align(real_yield)
    merged["dollar"] = align(dollar)
    merged["vix"] = align(vix)

    # ---- data-quality note (Checklist Day 1) ------------------------------- #
    n_before = len(merged)
    miss = merged.isna().sum()
    # Drop the warm-up rows at the start where lagged macro is still NaN.
    merged = merged.dropna()
    merged.index.name = "date"

    print("Data-quality note")
    print("-----------------")
    print(f"  Gold trading days spanned : {n_before:,}")
    print(f"  Missing per column (pre-drop):")
    for col, m in miss.items():
        print(f"      {col:12s} {m:>5,}")
    print(f"  Rows after dropping warm-up/NaN : {len(merged):,}")
    print(f"  Date range : {merged.index.min().date()} .. {merged.index.max().date()}")
    # Sanity: no non-positive gold prices (Pitfall — zero/negative prices)
    bad = (merged["gold"] <= 0).sum()
    print(f"  Non-positive gold prices : {bad}")

    merged.reset_index().to_csv(config.MERGED_CSV, index=False)
    print(f"\nSaved {config.MERGED_CSV}")
    return merged


if __name__ == "__main__":
    build()
