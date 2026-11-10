"""
regimes.py
==========
Find macro regimes (Checklist Day 3).

Three things live here:
  1. rule_based_regime  — a transparent benchmark (high/low vol x rising/falling
     real yield). This is the simple yardstick the model must beat.
  2. fit_kmeans / fit_gmm — unsupervised regimes, fit on the TRAINING PERIOD
     ONLY (Pitfall #2), then applied to the whole sample.
  3. label_by_stats — rename numeric clusters by what they look like (e.g.
     "calm", "stressed"), so labels are stable and meaningful (Pitfall #11).

Scaling (StandardScaler) is also fit on train only (Pitfall #2), because k-means
is sensitive to feature scale.

    python -m src.regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config
from .features import train_test_masks


# --------------------------------------------------------------------------- #
# 1. Rule-based baseline
# --------------------------------------------------------------------------- #
def rule_based_regime(df: pd.DataFrame) -> pd.Series:
    """High/low volatility crossed with rising/falling real yield.

    Thresholds use the TRAIN median only, so the rule does not peek at the
    test period. Returns a categorical label per row.
    """
    train, _ = train_test_masks(df)
    vol_thr = df.loc[train, "vol_20"].median()
    # "rising" real yield = positive 20d change
    hi_vol = df["vol_20"] > vol_thr
    rising = df["ry_chg_20"] > 0
    label = np.where(
        hi_vol & rising, "HiVol-RisingRY",
        np.where(hi_vol & ~rising, "HiVol-FallingRY",
        np.where(~hi_vol & rising, "LoVol-RisingRY", "LoVol-FallingRY")))
    return pd.Series(label, index=df.index, name="rule_regime")


# --------------------------------------------------------------------------- #
# 2. Unsupervised regimes (fit on train only)
# --------------------------------------------------------------------------- #
def _scale_train_only(df: pd.DataFrame, cols: list[str]):
    train, _ = train_test_masks(df)
    scaler = StandardScaler().fit(df.loc[train, cols])
    X = pd.DataFrame(scaler.transform(df[cols]), index=df.index, columns=cols)
    return scaler, X


def fit_kmeans(df: pd.DataFrame, k: int, cols: list[str] | None = None):
    """Fit k-means on the training rows, predict labels for every row."""
    cols = cols or config.CLUSTER_FEATURES
    train, _ = train_test_masks(df)
    _, X = _scale_train_only(df, cols)
    model = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE)
    model.fit(X.loc[train])
    labels = pd.Series(model.predict(X), index=df.index, name=f"kmeans_{k}")
    return model, labels


def fit_gmm(df: pd.DataFrame, k: int, cols: list[str] | None = None):
    """Gaussian mixture alternative, same train-only discipline."""
    cols = cols or config.CLUSTER_FEATURES
    train, _ = train_test_masks(df)
    _, X = _scale_train_only(df, cols)
    model = GaussianMixture(
        n_components=k, covariance_type="full", random_state=config.RANDOM_STATE
    )
    model.fit(X.loc[train])
    labels = pd.Series(model.predict(X), index=df.index, name=f"gmm_{k}")
    return model, labels


def choose_k_by_bic(df: pd.DataFrame, k_grid=config.K_GRID, cols=None) -> pd.DataFrame:
    """Compare k on TRAIN only, via GMM BIC and k-means inertia (Pitfall #4)."""
    cols = cols or config.CLUSTER_FEATURES
    train, _ = train_test_masks(df)
    _, X = _scale_train_only(df, cols)
    rows = []
    for k in k_grid:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=config.RANDOM_STATE).fit(X.loc[train])
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=config.RANDOM_STATE).fit(X.loc[train])
        rows.append({"k": k, "gmm_bic": gm.bic(X.loc[train]),
                     "kmeans_inertia": km.inertia_})
    return pd.DataFrame(rows).set_index("k")


# --------------------------------------------------------------------------- #
# 3. Label clusters by their statistics (stable, meaningful names)
# --------------------------------------------------------------------------- #
def label_by_stats(df: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Rename numeric clusters by average volatility: lowest vol -> 'calm',
    highest vol -> 'stressed', middle ones 'normal'/'elevated'.

    Naming by statistics (not by cluster number) keeps labels consistent across
    re-fits (Pitfall #11).
    """
    vol_by_cluster = df.assign(_c=labels).groupby("_c")["vol_20"].mean().sort_values()
    order = list(vol_by_cluster.index)
    k = len(order)
    if k == 2:
        names = ["calm", "stressed"]
    elif k == 3:
        names = ["calm", "normal", "stressed"]
    elif k == 4:
        names = ["calm", "normal", "elevated", "stressed"]
    else:
        names = [f"r{i}" for i in range(k)]
    mapping = {cluster: names[i] for i, cluster in enumerate(order)}
    named = labels.map(mapping).rename(labels.name + "_named")
    return named


def build_regimes(df: pd.DataFrame, k: int = config.K_REGIMES):
    """Convenience: fit the headline k-means model and attach named labels."""
    model, labels = fit_kmeans(df, k)
    named = label_by_stats(df, labels)
    out = df.copy()
    out["regime_id"] = labels
    out["regime"] = named
    out["rule_regime"] = rule_based_regime(df)
    out[["regime_id", "regime", "rule_regime"]].to_csv(config.LABELS_CSV)
    return model, out


if __name__ == "__main__":
    feats = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"]).set_index("date")
    _, out = build_regimes(feats)
    print(out["regime"].value_counts())
    print(f"Saved {config.LABELS_CSV}")
