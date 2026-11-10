# LinkedIn post — draft (fill the result before posting)

> Replace the bracketed verdict with your REAL result after running on real data.
> No profit claims. Honesty is the point.

---

Can macro regimes time gold? I built a small project to find out — and to
practise doing a backtest *honestly*.

The question: does gold (XAUUSD) behave differently across macro regimes —
volatility, 10y real yields, the dollar, and the VIX — and can a simple
long/flat rule built on those regimes beat buy-and-hold **after costs**?

How I kept myself honest:
• Split by date (train ≈2006–2018, test 2019→now) and locked it before looking.
• Fit the scaler + clustering on the training period only.
• Lagged every macro series a day and shifted the signal — no look-ahead.
• Charged a cost on every position change.
• Judged it on the test period only, then stress-tested with a 1,000× bootstrap.

The result: [No edge after costs — buy-and-hold was as good or better / a small
edge that didn't survive the robustness checks]. Not the exciting answer, but the
correct one — and spotting that is the actual skill.

Stack: Python, pandas, scikit-learn (k-means / Gaussian mixture), matplotlib.
Code + write-up: [repo link]

#DataScience #Python #Quant #Backtesting #Gold
