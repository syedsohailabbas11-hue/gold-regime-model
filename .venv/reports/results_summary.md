> ⚠️ **SYNTHETIC DEMO RUN — every number here is fake.** Re-run on real data (`python -m src.data_download`) before citing any result.

# Results summary

- **Verdict:** No edge after costs (buy & hold is as good or better)
- **Long regimes (chosen on train):** normal
- **Cost per position change:** 0.05%

## Headline — TEST period, net of costs

| Metric | Regime strategy | Buy & hold |
|---|---|---|
| CAGR | +1.45% | +6.07% |
| Sharpe | +0.18 | +0.47 |
| Max drawdown | -15.60% | -20.64% |
| Time in market | 42% | 100% |
| Position changes | 209 | 1 |

## Train period (for context only — not evidence)

- Strategy Sharpe +0.23 vs buy&hold -0.05

## Bootstrap (1,000 resamples of test net returns)

- Sharpe: mean +0.18, 90% interval [-0.41, +0.78]
- Max drawdown: mean -20.01%, 90% interval [-33.67%, -11.01%]

## Sensitivity (fraction of settings where strategy Sharpe > buy&hold)

- 4% of 27 (k × vol-window × cost) combinations beat buy & hold on test Sharpe.

## Regime statistics (whole sample)

| regime   |      days |   share |   mean_daily_ret |   ann_vol |   worst_drawdown |   episodes |
|:---------|----------:|--------:|-----------------:|----------:|-----------------:|-----------:|
| calm     | 1775.0000 |  0.3274 |           0.0001 |    0.1088 |          -0.2247 |   222.0000 |
| normal   | 2193.0000 |  0.4045 |           0.0002 |    0.1174 |          -0.2160 |   236.0000 |
| stressed | 1453.0000 |  0.2680 |          -0.0002 |    0.1662 |          -0.3731 |   133.0000 |

## Forward-return table (mean gold return per regime)

| regime   |   train_fwd1 |   test_fwd1 |   train_fwd5 |   test_fwd5 |   train_fwd20 |   test_fwd20 |
|:---------|-------------:|------------:|-------------:|------------:|--------------:|-------------:|
| calm     |      -0.0000 |      0.0003 |       0.0004 |      0.0003 |        0.0003 |       0.0003 |
| normal   |       0.0003 |      0.0003 |       0.0015 |      0.0021 |        0.0053 |       0.0063 |
| stressed |      -0.0004 |      0.0001 |      -0.0031 |      0.0007 |       -0.0098 |       0.0087 |
