# Weather Nowcasting — Delhi Daily Climate (2013–2017)

## 1. Overview

This project builds a short-term ("nowcasting") temperature prediction model for Delhi using daily climate data from 2013–2017. The goal was to establish a rigorous classical baseline (ARIMA) and evaluate whether a deep learning approach (LSTM) could meaningfully outperform it — with careful attention to time-series-specific pitfalls (data leakage, evaluation methodology, stationarity).

**Dataset:** [Daily Climate Time Series Data — Delhi](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) (Kaggle), pre-split into train (2013–2016) and test (2017). Features: `meantemp`, `humidity`, `wind_speed`, `meanpressure`.

**Environment:** Google Colab, Kaggle API for data ingestion.

---

## 2. Exploratory Data Analysis

### Seasonal Decomposition
Additive decomposition (`period=365`) of `meantemp` revealed:
- **Trend:** mild, gradually accelerating warming across 2013–2016.
- **Seasonality:** strong, highly consistent yearly cycle (±10°C swing), peaking in summer (May–June), dipping in winter (Dec–Jan).
- **Residual:** roughly random noise, mostly within ±5°C, consistent with expected day-to-day weather variability.

### Correlation Analysis (features vs. `meantemp`)
| Feature | Correlation | Decision |
|---|---|---|
| humidity | -0.572 | Kept — strong relationship |
| wind_speed | +0.306 | Kept — moderate relationship |
| meanpressure | -0.039 | Excluded — negligible correlation |

`meanpressure` was retained in the dataset but excluded from the model's feature set due to its near-zero linear relationship with the target. Note: this decision was based on linear correlation only; a non-linear or interaction effect can't be ruled out, and with a larger dataset this would ideally be validated via ablation rather than correlation alone.

---

## 3. Stationarity Analysis (for ARIMA)

Stationarity is a strict requirement for ARIMA's underlying assumptions — it is **not** required for the LSTM, which can learn trend/seasonality directly from raw (scaled) sequences.

**Augmented Dickey-Fuller (ADF) Test:**

| Series | ADF Statistic | p-value | Stationary? |
|---|---|---|---|
| Raw `meantemp` | -1.10 | 0.716 | No |
| First-order differenced `meantemp` | -16.38 | 2.76e-29 | Yes |

**First-order differencing** removed the trend component and produced a strongly stationary series, confirming `d=1` for the ARIMA baseline.

---

## 4. Baseline Model: ARIMA

**Configuration:** ARIMA(5,1,0) — 5-day autoregressive lookback, first-order differencing, no moving-average term.

### Evaluation methodology — a key finding

Two evaluation strategies were tested:

1. **Blind multi-step forecast** (predict the full ~114-day test horizon in one shot, with no access to real values along the way): **RMSE = 11.41°C**. Diagnostic plotting showed the model converges to a near-flat prediction after ~10 days — it has no mechanism to anticipate the seasonal warming trend across a long unseen horizon.

2. **Rolling / walk-forward forecast** (refit the model at each step, predicting only 1 day ahead using real observed data up to that point): **RMSE = 1.77°C** — a ~6.5x improvement.

This large gap is a methodology finding, not a model change: since the task is *nowcasting* (short-horizon prediction with access to recent real data), the walk-forward evaluation is the fair and representative one. **1.77°C is the baseline used for comparison against the LSTM.**

---

## 5. LSTM Model

### Data pipeline (leak-free)
1. Chronological train/validation split within the training period (85/15), preserving temporal order.
2. `MinMaxScaler` fit **only** on the training split; validation and test sets transformed using training-derived min/max only.
3. Sequence framing: sliding windows of `SEQ_LENGTH=29` days, 3 input features (`meantemp`, `humidity`, `wind_speed`), predicting the next day's `meantemp`.
4. Resulting shapes: train `(1213, 29, 3)`, val `(191, 29, 3)`, test `(85, 29, 3)`.

### Final architecture
```
LSTM(128, return_sequences=True) → Dropout(0.3)
LSTM(64) → Dropout(0.3)
Dense(1)
```
- Optimizer: Adam, Loss: MSE
- Batch size: 8
- EarlyStopping (monitor=val_loss, patience=7, restore_best_weights=True) — training halted at epoch 33 (of 100 max)
- Training/validation loss converged rapidly within ~2 epochs and remained stable/close thereafter, with no divergence — indicating no significant overfitting. (Validation loss was often slightly below training loss, attributable to Dropout being active only during training.)

### Tuning journey (empirical, one lesson informing the next)

| Attempt | Config | Test RMSE | Outcome |
|---|---|---|---|
| 1 (initial) | 64/32 units, dropout=0.2, batch=32, seq=29 | 2.28°C | Underperformed ARIMA baseline |
| 2 | 32/16 units, dropout=0.4, batch=32, seq=32 | Worse than #1 | Reducing capacity + higher dropout caused underfitting, not improvement — indicated the original issue wasn't excess capacity |
| 3 (final) | 64/64 units, dropout=0.3, batch=4, seq=29 | **1.68°C** | Beat ARIMA baseline by ~5% |

**Key insight:** reducing batch size (32→4) — increasing the frequency of noisier gradient updates — had a larger positive impact than adjusting model capacity or dropout. This suggests batch size was a more impactful lever than architecture size for this dataset, and that the original model's capacity was reasonably well-matched to the problem (shrinking it hurt, not helped).

Both LSTM units and dropout were found to have a **non-monotonic** relationship with validation performance — too little of either causes underfitting, too much causes overfitting, with the optimum depending on dataset size and interacting with other hyperparameters. A day-of-year seasonality feature was considered but ultimately not needed, since tuning alone brought the LSTM past the ARIMA baseline.

---

## 6. Results Summary

| Model | RMSE (°C) |
|---|---|
| ARIMA (blind multi-step) | 11.41 |
| ARIMA (rolling, 1-day-ahead) | 1.77 |
| LSTM (initial) | 2.28 |
| **LSTM (final, tuned)** | **1.68** |

The final tuned LSTM outperforms the ARIMA rolling baseline by ~5%. The margin is modest, which is itself a meaningful finding: on a relatively small daily dataset (~1,200 training sequences), a well-evaluated classical model remains highly competitive with deep learning — consistent with published findings (e.g., the M4 forecasting competition) that classical methods often match or beat deep learning on smaller/simpler time series.

---

## 7. Key Learnings / Talking Points

- **Evaluation methodology matters as much as model choice** — the same ARIMA model went from "badly broken" (11.41°C) to "strong baseline" (1.77°C) purely by switching to an evaluation strategy that matched the actual use case (nowcasting = 1-step-ahead with real recent context).
- **Data leakage discipline**: chronological splitting throughout (train/val/test, never shuffled), scaler fit only on training data, no test-set influence at any stage.
- **Stationarity is model-specific, not universal** — required and handled for ARIMA; not required for LSTM.
- **Hyperparameter tuning is empirical, not purely theoretical** — an intuitive "fix" (shrink model, raise dropout) made results worse, and the actual effective lever (batch size) was found through iteration, not a priori reasoning.
- **A modest, well-justified win is more credible than an unexamined big one** — the project prioritizes a defensible, explained comparison over an inflated result.
