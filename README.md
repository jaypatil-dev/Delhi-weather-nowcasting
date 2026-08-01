# Delhi Weather Nowcasting

Short-term (next-day) temperature prediction for Delhi using daily climate data (2013–2017). Built a classical ARIMA baseline and a tuned LSTM, with a focus on rigorous time-series evaluation methodology and leak-free data pipelines.

## Results

| Model | RMSE (°C) |
|---|---|
| ARIMA (blind multi-step forecast) | 11.41 |
| ARIMA (rolling, 1-day-ahead) | 1.77 |
| **LSTM (final, tuned)** | **1.68** |

The final LSTM model outperforms the ARIMA baseline by ~5%. See [`architecture.md`](./architecture.md) for the full methodology, EDA findings, tuning journey, and analysis.

## Key Highlights

- **Evaluation methodology matters as much as the model**: a naive blind-forecast evaluation made ARIMA look far worse (11.41°C) than a realistic walk-forward evaluation matching the actual nowcasting use case (1.77°C) — a ~6.5x difference from evaluation strategy alone.
- **Stationarity analysis**: used the Augmented Dickey-Fuller test to confirm raw temperature data was non-stationary (p=0.716), and that first-order differencing fixed it (p=2.76e-29), directly informing ARIMA's `d` parameter.
- **Leak-free pipeline**: chronological train/validation/test splits throughout, with the scaler fit only on training data.
- **Empirical hyperparameter tuning**: documented an iterative tuning process, including an attempt that made results worse, and the reasoning that led to the final, better configuration.

## Dataset

[Daily Climate Time Series Data — Delhi](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) (Kaggle), 2013–2017. Features: mean temperature, humidity, wind speed, mean pressure.

## Tech Stack

Python, pandas, numpy, statsmodels (ARIMA, seasonal decomposition, ADF test), TensorFlow/Keras (LSTM), scikit-learn (scaling, metrics), matplotlib.

## How to Run

1. Open `weather_nowcasting.ipynb` in Google Colab
2. Get a Kaggle API key from [kaggle.com/settings](https://www.kaggle.com/settings) (Account → API)
3. Run the first cell — it will prompt you to enter your Kaggle username and API key (not stored in the notebook)
4. Run all remaining cells in order

## Project Structure

```
├── weather_nowcasting.ipynb   # Full notebook: EDA, ARIMA, LSTM, evaluation
├── architecture.md            # Detailed methodology, findings, and results
└── README.md
```
