# Forecasting Daily Transit Ridership: Linear Regression vs. Gradient Boosting

A small, reproducible machine learning project that forecasts **daily NYC subway ridership one week ahead** and compares two models against the simple baselines analysts commonly use.

## Question
Does a gradient-boosted model forecast ridership more accurately than a linear regression, and do either of them beat "same weekday last week" and "same weekday last year"?

## Data
[MTA Daily Ridership and Traffic: Beginning 2020](https://data.ny.gov/d/sayj-mze2), published by the MTA on NY Open Data. Long-format daily counts by mode (`Date`, `Mode`, `Count`). This project uses the `Subway` mode. The notebook downloads the CSV on first run and caches it in `data/`.

## Approach
1. **Validate** the daily series: continuous daily index, short-gap interpolation, missing-day check.
2. **Baselines:** same weekday last week (lag 7) and same weekday one year ago (lag 364).
3. **Features:** weekday, month, day-of-year (sin/cos), federal holidays and adjacent days, lags of 7/14/28/364 days, and a 28-day rolling mean. Every lag is at least 7 days old, so no information from inside the forecast horizon leaks into the features.
4. **Models:** scikit-learn `LinearRegression` (one-hot calendar features, scaled numeric features) and `HistGradientBoostingRegressor`.
5. **Evaluation:** walk-forward cross-validation (`TimeSeriesSplit`, 5 folds of 60 days) on the training window, then a final holdout of the most recent 90 days. Metrics: MAE, RMSE, MAPE.
6. **Diagnostics:** error by day type (weekday, weekend, holiday) and permutation importance for the gradient-boosting model.

The 2020-2021 pandemic period is excluded from the modeling window because the weekly pattern was structurally different; earlier data is still used to compute the 364-day lag.

## Run it
```bash
git clone https://github.com/shayda-maher/transit-ridership-forecasting.git
cd transit-ridership-forecasting
pip install -r requirements.txt
jupyter notebook ridership_forecast.ipynb
```
Run all cells. Outputs (figures and CSV tables) are written to `outputs/`.

To use a local copy of the data, set the `RIDERSHIP_CSV` environment variable to its path.

## Results
Run the notebook to regenerate these. Tables are saved to `outputs/results_holdout.csv`, `outputs/results_cv.csv`, and `outputs/results_by_day_type.csv`.

<!-- After running the notebook, paste the holdout table here and add 2-3 sentences on what you found. -->

## Limitations
- One NYC system-wide series; it says nothing about other cities or individual stations.
- The MTA publishes estimates that can be revised, so results shift slightly as the dataset updates.
- Only federal holidays are modeled. Storms, strikes, parades, and other local events are unmodeled and appear as large errors.
- Weather and economic drivers are not included. They are the natural next step.

## Tech
Python, pandas, NumPy, scikit-learn, matplotlib.
