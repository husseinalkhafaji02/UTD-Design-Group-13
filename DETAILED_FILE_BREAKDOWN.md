# Detailed File Breakdown

This document provides a file-by-file technical summary of the current project.

## 1) `README`
### Purpose
Project overview and high-level architecture description for the UTD Design Group 13 real estate analyzer.

### Key Contents
- Problem statement and product intent.
- Hybrid architecture definition:
  - Micro-pricing with XGBoost.
  - Macro-forecasting with LSTM.
- Data source descriptions (Redfin, Zillow, dynamic APIs).
- Basic project structure.

### Notes
- Mentions dynamic APIs broadly; in code, only FRED mortgage-rate retrieval is currently live.

## 2) `requirements.txt`
### Purpose
Declares Python dependencies required by the scripts.

### Packages Listed
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `tensorflow`
- `matplotlib`
- `seaborn`

### Notes
- `pandas_datareader` is used in `src/data_preprocessing.py` but is not listed here. Add it if environment setup errors appear.

## 3) `src/data_preprocessing.py`
### Purpose
Prepares Redfin property data and engineers features used for XGBoost training.

### Main Functions
- `load_and_clean_redfin(filepath)`:
  - Loads CSV.
  - Drops non-essential listing columns.
  - Fills missing `HOA/MONTH` with `0` when present.
  - Drops rows missing critical columns (`PRICE`, `SQUARE FEET`, `BEDS`, `BATHS`, `YEAR BUILT`, `LATITUDE`, `LONGITUDE`).
- `haversine_distance(lat1, lon1, lat2, lon2)`:
  - Computes geospatial distance in miles.
- `fetch_local_crime_index(lat, lon)`:
  - Placeholder for future police API.
  - Currently returns deterministic pseudo-random integers.
- `engineer_xgboost_vectors(df, user_poi_lat, user_poi_lon)`:
  - Creates `PROPERTY_AGE` from `YEAR BUILT`.
  - Adds `DISTANCE_TO_POI`.
  - Adds `LOCAL_CRIME_INDEX`.
- `simulate_seasonality_test(df)`:
  - Simulates search month.
  - Applies seasonal price shift (+3% in summer, -2% in winter).
  - Encodes month cyclically as sine/cosine.
- `fetch_current_mortgage_rate()`:
  - Fetches live `MORTGAGE30US` from FRED via `pandas_datareader`.
  - Falls back to `6.5` on failure.

### Inputs and Outputs
- Input: `data/redfin_dfw.csv`.
- Output: `data/engineered_xgboost_data.csv`.

### Execution Flow (`__main__`)
1. Load and clean Redfin data.
2. Engineer dynamic features.
3. Apply seasonality simulation.
4. Pull current mortgage rate and append as column.
5. Save engineered CSV.

## 4) `src/xgboost_training.py`
### Purpose
Trains the micro-pricing model that predicts property value from engineered static/local features.

### Core Logic
- Loads `data/engineered_xgboost_data.csv`.
- Selects feature list and target `PRICE`.
- Removes rows with missing values.
- Applies outlier filtering:
  - Price outside 1st-99th percentile removed.
  - Top 1% square footage removed.
- Applies log transform to target: `y_log = log1p(price)`.
- Trains `XGBRegressor` with fixed hyperparameters.
- Predicts on holdout split and inverse-transforms predictions with `expm1`.
- Reports MAE and R2 in dollar space.
- Prints feature importances.

### Inputs and Outputs
- Input: `data/engineered_xgboost_data.csv`.
- Output model: `models/xgboost_micro.json`.

### Notes
- Uses random train/test split (`train_test_split`) rather than time-based split, which is acceptable for cross-sectional property snapshots.

## 5) `src/lstm_data_prep.py`
### Purpose
Transforms Zillow time-series into 3D tensors suitable for LSTM training.

### Core Logic
- Loads `data/City_time_series.csv`.
- Drops missing target/date rows and parses date.
- Filters to `Dallas` in `RegionName` and aggregates by date.
- Engineers macro features:
  - `HISTORICAL_MORTGAGE_RATE` (simulated sinusoidal trend).
  - `MACRO_SENTIMENT_SCORE` (simulated random range).
- Builds feature matrix with 3 channels:
  - `ZHVI_AllHomes`
  - `HISTORICAL_MORTGAGE_RATE`
  - `MACRO_SENTIMENT_SCORE`
- Scales all channels to `[0, 1]` via `MinMaxScaler`.
- Creates rolling windows:
  - `X`: previous `lookback_window` months.
  - `y`: next month ZHVI channel.

### Inputs and Outputs
- Input: `data/City_time_series.csv`.
- Output arrays:
  - `data/lstm_X.npy`
  - `data/lstm_y.npy`

## 6) `src/lstm_training.py`
### Purpose
Trains the macro-forecasting model that predicts future market trend from historical sequences.

### Core Logic
- Loads `data/lstm_X.npy` and `data/lstm_y.npy`.
- Performs chronological 80/20 split (no shuffle).
- Defines sequential model:
  - `LSTM(50, activation='relu')`
  - `Dropout(0.2)`
  - `Dense(1)`
- Compiles with Adam optimizer and MSE loss.
- Trains for 20 epochs with validation on future split.
- Evaluates with scaled MAE.
- Generates and displays plot of actual vs predicted trend.
- Saves trained model artifact.

### Inputs and Outputs
- Inputs:
  - `data/lstm_X.npy`
  - `data/lstm_y.npy`
- Output model: `models/lstm_macro.keras`.

### Notes
- Metric is reported in scaled units, not dollar units.

## 7) `src/hybrid_engine.py`
### Purpose
Combines micro and macro models into one inference engine for final valuation.

### Main Class
- `RealEstateAnalyzer`

### Key Methods
- `__init__()`:
  - Loads XGBoost model: `models/xgboost_micro.json`.
  - Loads LSTM model: `models/lstm_macro.keras`.
- `predict_baseline_value(house_features)`:
  - Runs XGBoost on single-house feature vector.
  - Applies `expm1` to return dollar value.
- `scale_live_api_rate(raw_api_rate)`:
  - Min-max scales mortgage rate into assumed LSTM range `[2.5, 8.5]`.
- `predict_market_momentum(historical_macro_sequence, months_in_future, live_interest_rate)`:
  - Gets current index from LSTM.
  - Iteratively forecasts forward by feeding predicted index + provided rate + neutral sentiment (0.5).
  - Converts scaled index delta to growth using multiplier `0.05`.
  - Applies hard guardrail to growth in range `[-10%, +10%]`.
- `generate_final_valuation(house_features, macro_sequence, months_in_future, interest_rate)`:
  - Computes `baseline_price` from XGBoost.
  - Computes `growth_rate` from LSTM.
  - Returns formatted baseline, shift percentage, and final value.

### Inputs and Outputs
- Inputs:
  - House-level feature dictionary.
  - Recent macro sequence tensor.
  - Forecast horizon.
  - Interest rate.
- Output:
  - Dictionary with formatted valuation fields.

### Notes
- File currently ends with an incomplete `__main__` demonstration block (loads `recent_macro_data` but does not print final result).
- Growth shift depends on macro inputs, horizon, and interest rate; it does not directly vary by house features.

## 8) `demo_sample_test.py`
### Purpose
Scenario-based smoke test for the hybrid architecture.

### Current Test Design
- Loads `RealEstateAnalyzer`.
- Loads latest macro sequence: `np.load('data/lstm_X.npy')[-1:]`.
- Defines two sample homes:
  - Starter home.
  - Luxury estate.
- Runs grid across:
  - Horizons: `3, 6, 12` months.
  - Interest rates: `4.5, 6.0, 7.5`.
- Prints table with baseline, shift, and final valuation.
- Runs sanity checks:
  - Baseline stability across macro scenarios.
  - Monotonic rate sensitivity check (higher rates should not increase shift).

### Notes
- Useful for fast behavioral checks, but not a full validation/backtest framework.

## 9) `PIPELINE_SUMMARY.md`
### Purpose
One-page Mermaid flow diagram from raw datasets to final hybrid valuation output.

### Contents
- End-to-end flow for preprocessing, training, model artifacts, and hybrid inference.
- Includes where `demo_sample_test.py` plugs into the pipeline.

## 10) Data and Model Artifacts (Non-code)
### `data/`
- `redfin_dfw.csv`: raw micro-level listing data.
- `engineered_xgboost_data.csv`: processed micro-feature training table.
- `City_time_series.csv`: raw Zillow time-series.
- `lstm_X.npy`, `lstm_y.npy`: LSTM-ready tensors.

### `models/`
- `xgboost_micro.json`: trained XGBoost micro-pricing model.
- `lstm_macro.keras`: trained LSTM macro-forecast model.

## Recommended Run Order
1. `python src/data_preprocessing.py`
2. `python src/xgboost_training.py`
3. `python src/lstm_data_prep.py`
4. `python src/lstm_training.py`
5. `python demo_sample_test.py`

## Current Architecture Reality Check
- Live external data in use: FRED mortgage rate call in preprocessing.
- Placeholder simulations still present:
  - Local crime index.
  - Historical sentiment.
  - Historical mortgage trend in LSTM prep.
