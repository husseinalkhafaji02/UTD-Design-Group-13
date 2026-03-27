# Pipeline Summary: Raw Data to Hybrid Valuation

```mermaid
flowchart TD
    A1[Raw CSV: data/redfin_dfw.csv] --> B1[src/data_preprocessing.py]
    B1 --> B2[Clean listing rows and required fields]
    B2 --> B3[Engineer micro features]
    B3 --> B4[PROPERTY_AGE, DISTANCE_TO_POI, LOCAL_CRIME_INDEX]
    B4 --> B5[Seasonality simulation: SEARCH_MONTH_SIN/COS + adjusted PRICE]
    B5 --> B6[Fetch mortgage rate from FRED: MORTGAGE30US]
    B6 --> C1[Engineered CSV: data/engineered_xgboost_data.csv]

    A2[Raw CSV: data/City_time_series.csv] --> D1[src/lstm_data_prep.py]
    D1 --> D2[Filter Dallas market and aggregate by Date]
    D2 --> D3[Macro features: ZHVI + historical mortgage + sentiment]
    D3 --> D4[MinMax scaling and rolling windows lookback=12]
    D4 --> D5[Numpy tensors: data/lstm_X.npy, data/lstm_y.npy]

    C1 --> E1[src/xgboost_training.py]
    E1 --> E2[Outlier trimming + log1p target transform]
    E2 --> E3[Train XGBRegressor micro-pricing model]
    E3 --> E4[Model artifact: models/xgboost_micro.json]

    D5 --> F1[src/lstm_training.py]
    F1 --> F2[Chronological train/test split 80/20]
    F2 --> F3[Train LSTM macro-forecast model]
    F3 --> F4[Model artifact: models/lstm_macro.keras]

    E4 --> G1[src/hybrid_engine.py: RealEstateAnalyzer]
    F4 --> G1
    H1[User house features] --> G2[predict_baseline_value via XGBoost]
    H2[Recent macro sequence + horizon + interest rate] --> G3[predict_market_momentum via LSTM]
    G2 --> G4[Combine baseline and growth]
    G3 --> G4
    G4 --> I1[Final future valuation output]

    J1[demo_sample_test.py] --> G1
    J1 --> I2[Scenario table and sanity checks]
```

## Notes
- Micro model (XGBoost) estimates the present-day property baseline price from engineered listing features.
- Macro model (LSTM) estimates market momentum from historical Zillow trend plus macro signals.
- Hybrid engine multiplies baseline value by macro growth shift to produce future valuation.
