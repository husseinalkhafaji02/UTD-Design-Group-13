import numpy as np
import json
import os
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

print("=" * 60)
print("LOADING PRE-TRAINED MODELS")
print("=" * 60)

# Load data
X = np.load('data/lstm_X.npy')
y = np.load('data/lstm_y.npy')
train_end_idx = int(len(X) * 0.7)
val_end_idx = int(len(X) * 0.8)

X_val = X[train_end_idx:val_end_idx]
y_val = y[train_end_idx:val_end_idx]
X_test = X[val_end_idx:]
y_test = y[val_end_idx:]

print(f"\nData loaded:")
print(f"  Validation samples: {len(X_val)}")
print(f"  Test samples: {len(X_test)}")

# ==================== LSTM ====================
print("\n" + "=" * 60)
print("LSTM MODEL EVALUATION")
print("=" * 60)

lstm_model = load_model('models/lstm_macro.keras')
print("✓ Loaded: models/lstm_macro.keras")

y_val_pred = lstm_model.predict(X_val, verbose=0)
y_test_pred = lstm_model.predict(X_test, verbose=0)

lstm_val_mae = mean_absolute_error(y_val, y_val_pred)
lstm_test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nLSTM Results (Scaled):")
print(f"  Validation MAE: {lstm_val_mae:.4f}")
print(f"  Unseen Test MAE: {lstm_test_mae:.4f}")

# ==================== XGBoost ====================
print("\n" + "=" * 60)
print("XGBOOST MODEL EVALUATION")
print("=" * 60)

xgb_model = XGBRegressor()
xgb_model.load_model('models/xgboost_micro_core_plus_poi_v1.json')
print("✓ Loaded: models/xgboost_micro_core_plus_poi_v1.json")

# Load engineered data for XGBoost
import pandas as pd
df = pd.read_csv('data/engineered_xgboost_data.csv')

# Get features used in training
try:
    with open('models/xgboost_micro_metadata.json', 'r') as f:
        metadata = json.load(f)
    features = metadata.get('features', [])
    print(f"✓ Loaded: models/xgboost_micro_metadata.json")
except:
    features = ['SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS', 'PROPERTY_AGE', 
                'HOA/MONTH', 'LATITUDE', 'LONGITUDE', 'SEARCH_MONTH_SIN', 
                'SEARCH_MONTH_COS', 'DISTANCE_TO_POI_SINGLE', 
                'DISTANCE_TO_POI_MULTI_MIN', 'DISTANCE_TO_POI_MULTI_WEIGHTED',
                'DISTANCE_TO_POI_MULTI_MEAN_TOP_N', 'POI_COUNT_WITHIN_1_MI',
                'POI_COUNT_WITHIN_3_MI']

# Prepare data (same split as training)
from sklearn.model_selection import train_test_split
run_df = df[features + ['PRICE']].dropna()
X = run_df[features]
y_log = np.log1p(run_df['PRICE'])

X_train_full, X_test_xgb, y_train_full, y_test_xgb = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Predict
y_val_pred_xgb = np.expm1(xgb_model.predict(X_val_xgb))
y_test_pred_xgb = np.expm1(xgb_model.predict(X_test_xgb))

y_val_true_xgb = np.expm1(y_val_xgb)
y_test_true_xgb = np.expm1(y_test_xgb)

xgb_val_mae = mean_absolute_error(y_val_true_xgb, y_val_pred_xgb)
xgb_val_r2 = r2_score(y_val_true_xgb, y_val_pred_xgb)
xgb_test_mae = mean_absolute_error(y_test_true_xgb, y_test_pred_xgb)
xgb_test_r2 = r2_score(y_test_true_xgb, y_test_pred_xgb)

print(f"\nXGBoost Results (Real $ values):")
print(f"  Validation MAE: ${xgb_val_mae:,.2f}")
print(f"  Validation R²: {xgb_val_r2:.4f}")
print(f"  Unseen Test MAE: ${xgb_test_mae:,.2f}")
print(f"  Unseen Test R²: {xgb_test_r2:.4f}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\nLSTM (Macro-Forecasting):")
print(f"  • Validation MAE: {lstm_val_mae:.4f}")
print(f"  • Test MAE: {lstm_test_mae:.4f}")
print(f"\nXGBoost (Micro-Pricing):")
print(f"  • Validation MAE: ${xgb_val_mae:,.2f} (R² = {xgb_val_r2:.4f})")
print(f"  • Test MAE: ${xgb_test_mae:,.2f} (R² = {xgb_test_r2:.4f})")
print("\n" + "=" * 60)
