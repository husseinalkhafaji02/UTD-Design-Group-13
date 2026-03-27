import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

print("Loading Engineered Feature Vectors...")
df = pd.read_csv('data/engineered_xgboost_data.csv')

features = [
    'SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS', 
    'PROPERTY_AGE', 'HOA/MONTH', 
    'LATITUDE', 'LONGITUDE', 
    'SEARCH_MONTH_SIN', 'SEARCH_MONTH_COS',
    'DISTANCE_TO_POI', 'LOCAL_CRIME_INDEX'  # <--- Added our new dynamic vectors!
]
target = 'PRICE'

df = df.dropna(subset=features + [target])
print(f"Original dataset size: {len(df)} properties")

# ==========================================
# ADVANCED DATA SCIENCE TECHNIQUE 1: OUTLIER REMOVAL
# ==========================================
# We trim the top 1% and bottom 1% of prices. 
# This prevents extreme $10M mega-mansions from skewing the math and confusing the model's baseline rules.
lower_price_bound = df[target].quantile(0.01)
upper_price_bound = df[target].quantile(0.99)

# We also remove extreme square footage anomalies
upper_sqft_bound = df['SQUARE FEET'].quantile(0.99)

df = df[(df[target] >= lower_price_bound) & 
        (df[target] <= upper_price_bound) & 
        (df['SQUARE FEET'] <= upper_sqft_bound)]

print(f"Dataset size after outlier removal: {len(df)} properties")

X = df[features]

# ==========================================
# ADVANCED DATA SCIENCE TECHNIQUE 2: LOG TRANSFORMATION
# ==========================================
# Instead of predicting raw dollars, we compress the massive exponential scale using log(1 + x).
# A $300k house and a $3M house are now mathematically closer together, making pattern recognition much easier for XGBoost.
y_log = np.log1p(df[target])

# Train/Test Split
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

print("\nTraining the Advanced XGBoost Model (with Log-Transform)...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train_log)

# Generate predictions (Note: these predictions are currently in log scale!)
y_pred_log = model.predict(X_test)

# ==========================================
# REVERSING THE LOG (Back to Real Dollars)
# ==========================================
# We must convert the log predictions back to real money using exp(x) - 1 so our MAE is readable to a human user.
y_pred_dollars = np.expm1(y_pred_log)
y_test_dollars = np.expm1(y_test_log)

mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
r2 = r2_score(y_test_dollars, y_pred_dollars)

print("\n======================================")
print("PHASE 2: ADVANCED XGBOOST RESULTS")
print("======================================")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared (Accuracy Score): {r2:.4f}")
print("======================================\n")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances (What drives the price?):")
print(feature_importance_df.to_string(index=False))
model.save_model('models/xgboost_micro.json')