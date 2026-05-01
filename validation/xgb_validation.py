import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("data/engineered_xgboost_data.csv")

features = [
    'SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS',
    'PROPERTY_AGE', 'HOA/MONTH',
    'LATITUDE', 'LONGITUDE',
    'SEARCH_MONTH_SIN', 'SEARCH_MONTH_COS',
    'DISTANCE_TO_POI', 'LOCAL_CRIME_INDEX'
]

target = "PRICE"
df = df.dropna(subset=features + [target])

lower = df[target].quantile(0.01)
upper = df[target].quantile(0.99)
upper_sqft = df["SQUARE FEET"].quantile(0.99)

df = df[
    (df[target] >= lower) &
    (df[target] <= upper) &
    (df["SQUARE FEET"] <= upper_sqft)
]

X = df[features]
y_log = np.log1p(df[target])

# 60% train, 20% validation, 20% unseen test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_log, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

def evaluate(name, X_data, y_data):
    pred_log = model.predict(X_data)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_data)

    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    print(f"{name} MAE: ${mae:,.2f}")
    print(f"{name} R²: {r2:.4f}")

print("\nXGBoost Model Results (Micro-Pricing - Real $)")
evaluate("Validation", X_val, y_val)
evaluate("Unseen Test", X_test, y_test)

print("\nTop Features:")
importance = model.get_booster().get_score(importance_type="gain")

importance_df = pd.DataFrame([
    {"Feature": feature, "Gain": importance.get(feature, 0)}
    for feature in features
]).sort_values("Gain", ascending=False)

for _, row in importance_df.head(5).iterrows():
    print(f"{row['Feature']} ({row['Gain']:.4f} gain)")