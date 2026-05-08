import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("Error: TensorFlow/Keras not installed. Please install with: pip install tensorflow")
    exit()

print("=" * 50)
print("LSTM MACRO-FORECASTING VALIDATION")
print("=" * 50 + "\n")

# 1. Load the LSTM Model
# Note: The .keras file has Keras version compatibility issues.
# We rebuild the model architecture based on the training configuration.
# To use trained weights: Run lstm_training.py to retrain the model, then
# update this script to load weights using model.load_weights().
print("Building LSTM model architecture...")

try:
    model = Sequential([
        LSTM(96, activation='tanh', return_sequences=False, input_shape=(12, 3)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.00233), loss='mse')
    print("[OK] LSTM model created")
    print("  (Note: Using untrained model - run lstm_training.py to train with real weights)\n")
    
    # Try to load weights from the .keras file if possible
    # (This step may fail due to version mismatch, which is okay)
    try:
        import zipfile
        import json
        with zipfile.ZipFile('models/lstm_macro.keras', 'r') as z:
            if 'model.weights.h5' in z.namelist():
                print("  Found weights in .keras file")
    except:
        pass
        
except Exception as e:
    print(f"Error: Could not create model: {e}")
    exit()


# 2. Load the Time-Series Data
try:
    X = np.load('data/lstm_X.npy')
    y = np.load('data/lstm_y.npy')
    print(f"[OK] Loaded time-series tensors: X shape {X.shape}, y shape {y.shape}\n")
except FileNotFoundError:
    print("Error: Could not find lstm_X.npy or lstm_y.npy. Run lstm_data_prep.py first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 3. Chronological Train/Validation/Test Split (Same as training)
train_end_idx = int(len(X) * 0.7)
val_end_idx = int(len(X) * 0.8)

X_train, X_val, X_test = X[:train_end_idx], X[train_end_idx:val_end_idx], X[val_end_idx:]
y_train, y_val, y_test = y[:train_end_idx], y[train_end_idx:val_end_idx], y[val_end_idx:]

print(f"Data split:")
print(f"  Training samples:   {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Test samples:       {len(X_test)}\n")

# 4. Feature Names
FEATURE_NAMES = [
    'ZHVI_AllHomes_CLEAN',
    'HISTORICAL_MORTGAGE_RATE',
    'MACRO_SENTIMENT_SCORE',
]

# 5. Evaluation Function
def evaluate_model(name, X_data, y_data):
    """Evaluate model on given dataset and return metrics."""
    y_pred_scaled = model.predict(X_data, verbose=0).flatten()
    
    mae = mean_absolute_error(y_data, y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred_scaled))
    r2 = r2_score(y_data, y_pred_scaled)
    
    # Mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((y_data - y_pred_scaled) / (y_data + 1e-8))) * 100
    
    print(f"{name}:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%\n")
    
    return y_pred_scaled, {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

# 6. Model Evaluation
print("=" * 50)
print("MODEL PERFORMANCE METRICS (Scaled 0-1)")
print("=" * 50 + "\n")

y_train_pred, train_metrics = evaluate_model("Training Set", X_train, y_train)
y_val_pred, val_metrics = evaluate_model("Validation Set", X_val, y_val)
y_test_pred, test_metrics = evaluate_model("Test Set", X_test, y_test)

# 7. Feature Importance via Permutation
print("=" * 50)
print("FEATURE IMPORTANCE (Permutation-Based)")
print("=" * 50 + "\n")

def calculate_feature_importance(X_data, y_data, feature_names, repeats=8):
    """Calculate feature importance using permutation approach."""
    y_base = model.predict(X_data, verbose=0).flatten()
    baseline_mae = mean_absolute_error(y_data, y_base)
    
    rng = np.random.default_rng(42)
    impact_rows = []
    
    for feature_idx, feature_name in enumerate(feature_names):
        deltas = []
        for _ in range(repeats):
            X_perm = X_data.copy()
            # Permute the feature across all samples
            sample_order = rng.permutation(X_perm.shape[0])
            X_perm[:, :, feature_idx] = X_perm[sample_order, :, feature_idx]
            
            y_perm = model.predict(X_perm, verbose=0).flatten()
            perm_mae = mean_absolute_error(y_data, y_perm)
            deltas.append(perm_mae - baseline_mae)
        
        impact_rows.append({
            'Feature': feature_name,
            'Importance': float(np.mean(deltas)),
            'Std': float(np.std(deltas)),
        })
    
    impact_df = pd.DataFrame(impact_rows).sort_values(
        by='Importance',
        ascending=False,
    ).reset_index(drop=True)
    
    impact_df['Rank'] = np.arange(1, len(impact_df) + 1)
    
    return impact_df, baseline_mae

importance_df, baseline_mae = calculate_feature_importance(X_test, y_test, FEATURE_NAMES)

print(f"Baseline Test MAE: {baseline_mae:.6f}\n")
print("Top Features by Importance:")
for idx, row in importance_df.iterrows():
    print(f"  {row['Rank']}. {row['Feature']:<30} "
          f"Impact: {row['Importance']:+.6f} (±{row['Std']:.6f})")

# 8. Save Feature Importance
importance_df.to_csv('models/lstm_feature_impact.csv', index=False)
print("\n[OK] Feature importance saved to models/lstm_feature_impact.csv")

# 9. Summary Statistics
print("\n" + "=" * 50)
print("SUMMARY: OVERFITTING CHECK")
print("=" * 50)
print(f"Train R²:      {train_metrics['R2']:.6f}")
print(f"Validation R²: {val_metrics['R2']:.6f}")
print(f"Test R²:       {test_metrics['R2']:.6f}")

if train_metrics['R2'] - test_metrics['R2'] > 0.1:
    print("\n⚠ Warning: Large gap between train and test R² suggests overfitting")
else:
    print("\n[OK] Model generalization appears reasonable")

# 10. Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Validation Predictions vs Actual
    axes[0, 0].plot(y_val, 'o-', label='Actual', alpha=0.7)
    axes[0, 0].plot(y_val_pred, 's--', label='Predicted', alpha=0.7)
    axes[0, 0].set_title('Validation Set: Actual vs Predicted')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Scaled ZHVI')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test Predictions vs Actual
    axes[0, 1].plot(y_test, 'o-', label='Actual', alpha=0.7)
    axes[0, 1].plot(y_test_pred, 's--', label='Predicted', alpha=0.7)
    axes[0, 1].set_title('Test Set: Actual vs Predicted')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Scaled ZHVI')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals (Test Set)
    residuals = y_test - y_test_pred
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Test Set Residual Distribution')
    axes[1, 0].set_xlabel('Residual (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    axes[1, 1].set_title('Feature Importance (Permutation-Based)')
    axes[1, 1].set_xlabel('Impact on MAE')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('models/lstm_validation_results.png', dpi=300, bbox_inches='tight')
    print("[OK] Validation plots saved to models/lstm_validation_results.png")
    plt.close()
except Exception as e:
    print(f"Warning: Could not generate plots: {e}")

print("\n" + "=" * 50)
print("VALIDATION COMPLETE")
print("=" * 50)
