import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import keras_tuner as kt

print("==========================================")
print("PHASE 4: LSTM MACRO-FORECASTING TRAINING")
print("==========================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the macro LSTM model with optional tuner resume behavior.'
    )
    parser.add_argument(
        '--resume-tuner',
        action='store_true',
        help='Resume an existing Keras Tuner search from tuner_logs/project_name if available.'
    )
    parser.add_argument(
        '--tuner-project-name',
        default='lstm_macro_tuning',
        help='Keras Tuner project folder name under tuner_logs/.'
    )
    return parser.parse_args()


args = parse_args()

# 1. Load the 3D Tensors
try:
    X = np.load('data/lstm_X.npy')
    y = np.load('data/lstm_y.npy')
    print(f"Loaded Tensors successfully.")
    print(f"Input Shape: {X.shape} (Samples, Time Steps, Features)")
except FileNotFoundError:
    print("Error: Could not find the .npy files. Run lstm_data_prep.py first.")
    exit()

# 2. Chronological Train/Validation/Test Split
# 70% train, 10% validation, 20% test (no shuffling to preserve timeline)
train_end_idx = int(len(X) * 0.7)
val_end_idx = int(len(X) * 0.8)

X_train, X_val, X_test = X[:train_end_idx], X[train_end_idx:val_end_idx], X[val_end_idx:]
y_train, y_val, y_test = y[:train_end_idx], y[train_end_idx:val_end_idx], y[val_end_idx:]

print(f"Training on {len(X_train)} historical sequences.")
print(f"Validating on {len(X_val)} holdout sequences.")
print(f"Testing on {len(X_test)} future sequences.\n")

KNOWN_FEATURE_NAMES = [
    'ZHVI_AllHomes_CLEAN',
    'HISTORICAL_MORTGAGE_RATE',
    'MACRO_SENTIMENT_SCORE',
]


def infer_feature_names(num_features):
    if num_features <= len(KNOWN_FEATURE_NAMES):
        return KNOWN_FEATURE_NAMES[:num_features]
    dynamic_names = [f'FEATURE_{idx}' for idx in range(num_features)]
    for idx, name in enumerate(KNOWN_FEATURE_NAMES):
        if idx < len(dynamic_names):
            dynamic_names[idx] = name
    return dynamic_names


def build_lstm_feature_impact_table(model, X_eval, y_eval, feature_names, repeats=8):
    y_base = model.predict(X_eval, verbose=0)
    baseline_mae = mean_absolute_error(y_eval, y_base)
    rng = np.random.default_rng(42)
    impact_rows = []

    for feature_idx, feature_name in enumerate(feature_names):
        deltas = []
        for _ in range(repeats):
            X_perm = X_eval.copy()
            sample_order = rng.permutation(X_perm.shape[0])
            X_perm[:, :, feature_idx] = X_perm[sample_order, :, feature_idx]
            y_perm = model.predict(X_perm, verbose=0)
            perm_mae = mean_absolute_error(y_eval, y_perm)
            deltas.append(perm_mae - baseline_mae)

        impact_rows.append(
            {
                'Feature': feature_name,
                'Impact_to_Model_MAE_Delta': float(np.mean(deltas)),
                'Impact_to_Model_MAE_Delta_Std': float(np.std(deltas)),
            }
        )

    impact_df = pd.DataFrame(impact_rows).sort_values(
        by='Impact_to_Model_MAE_Delta',
        ascending=False,
    ).reset_index(drop=True)
    impact_df['Rank'] = np.arange(1, len(impact_df) + 1)
    return impact_df, baseline_mae


def build_tuner_model(hp):
    model = Sequential()
    
    # Advantage: Increasing layers allows the network to learn more complex hierarchical temporal patterns 
    # (e.g., short-term seasonality combined with long-term macro trends).
    # However, too many layers on small datasets can cause overfitting and vanishing gradients.
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=128, step=32)
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.4, step=0.1)
        
        # Intermediate LSTM layers must return sequences for the next LSTM layer
        return_seq = (i < num_layers - 1)
        
        if i == 0:
            model.add(LSTM(units, activation='tanh', return_sequences=return_seq, input_shape=(X.shape[1], X.shape[2])))
        else:
            model.add(LSTM(units, activation='tanh', return_sequences=return_seq))
            
        model.add(Dropout(dropout))
        
    model.add(Dense(1))

    # Advantage: Changing optimizers can help escape local minima and speed up convergence.
    # - Adam is a strong default.
    # - RMSprop sometimes handles recurrent neural networks better by normalizing gradients based on history.
    # - Nadam adds Nesterov momentum which helps 'look ahead' during gradient descent.
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Nadam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse')
    return model


# 3. Hyperparameter Tuning with Keras Tuner and Early Stopping
print("Beginning LSTM hyperparameter tuning with Keras Tuner...")

tuner = kt.Hyperband(
    build_tuner_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='tuner_logs',
    project_name=args.tuner_project_name,
    overwrite=not args.resume_tuner,
)

mode_label = 'RESUME' if args.resume_tuner else 'FRESH'
print(
    f"Keras Tuner mode: {mode_label} "
    f"(project=tuner_logs/{args.tuner_project_name})"
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    mode='min'
)

# Start the search
tuner.search(
    X_train, 
    y_train, 
    epochs=50, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping],
    verbose=2
)

print("\nBest LSTM configuration selected:")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Optimal layers: {best_hps.get('num_layers')}")
print(f"Optimal optimizer: {best_hps.get('optimizer')} @ LR: {best_hps.get('learning_rate'):.5f}")

# Rebuild the best model and train it optimally
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train, 
    epochs=200, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping], 
    verbose=1
)

# Save training history for visualization
import json
history_dict = {
    'loss': [float(x) for x in history.history.get('loss', [])],
    'val_loss': [float(x) for x in history.history.get('val_loss', [])]
}
with open('models/lstm_training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"Saved LSTM training history: models/lstm_training_history.json")

# 5. Evaluate Accuracy
print("\nGenerating Future Predictions...")
y_val_pred = model.predict(X_val)
y_pred = model.predict(X_test)

# Because our data is scaled between 0 and 1, the MAE will be a small decimal.
# In a full production app, we would inverse_transform this back into real dollars.
val_mae = mean_absolute_error(y_val, y_val_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n======================================")
print(f"LSTM Validation Scaled Mean Absolute Error: {val_mae:.4f}")
print(f"\n======================================")
print(f"LSTM Unseen Test Scaled Mean Absolute Error: {mae:.4f}")
print(f"======================================")
print("(Note: A score closer to 0.0 means the predicted trend matches the actual trend perfectly.)\n")

feature_names = infer_feature_names(X_test.shape[2])
feature_impact_df, baseline_perm_mae = build_lstm_feature_impact_table(
    model=model,
    X_eval=X_test,
    y_eval=y_test,
    feature_names=feature_names,
    repeats=8,
)

print("LSTM Feature Impact Table (Permutation on Evaluation Split):")
print(feature_impact_df.to_string(index=False))
print(f"Baseline eval MAE used for permutation impact: {baseline_perm_mae:.6f}\n")

os.makedirs('models', exist_ok=True)
lstm_feature_impact_path = 'models/lstm_feature_impact.csv'
feature_impact_df.to_csv(lstm_feature_impact_path, index=False)
print(f"Saved LSTM feature impact table: {lstm_feature_impact_path}")

# 6. Visualize the Results 
print("Generating visualization chart...")
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Market Trend', color='blue', linewidth=2)
plt.plot(y_pred, label='LSTM Predicted Trend', color='red', linestyle='dashed', linewidth=2)
plt.title('LSTM Macro-Forecasting: Actual vs Predicted Market Momentum (Dallas)')
plt.xlabel('Time Steps (Future Months)')
plt.ylabel('Scaled Zillow Home Value Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
model.save('models/lstm_macro.keras')