import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

print("==========================================")
print("PHASE 4: LSTM MACRO-FORECASTING TRAINING")
print("==========================================\n")

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


def build_lstm_model(units, dropout_rate, learning_rate):
    model = Sequential([
        # The LSTM layer reads the 12-month sequence and looks for patterns
        LSTM(units, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
        # Dropout prevents the neural network from memorizing the data (Overfitting)
        Dropout(dropout_rate),
        # The Dense layer outputs the single prediction for the 13th month
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


# 3. Hyperparameter Tuning with Early Stopping
print("Beginning LSTM hyperparameter tuning...")

search_space = [
    {'units': 32, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 16},
    {'units': 32, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
    {'units': 50, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 16},
    {'units': 50, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
    {'units': 64, 'dropout': 0.2, 'learning_rate': 0.0005, 'batch_size': 16},
    {'units': 64, 'dropout': 0.3, 'learning_rate': 0.0005, 'batch_size': 32},
]

best_trial = None
best_history = None

for trial_idx, params in enumerate(search_space, start=1):
    print(
        f"\nTrial {trial_idx}/{len(search_space)} | "
        f"units={params['units']}, dropout={params['dropout']}, "
        f"lr={params['learning_rate']}, batch={params['batch_size']}"
    )

    trial_model = build_lstm_model(
        units=params['units'],
        dropout_rate=params['dropout'],
        learning_rate=params['learning_rate']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    )

    trial_history = trial_model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )

    best_val_loss = float(np.min(trial_history.history['val_loss']))
    stopped_epoch = len(trial_history.history['loss'])

    print(f"Trial best val_loss: {best_val_loss:.6f} | epochs run: {stopped_epoch}")

    if best_trial is None or best_val_loss < best_trial['best_val_loss']:
        best_trial = {
            'params': params,
            'best_val_loss': best_val_loss,
            'epochs_run': stopped_epoch,
            'model': trial_model,
        }
        best_history = trial_history

print("\nBest LSTM configuration selected:")
print(
    f"units={best_trial['params']['units']}, "
    f"dropout={best_trial['params']['dropout']}, "
    f"learning_rate={best_trial['params']['learning_rate']}, "
    f"batch_size={best_trial['params']['batch_size']}"
)
print(
    f"Best validation loss: {best_trial['best_val_loss']:.6f} "
    f"(epochs run before early stopping: {best_trial['epochs_run']})"
)

model = best_trial['model']
history = best_history

# 5. Evaluate Accuracy
print("\nGenerating Future Predictions...")
y_pred = model.predict(X_test)

# Because our data is scaled between 0 and 1, the MAE will be a small decimal.
# In a full production app, we would inverse_transform this back into real dollars.
mae = mean_absolute_error(y_test, y_pred)
print(f"\n======================================")
print(f"LSTM Scaled Mean Absolute Error: {mae:.4f}")
print(f"======================================")
print("(Note: A score closer to 0.0 means the predicted trend matches the actual trend perfectly.)\n")

# 6. Visualize the Results (For the Professor/Team Meeting)
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