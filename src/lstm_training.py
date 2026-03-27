import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

# 2. Chronological Train/Test Split (80% Past, 20% Future)
# We do NOT shuffle. We cut the timeline cleanly at the 80% mark.
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training on {len(X_train)} historical sequences.")
print(f"Testing on {len(X_test)} future sequences.\n")

# 3. Build the Deep Learning Architecture
model = Sequential([
    # The LSTM layer reads the 12-month sequence and looks for patterns
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    # Dropout prevents the neural network from memorizing the data (Overfitting)
    Dropout(0.2),
    # The Dense layer outputs the single prediction for the 13th month
    Dense(1)
])

# Compile the model with the Adam optimizer and Mean Squared Error loss
model.compile(optimizer='adam', loss='mse')

# 4. Train the Network
print("Beginning Neural Network Training...")
# Epochs = how many times the network loops through the data to learn
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=16, 
    validation_data=(X_test, y_test),
    verbose=1
)

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