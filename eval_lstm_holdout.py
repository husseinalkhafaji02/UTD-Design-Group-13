import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

X = np.load('data/lstm_X.npy')
y = np.load('data/lstm_y.npy')

train_end_idx = int(len(X) * 0.7)
val_end_idx = int(len(X) * 0.8)

X_val = X[train_end_idx:val_end_idx]
y_val = y[train_end_idx:val_end_idx]
X_test = X[val_end_idx:]
y_test = y[val_end_idx:]

model = load_model('models/lstm_macro.keras')
y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f'LSTM Validation Scaled MAE: {val_mae:.4f}')
print(f'LSTM Unseen Test Scaled MAE: {test_mae:.4f}')
