import json
import matplotlib.pyplot as plt
import os

# Load LSTM training history
lstm_history_path = 'models/lstm_training_history.json'
xgboost_history_path = 'models/xgboost_core_plus_poi_v1_eval_history.json'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History: LSTM vs XGBoost', fontsize=14, fontweight='bold')

# Plot LSTM history
if os.path.exists(lstm_history_path):
    with open(lstm_history_path, 'r') as f:
        lstm_history = json.load(f)
    
    epochs = range(1, len(lstm_history['loss']) + 1)
    axes[0].plot(epochs, lstm_history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(epochs, lstm_history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('LSTM - Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, f'LSTM history not found.\nRun: python src/lstm_training.py', 
                 ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

# Plot XGBoost history
if os.path.exists(xgboost_history_path):
    with open(xgboost_history_path, 'r') as f:
        xgboost_history = json.load(f)
    
    iterations = range(1, len(xgboost_history['val_mae']) + 1)
    axes[1].plot(iterations, xgboost_history['val_mae'], label='Validation RMSE', linewidth=2, color='orange')
    axes[1].set_xlabel('Boosting Round')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('XGBoost - Validation Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, f'XGBoost history not found.\nRun: python src/xgboost_training.py', 
                 ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

plt.tight_layout()
plt.show()
print("Training history visualization complete!")
