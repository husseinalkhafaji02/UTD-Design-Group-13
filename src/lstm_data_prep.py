import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def prep_zillow_data(filepath, lookback_window=12):
    print("Loading raw Zillow time-series data...")
    df = pd.read_csv(filepath)
    
    # 1. Isolate the target variable and dates
    df = df.dropna(subset=['ZHVI_AllHomes', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Filter for the local Dallas market to align with our DFW XGBoost model
    if 'RegionName' in df.columns:
        dallas_data = df[df['RegionName'].astype(str).str.contains('Dallas', na=False, case=False)]
        if not dallas_data.empty:
            # If Dallas is found, group by date in case there are multiple Dallas sub-regions
            df = dallas_data.groupby('Date')['ZHVI_AllHomes'].mean().reset_index()
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 3. Engineer Historical Macro Vectors
    print("Engineering Historical Macro Vectors (Interest Rates & Sentiment)...")
    # To match historical Zillow dates, we simulate the historical trends of our dynamic features
    np.random.seed(42)
    
    # Simulating a historical mortgage rate cycle (fluctuating roughly between 3% and 7%)
    df['HISTORICAL_MORTGAGE_RATE'] = 5.0 + np.sin(np.arange(len(df)) / 12) * 2.0
    
    # Simulating historical regional news sentiment (-1.0 to 1.0)
    df['MACRO_SENTIMENT_SCORE'] = np.random.uniform(-0.5, 0.8, len(df))
    
    # Our 3 LSTM feature vectors
    features = ['ZHVI_AllHomes', 'HISTORICAL_MORTGAGE_RATE', 'MACRO_SENTIMENT_SCORE']
    data_matrix = df[features].values
    
    # 4. Scale the Data (Crucial for Deep Learning)
    print("Scaling data between 0 and 1...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_matrix)
    
    # 5. Create the 3D Time-Series Sequences
    print(f"Formatting sequences with a {lookback_window}-month rolling window...")
    X, y = [], []
    
    for i in range(len(scaled_data) - lookback_window):
        # The Input (X): The previous 12 months of all 3 features
        X.append(scaled_data[i:(i + lookback_window), :])
        
        # The Target (y): The 13th month's ZHVI (Index 0)
        y.append(scaled_data[i + lookback_window, 0])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    zillow_path = 'data/City_time_series.csv'
    output_dir = 'data/'
    
    try:
        X, y = prep_zillow_data(zillow_path, lookback_window=12)
        
        print(f"\nSuccess! LSTM Data Shaped.")
        print(f"X shape (Input): {X.shape} -> [Samples, Time Steps, Features]")
        print(f"y shape (Output): {y.shape} -> [Target ZHVI Predictions]")
        
        # Save the formatted numpy arrays so our training script can load them
        np.save(os.path.join(output_dir, 'lstm_X.npy'), X)
        np.save(os.path.join(output_dir, 'lstm_y.npy'), y)
        print("Saved time-series arrays to the data folder.")
        
    except FileNotFoundError:
        print(f"Error: Could not find {zillow_path}. Did you re-unzip the Zillow dataset into the data folder?")