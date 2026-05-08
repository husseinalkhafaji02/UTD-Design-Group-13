import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pandas_datareader.data as web
from datetime import datetime


def load_historical_mortgage_rates(start_date, end_date):
    """Load monthly 30-year mortgage rates from FRED and align to month-end."""
    try:
        rates = web.DataReader('MORTGAGE30US', 'fred', start_date, end_date)
        rates = rates.reset_index()
        rates.columns = ['Date', 'HISTORICAL_MORTGAGE_RATE']
        rates['Date'] = pd.to_datetime(rates['Date'])
        rates = rates.set_index('Date').resample('M').mean().reset_index()
        rates['HISTORICAL_MORTGAGE_RATE'] = rates['HISTORICAL_MORTGAGE_RATE'].ffill().bfill()
        return rates
    except Exception as e:
        print(f"Warning: Failed to load FRED mortgage rates, using fallback trend. Error: {e}")
        return None

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
            # Use a robust monthly median aggregation to reduce outlier-heavy cross-section distortion.
            monthly = dallas_data.groupby('Date').agg(
                ZHVI_AllHomes=('ZHVI_AllHomes', 'median'),
                OBS_COUNT=('ZHVI_AllHomes', 'size')
            ).reset_index()
            df = monthly
    
    df = df.sort_values('Date').reset_index(drop=True)

    # 2b. Sanity checks and anomaly clipping on monthly changes.
    if 'OBS_COUNT' in df.columns:
        min_obs = int(df['OBS_COUNT'].min())
        print(f"Dallas monthly sample-count min: {min_obs}")

    pct_change = df['ZHVI_AllHomes'].pct_change()
    valid_pct = pct_change.dropna()
    if len(valid_pct) > 0:
        clip_low, clip_high = valid_pct.quantile([0.01, 0.99])
        clipped_pct = pct_change.clip(lower=clip_low, upper=clip_high).fillna(0.0)
        df['ZHVI_AllHomes_CLEAN'] = df['ZHVI_AllHomes'].iloc[0] * (1 + clipped_pct).cumprod()
        clipped_count = int(((pct_change < clip_low) | (pct_change > clip_high)).sum())
        print(
            f"Clipped {clipped_count} monthly ZHVI jumps using 1%-99% pct-change bounds "
            f"[{clip_low:.4f}, {clip_high:.4f}]"
        )
    else:
        df['ZHVI_AllHomes_CLEAN'] = df['ZHVI_AllHomes']
    
    # 3. Engineer Historical Macro Vectors
    print("Engineering Historical Macro Vectors (Interest Rates & Sentiment)...")
    mortgage_df = load_historical_mortgage_rates(df['Date'].min(), df['Date'].max())
    if mortgage_df is not None:
        df = df.merge(mortgage_df, on='Date', how='left')
        df['HISTORICAL_MORTGAGE_RATE'] = df['HISTORICAL_MORTGAGE_RATE'].ffill().bfill()
    else:
        # Fallback trend if FRED is unavailable.
        df['HISTORICAL_MORTGAGE_RATE'] = 5.0 + np.sin(np.arange(len(df)) / 12) * 1.0

    # Keep sentiment neutral by default; attempt to load from cached sentiment time series
    df['MACRO_SENTIMENT_SCORE'] = 0.0
    
    # Try to load sentiment from external_features_manager output
    sentiment_file = 'data/sentiment_time_series.csv'
    if os.path.exists(sentiment_file):
        try:
            print(f"Loading sentiment time series from {sentiment_file}...")
            sentiment_df = pd.read_csv(sentiment_file)
            sentiment_df['DATE'] = pd.to_datetime(sentiment_df['DATE'])
            
            # Aggregate by month (average sentiment for each month)
            sentiment_df['YEAR_MONTH'] = sentiment_df['DATE'].dt.to_period('M')
            monthly_sentiment = sentiment_df.groupby('YEAR_MONTH')['SENTIMENT_SCORE'].mean().reset_index()
            monthly_sentiment['YEAR_MONTH'] = monthly_sentiment['YEAR_MONTH'].astype(str)
            
            print(f"  Loaded sentiment for {len(monthly_sentiment)} months")
            
            # For now, use the average sentiment (can be extended to match date ranges)
            avg_sentiment = sentiment_df['SENTIMENT_SCORE'].mean()
            df['MACRO_SENTIMENT_SCORE'] = avg_sentiment
            print(f"  Using average sentiment: {avg_sentiment:.3f}")
        except Exception as e:
            print(f"  Warning: Could not load sentiment from {sentiment_file}: {e}")
            print(f"  Keeping MACRO_SENTIMENT_SCORE as 0.0 (neutral)")
    else:
        print(f"  Note: {sentiment_file} not found. Run external_features_manager.py first.")
        print(f"  Keeping MACRO_SENTIMENT_SCORE as 0.0 (neutral)")
    
    # Our 3 LSTM feature vectors
    features = ['ZHVI_AllHomes_CLEAN', 'HISTORICAL_MORTGAGE_RATE', 'MACRO_SENTIMENT_SCORE']
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