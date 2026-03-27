import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as web

def load_and_clean_redfin(filepath):
    print("Loading raw Redfin data...")
    df = pd.read_csv(filepath)
    
    columns_to_drop = [
        '$/SQUARE FEET', 'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)', 
        'SOURCE', 'MLS#', 'FAVORITE', 'INTERESTED', 'STATUS', 'ADDRESS', 
        'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    if 'HOA/MONTH' in df.columns:
        df['HOA/MONTH'] = df['HOA/MONTH'].fillna(0)
    
    critical_features = ['PRICE', 'SQUARE FEET', 'BEDS', 'BATHS', 'YEAR BUILT', 'LATITUDE', 'LONGITUDE']
    df = df.dropna(subset=[col for col in critical_features if col in df.columns])
    
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two GPS coordinates in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def fetch_local_crime_index(lat, lon):
    """Acts as a wrapper for a future Police API integration."""
    # Removed the print statement here so it doesn't print 7,000 times!
    np.random.seed(int(abs(lat * 1000))) 
    return np.random.randint(10, 85)

def engineer_xgboost_vectors(df, user_poi_lat=32.9866, user_poi_lon=-96.7503):
    print("Engineering Dynamic Feature Vectors...")
    current_year = datetime.now().year
    
    if 'YEAR BUILT' in df.columns:
        df['PROPERTY_AGE'] = current_year - df['YEAR BUILT']
        df = df.drop(columns=['YEAR BUILT']) 
        
    print(f"Calculating dynamic distances to POI ({user_poi_lat}, {user_poi_lon})...")
    df['DISTANCE_TO_POI'] = haversine_distance(
        df['LATITUDE'], df['LONGITUDE'], 
        user_poi_lat, user_poi_lon
    )
    
    print("Pinging Simulated Local Municipal API for 90-day crime data...")
    df['LOCAL_CRIME_INDEX'] = df.apply(
        lambda row: fetch_local_crime_index(row['LATITUDE'], row['LONGITUDE']), axis=1
    )
    
    return df

def simulate_seasonality_test(df):
    print("Simulating 12-Month Search Seasonality for Testing...")
    np.random.seed(42) 
    df['SIMULATED_SEARCH_MONTH'] = np.random.randint(1, 13, size=len(df))
    
    def apply_seasonal_price_shift(row):
        month = row['SIMULATED_SEARCH_MONTH']
        price = row['PRICE']
        if month in [5, 6, 7, 8]:  
            return price * 1.03
        elif month in [11, 12, 1, 2]: 
            return price * 0.98
        else:
            return price
            
    df['PRICE'] = df.apply(apply_seasonal_price_shift, axis=1)
    
    df['SEARCH_MONTH_SIN'] = np.sin(2 * np.pi * df['SIMULATED_SEARCH_MONTH'] / 12)
    df['SEARCH_MONTH_COS'] = np.cos(2 * np.pi * df['SIMULATED_SEARCH_MONTH'] / 12)
    df = df.drop(columns=['SIMULATED_SEARCH_MONTH'])
    
    return df

def fetch_current_mortgage_rate():
    print("Fetching live 30-Year Mortgage Rate from FRED...")
    try:
        start_date = datetime(datetime.now().year, 1, 1)
        end_date = datetime.now()
        rates = web.DataReader('MORTGAGE30US', 'fred', start_date, end_date)
        current_rate = rates.iloc[-1]['MORTGAGE30US']
        print(f"Current 30-Year Fixed Rate: {current_rate}%")
        return current_rate
    except Exception as e:
        print(f"Could not fetch rate. Using fallback default. Error: {e}")
        return 6.5 

if __name__ == "__main__":
    # Assuming you are running this from your UTD-Design-Group-13 root folder
    redfin_path = 'data/redfin_dfw.csv'
    output_path = 'data/engineered_xgboost_data.csv'
    
    try:
        clean_df = load_and_clean_redfin(redfin_path)
        
        # 1. Engineer the age, POI distance, and crime index
        engineered_df = engineer_xgboost_vectors(clean_df, user_poi_lat=32.9866, user_poi_lon=-96.7503)
        
        # 2. RUN THE SIMULATOR (This is the line you were missing!)
        engineered_df = simulate_seasonality_test(engineered_df)
        
        # 3. Fetch live rates
        live_rate = fetch_current_mortgage_rate()
        engineered_df['CURRENT_MORTGAGE_RATE'] = live_rate
        
        engineered_df.to_csv(output_path, index=False)
        print(f"\nSuccess! Dynamic dataset saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file. Check your paths.")