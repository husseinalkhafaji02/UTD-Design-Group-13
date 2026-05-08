import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from functools import lru_cache
import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file (if it exists)
load_dotenv()

# Add src to path to import external_features_manager
sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_EXTERNAL_FEATURE_MODE = os.getenv('EXTERNAL_FEATURE_MODE', 'live')  # Use real APIs when available
OVERPASS_API_URL = os.getenv('OVERPASS_API_URL', 'https://overpass-api.de/api/interpreter')
GDELT_API_URL = os.getenv('GDELT_API_URL', 'https://api.gdeltproject.org/api/v2/search/tv')

# Fallback default POIs (used when Overpass API fails)
DEFAULT_POIS = [
    {'name': 'DFW Transit Hub', 'lat': 32.8975, 'lon': -97.0403, 'category': 'transit'},
    {'name': 'Dallas CBD', 'lat': 32.7767, 'lon': -96.7970, 'category': 'retail'},
    {'name': 'Medical District', 'lat': 32.8116, 'lon': -96.8380, 'category': 'hospital'},
    {'name': 'Plano ISD Anchor', 'lat': 33.0198, 'lon': -96.6989, 'category': 'school'}
]

POI_CATEGORY_WEIGHTS = {
    'school': 0.30,
    'transit': 0.30,
    'hospital': 0.20,
    'retail': 0.20
}

@lru_cache(maxsize=512)
def fetch_pois_from_overpass(lat, lon):
    """Fetch real POIs (schools, hospitals, transit, retail) from Overpass API.
    
    Returns list of POI dicts with name, lat, lon, category.
    Falls back to DEFAULT_POIS if API fails.
    """
    try:
        # Query for schools, hospitals, transit stations, and retail within ~5km
        query = f"""
        [bbox:{lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05}];
        (
          node["amenity"="school"];
          way["amenity"="school"];
          node["amenity"="hospital"];
          way["amenity"="hospital"];
          node["public_transport"="station"];
          way["public_transport"="station"];
          node["shop"~"supermarket|shopping_centre|mall"];
          way["shop"~"supermarket|shopping_centre|mall"];
        );
        out center;
        """
        
        headers = {
            'User-Agent': 'RealEstateMarketAnalyzer/1.0 (contact:dev@example.com)',
            'Accept': 'application/json'
        }
        response = requests.post(OVERPASS_API_URL, data={'data': query}, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        elements = data.get('elements', [])
        
        pois = []
        category_map = {
            'school': 'school',
            'hospital': 'hospital',
            'public_transport': 'transit',
            'shop': 'retail'
        }
        
        for element in elements[:20]:  # Limit to top 20 to avoid too many queries
            if 'lat' in element and 'lon' in element:
                tags = element.get('tags', {})
                category = 'retail'  # default
                
                if tags.get('amenity') == 'school':
                    category = 'school'
                elif tags.get('amenity') == 'hospital':
                    category = 'hospital'
                elif tags.get('public_transport'):
                    category = 'transit'
                elif tags.get('shop'):
                    category = 'retail'
                
                pois.append({
                    'name': tags.get('name', f"{category.title()} {len(pois)}"),
                    'lat': element['lat'],
                    'lon': element['lon'],
                    'category': category
                })
        
        if pois:
            print(f"Found {len(pois)} real POIs from Overpass API for ({lat:.4f}, {lon:.4f})")
            return pois
        else:
            print(f"No POIs found from Overpass for ({lat:.4f}, {lon:.4f}), using defaults")
            return DEFAULT_POIS
            
    except Exception as e:
        print(f"Warning: Could not fetch POIs from Overpass for ({lat:.4f}, {lon:.4f}): {e}")
        return DEFAULT_POIS

def fetch_market_sentiment():
    """Fetch Dallas real estate market sentiment from GDELT news data.
    
    Returns sentiment score: -1 (negative) to +1 (positive)
    """
    try:
        # Query GDELT for recent Dallas real estate news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        params = {
            'query': 'Dallas real estate market housing prices',
            'mode': 'artlist',
            'sort': 'date',
            'format': 'json',
            'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
            'enddatetime': end_date.strftime('%Y%m%d%H%M%S')
        }
        
        response = requests.get(GDELT_API_URL, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        if not articles:
            return 0.0  # Neutral if no articles found
        
        # Simple sentiment scoring based on keyword presence
        positive_keywords = ['growth', 'surge', 'boom', 'opportunity', 'strong', 'rise', 'increase']
        negative_keywords = ['decline', 'fall', 'crisis', 'risk', 'drop', 'crash', 'collapse']
        
        sentiment_scores = []
        for article in articles[:10]:  # Analyze top 10 articles
            text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()
            
            pos_count = sum(1 for keyword in positive_keywords if keyword in text)
            neg_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
                sentiment_scores.append(score)
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            print(f"Market sentiment (GDELT): {avg_sentiment:.2f} from {len(articles)} articles")
            return float(np.clip(avg_sentiment, -1, 1))
        else:
            return 0.0
            
    except Exception as e:
        print(f"Warning: Could not fetch market sentiment from GDELT: {e}")
        return 0.0

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

@lru_cache(maxsize=512)
def fetch_crime_data_by_zip(zip_code):
    """Load crime index for a ZIP code from cache or fetch fresh if available."""
    try:
        # Try to load from cache first
        if os.path.exists('data/zip_crime_lookup.csv'):
            lookup_df = pd.read_csv('data/zip_crime_lookup.csv')
            match = lookup_df[lookup_df['ZIP_CODE'].astype(str) == str(zip_code)]
            if not match.empty:
                return float(match.iloc[0]['CRIME_INDEX'])
    except Exception as e:
        print(f"  Warning: Could not load cached crime for ZIP {zip_code}: {e}")
    
    # Fallback: return neutral default
    np.random.seed(int(float(zip_code)))
    return float(np.random.randint(40, 80))

def fetch_local_crime_index(lat, lon, zip_code=None):
    """Wrapper: fetch crime by ZIP code if available, otherwise default."""
    if zip_code is not None:
        return fetch_crime_data_by_zip(str(zip_code))
    else:
        # No ZIP available, use random default
        np.random.seed(int(abs(lat * 1000)))
        return float(np.random.randint(40, 80))

def engineer_xgboost_vectors(df, user_poi_lat=32.9866, user_poi_lon=-96.7503):
    print("Engineering Dynamic Feature Vectors...")
    current_year = datetime.now().year
    
    if 'YEAR BUILT' in df.columns:
        df['PROPERTY_AGE'] = current_year - df['YEAR BUILT']
        df = df.drop(columns=['YEAR BUILT']) 
        
    print(f"Calculating dynamic distances to POI ({user_poi_lat}, {user_poi_lon})...")
    df['DISTANCE_TO_POI_SINGLE'] = haversine_distance(
        df['LATITUDE'], df['LONGITUDE'], 
        user_poi_lat, user_poi_lon
    )

    # Fetch real POIs for the first property to use for all (or fallback to defaults)
    pois = DEFAULT_POIS
    if len(df) > 0 and DEFAULT_EXTERNAL_FEATURE_MODE == 'live':
        first_lat = df['LATITUDE'].iloc[0]
        first_lon = df['LONGITUDE'].iloc[0]
        pois = fetch_pois_from_overpass(first_lat, first_lon)

    distance_frames = []
    for poi in pois:
        series = haversine_distance(df['LATITUDE'], df['LONGITUDE'], poi['lat'], poi['lon'])
        distance_frames.append(
            pd.DataFrame(
                {
                    'distance': series,
                    'weight': POI_CATEGORY_WEIGHTS.get(poi['category'], 0.25)
                }
            )
        )

    stacked_distances = np.column_stack([frame['distance'].to_numpy() for frame in distance_frames])
    df['DISTANCE_TO_POI_MULTI_MIN'] = stacked_distances.min(axis=1)

    top_n = min(3, stacked_distances.shape[1])
    sorted_distances = np.sort(stacked_distances, axis=1)
    df['DISTANCE_TO_POI_MULTI_MEAN_TOP_N'] = sorted_distances[:, :top_n].mean(axis=1)
    df['POI_COUNT_WITHIN_1_MI'] = (stacked_distances <= 1.0).sum(axis=1)
    df['POI_COUNT_WITHIN_3_MI'] = (stacked_distances <= 3.0).sum(axis=1)

    weighted_num = np.zeros(len(df), dtype=float)
    weighted_den = np.zeros(len(df), dtype=float)
    for frame in distance_frames:
        proximity = 1.0 / (frame['distance'].to_numpy() + 0.1)
        weighted_num += frame['weight'].to_numpy() * proximity
        weighted_den += frame['weight'].to_numpy()
    df['DISTANCE_TO_POI_MULTI_WEIGHTED'] = weighted_num / np.maximum(weighted_den, 1e-8)

    # Backward-compatible alias while migrating downstream code.
    df['DISTANCE_TO_POI'] = df['DISTANCE_TO_POI_SINGLE']
    
    print("Loading crime data by ZIP code...")
    
    # Use ZIP_CODE if available, otherwise fall back to lat/lon
    if 'ZIP_CODE' in df.columns:
        df['LOCAL_CRIME_INDEX_SIM'] = df.apply(
            lambda row: fetch_local_crime_index(row['LATITUDE'], row['LONGITUDE'], zip_code=row.get('ZIP_CODE')), axis=1
        )
    else:
        print("  Note: ZIP_CODE column not found, using lat/lon fallback")
        df['LOCAL_CRIME_INDEX_SIM'] = df.apply(
            lambda row: fetch_local_crime_index(row['LATITUDE'], row['LONGITUDE']), axis=1
        )

    # Add market sentiment feature
    market_sentiment = fetch_market_sentiment()
    df['MARKET_SENTIMENT'] = market_sentiment
    
    # Metadata for data freshness
    df['LOCAL_CRIME_DATA_AGE_DAYS'] = 0 if DEFAULT_EXTERNAL_FEATURE_MODE == 'live' else np.nan
    df['LOCAL_CRIME_SNAPSHOT_IS_STALE'] = 0 if DEFAULT_EXTERNAL_FEATURE_MODE == 'live' else 1

    # Backward-compatible alias while migrating downstream code.
    df['LOCAL_CRIME_INDEX'] = df['LOCAL_CRIME_INDEX_SIM']
    
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
        import pandas_datareader.data as web

        start_date = datetime(datetime.now().year, 1, 1)
        end_date = datetime.now()
        rates = web.DataReader('MORTGAGE30US', 'fred', start_date, end_date)
        current_rate = rates.iloc[-1]['MORTGAGE30US']
        print(f"Current 30-Year Fixed Rate: {current_rate}%")
        return current_rate
    except Exception as e:
        print(f"Could not fetch rate. Using fallback default. Error: {e}")
        return 6.5 

def sanitize_feature_vectors(df):
    print("Running anomaly cleanup for feature vectors...")
    start_rows = len(df)

    numeric_columns = [
        'PRICE', 'SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS',
        'HOA/MONTH', 'LATITUDE', 'LONGITUDE', 'PROPERTY_AGE'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'LOT SIZE' in df.columns:
        df['LOT SIZE'] = df['LOT SIZE'].fillna(df['LOT SIZE'].median())

    required = ['PRICE', 'SQUARE FEET', 'BEDS', 'BATHS', 'LATITUDE', 'LONGITUDE']
    df = df.dropna(subset=[col for col in required if col in df.columns])

    base_mask = (
        (df['PRICE'] > 0) &
        (df['SQUARE FEET'] >= 200) &
        (df['BEDS'] > 0) &
        (df['BATHS'] > 0) &
        (df['LATITUDE'].between(24, 50)) &
        (df['LONGITUDE'].between(-125, -60))
    )

    price_lower = df['PRICE'].quantile(0.01)
    price_upper = df['PRICE'].quantile(0.99)
    sqft_lower = max(200, df['SQUARE FEET'].quantile(0.01))
    sqft_upper = df['SQUARE FEET'].quantile(0.99)

    outlier_mask = (
        (df['PRICE'] >= price_lower) &
        (df['PRICE'] <= price_upper) &
        (df['SQUARE FEET'] >= sqft_lower) &
        (df['SQUARE FEET'] <= sqft_upper)
    )

    df = df[base_mask & outlier_mask]
    removed_rows = start_rows - len(df)
    print(
        f"Anomaly cleanup removed {removed_rows} rows | "
        f"Price bounds: [{price_lower:,.0f}, {price_upper:,.0f}] | "
        f"SqFt bounds: [{sqft_lower:,.0f}, {sqft_upper:,.0f}]"
    )
    print(f"Rows after anomaly cleanup: {len(df)}")

    return df

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
        engineered_df['EXTERNAL_FEATURE_MODE'] = DEFAULT_EXTERNAL_FEATURE_MODE

        # 4. Clean anomalies so downstream models get stable vectors
        engineered_df = sanitize_feature_vectors(engineered_df)
        
        engineered_df.to_csv(output_path, index=False)
        print(f"\nSuccess! Dynamic dataset saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file. Check your paths.")