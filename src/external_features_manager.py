"""
External Features Manager
Unified pipeline for fetching and caching crime data (by ZIP code) and sentiment data (time-indexed).
This module is designed to be run once to populate CSVs that both XGBoost and LSTM pipelines can reuse.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from functools import lru_cache
from external_apis.rapidapi_crime import fetch_crime_via_rapidapi
from external_apis.finnhub_client import fetch_sentiment_time_series_finnhub
from dotenv import load_dotenv

load_dotenv()

OVERPASS_API_URL = os.getenv('OVERPASS_API_URL', 'https://overpass-api.de/api/interpreter')
GDELT_API_URL = os.getenv('GDELT_API_URL', 'https://api.gdeltproject.org/api/v2/search/tv')

# Dallas DFW common ZIP codes and their approximate centers (for Overpass queries)
DFW_ZIP_BOUNDS = {
    '75001': (32.7438, -96.8311),  # Arlington
    '75006': (32.7259, -96.7859),  # Arlington
    '75010': (32.7153, -96.6731),  # Arlington
    '75013': (32.7044, -96.5975),  # Arlington
    '75014': (32.7452, -96.6706),  # Arlington
    '75015': (32.6869, -96.7038),  # Arlington
    '75016': (32.7069, -96.6569),  # Arlington
    '75017': (32.7226, -96.6189),  # Arlington
    '75018': (32.6669, -96.6231),  # Arlington
    '75019': (32.6568, -96.7294),  # Arlington
    '75201': (32.7767, -96.7970),  # Dallas (Highland Park area)
    '75202': (32.7851, -96.8084),  # Dallas
    '75203': (32.7920, -96.7760),  # Dallas
    '75204': (32.7960, -96.8244),  # Dallas
    '75205': (32.8000, -96.7990),  # Dallas
    '75206': (32.7660, -96.8530),  # Dallas
    '75207': (32.7790, -96.8140),  # Dallas
    '75208': (32.7559, -96.8560),  # Dallas
    '75209': (32.7691, -96.7949),  # Dallas
    '75210': (32.7862, -96.7625),  # Dallas
    '75211': (32.7703, -96.8238),  # Dallas
    '75212': (32.7610, -96.8390),  # Dallas
    '75214': (32.8170, -96.7840),  # Dallas (East Dallas)
    '75215': (32.7852, -96.7580),  # Dallas
    '75216': (32.7684, -96.8840),  # Dallas
    '75217': (32.7561, -96.9010),  # Dallas
    '75218': (32.8340, -96.7450),  # Dallas
    '75219': (32.8490, -96.7710),  # Dallas
    '75220': (32.8210, -96.8030),  # Dallas
    '75223': (32.7380, -96.8620),  # Dallas
    '75224': (32.7220, -96.9210),  # Dallas
    '75225': (32.8160, -96.8280),  # Dallas
    '75226': (32.7960, -96.9190),  # Dallas
    '75227': (32.7688, -96.9255),  # Dallas
    '75228': (32.7495, -96.9380),  # Dallas
    '75229': (32.8620, -96.8950),  # Dallas
    '75230': (32.8840, -96.8190),  # Dallas
    '75231': (32.8950, -96.8380),  # Dallas
    '75232': (32.7210, -96.8990),  # Dallas
    '75233': (32.7080, -96.9140),  # Dallas
    '75234': (32.6980, -96.8710),  # Dallas
    '75235': (32.7250, -96.7780),  # Dallas
    '75236': (32.7340, -96.9490),  # Dallas
    '75237': (32.7510, -96.9650),  # Dallas
    '75238': (32.7640, -96.9880),  # Dallas
    '75240': (32.9060, -96.8890),  # Dallas (North Dallas)
    '75241': (32.8760, -96.9390),  # Dallas
    '75243': (32.7180, -96.9720),  # Dallas
    '75244': (32.6880, -96.7550),  # Dallas
    '75245': (32.7350, -96.6690),  # Dallas
    '75246': (32.7470, -96.6050),  # Dallas
    '75247': (32.7620, -96.6480),  # Dallas
    '75248': (32.8620, -96.8210),  # Dallas
    '75249': (32.8740, -96.8640),  # Dallas
    '75250': (32.8840, -96.7860),  # Dallas
    '75251': (32.8940, -96.8050),  # Dallas
    '75252': (32.9050, -96.7510),  # Dallas
    '75253': (32.7190, -96.5970),  # Arlington/Grand Prairie area
    '75254': (32.6850, -96.7210),  # Dallas
    '75287': (32.7767, -96.7970),  # Dallas (Dallas CBD alternative)
    '75260': (32.7767, -96.7970),  # Dallas
    '75261': (32.7767, -96.7970),  # Dallas
}


@lru_cache(maxsize=512)
def fetch_crime_data_by_zip(zip_code):
    """
    Fetch crime index for a single ZIP code from Overpass API.
    Returns an integer 0-100 representing safety (higher = safer).
    Returns None if API call fails (will use fallback).
    """
    try:
        # Prefer RapidAPI provider if configured (faster/more reliable than Overpass mirrors)
        try:
            rapid_val = fetch_crime_via_rapidapi(zip_code)
            if rapid_val is not None:
                print(f"ZIP {zip_code}: RapidAPI provided crime index -> {rapid_val}")
                return int(rapid_val)
        except Exception:
            # non-fatal: fall back to Overpass
            pass

        if zip_code not in DFW_ZIP_BOUNDS:
            print(f"Warning: ZIP {zip_code} not in DFW bounds, using random default")
            np.random.seed(int(zip_code))
            return np.random.randint(40, 80)
        
        lat, lon = DFW_ZIP_BOUNDS[zip_code]
        
        # Query Overpass for police stations near ZIP center
        query = f"""
        [bbox:{lat-0.015},{lon-0.015},{lat+0.015},{lon+0.015}];
        (
          node["amenity"="police"];
          way["amenity"="police"];
          relation["amenity"="police"];
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
        
        # Crime index: more police stations = safer area (higher score)
        # 0 stations = 50 (neutral), each station +5 (max ~95)
        crime_index = 50 + min(len(elements) * 5, 45)
        print(f"ZIP {zip_code}: {len(elements)} police stations found -> Crime Index: {crime_index}")
        
        return int(crime_index)
        
    except Exception as e:
        print(f"Warning: Could not fetch crime for ZIP {zip_code}: {e}")
        return None


def build_zip_crime_lookup(property_df=None, output_file='data/zip_crime_lookup.csv', 
                           use_all_dfw_zips=True):
    """
    Build and cache crime index for each unique ZIP code.
    
    Args:
        property_df: DataFrame with 'ZIP_CODE' column. If None and use_all_dfw_zips=True,
                     will fetch all DFW_ZIP_BOUNDS.
        output_file: Path to save the lookup CSV.
        use_all_dfw_zips: If True, fetch all known DFW ZIP codes, not just those in property_df.
    
    Returns:
        DataFrame with columns: ZIP_CODE, CRIME_INDEX, DATA_SOURCE, FETCH_DATE
    """
    print("Building ZIP-code crime lookup...")
    
    if use_all_dfw_zips:
        unique_zips = list(DFW_ZIP_BOUNDS.keys())
        print(f"Fetching crime data for {len(unique_zips)} known DFW ZIP codes")
    else:
        unique_zips = sorted(property_df['ZIP_CODE'].dropna().unique())
        print(f"Fetching crime data for {len(unique_zips)} ZIP codes in property dataset")
    
    results = []
    for i, zip_code in enumerate(unique_zips):
        print(f"  [{i+1}/{len(unique_zips)}] Fetching ZIP {zip_code}...")
        crime_index = fetch_crime_data_by_zip(str(zip_code))
        
        results.append({
            'ZIP_CODE': str(zip_code),
            'CRIME_INDEX': crime_index if crime_index is not None else 50,
            'DATA_SOURCE': 'Overpass' if crime_index is not None else 'default',
            'FETCH_DATE': datetime.now().strftime('%Y-%m-%d')
        })
    
    df_lookup = pd.DataFrame(results)
    df_lookup.to_csv(output_file, index=False)
    print(f"\nSaved ZIP crime lookup to: {output_file}")
    print(f"  Total ZIPs: {len(df_lookup)}")
    print(f"  ZIPs with data: {(df_lookup['DATA_SOURCE'] == 'Overpass').sum()}")
    
    return df_lookup


def fetch_sentiment_time_series(start_date=None, end_date=None, 
                                output_file='data/sentiment_time_series.csv'):
    """
    Fetch historical market sentiment from GDELT and save as time series.
    
    Args:
        start_date: datetime object. If None, defaults to 1 year ago.
        end_date: datetime object. If None, defaults to today.
        output_file: Path to save the sentiment CSV.
    
    Returns:
        DataFrame with columns: DATE, SENTIMENT_SCORE, SOURCE, ARTICLE_COUNT
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Fetching sentiment time series from Finnhub/GDELT ({start_date.date()} to {end_date.date()})...")

    # Try Finnhub first (requires FINNHUB_KEY and FINNHUB_SYMBOL)
    try:
        fh_df = fetch_sentiment_time_series_finnhub(start_date, end_date)
        if fh_df is not None and len(fh_df) > 0:
            fh_df.to_csv(output_file, index=False)
            print(f"\nSaved sentiment time series to: {output_file} (source=Finnhub)")
            print(f"  Total weeks: {len(fh_df)}")
            return fh_df
    except Exception as e:
        print(f"Finnhub fetch error: {e}")
    
    sentiments = []
    current_date = start_date
    week_count = 0
    
    while current_date <= end_date:
        week_count += 1
        try:
            print(f"  Week {week_count}: {current_date.date()}...", end=' ', flush=True)
            
            params = {
                'query': 'Dallas real estate housing market prices',
                'mode': 'artlist',
                'sort': 'date',
                'format': 'json',
                'startdatetime': current_date.strftime('%Y%m%d000000'),
                'enddatetime': (current_date + timedelta(days=7)).strftime('%Y%m%d235959')
            }
            
            response = requests.get(GDELT_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            
            # Simple sentiment: count positive and negative keywords
            pos_keywords = ['growth', 'surge', 'boom', 'strong', 'rise', 'increase', 'positive', 'up']
            neg_keywords = ['decline', 'fall', 'risk', 'drop', 'crash', 'collapse', 'negative', 'down']
            
            pos_count = 0
            neg_count = 0
            for article in articles:
                title = article.get('title', '').lower()
                for kw in pos_keywords:
                    if kw in title:
                        pos_count += 1
                for kw in neg_keywords:
                    if kw in title:
                        neg_count += 1
            
            total = pos_count + neg_count
            if total > 0:
                sentiment = (pos_count - neg_count) / total
            else:
                sentiment = 0.0
            
            sentiment = max(-1.0, min(1.0, sentiment))  # Clip to [-1, 1]
            
            sentiments.append({
                'DATE': current_date.strftime('%Y-%m-%d'),
                'SENTIMENT_SCORE': float(sentiment),
                'SOURCE': 'GDELT',
                'ARTICLE_COUNT': len(articles)
            })
            
            print(f"Found {len(articles)} articles, sentiment={sentiment:.2f}")
            
        except Exception as e:
            print(f"Error fetching sentiment for {current_date.date()}: {e}")
            # Still add a neutral entry
            sentiments.append({
                'DATE': current_date.strftime('%Y-%m-%d'),
                'SENTIMENT_SCORE': 0.0,
                'SOURCE': 'GDELT_error',
                'ARTICLE_COUNT': 0
            })
        
        current_date += timedelta(days=7)
    
    df_sentiment = pd.DataFrame(sentiments)
    df_sentiment.to_csv(output_file, index=False)
    print(f"\nSaved sentiment time series to: {output_file} (source=GDELT)")
    print(f"  Total weeks: {len(df_sentiment)}")
    print(f"  Date range: {df_sentiment['DATE'].min()} to {df_sentiment['DATE'].max()}")
    
    return df_sentiment


def merge_crime_into_properties(property_df, zip_lookup_df):
    """Join properties to crime index by ZIP code."""
    print("Merging crime index into properties by ZIP code...")
    
    property_df['ZIP_CODE'] = property_df['ZIP_CODE'].astype(str)
    zip_lookup_df['ZIP_CODE'] = zip_lookup_df['ZIP_CODE'].astype(str)
    
    merged = property_df.merge(
        zip_lookup_df[['ZIP_CODE', 'CRIME_INDEX']],
        on='ZIP_CODE',
        how='left'
    ).rename(columns={'CRIME_INDEX': 'LOCAL_CRIME_INDEX'})
    
    missing_count = merged['LOCAL_CRIME_INDEX'].isna().sum()
    if missing_count > 0:
        print(f"  Warning: {missing_count} properties missing crime data (ZIPs not in lookup)")
        merged['LOCAL_CRIME_INDEX'].fillna(50, inplace=True)  # Default to neutral
    
    print(f"  Merged {len(merged)} properties with crime index")
    return merged


def merge_sentiment_into_properties(property_df, sentiment_df, date_column='LISTING_DATE'):
    """
    Join properties to sentiment by date (backward fill: use most recent sentiment before/on listing).
    
    Args:
        property_df: DataFrame with listing dates.
        sentiment_df: DataFrame with DATE and SENTIMENT_SCORE.
        date_column: Name of the date column in property_df.
    
    Returns:
        Merged DataFrame with new column MARKET_SENTIMENT.
    """
    print(f"Merging sentiment into properties by {date_column}...")
    
    if date_column not in property_df.columns:
        print(f"  Warning: {date_column} not found. Adding MARKET_SENTIMENT with default value 0.0")
        property_df['MARKET_SENTIMENT'] = 0.0
        return property_df
    
    # Convert to datetime
    property_df = property_df.copy()
    property_df[date_column] = pd.to_datetime(property_df[date_column], errors='coerce')
    sentiment_df = sentiment_df.copy()
    sentiment_df['DATE'] = pd.to_datetime(sentiment_df['DATE'])
    
    # Sort for merge_asof
    property_df = property_df.sort_values(date_column)
    sentiment_df = sentiment_df.sort_values('DATE')
    
    # Merge: for each property date, get the most recent sentiment
    merged = pd.merge_asof(
        property_df,
        sentiment_df[['DATE', 'SENTIMENT_SCORE']],
        left_on=date_column,
        right_on='DATE',
        direction='backward'
    )
    
    merged.rename(columns={'SENTIMENT_SCORE': 'MARKET_SENTIMENT'}, inplace=True)
    merged.drop(columns=['DATE'], inplace=True)
    
    # Fill any missing values with 0
    merged['MARKET_SENTIMENT'].fillna(0.0, inplace=True)
    
    print(f"  Merged {len(merged)} properties with market sentiment")
    return merged


if __name__ == '__main__':
    print("=" * 60)
    print("EXTERNAL FEATURES MANAGER")
    print("=" * 60 + "\n")
    
    # Step 1: Build ZIP crime lookup (no property data needed, uses all DFW ZIPs)
    print("STEP 1: Building ZIP-code crime lookup")
    print("-" * 60)
    zip_lookup = build_zip_crime_lookup(use_all_dfw_zips=True)
    print()
    
    # Step 2: Fetch sentiment time series
    print("STEP 2: Fetching sentiment time series")
    print("-" * 60)
    sentiment = fetch_sentiment_time_series(
        start_date=datetime(2025, 1, 1),
        end_date=datetime.now()
    )
    print()
    
    print("=" * 60)
    print("SUCCESS! External features built and cached.")
    print("  - Zip crime lookup: data/zip_crime_lookup.csv")
    print("  - Sentiment time series: data/sentiment_time_series.csv")
    print("=" * 60)
