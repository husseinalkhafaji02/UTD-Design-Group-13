import numpy as np
import pandas as pd
import json
import os
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model


LEGACY_FEATURES = [
    'SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS',
    'PROPERTY_AGE', 'HOA/MONTH',
    'LATITUDE', 'LONGITUDE',
    'SEARCH_MONTH_SIN', 'SEARCH_MONTH_COS',
    'DISTANCE_TO_POI', 'LOCAL_CRIME_INDEX'
]


DEFAULT_FEATURE_VALUES = {
    'DISTANCE_TO_POI_SINGLE': 10.0,
    'DISTANCE_TO_POI_MULTI_MIN': 10.0,
    'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.1,
    'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 10.0,
    'POI_COUNT_WITHIN_1_MI': 0,
    'POI_COUNT_WITHIN_3_MI': 0,
    'LOCAL_CRIME_INDEX_SIM': 40,
    'LOCAL_CRIME_INDEX_REAL': np.nan,
    'LOCAL_CRIME_INDEX_REAL_IMPUTED': 40,
    'LOCAL_CRIME_INDEX_REAL_MISSING': 1,
}

class RealEstateAnalyzer:
    def __init__(self):
        print("Loading Hybrid Architecture Models...")
        self.feature_names = LEGACY_FEATURES

        metadata_path = 'models/xgboost_micro_metadata.json'
        model_path = 'models/xgboost_micro.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)
            feature_set_id = metadata.get('feature_set_id')
            if feature_set_id:
                candidate_model = f"models/xgboost_micro_{feature_set_id}.json"
                if os.path.exists(candidate_model):
                    model_path = candidate_model
            self.feature_names = metadata.get('features', LEGACY_FEATURES)

        self.xgb_model = XGBRegressor()
        self.xgb_model.load_model(model_path)
        self.lstm_model = load_model('models/lstm_macro.keras')

    def _prepare_feature_vector(self, house_features):
        row = house_features.copy()

        # Backward compatibility from old demo dictionaries.
        if 'DISTANCE_TO_POI_SINGLE' in self.feature_names and 'DISTANCE_TO_POI_SINGLE' not in row:
            poi_distance = row.get('DISTANCE_TO_POI', DEFAULT_FEATURE_VALUES['DISTANCE_TO_POI_SINGLE'])
            row['DISTANCE_TO_POI_SINGLE'] = poi_distance
            row.setdefault('DISTANCE_TO_POI_MULTI_MIN', poi_distance)
            row.setdefault('DISTANCE_TO_POI_MULTI_MEAN_TOP_N', poi_distance)
            row.setdefault('DISTANCE_TO_POI_MULTI_WEIGHTED', 1.0 / (float(poi_distance) + 0.1))
            row.setdefault('POI_COUNT_WITHIN_1_MI', 1 if float(poi_distance) <= 1.0 else 0)
            row.setdefault('POI_COUNT_WITHIN_3_MI', 1 if float(poi_distance) <= 3.0 else 0)

        if 'LOCAL_CRIME_INDEX_SIM' in self.feature_names and 'LOCAL_CRIME_INDEX_SIM' not in row:
            row['LOCAL_CRIME_INDEX_SIM'] = row.get('LOCAL_CRIME_INDEX', DEFAULT_FEATURE_VALUES['LOCAL_CRIME_INDEX_SIM'])

        for feature_name in self.feature_names:
            if feature_name not in row:
                row[feature_name] = DEFAULT_FEATURE_VALUES.get(feature_name, 0)

        return pd.DataFrame([[row[feature_name] for feature_name in self.feature_names]], columns=self.feature_names)
        
    def predict_baseline_value(self, house_features):
        """Runs the XGBoost Micro-Model"""
        feature_vector = self._prepare_feature_vector(house_features)
        log_prediction = self.xgb_model.predict(feature_vector)[0]
        return np.expm1(log_prediction)

    def scale_live_api_rate(self, raw_api_rate):
        """Standardizes raw % (e.g. 6.5) to LSTM scale (0.0-1.0)"""
        hist_min, hist_max = 2.5, 8.5
        scaled_rate = (raw_api_rate - hist_min) / (hist_max - hist_min)
        return max(0.0, min(scaled_rate, 1.0))

    def predict_market_momentum(self, historical_macro_sequence, months_in_future, live_interest_rate=6.38):
        """
        Runs the LSTM Macro-Model. 
        Now takes a live_interest_rate and scales it automatically.
        """
        # 1. Scale the input so the LSTM doesn't explode
        scaled_rate = self.scale_live_api_rate(live_interest_rate)
        
        current_index_scaled = self.lstm_model.predict(historical_macro_sequence, verbose=0)[0][0]
        future_sequence = historical_macro_sequence.copy()
        
        for _ in range(months_in_future):
            next_step = self.lstm_model.predict(future_sequence, verbose=0)
            
            # Use the NEW scaled rate for the forecast steps
            # Sentiment is hardcoded to 0.5 (neutral) for the forecast
            new_step_reshaped = np.array([[[next_step[0][0], scaled_rate, 0.5]]])
            future_sequence = np.append(future_sequence[:, 1:, :], new_step_reshaped, axis=1)
            
        future_index_scaled = next_step[0][0]
        
        # Calculate realistic growth
        raw_scaled_difference = future_index_scaled - current_index_scaled
        growth_rate = raw_scaled_difference * 0.05 

        # Economic monotonicity guardrail:
        # higher mortgage rates should reduce forward growth, especially at short/medium horizons.
        neutral_rate = 6.0
        horizon_factor = max(months_in_future, 1) / 12.0
        rate_sensitivity = 0.005
        growth_rate -= (live_interest_rate - neutral_rate) * rate_sensitivity * horizon_factor
        
        # Safety guardrail (+/- 10%)
        return max(min(growth_rate, 0.10), -0.10)

    def generate_final_valuation(self, house_features, macro_sequence, months_in_future=6, interest_rate=6.38):
        """Main entry point for GUI"""
        baseline_price = self.predict_baseline_value(house_features)
        
        # Pass the interest rate into the momentum predictor
        growth_rate = self.predict_market_momentum(macro_sequence, months_in_future, interest_rate)
        
        final_price = baseline_price * (1 + growth_rate)
        
        return {
            "Baseline Value (Today)": f"${baseline_price:,.2f}",
            f"Forecasted Market Shift ({months_in_future} Months)": f"{growth_rate * 100:.2f}%",
            "Final Future Valuation": f"${final_price:,.2f}"
        }

if __name__ == "__main__":
    analyzer = RealEstateAnalyzer()
    
    user_house = {
        'SQUARE FEET': 2500, 'LOT SIZE': 7000, 'BEDS': 4, 'BATHS': 3, 
        'PROPERTY_AGE': 10, 'HOA/MONTH': 50, 'LATITUDE': 32.9866, 'LONGITUDE': -96.7503, 
        'SEARCH_MONTH_SIN': 0.5, 'SEARCH_MONTH_COS': 0.866,
        'DISTANCE_TO_POI': 2.5, 'LOCAL_CRIME_INDEX': 35
    }
    
    recent_macro_data = np.load('data/lstm_X.npy')[-1:]
