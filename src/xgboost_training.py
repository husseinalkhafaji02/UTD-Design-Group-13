import json
import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

FEATURE_MODE = os.getenv('FEATURE_MODE', 'simulated').strip().lower()
PRIMARY_FEATURE_SET = os.getenv('PRIMARY_FEATURE_SET', 'core_plus_poi_v1').strip()
CRIME_COVERAGE_THRESHOLD = 0.90
CRIME_LOW_COVERAGE_STRATEGY = os.getenv('CRIME_LOW_COVERAGE_STRATEGY', 'exclude').strip().lower()
RUN_ABLATION_GATE = True

N_ITER_TUNING = 100
EARLY_STOPPING_ROUNDS = 5
TARGET = 'PRICE'

CORE_FEATURES = [
    'SQUARE FEET', 'LOT SIZE', 'BEDS', 'BATHS',
    'PROPERTY_AGE', 'HOA/MONTH',
    'LATITUDE', 'LONGITUDE',
    'SEARCH_MONTH_SIN', 'SEARCH_MONTH_COS'
]

POI_FEATURES = [
    'DISTANCE_TO_POI_SINGLE',
    'DISTANCE_TO_POI_MULTI_MIN',
    'DISTANCE_TO_POI_MULTI_WEIGHTED',
    'DISTANCE_TO_POI_MULTI_MEAN_TOP_N',
    'POI_COUNT_WITHIN_1_MI',
    'POI_COUNT_WITHIN_3_MI'
]


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected_defaults = {
        'DISTANCE_TO_POI_SINGLE': np.nan,
        'DISTANCE_TO_POI_MULTI_MIN': np.nan,
        'DISTANCE_TO_POI_MULTI_WEIGHTED': np.nan,
        'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': np.nan,
        'POI_COUNT_WITHIN_1_MI': 0,
        'POI_COUNT_WITHIN_3_MI': 0,
        'LOCAL_CRIME_INDEX_SIM': np.nan,
        'LOCAL_CRIME_INDEX_REAL': np.nan,
        'LOCAL_CRIME_DATA_AGE_DAYS': np.nan,
        'LOCAL_CRIME_SNAPSHOT_IS_STALE': 1,
        'EXTERNAL_FEATURE_MODE': FEATURE_MODE,
    }
    for column, default_value in expected_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    # Backward compatibility for older engineered files.
    if 'DISTANCE_TO_POI' in df.columns and 'DISTANCE_TO_POI_SINGLE' not in df.columns:
        df['DISTANCE_TO_POI_SINGLE'] = df['DISTANCE_TO_POI']
    if 'LOCAL_CRIME_INDEX' in df.columns and 'LOCAL_CRIME_INDEX_SIM' not in df.columns:
        df['LOCAL_CRIME_INDEX_SIM'] = df['LOCAL_CRIME_INDEX']

    return df


def build_crime_feature_config(df: pd.DataFrame):
    coverage = float(df['LOCAL_CRIME_INDEX_REAL'].notna().mean())
    strategy = CRIME_LOW_COVERAGE_STRATEGY
    selected_crime_features = []
    decision = ''

    if FEATURE_MODE == 'production' and coverage >= CRIME_COVERAGE_THRESHOLD:
        selected_crime_features = ['LOCAL_CRIME_INDEX_REAL']
        decision = (
            f"Using LOCAL_CRIME_INDEX_REAL (coverage={coverage:.1%} >= {CRIME_COVERAGE_THRESHOLD:.0%})."
        )
    elif FEATURE_MODE == 'production' and strategy == 'impute':
        df['LOCAL_CRIME_INDEX_REAL_MISSING'] = df['LOCAL_CRIME_INDEX_REAL'].isna().astype(int)

        if 'ZIP OR POSTAL CODE' in df.columns:
            df['LOCAL_CRIME_INDEX_REAL_IMPUTED'] = (
                df.groupby('ZIP OR POSTAL CODE')['LOCAL_CRIME_INDEX_REAL'].transform(
                    lambda s: s.fillna(s.median())
                )
            )
        else:
            df['LOCAL_CRIME_INDEX_REAL_IMPUTED'] = df['LOCAL_CRIME_INDEX_REAL']

        df['LOCAL_CRIME_INDEX_REAL_IMPUTED'] = df['LOCAL_CRIME_INDEX_REAL_IMPUTED'].fillna(
            df['LOCAL_CRIME_INDEX_REAL'].median()
        )

        selected_crime_features = ['LOCAL_CRIME_INDEX_REAL_IMPUTED', 'LOCAL_CRIME_INDEX_REAL_MISSING']
        decision = (
            f"Coverage={coverage:.1%} < {CRIME_COVERAGE_THRESHOLD:.0%}; "
            "using imputed real-crime feature with missing indicator."
        )
    else:
        selected_crime_features = []
        decision = (
            f"Coverage={coverage:.1%} < {CRIME_COVERAGE_THRESHOLD:.0%} or mode={FEATURE_MODE}; "
            "excluding real-crime feature for this run."
        )

    print("Crime reliability gate:", decision)
    return df, coverage, selected_crime_features, decision


def get_param_distributions():
    return {
        'n_estimators': [250, 400, 600, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 2, 4, 6, 8],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.75, 0.9, 1.0],
        'gamma': [0, 0.05, 0.1, 0.2],
        'reg_alpha': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    }


def train_and_evaluate(df: pd.DataFrame, feature_set_name: str, feature_list, tune: bool, fixed_params=None):
    usable_features = [feature for feature in feature_list if feature in df.columns]
    if not usable_features:
        return None

    run_df = df.dropna(subset=usable_features + [TARGET]).copy()
    if len(run_df) < 200:
        print(f"Skipping {feature_set_name}: not enough rows after NA filtering ({len(run_df)}).")
        return None

    X = run_df[usable_features]
    y_log = np.log1p(run_df[TARGET])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    cv_score = None
    chosen_params = fixed_params
    if tune:
        print(f"\nTuning XGBoost hyperparameters for {feature_set_name}...")
        search = RandomizedSearchCV(
            estimator=XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            param_distributions=get_param_distributions(),
            n_iter=N_ITER_TUNING,
            scoring='neg_mean_absolute_error',
            cv=3,
            verbose=1,
            n_jobs=1,
            random_state=42,
            refit=True
        )
        search.fit(X_train, y_train)
        chosen_params = search.best_params_.copy()
        cv_score = -search.best_score_
        print(f"Best CV score (log MAE) for {feature_set_name}: {cv_score:.5f}")
        print(f"Best params for {feature_set_name}: {search.best_params_}")

    if not chosen_params:
        chosen_params = {
            'n_estimators': 600,
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_child_weight': 6,
            'subsample': 0.9,
            'colsample_bytree': 0.6,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 0.5,
        }

    model_params = chosen_params.copy()
    model_params.update(
        {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': 1,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        }
    )

    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
    
    # Save eval results for visualization
    eval_results = model.evals_result()
    if eval_results:
        import json
        history_dict = {
            'val_mae': [float(x) for x in eval_results.get('validation_0', {}).get('rmse', [])]
        }
        history_path = f'models/xgboost_{feature_set_name}_eval_history.json'
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    val_pred = np.expm1(model.predict(X_val))
    val_true = np.expm1(y_val)
    val_mae = mean_absolute_error(val_true, val_pred)
    val_r2 = r2_score(val_true, val_pred)

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'feature_set_id': feature_set_name,
        'features': usable_features,
        'rows_used': len(run_df),
        'cv_log_mae': cv_score,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'mae': mae,
        'r2': r2,
        'params': chosen_params,
        'model': model,
        'X_test': X_test,
        'y_test_log': y_test,
    }


def build_feature_impact_table(primary_result: dict) -> pd.DataFrame:
    model = primary_result['model']
    features = primary_result['features']
    X_test = primary_result['X_test']
    y_test_log = primary_result['y_test_log']

    perm_result = permutation_importance(
        model,
        X_test,
        y_test_log,
        scoring='neg_mean_absolute_error',
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    impact_df = pd.DataFrame(
        {
            'Feature': features,
            'Model_Impact_Gain': model.feature_importances_,
            'Eval_Impact_MAE_Delta_Log': perm_result.importances_mean,
            'Eval_Impact_MAE_Delta_Log_Std': perm_result.importances_std,
        }
    )
    impact_df = impact_df.sort_values(
        by=['Eval_Impact_MAE_Delta_Log', 'Model_Impact_Gain'],
        ascending=False,
    ).reset_index(drop=True)
    impact_df['Rank'] = np.arange(1, len(impact_df) + 1)
    return impact_df


print("Loading Engineered Feature Vectors...")
df = pd.read_csv('data/engineered_xgboost_data.csv')
df = ensure_feature_columns(df)

print(f"Original dataset size: {len(df)} properties")
df, crime_coverage, gated_crime_features, crime_decision = build_crime_feature_config(df)
print(f"Dataset size available for feature-set builds: {len(df)} properties")

feature_sets = {
    'core_v1': CORE_FEATURES,
    'core_plus_poi_v1': CORE_FEATURES + POI_FEATURES,
    'core_plus_crime_v1': CORE_FEATURES + gated_crime_features,
    'full_external_v1': CORE_FEATURES + POI_FEATURES + gated_crime_features,
}

if PRIMARY_FEATURE_SET not in feature_sets:
    print(f"Unknown PRIMARY_FEATURE_SET={PRIMARY_FEATURE_SET}. Falling back to core_plus_poi_v1.")
    PRIMARY_FEATURE_SET = 'core_plus_poi_v1'

primary_result = train_and_evaluate(
    df,
    PRIMARY_FEATURE_SET,
    feature_sets[PRIMARY_FEATURE_SET],
    tune=False,
)

if primary_result is None:
    raise RuntimeError("Primary feature set could not be trained. Check data coverage and feature availability.")

print("\n======================================")
print("PRIMARY MODEL RESULTS")
print("======================================")
print(f"Feature Set ID: {primary_result['feature_set_id']}")
print(f"Rows used: {primary_result['rows_used']}")
print(f"Validation MAE: ${primary_result['val_mae']:,.2f}")
print(f"Validation R-squared: {primary_result['val_r2']:.4f}")
print(f"Unseen Test MAE: ${primary_result['mae']:,.2f}")
print(f"R-squared (Accuracy Score): {primary_result['r2']:.4f}")
print("======================================\n")

if RUN_ABLATION_GATE:
    print("Ablation gate (baseline / crime / poi / both):")
    print("FeatureSet | Rows | MAE | R2")
    ablation_order = ['core_v1', 'core_plus_crime_v1', 'core_plus_poi_v1', 'full_external_v1']
    for set_name in ablation_order:
        result = train_and_evaluate(
            df,
            set_name,
            feature_sets[set_name],
            tune=False,
            fixed_params=primary_result['params'],
        )
        if result is None:
            print(f"{set_name} | SKIPPED")
            continue
        print(
            f"{set_name} | {result['rows_used']} | "
            f"${result['mae']:,.2f} | {result['r2']:.4f}"
        )

feature_impact_df = build_feature_impact_table(primary_result)

print("\nFeature Impact Table (Primary Model):")
print(feature_impact_df.to_string(index=False))

feature_impact_path = 'models/xgboost_primary_feature_impact.csv'
feature_impact_df.to_csv(feature_impact_path, index=False)
print(f"Saved feature impact table: {feature_impact_path}")

artifact_name = f"models/xgboost_micro_{primary_result['feature_set_id']}.json"
primary_result['model'].save_model(artifact_name)
print(f"Saved model artifact: {artifact_name}")

metadata = {
    'created_at_utc': pd.Timestamp.utcnow().isoformat(),
    'feature_mode': FEATURE_MODE,
    'feature_set_id': primary_result['feature_set_id'],
    'features': primary_result['features'],
    'rows_used': int(primary_result['rows_used']),
    'val_mae': float(primary_result['val_mae']),
    'val_r2': float(primary_result['val_r2']),
    'mae': float(primary_result['mae']),
    'r2': float(primary_result['r2']),
    'tuned_params': primary_result['params'],
    'crime_coverage': crime_coverage,
    'crime_gate_decision': crime_decision,
    'crime_low_coverage_strategy': CRIME_LOW_COVERAGE_STRATEGY,
    'crime_coverage_threshold': CRIME_COVERAGE_THRESHOLD,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    'feature_impact_table_path': feature_impact_path,
}

metadata_path = 'models/xgboost_micro_metadata.json'
with open(metadata_path, 'w', encoding='utf-8') as handle:
    json.dump(metadata, handle, indent=2)
print(f"Saved model metadata: {metadata_path}")