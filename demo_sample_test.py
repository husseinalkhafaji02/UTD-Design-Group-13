import numpy as np
from src.hybrid_engine import RealEstateAnalyzer

print("Initializing Hybrid Architecture for Professor Demo...\n")
analyzer = RealEstateAnalyzer()

# Grab the latest macro sequence for the LSTM
recent_macro_data = np.load('data/lstm_X.npy')[-1:] 

def _parse_money(money_str):
    return float(money_str.replace('$', '').replace(',', '').strip())


def _parse_percent(percent_str):
    return float(percent_str.replace('%', '').strip())


def run_richardson_user_demo(analyzer_obj, macro_sequence, users, horizons):
    results = []

    print("\n====================================================================")
    print("RICHARDSON USER DEMO: USER INPUTS + MODEL OUTPUTS (3M/6M/12M)")
    print("====================================================================")

    for user in users:
        home_features = user['home_features']
        rate = user['interest_rate']

        print(f"\nUser: {user['user_id']} ({user['name']})")
        print("Input (user-provided):")
        print(
            f"  Area: Richardson | SqFt: {home_features['SQUARE FEET']} | "
            f"Lot: {home_features['LOT SIZE']} | Beds: {home_features['BEDS']} | "
            f"Baths: {home_features['BATHS']} | Age: {home_features['PROPERTY_AGE']} | "
            f"HOA: {home_features['HOA/MONTH']} | Search Rate: {rate:.2f}%"
        )

        for months in horizons:
            output = analyzer_obj.generate_final_valuation(
                home_features,
                macro_sequence,
                months_in_future=months,
                interest_rate=rate
            )

            baseline = _parse_money(output["Baseline Value (Today)"])
            shift_pct = _parse_percent(output[f"Forecasted Market Shift ({months} Months)"])
            final_value = _parse_money(output["Final Future Valuation"])

            results.append({
                "UserID": user['user_id'],
                "Name": user['name'],
                "Months": months,
                "Rate": rate,
                "Baseline": baseline,
                "ShiftPct": shift_pct,
                "Final": final_value
            })

    return results


def print_user_results_table(results):
    print("\nOutput (predicted baseline price + 3/6/12 month shifts):")
    print(f"{'User':<6} {'Name':<14} {'Months':<7} {'Rate(%)':<8} {'Baseline($)':<14} {'Shift(%)':<9} {'Final($)':<14}")
    print("-" * 96)

    for row in results:
        print(
            f"{row['UserID']:<6} "
            f"{row['Name']:<14} "
            f"{row['Months']:<7} "
            f"{row['Rate']:<8.2f} "
            f"{row['Baseline']:<14,.2f} "
            f"{row['ShiftPct']:<9.2f} "
            f"{row['Final']:<14,.2f}"
        )


def run_sanity_checks(results):
    print("\nSanity Checks:")

    # 1) Baseline should remain stable per user across horizons at fixed search rate.
    for user_id in sorted(set(r["UserID"] for r in results)):
        user_rows = [r for r in results if r["UserID"] == user_id]
        baseline_values = [r["Baseline"] for r in user_rows]
        baseline_span = max(baseline_values) - min(baseline_values)
        if baseline_span < 1e-6:
            print(f"[PASS] {user_id} baseline is stable across horizons.")
        else:
            print(f"[WARN] {user_id} baseline changed across scenarios (span=${baseline_span:,.2f}).")

    # 2) For each user, longer horizons should usually have larger magnitude shifts.
    for user_id in sorted(set(r["UserID"] for r in results)):
        subset = sorted([r for r in results if r["UserID"] == user_id], key=lambda x: x["Months"])
        shifts = [r["ShiftPct"] for r in subset]
        monotonic_non_decreasing = all(shifts[i] <= shifts[i + 1] for i in range(len(shifts) - 1))
        if monotonic_non_decreasing:
            print(f"[PASS] {user_id}: shift grows across 3m/6m/12m horizons.")
        else:
            print(f"[WARN] {user_id}: shift is not monotonic across horizons.")


users_to_test = [
    {
        'user_id': 'U1',
        'name': 'Richardson Starter',
        'interest_rate': 6.10,
        'home_features': {
            'SQUARE FEET': 1400, 'LOT SIZE': 6200, 'BEDS': 3, 'BATHS': 2,
            'PROPERTY_AGE': 40, 'HOA/MONTH': 0, 'LATITUDE': 32.9600, 'LONGITUDE': -96.7200,
            'SEARCH_MONTH_SIN': 0.5, 'SEARCH_MONTH_COS': 0.866,
            'DISTANCE_TO_POI_SINGLE': 5.8, 'DISTANCE_TO_POI_MULTI_MIN': 2.0,
            'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.36, 'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 3.2,
            'POI_COUNT_WITHIN_1_MI': 1, 'POI_COUNT_WITHIN_3_MI': 3,
            'LOCAL_CRIME_INDEX_SIM': 42
        }
    },
    {
        'user_id': 'U2',
        'name': 'Family Upgrade',
        'interest_rate': 6.45,
        'home_features': {
            'SQUARE FEET': 2100, 'LOT SIZE': 7800, 'BEDS': 4, 'BATHS': 2.5,
            'PROPERTY_AGE': 25, 'HOA/MONTH': 35, 'LATITUDE': 32.9750, 'LONGITUDE': -96.7100,
            'SEARCH_MONTH_SIN': -0.5, 'SEARCH_MONTH_COS': 0.866,
            'DISTANCE_TO_POI_SINGLE': 4.3, 'DISTANCE_TO_POI_MULTI_MIN': 1.7,
            'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.48, 'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 2.7,
            'POI_COUNT_WITHIN_1_MI': 1, 'POI_COUNT_WITHIN_3_MI': 4,
            'LOCAL_CRIME_INDEX_SIM': 35
        }
    },
    {
        'user_id': 'U3',
        'name': 'Townhome Buyer',
        'interest_rate': 5.90,
        'home_features': {
            'SQUARE FEET': 1650, 'LOT SIZE': 2900, 'BEDS': 3, 'BATHS': 2.5,
            'PROPERTY_AGE': 12, 'HOA/MONTH': 180, 'LATITUDE': 32.9900, 'LONGITUDE': -96.7000,
            'SEARCH_MONTH_SIN': 0.0, 'SEARCH_MONTH_COS': 1.0,
            'DISTANCE_TO_POI_SINGLE': 3.6, 'DISTANCE_TO_POI_MULTI_MIN': 1.1,
            'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.62, 'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 2.1,
            'POI_COUNT_WITHIN_1_MI': 2, 'POI_COUNT_WITHIN_3_MI': 5,
            'LOCAL_CRIME_INDEX_SIM': 28
        }
    },
    {
        'user_id': 'U4',
        'name': 'Move-Up Buyer',
        'interest_rate': 6.70,
        'home_features': {
            'SQUARE FEET': 2800, 'LOT SIZE': 9100, 'BEDS': 4, 'BATHS': 3,
            'PROPERTY_AGE': 18, 'HOA/MONTH': 65, 'LATITUDE': 32.9450, 'LONGITUDE': -96.7300,
            'SEARCH_MONTH_SIN': -0.866, 'SEARCH_MONTH_COS': -0.5,
            'DISTANCE_TO_POI_SINGLE': 6.5, 'DISTANCE_TO_POI_MULTI_MIN': 2.4,
            'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.31, 'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 3.8,
            'POI_COUNT_WITHIN_1_MI': 0, 'POI_COUNT_WITHIN_3_MI': 2,
            'LOCAL_CRIME_INDEX_SIM': 39
        }
    },
    {
        'user_id': 'U5',
        'name': 'Luxury Relocator',
        'interest_rate': 6.25,
        'home_features': {
            'SQUARE FEET': 3600, 'LOT SIZE': 11500, 'BEDS': 5, 'BATHS': 4,
            'PROPERTY_AGE': 8, 'HOA/MONTH': 140, 'LATITUDE': 32.9700, 'LONGITUDE': -96.6900,
            'SEARCH_MONTH_SIN': 0.866, 'SEARCH_MONTH_COS': -0.5,
            'DISTANCE_TO_POI_SINGLE': 2.8, 'DISTANCE_TO_POI_MULTI_MIN': 0.9,
            'DISTANCE_TO_POI_MULTI_WEIGHTED': 0.77, 'DISTANCE_TO_POI_MULTI_MEAN_TOP_N': 1.8,
            'POI_COUNT_WITHIN_1_MI': 2, 'POI_COUNT_WITHIN_3_MI': 5,
            'LOCAL_CRIME_INDEX_SIM': 22
        }
    },
]

forecast_horizons = [3, 6, 12]
user_results = run_richardson_user_demo(
    analyzer,
    recent_macro_data,
    users_to_test,
    forecast_horizons
)

print_user_results_table(user_results)
run_sanity_checks(user_results)

print("\nDone. This output shows expected user inputs and resulting valuations/shifts for Richardson scenarios.")