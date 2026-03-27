import numpy as np
from src.hybrid_engine import RealEstateAnalyzer

print("Initializing Hybrid Architecture for Professor Demo...\n")
analyzer = RealEstateAnalyzer()

# Grab the latest macro sequence for the LSTM
recent_macro_data = np.load('data/lstm_X.npy')[-1:] 

# --- SAMPLE DATA 1: The Starter Home ---
house_1 = {
    'SQUARE FEET': 1200, 'LOT SIZE': 4000, 'BEDS': 3, 'BATHS': 2, 
    'PROPERTY_AGE': 45, 'HOA/MONTH': 0, 'LATITUDE': 32.7, 'LONGITUDE': -96.8, 
    'SEARCH_MONTH_SIN': 0.5, 'SEARCH_MONTH_COS': 0.866, # Spring search
    'DISTANCE_TO_POI': 25.0, 'LOCAL_CRIME_INDEX': 65 # Far away, higher crime
}

# --- SAMPLE DATA 2: The Luxury Estate ---
house_2 = {
    'SQUARE FEET': 4500, 'LOT SIZE': 12000, 'BEDS': 5, 'BATHS': 4.5, 
    'PROPERTY_AGE': 2, 'HOA/MONTH': 250, 'LATITUDE': 33.1, 'LONGITUDE': -96.8, 
    'SEARCH_MONTH_SIN': -1.0, 'SEARCH_MONTH_COS': 0.0, # Peak Summer search
    'DISTANCE_TO_POI': 3.0, 'LOCAL_CRIME_INDEX': 15 # Very close, low crime
}

def _parse_money(money_str):
    return float(money_str.replace('$', '').replace(',', '').strip())


def _parse_percent(percent_str):
    return float(percent_str.replace('%', '').strip())


def run_scenario_grid(analyzer_obj, macro_sequence, homes, horizons, interest_rates):
    results = []

    print("\n==============================================================")
    print("SCENARIO GRID TEST: HOME x FORECAST HORIZON x INTEREST RATE")
    print("==============================================================")

    for home_name, home_features in homes.items():
        for months in horizons:
            for rate in interest_rates:
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
                    "Home": home_name,
                    "Months": months,
                    "Rate": rate,
                    "Baseline": baseline,
                    "ShiftPct": shift_pct,
                    "Final": final_value
                })

    return results


def print_results_table(results):
    print("\nScenario Results:")
    print(f"{'Home':<14} {'Months':<7} {'Rate(%)':<8} {'Baseline($)':<14} {'Shift(%)':<9} {'Final($)':<14}")
    print("-" * 74)

    for row in results:
        print(
            f"{row['Home']:<14} "
            f"{row['Months']:<7} "
            f"{row['Rate']:<8.2f} "
            f"{row['Baseline']:<14,.2f} "
            f"{row['ShiftPct']:<9.2f} "
            f"{row['Final']:<14,.2f}"
        )


def run_sanity_checks(results):
    print("\nSanity Checks:")

    # 1) Baseline should remain stable per home across macro scenarios.
    for home in sorted(set(r["Home"] for r in results)):
        home_rows = [r for r in results if r["Home"] == home]
        baseline_values = [r["Baseline"] for r in home_rows]
        baseline_span = max(baseline_values) - min(baseline_values)
        if baseline_span < 1e-6:
            print(f"[PASS] {home} baseline is stable across macro scenarios.")
        else:
            print(f"[WARN] {home} baseline changed across scenarios (span=${baseline_span:,.2f}).")

    # 2) For each home and horizon, higher rates should not produce higher shift than low rates.
    for home in sorted(set(r["Home"] for r in results)):
        for months in sorted(set(r["Months"] for r in results)):
            subset = [r for r in results if r["Home"] == home and r["Months"] == months]
            subset = sorted(subset, key=lambda x: x["Rate"])
            shifts = [r["ShiftPct"] for r in subset]

            monotonic_non_increasing = all(shifts[i] >= shifts[i + 1] for i in range(len(shifts) - 1))
            if monotonic_non_increasing:
                print(f"[PASS] {home}, {months}m: shift decreases as rates rise.")
            else:
                print(f"[WARN] {home}, {months}m: shift is not monotonic vs rates.")


homes_to_test = {
    "StarterHome": house_1,
    "LuxuryEstate": house_2
}

forecast_horizons = [3, 6, 12]
interest_rate_scenarios = [4.5, 6.0, 7.5]

grid_results = run_scenario_grid(
    analyzer,
    recent_macro_data,
    homes_to_test,
    forecast_horizons,
    interest_rate_scenarios
)

print_results_table(grid_results)
run_sanity_checks(grid_results)

print("\nDone. Use this grid output as a quick architecture smoke test.")