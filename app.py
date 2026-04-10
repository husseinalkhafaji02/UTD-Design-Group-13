from flask import Flask, request, render_template_string
import numpy as np
from math import sin, cos, pi
from datetime import datetime

from src.hybrid_engine import RealEstateAnalyzer
from src.data_preprocessing import haversine_distance, fetch_local_crime_index

app = Flask(__name__)

# loading models
analyzer = RealEstateAnalyzer()
LATEST_MACRO_SEQUENCE = np.load('data/lstm_X.npy')[-1:]

# poi used for preproccessing pipeline
DEFAULT_POI_LAT = 32.9866
DEFAULT_POI_LON = -96.7503

INPUT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Real Estate Analyzer - Input</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            margin: 0;
            padding: 32px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }
        h1 { margin-top: 0; }
        p { color: #444; }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        .field {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 6px;
        }
        input, select {
            padding: 12px;
            border: 1px solid #d0d7e2;
            border-radius: 10px;
            font-size: 14px;
        }
        .actions {
            margin-top: 24px;
        }
        button {
            background: #1f6feb;
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 10px;
            font-size: 15px;
            cursor: pointer;
        }
        .hint {
            font-size: 13px;
            color: #666;
            margin-top: 18px;
        }
        .error {
            background: #fff1f1;
            color: #9f1d1d;
            border: 1px solid #f3c2c2;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 16px;
        }
        @media (max-width: 700px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real Estate Analyzer</h1>
        <p>Enter one house, then the app will generate the same scenario grid your demo script currently produces for different forecast horizons and interest rates.</p>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <form method="post" action="/predict">
            <div class="grid">
                <div class="field">
                    <label for="square_feet">Square Feet</label>
                    <input type="number" step="any" name="square_feet" id="square_feet" required value="{{ values.square_feet if values else '' }}">
                </div>
                <div class="field">
                    <label for="lot_size">Lot Size</label>
                    <input type="number" step="any" name="lot_size" id="lot_size" required value="{{ values.lot_size if values else '' }}">
                </div>
                <div class="field">
                    <label for="beds">Beds</label>
                    <input type="number" step="any" name="beds" id="beds" required value="{{ values.beds if values else '' }}">
                </div>
                <div class="field">
                    <label for="baths">Baths</label>
                    <input type="number" step="any" name="baths" id="baths" required value="{{ values.baths if values else '' }}">
                </div>
                <div class="field">
                    <label for="year_built">Year Built</label>
                    <input type="number" name="year_built" id="year_built" required value="{{ values.year_built if values else '' }}">
                </div>
                <div class="field">
                    <label for="hoa_month">HOA / Month</label>
                    <input type="number" step="any" name="hoa_month" id="hoa_month" required value="{{ values.hoa_month if values else '0' }}">
                </div>
                <div class="field">
                    <label for="latitude">Latitude</label>
                    <input type="number" step="any" name="latitude" id="latitude" required value="{{ values.latitude if values else '' }}">
                </div>
                <div class="field">
                    <label for="longitude">Longitude</label>
                    <input type="number" step="any" name="longitude" id="longitude" required value="{{ values.longitude if values else '' }}">
                </div>
                <div class="field">
                    <label for="search_month">Search Month</label>
                    <select name="search_month" id="search_month" required>
                        {% for month_num, month_name in months %}
                        <option value="{{ month_num }}" {% if values and values.search_month|int == month_num %}selected{% endif %}>{{ month_name }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="actions">
                <button type="submit">Generate Scenario Table</button>
            </div>
        </form>

        <div class="hint">
            The app computes property age, distance to the project point-of-interest, crime index, and seasonal month features automatically.
        </div>
    </div>
</body>
</html>
"""

OUTPUT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Real Estate Analyzer - Output</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            margin: 0;
            padding: 32px;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }
        h1, h2 { margin-top: 0; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background: white;
        }
        th, td {
            padding: 12px;
            border-bottom: 1px solid #e5eaf1;
            text-align: left;
        }
        th {
            background: #eef4ff;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        .card {
            background: #f8fbff;
            border: 1px solid #d9e7ff;
            border-radius: 12px;
            padding: 16px;
        }
        .label { color: #555; font-size: 13px; }
        .value { font-size: 22px; font-weight: 700; margin-top: 6px; }
        .actions { margin-top: 24px; }
        a.button {
            display: inline-block;
            text-decoration: none;
            background: #1f6feb;
            color: white;
            padding: 12px 18px;
            border-radius: 10px;
        }
        @media (max-width: 700px) {
            .summary { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Scenario Results for One House</h1>
        <p>The table below applies the same forecast grid as your demo: 3, 6, and 12 months across 4.5%, 6.0%, and 7.5% interest-rate scenarios.</p>

        <div class="summary">
            <div class="card">
                <div class="label">Baseline Value (same across scenarios)</div>
                <div class="value">${{ "{:,.2f}".format(baseline) }}</div>
            </div>
            <div class="card">
                <div class="label">Lowest Scenario</div>
                <div class="value">${{ "{:,.2f}".format(lowest_final) }}</div>
            </div>
            <div class="card">
                <div class="label">Highest Scenario</div>
                <div class="value">${{ "{:,.2f}".format(highest_final) }}</div>
            </div>
        </div>

        <h2>Scenario Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Months</th>
                    <th>Rate (%)</th>
                    <th>Baseline ($)</th>
                    <th>Shift (%)</th>
                    <th>Final ($)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row['Months'] }}</td>
                    <td>{{ "{:.2f}".format(row['Rate']) }}</td>
                    <td>{{ "{:,.2f}".format(row['Baseline']) }}</td>
                    <td>{{ "{:.2f}".format(row['ShiftPct']) }}</td>
                    <td>{{ "{:,.2f}".format(row['Final']) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2 style="margin-top: 28px;">Submitted House Details</h2>
        <table>
            <tbody>
                {% for label, value in submitted_fields %}
                <tr>
                    <th style="width: 40%;">{{ label }}</th>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="actions">
            <a href="/" class="button">Analyze Another House</a>
        </div>
    </div>
</body>
</html>
"""


def _parse_money(money_str: str) -> float:
    return float(money_str.replace('$', '').replace(',', '').strip())



def _parse_percent(percent_str: str) -> float:
    return float(percent_str.replace('%', '').strip())



def build_house_features(form_data):
    current_year = datetime.now().year

    square_feet = float(form_data['square_feet'])
    lot_size = float(form_data['lot_size'])
    beds = float(form_data['beds'])
    baths = float(form_data['baths'])
    year_built = int(form_data['year_built'])
    hoa_month = float(form_data['hoa_month'])
    latitude = float(form_data['latitude'])
    longitude = float(form_data['longitude'])
    search_month = int(form_data['search_month'])

    property_age = current_year - year_built
    search_month_sin = sin(2 * pi * search_month / 12)
    search_month_cos = cos(2 * pi * search_month / 12)
    distance_to_poi = haversine_distance(latitude, longitude, DEFAULT_POI_LAT, DEFAULT_POI_LON)
    local_crime_index = fetch_local_crime_index(latitude, longitude)

    house_features = {
        'SQUARE FEET': square_feet,
        'LOT SIZE': lot_size,
        'BEDS': beds,
        'BATHS': baths,
        'PROPERTY_AGE': property_age,
        'HOA/MONTH': hoa_month,
        'LATITUDE': latitude,
        'LONGITUDE': longitude,
        'SEARCH_MONTH_SIN': search_month_sin,
        'SEARCH_MONTH_COS': search_month_cos,
        'DISTANCE_TO_POI': distance_to_poi,
        'LOCAL_CRIME_INDEX': local_crime_index,
    }

    submitted_fields = [
        ('Square Feet', square_feet),
        ('Lot Size', lot_size),
        ('Beds', beds),
        ('Baths', baths),
        ('Year Built', year_built),
        ('Property Age', property_age),
        ('HOA / Month', hoa_month),
        ('Latitude', latitude),
        ('Longitude', longitude),
        ('Search Month', search_month),
        ('Distance to POI', f'{distance_to_poi:.2f} miles'),
        ('Local Crime Index', local_crime_index),
    ]

    return house_features, submitted_fields



def run_scenario_grid_for_one_house(house_features):
    horizons = [3, 6, 12]
    interest_rates = [4.5, 6.0, 7.5]
    results = []

    for months in horizons:
        for rate in interest_rates:
            output = analyzer.generate_final_valuation(
                house_features,
                LATEST_MACRO_SEQUENCE,
                months_in_future=months,
                interest_rate=rate
            )

            baseline = _parse_money(output['Baseline Value (Today)'])
            shift_pct = _parse_percent(output[f'Forecasted Market Shift ({months} Months)'])
            final_value = _parse_money(output['Final Future Valuation'])

            results.append({
                'Months': months,
                'Rate': rate,
                'Baseline': baseline,
                'ShiftPct': shift_pct,
                'Final': final_value,
            })

    return results


@app.route('/', methods=['GET'])
def index():
    months = [
        (1, 'January'), (2, 'February'), (3, 'March'), (4, 'April'),
        (5, 'May'), (6, 'June'), (7, 'July'), (8, 'August'),
        (9, 'September'), (10, 'October'), (11, 'November'), (12, 'December')
    ]
    return render_template_string(INPUT_TEMPLATE, months=months, error=None, values=None)


@app.route('/predict', methods=['POST'])
def predict():
    months = [
        (1, 'January'), (2, 'February'), (3, 'March'), (4, 'April'),
        (5, 'May'), (6, 'June'), (7, 'July'), (8, 'August'),
        (9, 'September'), (10, 'October'), (11, 'November'), (12, 'December')
    ]

    try:
        house_features, submitted_fields = build_house_features(request.form)
        results = run_scenario_grid_for_one_house(house_features)

        baseline = results[0]['Baseline'] if results else 0.0
        lowest_final = min(r['Final'] for r in results)
        highest_final = max(r['Final'] for r in results)

        return render_template_string(
            OUTPUT_TEMPLATE,
            results=results,
            baseline=baseline,
            lowest_final=lowest_final,
            highest_final=highest_final,
            submitted_fields=submitted_fields,
        )
    except Exception as e:
        return render_template_string(
            INPUT_TEMPLATE,
            months=months,
            error=f'Could not generate prediction: {e}',
            values=request.form,
        )


if __name__ == '__main__':
    app.run(debug=True)
