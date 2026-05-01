from flask import Flask, request, render_template_string
import numpy as np
from math import sin, cos, pi
from datetime import datetime
from functools import lru_cache
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

from src.hybrid_engine import RealEstateAnalyzer
from src.data_preprocessing import haversine_distance, fetch_local_crime_index

app = Flask(__name__)
geocoder = Nominatim(user_agent='realestate_market_analyzer')

# load models
analyzer = RealEstateAnalyzer()
LATEST_MACRO_SEQUENCE = np.load('data/lstm_X.npy')[-1:]

# default poi used by pipeline
DEFAULT_POI_LAT = 32.9866
DEFAULT_POI_LON = -96.7503

INPUT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Real Estate Scenario Analyzer</title>
    <style>
        :root {
            --bg: #eef3fb;
            --card: #ffffff;
            --text: #172033;
            --muted: #64748b;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --line: #e2e8f0;
            --soft: #f8fafc;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 34%),
                linear-gradient(135deg, #eef3fb 0%, #f8fafc 100%);
            min-height: 100vh;
        }

        .shell {
            max-width: 1080px;
            margin: 0 auto;
            padding: 42px 22px;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.3fr 0.7fr;
            gap: 22px;
            align-items: stretch;
            margin-bottom: 24px;
        }

        .hero-card, .side-card, .form-card {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(226,232,240,0.9);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        }

        .hero-card { padding: 34px; }

        
        h1 {
            font-size: clamp(34px, 5vw, 54px);
            line-height: 1;
            margin: 0 0 14px;
            letter-spacing: -1.5px;
        }

        .lead {
            color: var(--muted);
            font-size: 17px;
            line-height: 1.65;
            max-width: 680px;
            margin: 0;
        }

        .side-card {
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: rgba(255,255,255,0.94);
            color: var(--text);
            overflow: hidden;
            position: relative;
        }


        .side-card h3 {
            margin: 0 0 10px;
            font-size: 18px;
        }

        .side-card p {
            color: var(--muted);
            line-height: 1.6;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .stat-row {
            display: grid;
            gap: 10px;
            position: relative;
            z-index: 1;
            margin-top: 22px;
        }

        .stat {
            background: var(--soft);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 14px;
        }

        .stat strong {
            display: block;
            font-size: 20px;
        }

        .stat span {
            color: var(--muted);
            font-size: 13px;
        }

        .form-card { padding: 28px; }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 18px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 14px;
        }

        .section-title h2 {
            margin: 0;
            font-size: 24px;
            letter-spacing: -0.5px;
        }

        .section-title span {
            color: var(--muted);
            font-size: 14px;
        }

        .form-section { margin-top: 24px; }

        .form-section h3 {
            margin: 0 0 14px;
            font-size: 16px;
            color: #334155;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
        }

        .grid .wide { grid-column: span 2; }

        .field {
            display: flex;
            flex-direction: column;
            gap: 7px;
        }

        label {
            font-weight: 800;
            font-size: 13px;
            color: #334155;
        }

        input, select {
            width: 100%;
            padding: 13px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            font-size: 15px;
            outline: none;
            background: white;
            transition: border-color 0.15s, box-shadow 0.15s;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
        }

        .actions {
            margin-top: 28px;
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 14px 20px;
            border-radius: 14px;
            font-size: 15px;
            font-weight: 900;
            cursor: pointer;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28);
        }

        button:hover { transform: translateY(-1px); }

        .hint {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.55;
        }

        .error {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 14px;
            border-radius: 16px;
            margin-bottom: 18px;
            font-weight: 700;
        }

        @media (max-width: 850px) {
            .hero { grid-template-columns: 1fr; }
            .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .grid .wide { grid-column: span 2; }
        }

        @media (max-width: 600px) {
            .shell { padding: 24px 14px; }
            .hero-card, .form-card { padding: 22px; }
            .grid { grid-template-columns: 1fr; }
            .grid .wide { grid-column: span 1; }
            .section-title { align-items: flex-start; flex-direction: column; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="hero-card">
                                <h1>Real Estate Scenario Analyzer</h1>
                <p class="lead">Enter one property address and house details, then generate a scenario table showing how its estimated value changes across forecast horizons and mortgage-rate assumptions.</p>
            </div>
            <aside class="side-card">
                <div>
                    <h3>What the system returns</h3>
                    <p>A baseline value, market shifts, and final future valuations for each scenario.</p>
                </div>
                <div class="stat-row">
                    <div class="stat"><strong>9</strong><span>Scenarios per house</span></div>
                    <div class="stat"><strong>3 / 6 / 12</strong><span>Forecast months</span></div>
                    <div class="stat"><strong>Address → Coordinates</strong><span>Geocoded automatically</span></div>
                </div>
            </aside>
        </section>

        <section class="form-card">
            <div class="section-title">
                <h2>Property Input</h2>
                <span>Coordinates and engineered features are calculated automatically.</span>
            </div>

            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}

            <form method="post" action="/predict">
                <div class="form-section">
                    <h3>Home Details</h3>
                    <div class="grid">
                        <div class="field">
                            <label for="square_feet">Square Feet</label>
                            <input type="number" step="any" name="square_feet" id="square_feet" required value="{{ values.square_feet if values else '' }}" placeholder="1200">
                        </div>
                        <div class="field">
                            <label for="lot_size">Lot Size</label>
                            <input type="number" step="any" name="lot_size" id="lot_size" required value="{{ values.lot_size if values else '' }}" placeholder="4000">
                        </div>
                        <div class="field">
                            <label for="year_built">Year Built</label>
                            <input type="number" name="year_built" id="year_built" required value="{{ values.year_built if values else '' }}" placeholder="1980">
                        </div>
                        <div class="field">
                            <label for="beds">Beds</label>
                            <input type="number" step="any" name="beds" id="beds" required value="{{ values.beds if values else '' }}" placeholder="3">
                        </div>
                        <div class="field">
                            <label for="baths">Baths</label>
                            <input type="number" step="any" name="baths" id="baths" required value="{{ values.baths if values else '' }}" placeholder="2">
                        </div>
                        <div class="field">
                            <label for="hoa_month">HOA / Month</label>
                            <input type="number" step="any" name="hoa_month" id="hoa_month" required value="{{ values.hoa_month if values else '0' }}" placeholder="0">
                        </div>
                    </div>
                </div>

                <div class="form-section">
                    <h3>Location & Timing</h3>
                    <div class="grid">
                        <div class="field wide">
                            <label for="address">Property Address</label>
                            <input type="text" name="address" id="address" required placeholder="123 Main St, Dallas, TX" value="{{ values.address if values else '' }}">
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
                </div>

                <div class="actions">
                    <button type="submit">Generate Scenario Table →</button>
                    <div class="hint">The address is converted into latitude/longitude, then the app calculates distance to POI, property age, crime index, and seasonal month features.</div>
                </div>
            </form>
        </section>
    </main>
</body>
</html>
"""

OUTPUT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Real Estate Scenario Results</title>
    <style>
        :root {
            --bg: #eef3fb;
            --card: #ffffff;
            --text: #172033;
            --muted: #64748b;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --line: #e2e8f0;
            --soft: #f8fafc;
            --good: #059669;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 34%),
                linear-gradient(135deg, #eef3fb 0%, #f8fafc 100%);
            min-height: 100vh;
        }

        .shell {
            max-width: 1180px;
            margin: 0 auto;
            padding: 42px 22px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 22px;
        }

        .title h1 {
            font-size: clamp(32px, 5vw, 48px);
            line-height: 1;
            letter-spacing: -1.2px;
            margin: 0 0 10px;
        }

        .title p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 13px 18px;
            border-radius: 14px;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
            white-space: nowrap;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 22px;
        }

        .card, .table-card {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(226,232,240,0.9);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        }

        .card {
            padding: 22px;
            position: relative;
            overflow: hidden;
        }

        .card::after {
            content: "";
            position: absolute;
            right: -24px;
            bottom: -24px;
            width: 90px;
            height: 90px;
            background: rgba(37, 99, 235, 0.08);
            border-radius: 50%;
        }

        .label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .value {
            margin-top: 8px;
            font-size: clamp(24px, 4vw, 34px);
            font-weight: 950;
            letter-spacing: -0.8px;
        }

        .subtext {
            margin-top: 8px;
            color: var(--muted);
            font-size: 14px;
        }

        .table-card {
            padding: 24px;
            margin-bottom: 22px;
        }

        .table-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 16px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 16px;
            margin-bottom: 10px;
        }

        .table-head h2 {
            margin: 0;
            font-size: 24px;
            letter-spacing: -0.5px;
        }

        .table-head p {
            margin: 6px 0 0;
            color: var(--muted);
            line-height: 1.5;
        }

        .pill {
            display: inline-flex;
            background: #dbeafe;
            color: #1e40af;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 13px;
            white-space: nowrap;
        }

        .table-wrap { overflow-x: auto; }

        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 720px;
        }

        th, td {
            padding: 15px 14px;
            text-align: left;
            border-bottom: 1px solid var(--line);
        }

        th {
            color: #334155;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: #f8fafc;
        }

        td { font-size: 15px; }

        tbody tr:hover { background: #f8fbff; }

        .money { font-weight: 900; }

        .shift {
            display: inline-flex;
            padding: 6px 10px;
            border-radius: 999px;
            background: #ecfdf5;
            color: #047857;
            font-weight: 900;
            font-size: 13px;
        }

        .details-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }

        .detail {
            background: var(--soft);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px;
        }

        .detail span {
            display: block;
            color: var(--muted);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }

        .detail strong {
            font-size: 16px;
            word-break: break-word;
        }

        @media (max-width: 850px) {
            .topbar { align-items: flex-start; flex-direction: column; }
            .summary { grid-template-columns: 1fr; }
            .details-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 600px) {
            .shell { padding: 24px 14px; }
            .table-card { padding: 18px; }
            .details-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <div class="topbar">
            <div class="title">
                <h1>Scenario Results</h1>
                <p>One property analyzed across 3 forecast horizons and 3 mortgage-rate assumptions.</p>
            </div>
            <a href="/" class="button">← Analyze Another House</a>
        </div>

        <section class="summary">
            <div class="card">
                <div class="label">Baseline Value</div>
                <div class="value">${{ "{:,.2f}".format(baseline) }}</div>
                <div class="subtext">Current estimate from XGBoost</div>
            </div>
            <div class="card">
                <div class="label">Lowest Scenario</div>
                <div class="value">${{ "{:,.2f}".format(lowest_final) }}</div>
                <div class="subtext">Lowest final valuation in the grid</div>
            </div>
            <div class="card">
                <div class="label">Highest Scenario</div>
                <div class="value">${{ "{:,.2f}".format(highest_final) }}</div>
                <div class="subtext">Highest final valuation in the grid</div>
            </div>
        </section>

        <section class="table-card">
            <div class="table-head">
                <div>
                    <h2>Forecast Scenario Table</h2>
                    <p>Final value equals baseline value adjusted by the LSTM market shift for each scenario.</p>
                </div>
                <span class="pill">9 scenarios</span>
            </div>

            <div class="table-wrap">
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
                            <td class="money">{{ "{:,.2f}".format(row['Baseline']) }}</td>
                            <td><span class="shift">{{ "{:.2f}".format(row['ShiftPct']) }}%</span></td>
                            <td class="money">{{ "{:,.2f}".format(row['Final']) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="table-card">
            <div class="table-head">
                <div>
                    <h2>Submitted Property Details</h2>
                    <p>These include user inputs plus address-derived and engineered features.</p>
                </div>
            </div>
            <div class="details-grid">
                {% for label, value in submitted_fields %}
                <div class="detail">
                    <span>{{ label }}</span>
                    <strong>{{ value }}</strong>
                </div>
                {% endfor %}
            </div>
        </section>
    </main>
</body>
</html>
"""


def _parse_money(money_str: str) -> float:
    return float(money_str.replace('$', '').replace(',', '').strip())


def _parse_percent(percent_str: str) -> float:
    return float(percent_str.replace('%', '').strip())


@lru_cache(maxsize=256)
def geocode_address(address: str):
    try:
        location = geocoder.geocode(address, timeout=10)
    except GeopyError as exc:
        raise ValueError(f'Address lookup failed: {exc}') from exc

    if not location:
        raise ValueError('Could not find that address. Try a more complete street/city/state format.')

    return float(location.latitude), float(location.longitude)


def build_house_features(form_data):
    current_year = datetime.now().year

    square_feet = float(form_data['square_feet'])
    lot_size = float(form_data['lot_size'])
    beds = float(form_data['beds'])
    baths = float(form_data['baths'])
    year_built = int(form_data['year_built'])
    hoa_month = float(form_data['hoa_month'])
    address = str(form_data['address']).strip()
    if not address:
        raise ValueError('Address is required.')

    latitude, longitude = geocode_address(address)
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
        ('Square Feet', f'{square_feet:,.0f}'),
        ('Lot Size', f'{lot_size:,.0f}'),
        ('Beds', beds),
        ('Baths', baths),
        ('Year Built', year_built),
        ('Property Age', property_age),
        ('HOA / Month', f'${hoa_month:,.2f}'),
        ('Address', address),
        ('Latitude', f'{latitude:.6f}'),
        ('Longitude', f'{longitude:.6f}'),
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
