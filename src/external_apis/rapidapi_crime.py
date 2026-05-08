"""RapidAPI wrapper for crime data providers.
This module provides a thin adapter that calls a configurable RapidAPI endpoint
to fetch crime information by ZIP code. Configure via environment variables:

- RAPIDAPI_KEY: your RapidAPI key
- RAPIDAPI_HOST: rapidapi host header (provider-specific)
- RAPIDAPI_CRIME_URL: full URL to the provider's crime endpoint, with a placeholder
  for the ZIP code expressed as `{zip}` or using query param `zip`.

The wrapper returns a normalized crime index 0-100 (higher = safer), or None
if the call cannot be made or the response is unexpected.
"""
import os
import requests

RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST')
RAPIDAPI_CRIME_URL = os.getenv('RAPIDAPI_CRIME_URL')


def fetch_crime_via_rapidapi(zip_code, timeout=10):
    """Fetch crime index for a ZIP code using a configured RapidAPI provider.

    Returns integer 0-100 or None on failure.
    """
    # Read environment at call time to support runtime test monkeypatching
    key = os.getenv('RAPIDAPI_KEY')
    host = os.getenv('RAPIDAPI_HOST')
    url_template = os.getenv('RAPIDAPI_CRIME_URL')
    if not (key and host and url_template):
        return None

    headers = {
        'x-rapidapi-key': key,
        'x-rapidapi-host': host,
        'Accept': 'application/json'
    }

    try:
        # If the URL contains a {zip} placeholder, format it; otherwise send as query param
        if '{zip}' in url_template:
            url = url_template.format(zip=zip_code)
            resp = requests.get(url, headers=headers, timeout=timeout)
        else:
            resp = requests.get(url_template, headers=headers, params={'zip': zip_code}, timeout=timeout)

        resp.raise_for_status()
        data = resp.json()

        # Normalization heuristics for common provider responses:
        # - If provider returns a numeric "crime_index" or "safety_index"
        # - If provider returns an "incidents" array, derive index from count
        if isinstance(data, dict):
            if 'crime_index' in data:
                return int(max(0, min(100, data['crime_index'])))
            if 'safety_index' in data:
                return int(max(0, min(100, data['safety_index'])))
            if 'incidents' in data and isinstance(data['incidents'], (list, tuple)):
                # fewer incidents -> higher safety score
                count = len(data['incidents'])
                score = max(0, min(100, 80 - count))
                return int(score)

        # Unknown response shape
        return None

    except Exception:
        return None
