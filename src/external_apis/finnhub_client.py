"""Finnhub client wrapper for fetching news sentiment.

This wrapper uses the Finnhub company-news endpoint when a `FINNHUB_SYMBOL`
is supplied. Configure via environment variables:

- FINNHUB_KEY: your Finnhub API key
- FINNHUB_SYMBOL: optional symbol to query company-news (e.g., a real-estate ETF)

If no `FINNHUB_SYMBOL` is supplied this client will return None (fall back
to other providers). The client returns a pandas DataFrame with columns:
`DATE`, `SENTIMENT_SCORE`, `SOURCE`, `ARTICLE_COUNT`.
"""
import os
import requests
from datetime import datetime, timedelta
import pandas as pd

FINNHUB_KEY = os.getenv('FINNHUB_KEY')
FINNHUB_SYMBOL = os.getenv('FINNHUB_SYMBOL')


def fetch_sentiment_time_series_finnhub(start_date, end_date, query=None):
    """Fetch weekly sentiment via Finnhub using company-news for FINNHUB_SYMBOL.

    Returns DataFrame or None if not configured.
    """
    # Read keys at call time so runtime env changes (tests) take effect
    key = os.getenv('FINNHUB_KEY')
    symbol = os.getenv('FINNHUB_SYMBOL')
    if not key or not symbol:
        return None

    base = 'https://finnhub.io/api/v1/company-news'
    current = start_date
    sentiments = []

    pos_keywords = ['growth', 'surge', 'boom', 'strong', 'rise', 'increase', 'positive', 'up']
    neg_keywords = ['decline', 'fall', 'risk', 'drop', 'crash', 'collapse', 'negative', 'down']

    while current <= end_date:
        to_date = current + timedelta(days=7)
        params = {
            'symbol': symbol,
            'from': current.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': key
        }
        try:
            resp = requests.get(base, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json() or []

            pos_count = 0
            neg_count = 0
            for a in articles:
                title = (a.get('headline') or a.get('summary') or '').lower()
                for kw in pos_keywords:
                    if kw in title:
                        pos_count += 1
                for kw in neg_keywords:
                    if kw in title:
                        neg_count += 1

            total = pos_count + neg_count
            sentiment = (pos_count - neg_count) / total if total > 0 else 0.0

            sentiments.append({
                'DATE': current.strftime('%Y-%m-%d'),
                'SENTIMENT_SCORE': float(max(-1.0, min(1.0, sentiment))),
                'SOURCE': 'Finnhub',
                'ARTICLE_COUNT': len(articles)
            })

        except Exception:
            sentiments.append({
                'DATE': current.strftime('%Y-%m-%d'),
                'SENTIMENT_SCORE': 0.0,
                'SOURCE': 'Finnhub_error',
                'ARTICLE_COUNT': 0
            })

        current = to_date

    df = pd.DataFrame(sentiments)
    return df
