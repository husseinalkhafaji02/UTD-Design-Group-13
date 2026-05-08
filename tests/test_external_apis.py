import sys
import os
from datetime import datetime

# Ensure src/ is importable when running tests from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest.mock import patch, Mock

import external_apis.rapidapi_crime as rapidapi_crime
import external_apis.finnhub_client as finnhub_client


def test_fetch_crime_via_rapidapi_no_config(monkeypatch):
    # Ensure environment keys are not present
    monkeypatch.delenv('RAPIDAPI_KEY', raising=False)
    monkeypatch.delenv('RAPIDAPI_HOST', raising=False)
    monkeypatch.delenv('RAPIDAPI_CRIME_URL', raising=False)

    # Reload module constants to reflect environment (optional safe check)
    # The function should return None if not configured
    assert rapidapi_crime.fetch_crime_via_rapidapi('75001') is None


def test_fetch_crime_via_rapidapi_success(monkeypatch):
    monkeypatch.setenv('RAPIDAPI_KEY', 'testkey')
    monkeypatch.setenv('RAPIDAPI_HOST', 'testhost')
    monkeypatch.setenv('RAPIDAPI_CRIME_URL', 'https://api.example.com/crime/{zip}')

    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {'crime_index': 72}

    with patch('external_apis.rapidapi_crime.requests.get', return_value=mock_resp):
        val = rapidapi_crime.fetch_crime_via_rapidapi('75001')
        assert val == 72


def test_fetch_sentiment_time_series_finnhub(monkeypatch):
    # Configure Finnhub env
    monkeypatch.setenv('FINNHUB_KEY', 'fakekey')
    monkeypatch.setenv('FINNHUB_SYMBOL', 'TESTSYM')

    # One week of articles: one positive, one negative
    articles = [
        {'headline': 'Market shows strong growth this quarter'},
        {'headline': 'Concerns about decline and risk in housing'}
    ]

    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = articles

    with patch('external_apis.finnhub_client.requests.get', return_value=mock_resp):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 8)
        df = finnhub_client.fetch_sentiment_time_series_finnhub(start, end)

        assert df is not None
        assert list(df.columns) == ['DATE', 'SENTIMENT_SCORE', 'SOURCE', 'ARTICLE_COUNT']
        assert int(df.iloc[0]['ARTICLE_COUNT']) == 2
        # With 1 pos and 1 neg, sentiment should be 0.0
        assert float(df.iloc[0]['SENTIMENT_SCORE']) == 0.0
