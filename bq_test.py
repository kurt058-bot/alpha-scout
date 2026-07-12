"""
Business Quant Dividends API test — confirms the key works and shows the
real response shape before we wire it into the main server.

Run this on your machine:
    python bq_test.py AAPL
    python bq_test.py SPYI
    python bq_test.py CGDV
"""
import sys
import requests
import json

API_KEY = "552aca2c9d930d436cecd684f1e9dc54"
BASE_URL = "https://data.businessquant.com"


def test_dividends(ticker):
    print(f"\n{'='*60}\nTesting dividends for: {ticker}\n{'='*60}")
    url = f"{BASE_URL}/dividends"
    params = {"ticker": ticker, "mode": "dps", "api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        print(f"Status code: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print("\nMetadata:")
            print(json.dumps(data.get("metadata", {}), indent=2))
            print(f"\nFirst 5 dividend records:")
            for rec in data.get("data", [])[:5]:
                print(" ", rec)
            print(f"\nTotal records: {len(data.get('data', []))}")
        else:
            print("Response body:", r.text[:500])
    except Exception as e:
        print(f"ERROR: {e}")


def test_funds_profile(ticker):
    print(f"\n{'='*60}\nTesting funds-profile for: {ticker}\n{'='*60}")
    url = f"{BASE_URL}/funds-profile"
    params = {"ticker": ticker, "api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        print(f"Status code: {r.status_code}")
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2)[:1500])
        else:
            print("Response body:", r.text[:500])
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "SPYI"]
    for t in tickers:
        test_dividends(t)
        test_funds_profile(t)
