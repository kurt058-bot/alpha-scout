"""
Standalone diagnostic script — tests Finnhub's ETF Profile endpoint for
AUM and expense ratio, since their docs suggest premium only adds extra
disclosure detail (not gating the core profile fields themselves) —
different in kind from FMP/EODHD/API Ninjas, which all gated these fields
entirely on the free tier. Prints the FULL raw response for confirmation.

Run this directly:  python test_finnhub_etf_endpoint.py
"""
import requests
import json

API_KEY = 'd8d4qc9r01qub2puhi4gd8d4qc9r01qub2puhi50'  # your Finnhub key
TEST_TICKERS = ['SCHD', 'AIS', 'SPY']  # SPY as a well-covered sanity check

def test_ticker(ticker):
    url = 'https://finnhub.io/api/v1/etf/profile'
    params = {'symbol': ticker, 'token': API_KEY}
    print(f'\n{"="*70}')
    print(f'Testing: {ticker}')
    print(f'URL: {url}?symbol={ticker}&token=***')
    try:
        r = requests.get(url, params=params, timeout=15)
        print(f'Status: {r.status_code}')
        try:
            body = r.json()
            print(f'Full response:\n{json.dumps(body, indent=2)[:2000]}')

            if isinstance(body, dict):
                print(f'\n--- Fields Alpha Scout needs ---')
                print(f'Name: {body.get("name")}')
                print(f'AUM: {body.get("aum")}')
                print(f'Expense Ratio: {body.get("expenseRatio")}')
                print(f'Yield: {body.get("yield")}')
                print(f'Inception Date: {body.get("inceptionDate")}')
        except Exception as je:
            print(f'Body is not valid JSON ({je}). Raw text: {r.text[:500]}')
    except Exception as e:
        print(f'REQUEST FAILED: {e}')

if __name__ == '__main__':
    print('Finnhub ETF Profile Diagnostic — checking AUM/expense-ratio coverage on free tier')
    print(f'Using API key ending in: ...{API_KEY[-6:]}')
    for ticker in TEST_TICKERS:
        test_ticker(ticker)
    print(f'\n{"="*70}')
    print('DONE. Paste this full output back for diagnosis.')
