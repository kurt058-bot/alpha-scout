"""
Standalone diagnostic script — tests FMP endpoints that might fill the
AUM / expense ratio / dividend yield gaps in the ETF Lab, with FULL response
visibility (status code + raw body for every call, success or failure).

Run this directly:  python test_fmp_etf_endpoints.py

This is intentionally separate from macro_server.py so you can get fast,
isolated answers without restarting the whole app or wading through
scanner/macro background-thread log noise.
"""
import requests
import json

API_KEY = 'rBiufns7lKMOcoPVxsWdvgD78JkZfDYS'
TEST_TICKERS = ['SCHD', 'SPY']  # SPY as a well-covered sanity-check ticker

ENDPOINTS_TO_TEST = [
    ('ETF & Mutual Fund Info',  'https://financialmodelingprep.com/stable/etf/info'),
    ('Ratios TTM (known 402)',  'https://financialmodelingprep.com/stable/ratios-ttm'),
    ('Company Profile',        'https://financialmodelingprep.com/stable/profile'),
    ('Key Metrics TTM',        'https://financialmodelingprep.com/stable/key-metrics-ttm'),
]

def test_endpoint(label, base_url, ticker):
    url = f'{base_url}?symbol={ticker}&apikey={API_KEY}'
    print(f'\n{"="*70}')
    print(f'{label} — {ticker}')
    print(f'URL: {base_url}?symbol={ticker}&apikey=***')
    try:
        r = requests.get(url, timeout=10)
        print(f'Status: {r.status_code}')
        try:
            body = r.json()
            print(f'Body (JSON): {json.dumps(body, indent=2)[:1500]}')
        except Exception:
            print(f'Body (raw, not JSON): {r.text[:500]}')
    except Exception as e:
        print(f'REQUEST FAILED: {e}')

if __name__ == '__main__':
    print('FMP Endpoint Diagnostic — testing potential AUM/expense-ratio/dividend fallback sources')
    print(f'Using API key ending in: ...{API_KEY[-6:]}')
    for label, base_url in ENDPOINTS_TO_TEST:
        for ticker in TEST_TICKERS:
            test_endpoint(label, base_url, ticker)
    print(f'\n{"="*70}')
    print('DONE. Paste this full output back for diagnosis.')
