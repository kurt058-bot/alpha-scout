"""
Standalone diagnostic script — tests API Ninjas' ETF API for AUM and
expense ratio, since third-party documentation suggests these fields
return null on the free tier even though they appear in the schema.
Prints the FULL raw response so we can see exactly what comes back.

Run this directly:  python test_api_ninjas_etf.py
"""
import requests
import json

API_KEY = 'q7Wi1JyEvNZg7eeYgfIcyWagWHojYgsSs8NDu4Sl'
TEST_TICKERS = ['SCHD', 'AIS', 'SPY']  # SPY as a well-covered sanity check

def test_ticker(ticker):
    url = 'https://api.api-ninjas.com/v1/etf'
    headers = {'X-Api-Key': API_KEY}
    params = {'ticker': ticker}
    print(f'\n{"="*70}')
    print(f'Testing: {ticker}')
    print(f'URL: {url}?ticker={ticker}')
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        print(f'Status: {r.status_code}')
        try:
            body = r.json()
            print(f'Full response:\n{json.dumps(body, indent=2)[:2000]}')

            # Pull out exactly the fields we care about, if present
            if isinstance(body, dict):
                print(f'\n--- Fields Alpha Scout needs ---')
                print(f'Name: {body.get("name")}')
                print(f'AUM: {body.get("aum")}')
                print(f'Expense Ratio: {body.get("expense_ratio")}')
                print(f'Num Holdings: {body.get("num_holdings")}')
                print(f'Price: {body.get("price")}')
        except Exception as je:
            print(f'Body is not valid JSON ({je}). Raw text: {r.text[:500]}')
    except Exception as e:
        print(f'REQUEST FAILED: {e}')

if __name__ == '__main__':
    print('API Ninjas ETF Diagnostic — checking whether AUM/expense_ratio are null on free tier')
    print(f'Using API key ending in: ...{API_KEY[-6:]}')
    for ticker in TEST_TICKERS:
        test_ticker(ticker)
    print(f'\n{"="*70}')
    print('DONE. Paste this full output back for diagnosis.')
