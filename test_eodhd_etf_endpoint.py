"""
Standalone diagnostic script — tests EODHD's Fundamentals API for the exact
fields Alpha Scout's ETF Lab needs as fallbacks: AUM, expense ratio,
dividend yield, beta, 52-week high/low. Prints the FULL raw response so we
can see field names and values directly, not just a success/fail status.

Run this directly:  python test_eodhd_etf_endpoint.py
"""
import requests
import json

API_TOKEN = '6a49b5df837018.52615689'
TEST_TICKERS = ['SCHD.US', 'AIS.US', 'SPY.US']  # SPY as a well-covered sanity check

def test_ticker(ticker):
    url = f'https://eodhd.com/api/fundamentals/{ticker}'
    params = {
        'filter': 'General,ETF_Data,Technicals',
        'api_token': API_TOKEN,
        'fmt': 'json',
    }
    print(f'\n{"="*70}')
    print(f'Testing: {ticker}')
    print(f'URL: {url}?filter=General,ETF_Data,Technicals&api_token=***&fmt=json')
    try:
        r = requests.get(url, params=params, timeout=15)
        print(f'Status: {r.status_code}')
        try:
            body = r.json()
            print(f'Full response:\n{json.dumps(body, indent=2)}')

            # Pull out exactly the fields we care about, if present
            print(f'\n--- Fields Alpha Scout needs ---')
            general = body.get('General', {})
            etf_data = body.get('ETF_Data', {})
            technicals = body.get('Technicals', {})
            print(f'Name: {general.get("Name")}')
            print(f'AUM (TotalAssets): {etf_data.get("TotalAssets")}')
            print(f'Expense Ratio (NetExpenseRatio): {etf_data.get("NetExpenseRatio")}')
            print(f'Dividend Yield: {etf_data.get("Yield")}')
            print(f'Beta: {technicals.get("Beta")}')
            print(f'52-Wk High: {technicals.get("52WeekHigh")}')
            print(f'52-Wk Low: {technicals.get("52WeekLow")}')
        except Exception as je:
            print(f'Body is not valid JSON ({je}). Raw text: {r.text[:500]}')
    except Exception as e:
        print(f'REQUEST FAILED: {e}')

if __name__ == '__main__':
    print('EODHD Fundamentals Diagnostic — testing AUM/expense-ratio/dividend/beta fallback coverage')
    print(f'Using API token ending in: ...{API_TOKEN[-6:]}')
    for ticker in TEST_TICKERS:
        test_ticker(ticker)
    print(f'\n{"="*70}')
    print('DONE. Paste this full output back for diagnosis.')
