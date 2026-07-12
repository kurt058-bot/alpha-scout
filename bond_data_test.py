"""
Bond Fund Data Test — checks what yfinance actually returns for bond ETFs
Run this on your machine (not the sandbox) since it needs real Yahoo access.

    python bond_data_test.py TLT
    python bond_data_test.py BND
    python bond_data_test.py TLTW
"""

import sys
import yfinance as yf


def test_ticker(ticker):
    print(f"\n{'='*60}\nTesting: {ticker}\n{'='*60}")
    try:
        t = yf.Ticker(ticker)
        fd = t.funds_data

        print("\n--- asset_classes ---")
        try:
            print(fd.asset_classes)
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- bond_ratings ---")
        try:
            print(fd.bond_ratings)
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- bond_holdings ---")
        try:
            print(fd.bond_holdings)
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- fund_overview (duration/maturity sometimes lives here) ---")
        try:
            print(fd.fund_overview)
        except Exception as e:
            print(f"  Error: {e}")

        print("\n--- .info yield/duration fields ---")
        try:
            info = t.info
            relevant_keys = ['yield', 'dividendYield', 'trailingAnnualDividendYield',
                            'thirtyDayAverageVolume', 'category', 'fundFamily']
            for k in relevant_keys:
                if k in info:
                    print(f"  {k}: {info[k]}")
        except Exception as e:
            print(f"  Error: {e}")

    except Exception as e:
        print(f"OVERALL ERROR for {ticker}: {e}")


if __name__ == '__main__':
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['TLT', 'BND', 'TLTW']
    for t in tickers:
        test_ticker(t)
