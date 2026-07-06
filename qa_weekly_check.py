#!/usr/bin/env python3
"""
Alpha Scout — Weekly API QA Check
==================================
Runs live, timed checks against every external API the app depends on,
grouped by the page that uses them, and emails a pass/fail report.

Run manually:
    python qa_weekly_check.py

Run on a schedule:
  - Windows Task Scheduler: trigger weekly, Sunday 11:00 PM, action =
    `python C:\\Users\\kurth\\Downloads\\qa_weekly_check.py`
  - GitHub Actions: see .github/workflows/weekly_qa.yml (public APIs only —
    this script never needs your local machine to be on, since none of
    these endpoints are localhost-only).

Requires a .env file (see .env.example) with:
    FMP_API_KEY, FRED_API_KEY, UNSPLASH_ACCESS_KEY, ANTHROPIC_API_KEY,
    GMAIL_ADDRESS, GMAIL_APP_PASSWORD
"""
import os
import time
import smtplib
import traceback
from email.mime.text import MIMEText
from datetime import datetime

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GMAIL_ADDRESS = os.environ.get('GMAIL_ADDRESS', '')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', '')
REPORT_TO = os.environ.get('QA_REPORT_TO', 'kurt058@gmail.com')

SPOT_CHECK_TICKERS = ['SCHD', 'AIS', 'IWM', 'COWZ', 'CALF', 'DGRO', 'QMOM', 'VUG', 'VIG']

EDGAR_HEADERS = {'User-Agent': 'AlphaScout QA kurth-personal-use@example.com'}

results = []  # list of dicts: page, api, check, status, seconds, detail, fix_plan


def record(page, api, check, status, seconds, detail='', fix_plan=''):
    results.append({
        'page': page, 'api': api, 'check': check, 'status': status,
        'seconds': round(seconds, 2), 'detail': detail, 'fix_plan': fix_plan,
    })
    print(f'[{status}] {page} / {api} / {check} — {seconds:.2f}s — {detail}')


def timed(fn):
    t0 = time.time()
    try:
        out = fn()
        return True, out, time.time() - t0, ''
    except Exception as e:
        return False, None, time.time() - t0, f'{type(e).__name__}: {e}'


# ─────────────────────────────────────────────────────────────
# HOME PAGE — Unsplash hero image, headlines, market movers, earnings,
# AI beneficiaries/bottlenecks (Anthropic), channel episodes
# ─────────────────────────────────────────────────────────────
def check_home():
    page = 'Home'

    def _unsplash():
        r = requests.get('https://api.unsplash.com/search/photos',
                          params={'query': 'stock market', 'per_page': 1},
                          headers={'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'},
                          timeout=10)
        r.raise_for_status()
        data = r.json()
        assert data.get('results'), 'empty results array'
        return data
    ok, _, dt, err = timed(_unsplash)
    record(page, 'Unsplash', 'search/photos returns image', 'PASS' if ok else 'FAIL', dt,
           err or 'OK', '' if ok else 'Check UNSPLASH_ACCESS_KEY validity/rate limit; fall back to a static hero image if repeatedly failing.')

    def _anthropic():
        r = requests.post('https://api.anthropic.com/v1/messages',
                           headers={'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01',
                                    'content-type': 'application/json'},
                           json={'model': 'claude-sonnet-4-6', 'max_tokens': 16,
                                 'messages': [{'role': 'user', 'content': 'ping'}]},
                           timeout=15)
        r.raise_for_status()
        return r.json()
    ok, _, dt, err = timed(_anthropic)
    record(page, 'Anthropic', 'AI beneficiaries/bottlenecks generation', 'PASS' if ok else 'FAIL', dt,
           err or 'OK', '' if ok else 'Check ANTHROPIC_API_KEY validity/billing; this key is used for the AI Beneficiaries/Bottlenecks panels on Home.')

    def _yahoo_movers():
        t = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/SPY', timeout=10)
        t.raise_for_status()
        return t.json()
    ok, _, dt, err = timed(_yahoo_movers)
    record(page, 'Yahoo Finance', 'market movers price pull (SPY)', 'PASS' if ok else 'FAIL', dt,
           err or 'OK', '' if ok else 'Yahoo rate-limited — verifying FMP fallback below.')
    if not ok:
        def _fmp_spy():
            r = requests.get('https://financialmodelingprep.com/stable/profile',
                              params={'symbol': 'SPY', 'apikey': FMP_API_KEY}, timeout=10)
            r.raise_for_status()
            data = r.json()
            assert data and isinstance(data, list) and data[0].get('marketCap'), 'empty/placeholder profile'
            return data[0]
        fok, _, fdt, ferr = timed(_fmp_spy)
        record(page, 'FMP', 'SPY /stable/profile fallback (triggered by Yahoo failure)', 'PASS' if fok else 'FAIL', fdt,
               ferr or 'OK — fallback covers the outage', '' if fok else 'Both Yahoo AND FMP failed for SPY — this is a real gap, not just Yahoo rate-limiting.')


# ─────────────────────────────────────────────────────────────
# MACRO DASHBOARD — FRED (rates, inflation, labor, liquidity), VIX/sector
# ETFs via Yahoo
# ─────────────────────────────────────────────────────────────
def check_macro():
    page = 'Macro Dashboard'
    fred_series = {
        'MORTGAGE30US': 'rates', 'CPIAUCSL': 'inflation', 'UNRATE': 'labor',
        'PAYEMS': 'labor', 'M2SL': 'liquidity', 'WALCL': 'liquidity', 'T5YIE': 'inflation',
    }
    for series_id, group in fred_series.items():
        def _f(sid=series_id):
            r = requests.get('https://api.stlouisfed.org/fred/series/observations',
                              params={'series_id': sid, 'api_key': FRED_API_KEY,
                                      'file_type': 'json', 'sort_order': 'desc', 'limit': 1},
                              timeout=10)
            r.raise_for_status()
            data = r.json()
            assert data.get('observations'), 'no observations returned'
            return data['observations'][0]['value']
        ok, val, dt, err = timed(_f)
        record(page, f'FRED:{series_id}', f'{group} — latest observation', 'PASS' if ok else 'FAIL', dt,
               err or f'value={val}', '' if ok else f'Check FRED_API_KEY and series ID {series_id} is still valid on fred.stlouisfed.org.')

    def _vix():
        r = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX', timeout=10)
        r.raise_for_status()
        return r.json()
    ok, _, dt, err = timed(_vix)
    record(page, 'Yahoo Finance', 'VIX pull', 'PASS' if ok else 'FAIL', dt,
           err or 'OK', '' if ok else 'Yahoo rate-limited — verifying FMP fallback below.')
    if not ok:
        def _fmp_vix():
            r = requests.get('https://financialmodelingprep.com/stable/quote',
                              params={'symbol': '^VIX', 'apikey': FMP_API_KEY}, timeout=10)
            r.raise_for_status()
            data = r.json()
            assert data and isinstance(data, list) and data[0].get('price') is not None, 'empty/placeholder quote'
            return data[0]
        fok, _, fdt, ferr = timed(_fmp_vix)
        record(page, 'FMP', 'VIX /stable/quote fallback (triggered by Yahoo failure)', 'PASS' if fok else 'FAIL', fdt,
               ferr or 'OK — fallback covers the outage', '' if fok else 'Both Yahoo AND FMP failed for VIX — this is a real gap, not just Yahoo rate-limiting.')


# ─────────────────────────────────────────────────────────────
# ETF SCANNER / EQUITY SCANNER — bulk yfinance pulls, FMP profile fallback
# NOTE: Equity Scanner's nav link points to localhost:5000 (home route), not
# a distinct page/API surface as of this codebase — flagged separately.
# ─────────────────────────────────────────────────────────────
def check_scanner():
    page = 'ETF Scanner'
    for ticker in SPOT_CHECK_TICKERS[:3]:
        def _yf(tk=ticker):
            r = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}', timeout=10)
            r.raise_for_status()
            return r.json()
        ok, _, dt, err = timed(_yf)
        record(page, 'Yahoo Finance', f'{ticker} quote', 'PASS' if ok else 'FAIL', dt,
               err or 'OK', '' if ok else f'Fall back to FMP /stable/profile for {ticker}.')

        def _fmp(tk=ticker):
            r = requests.get('https://financialmodelingprep.com/stable/profile',
                              params={'symbol': tk, 'apikey': FMP_API_KEY}, timeout=10)
            r.raise_for_status()
            data = r.json()
            assert data and isinstance(data, list) and data[0].get('marketCap'), 'empty/placeholder profile'
            return data[0]
        ok, _, dt, err = timed(_fmp)
        record(page, 'FMP', f'{ticker} /stable/profile fallback', 'PASS' if ok else 'FAIL', dt,
               err or 'OK', '' if ok else f'FMP profile fallback for {ticker} is broken — this is the last resort for AUM/Beta/52wk/div yield.')


# ─────────────────────────────────────────────────────────────
# ETF LAB — SEC EDGAR N-PORT holdings + rebalance detection
# ─────────────────────────────────────────────────────────────
def check_etf_lab():
    page = 'ETF Lab'

    def _edgar_map():
        r = requests.get('https://www.sec.gov/files/company_tickers.json',
                          headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        assert len(data) > 1000, 'ticker map suspiciously small'
        return len(data)
    ok, n, dt, err = timed(_edgar_map)
    record(page, 'SEC EDGAR', 'ticker→CIK map load', 'PASS' if ok else 'FAIL', dt,
           err or f'{n} tickers loaded', '' if ok else 'sec.gov may be rate-limiting — confirm User-Agent header is set and requests are throttled.')

    def _edgar_filing():
        # Apple Inc CIK — used as a stable smoke test for the EDGAR filing
        # index pathway (actual fund CIKs are ticker-specific in the app).
        r = requests.get('https://data.sec.gov/submissions/CIK0000320193.json',
                          headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    ok, _, dt, err = timed(_edgar_filing)
    record(page, 'SEC EDGAR', 'submissions/filing index reachable', 'PASS' if ok else 'FAIL', dt,
           err or 'OK', '' if ok else 'data.sec.gov may be down or blocking — N-PORT holdings + empirical rebalance detection depend on this.')


# ─────────────────────────────────────────────────────────────
# EQUITY SCANNER — served at "/" as entry-point-scanner-v12.html.
# This page calls Kurt's own Flask routes (/quote/<ticker>,
# /api/scanner/data, /api/scanner/status, /alerts) rather than third-party
# APIs directly — those routes' underlying Yahoo/FMP dependency is already
# covered under ETF Scanner above. This check just confirms the local
# server itself is up and those routes respond. Only meaningful when run
# on/against the machine actually running macro_server.py (e.g. locally,
# or via a tunnel) — skipped rather than failed if unreachable, since a
# GitHub Actions runner has no route to your localhost.
# ─────────────────────────────────────────────────────────────
LOCAL_SERVER_BASE = os.environ.get('ALPHA_SCOUT_BASE_URL', 'http://localhost:5000')


def check_equity_scanner():
    page = 'Equity Scanner'
    local_routes = [
        ('/', 'root — serves entry-point-scanner-v12.html'),
        ('/api/scanner/data', 'scanner data payload'),
        ('/api/scanner/status', 'scanner background-loop status'),
        ('/quote/SCHD', 'single-ticker quote route'),
        ('/alerts', 'alerts feed'),
    ]
    for route, desc in local_routes:
        def _local(r=route):
            resp = requests.get(f'{LOCAL_SERVER_BASE}{r}', timeout=5)
            resp.raise_for_status()
            return resp
        try:
            requests.get(LOCAL_SERVER_BASE, timeout=2)
        except requests.exceptions.ConnectionError:
            record(page, 'Local Flask server', desc, 'SKIP', 0.0,
                   f'{LOCAL_SERVER_BASE} not reachable from this host — run this script on the machine hosting macro_server.py, or set ALPHA_SCOUT_BASE_URL.', '')
            continue
        ok, _, dt, err = timed(_local)
        record(page, 'Local Flask server', desc, 'PASS' if ok else 'FAIL', dt,
               err or 'OK', '' if ok else f'Route {route} is erroring — check macro_server.py logs directly.')


# ─────────────────────────────────────────────────────────────
# PORTFOLIO — reuses Yahoo/FMP quote pulls for held positions
# ─────────────────────────────────────────────────────────────
def check_portfolio():
    page = 'Portfolio'
    for ticker in SPOT_CHECK_TICKERS[3:6]:
        def _yf(tk=ticker):
            r = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{tk}', timeout=10)
            r.raise_for_status()
            return r.json()
        ok, _, dt, err = timed(_yf)
        record(page, 'Yahoo Finance', f'{ticker} position price pull', 'PASS' if ok else 'FAIL', dt,
               err or 'OK', '' if ok else 'Yahoo rate-limited — verifying FMP fallback below.')
        if not ok:
            def _fmp(tk=ticker):
                r = requests.get('https://financialmodelingprep.com/stable/profile',
                                  params={'symbol': tk, 'apikey': FMP_API_KEY}, timeout=10)
                r.raise_for_status()
                data = r.json()
                assert data and isinstance(data, list) and data[0].get('marketCap'), 'empty/placeholder profile'
                return data[0]
            fok, _, fdt, ferr = timed(_fmp)
            record(page, 'FMP', f'{ticker} /stable/profile fallback (triggered by Yahoo failure)', 'PASS' if fok else 'FAIL', fdt,
                   ferr or 'OK — fallback covers the outage', '' if fok else f'Both Yahoo AND FMP failed for {ticker} — this is a real gap, not just Yahoo rate-limiting.')


def build_report():
    total_time = sum(r['seconds'] for r in results)
    fails = [r for r in results if r['status'] == 'FAIL']
    skips = [r for r in results if r['status'] == 'SKIP']
    scored = [r for r in results if r['status'] != 'SKIP']
    lines = []
    lines.append(f"Alpha Scout — Weekly API QA Report")
    lines.append(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Overall: {len(scored) - len(fails)}/{len(scored)} checks passed"
                 f"{f', {len(skips)} skipped (local server unreachable)' if skips else ''}"
                 f" — total time {total_time:.1f}s")
    lines.append('')

    pages = []
    for r in results:
        if r['page'] not in pages:
            pages.append(r['page'])

    for page in pages:
        page_results = [r for r in results if r['page'] == page]
        page_time = sum(r['seconds'] for r in page_results)
        lines.append(f"── {page} ({page_time:.1f}s) ──")
        for r in page_results:
            lines.append(f"  [{r['status']}] {r['api']} — {r['check']} — {r['seconds']:.2f}s — {r['detail']}")
            if r['status'] == 'FAIL' and r['fix_plan']:
                lines.append(f"        Fix plan: {r['fix_plan']}")
        lines.append('')

    if fails:
        lines.append(f"⚠ {len(fails)} failing connection(s) need attention — see fix plans above.")
    else:
        lines.append("✅ All connections healthy this week.")

    slow = [r for r in results if r['seconds'] > 3]
    if slow:
        lines.append('')
        lines.append("Latency notes:")
        for r in slow:
            lines.append(f"  - {r['page']}/{r['api']} took {r['seconds']:.1f}s — consider caching or a faster fallback if this recurs.")

    return '\n'.join(lines), len(fails)


def send_email(body, fail_count):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print('\n[WARN] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — skipping send. Report printed above.')
        return
    subject = f"Alpha Scout QA — {'ALL PASS' if fail_count == 0 else f'{fail_count} FAIL(S)'} — {datetime.now().strftime('%Y-%m-%d')}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = REPORT_TO
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, [REPORT_TO], msg.as_string())
    print(f'[OK] Report emailed to {REPORT_TO}')


if __name__ == '__main__':
    for check_fn in (check_home, check_macro, check_scanner, check_etf_lab, check_equity_scanner, check_portfolio):
        try:
            check_fn()
        except Exception:
            print(f'[ERROR] {check_fn.__name__} crashed:')
            traceback.print_exc()

    report, fail_count = build_report()
    print('\n' + '=' * 60)
    print(report)
    send_email(report, fail_count)
