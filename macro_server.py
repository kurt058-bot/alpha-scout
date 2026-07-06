from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import threading
import time
import statistics
from datetime import datetime, timedelta

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── API keys are loaded from environment variables (see .env.example). ──
# ── NEVER hardcode keys here — this file may end up in a git repo / CI runner. ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))
except ImportError:
    pass  # dotenv is optional; env vars can also be set at the OS level

UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_ACCESS_KEY', '')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_CACHE_DIR = os.path.join(BASE_DIR, 'fmp_cache')
UNSPLASH_CACHE_DIR = os.path.join(BASE_DIR, 'unsplash_cache')

# ── FRED API Key (free at fred.stlouisfed.org) ──
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

# ── Anthropic API ──
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

for _name, _val in [('UNSPLASH_ACCESS_KEY', UNSPLASH_ACCESS_KEY), ('FMP_API_KEY', FMP_API_KEY),
                     ('FRED_API_KEY', FRED_API_KEY), ('ANTHROPIC_API_KEY', ANTHROPIC_API_KEY)]:
    if not _val:
        print(f'  [Config] WARNING: {_name} is not set — related features will be degraded/disabled.')

# Cache
_cache = {}
_cache_time = {}
CACHE_TTL = 3600  # 1 hour

def cached(key, fn):
    now = time.time()
    if key in _cache and (now - _cache_time.get(key, 0)) < CACHE_TTL:
        return _cache[key]
    try:
        result = fn()
        _cache[key] = result
        _cache_time[key] = now
        return result
    except Exception as e:
        print(f'  Cache error [{key}]: {e}')
        return _cache.get(key, None)

def safe_float(val, decimals=2):
    try:
        return round(float(val), decimals)
    except:
        return None


def normalize_expense_ratio(raw):
    """Yahoo/yfinance inconsistently returns expense ratio either as a true
    decimal fraction (0.0006 = 0.06%) or as an already-scaled percentage
    (0.06 meaning 0.06% directly) — this varies by ticker and even by which
    underlying Yahoo endpoint served the data, so the source can't be
    trusted to disambiguate (confirmed: SCHD and DGRO both hit this
    inconsistently in practice). Instead, pick whichever interpretation
    lands within a realistic ETF expense ratio range (0.01%-3.00%, which
    covers virtually all real funds from ultra-cheap index funds up through
    leveraged/actively-managed strategies)."""
    if raw is None:
        return None
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return None
    if raw <= 0:
        return None

    REALISTIC_MIN, REALISTIC_MAX = 0.01, 3.0
    as_is, scaled = raw, raw * 100
    as_is_ok  = REALISTIC_MIN <= as_is  <= REALISTIC_MAX
    scaled_ok = REALISTIC_MIN <= scaled <= REALISTIC_MAX

    if as_is_ok and not scaled_ok:
        return round(as_is, 4)
    if scaled_ok and not as_is_ok:
        return round(scaled, 4)
    if as_is_ok and scaled_ok:
        # Both technically plausible (narrow overlap band) — a true fraction
        # this small would imply a near-zero fee, rarer than a normal fee
        # already expressed as a percent, so prefer the as-is interpretation
        return round(as_is, 4)
    # Neither lands in a realistic range — likely bad/mismatched data;
    # return the smaller-magnitude interpretation rather than an absurd one
    smaller = min(as_is, scaled)
    return round(smaller, 4) if smaller > 0 else None


def normalize_dividend_yield(raw):
    """Same fraction-vs-percent ambiguity as expense ratio applies to
    Yahoo's dividend yield field too (confirmed: the codebase had the
    identical flawed `if div_yield and div_yield < 1: *100` pattern
    elsewhere, which silently drops genuine 0.0 yields via Python
    truthiness AND misfires on the same fraction/percent ambiguity).
    Bounds widened vs. expense ratio's 0.01%-3.00% since dividend yields
    legitimately run much higher — this app already tracks aggressive
    option-income ETFs yielding 35-49%."""
    if raw is None:
        return None
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return None
    if raw < 0:
        return None
    if raw == 0:
        return 0.0  # a genuine zero yield (non-dividend-paying fund) is real and common

    REALISTIC_MIN, REALISTIC_MAX = 0.01, 60.0
    as_is, scaled = raw, raw * 100
    as_is_ok  = REALISTIC_MIN <= as_is  <= REALISTIC_MAX
    scaled_ok = REALISTIC_MIN <= scaled <= REALISTIC_MAX

    if as_is_ok and not scaled_ok:
        return round(as_is, 2)
    if scaled_ok and not as_is_ok:
        return round(scaled, 2)
    if as_is_ok and scaled_ok:
        # Ambiguous overlap (raw between 0.01 and 0.60) — split further:
        # values under 0.10 are almost never a genuine as-is yield (no real
        # equity fund yields 0.035%), so those are almost certainly a true
        # fraction -> prefer scaled. Values 0.10-0.60 ARE a realistic as-is
        # yield range for low-yield growth ETFs (QQQ ~0.6%, VUG ~0.5%),
        # which is more common in practice than a true fraction implying a
        # 10%-60% yield -> prefer as-is in that sub-range.
        if raw < 0.10:
            return round(scaled, 2)
        else:
            return round(as_is, 2)
    smaller = min(as_is, scaled)
    return round(smaller, 2) if smaller > 0 else None


def get_ticker_fundamentals(ticker):
    """Look up a single stock's industry classification, individual
    valuation ratios (P/B, PEG, P/S), and beta — disk-cached indefinitely
    (these essentially never change day-to-day). One Yahoo .info call
    already contains all these fields, so this stays at zero extra network
    cost — and enables computing ETF-level weighted averages from holdings
    for all of them, since ETFs themselves don't file the financial
    statements these figures are normally built from (they file N-PORT
    holdings reports instead, not income statements or balance sheets).

    Uses a hard 4-second timeout via a worker thread — yfinance's .info
    property has no built-in timeout, so a single silently-hanging Yahoo
    request here could otherwise block far longer than expected, and with
    up to 15 sequential lookups per Lab request, one stuck call could
    plausibly consume the entire client-side request budget by itself."""
    ticker = ticker.upper().strip()
    os.makedirs(INDUSTRY_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(INDUSTRY_CACHE_DIR, f'{ticker}.json')
    if os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path))
            # Old cache format didn't have 'beta' — treat as a partial hit
            # and let it be refreshed below so beta gets backfilled once
            if 'beta' in cached:
                return cached
        except Exception:
            pass

    result = {'industry': None, 'price_to_book': None, 'peg_ratio': None, 'price_to_sales': None, 'beta': None}
    import concurrent.futures as _cf
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: yf.Ticker(ticker).info)
            info = future.result(timeout=4)
            result['industry']       = info.get('industry') or info.get('sector')
            result['price_to_book']  = info.get('priceToBook')
            result['peg_ratio']      = info.get('trailingPegRatio') or info.get('pegRatio')
            result['price_to_sales'] = info.get('priceToSalesTrailing12Months')
            # Same zero-placeholder guard as the ETF-level beta fix — Yahoo
            # sometimes returns an exact 0.0 for beta on thinly-covered
            # tickers rather than omitting the field, which reads as real
            # data (zero market correlation) when it's really just missing
            beta_raw = info.get('beta')
            if beta_raw is None or beta_raw == 0:
                beta_raw = info.get('beta3Year')
            result['beta'] = round(float(beta_raw), 3) if beta_raw not in (None, 0) else None
    except _cf.TimeoutError:
        print(f'  [Fundamentals] Lookup timed out for {ticker} after 4s')
    except Exception:
        pass
    try:
        json.dump(result, open(cache_path, 'w'))
    except Exception:
        pass
    return result


def compute_weighted_etf_ratios(holdings):
    """Compute ETF-level P/B, PEG, P/S, and Beta from the underlying
    holdings — since an ETF itself doesn't have its own earnings/book
    value/sales the way a single company does. Uses the CORRECT
    methodology per metric, not one blanket formula:

    - P/B, P/S: weighted HARMONIC mean, not simple arithmetic averaging.
      Verified by derivation (weight w_i is proportional to price_i *
      shares_i, so the implied fundamental exposure is proportional to
      w_i/ratio_i — summing that recovers the portfolio's true blended
      ratio) and by testing: a single holding at just 2% weight with a
      distorted ratio (e.g. P/B of 150x from a near-wiped-out book value)
      inflated a simple arithmetic average by 106% versus the harmonic
      mean, which matched the true representative value.

    - PEG: same harmonic approach, but holdings with NEGATIVE PEG
      (shrinking earnings) are excluded from the blend entirely first.
      A negative PEG isn't "a low ratio" — it's a different signal
      (deteriorating earnings), and averaging it in directly can flip the
      overall reading (tested case: a fund that should read ~1.49 PEG
      read as 0.83 — looking like a bargain — purely because one 8%-weight
      holding's negative PEG dragged the blend down).

    - Beta: kept as simple arithmetic weighted average — this one IS
      mathematically correct, confirmed via a full return-series
      simulation (portfolio beta = sum(w_i * beta_i) follows directly
      from CAPM linearity, unlike P/B, P/S, or PEG).

    Coverage % (and, for PEG, excluded-weight %) are reported for
    transparency about how much of the fund's weight the estimate is
    actually built on."""

    def harmonic_weighted_avg(pairs):
        total_w = sum(w for w, v in pairs if v is not None and v > 0)
        if total_w <= 0:
            return None, 0.0
        inv_sum = sum(w * (1.0 / v) for w, v in pairs if v is not None and v > 0)
        if inv_sum <= 0:
            return None, 0.0
        return round(total_w / inv_sum, 2), round(total_w, 1)

    def arithmetic_weighted_avg(pairs):
        valid = [(w, v) for w, v in pairs if v is not None]
        total_w = sum(w for w, _ in valid)
        if total_w <= 0:
            return None, 0.0
        return round(sum(w * v for w, v in valid) / total_w, 2), round(total_w, 1)

    def get_pairs(field):
        return [(h.get('weight', 0), h.get(field)) for h in holdings]

    pb_avg, pb_cov = harmonic_weighted_avg(get_pairs('price_to_book'))
    ps_avg, ps_cov = harmonic_weighted_avg(get_pairs('price_to_sales'))
    beta_avg, beta_cov = arithmetic_weighted_avg(get_pairs('beta'))

    # PEG: exclude negative-growth holdings first, then harmonic-average
    # the survivors, tracking how much weight was excluded and why
    peg_pairs = get_pairs('peg_ratio')
    total_peg_w = sum(w for w, v in peg_pairs if v is not None)
    peg_excluded_w = sum(w for w, v in peg_pairs if v is not None and v <= 0)
    peg_avg, peg_cov = harmonic_weighted_avg(peg_pairs)  # already skips v<=0 internally

    return {
        'price_to_book': pb_avg, 'price_to_book_coverage': pb_cov,
        'peg_ratio': peg_avg, 'peg_ratio_coverage': peg_cov,
        'peg_ratio_excluded_negative_pct': round(peg_excluded_w, 1),
        'price_to_sales': ps_avg, 'price_to_sales_coverage': ps_cov,
        'beta': beta_avg, 'beta_coverage': beta_cov,
    }


def get_fmp_ratios(ticker):
    """Fetch trailing-twelve-month valuation ratios (P/B, PEG, P/S) from
    Financial Modeling Prep as a fallback when yfinance doesn't have them.
    Disk-cached 24h. NOTE: FMP's ratios are computed from company financial
    statements (balance sheet, income statement) — ETFs don't file these
    (they file N-PORT holdings reports instead), so this may return nothing
    for the ETF ticker itself. Logged clearly either way so this gets
    confirmed by real results rather than assumed."""
    ticker = ticker.upper().strip()
    os.makedirs(FMP_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(FMP_CACHE_DIR, f'{ticker}.json')
    if os.path.exists(cache_path):
        age_hrs = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hrs < 24:
            try:
                return json.load(open(cache_path))
            except Exception:
                pass

    result = {'price_to_book': None, 'peg_ratio': None, 'price_to_sales': None}
    try:
        r = requests.get(
            'https://financialmodelingprep.com/stable/ratios-ttm',
            params={'symbol': ticker, 'apikey': FMP_API_KEY},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            print(f'  [FMP] {ticker} ratios-ttm raw response: {data}')
            if isinstance(data, list) and data:
                row = data[0]
                result['price_to_book']  = row.get('priceToBookRatioTTM')
                result['peg_ratio']      = row.get('priceToEarningsGrowthRatioTTM')
                result['price_to_sales'] = row.get('priceToSalesRatioTTM')
            elif isinstance(data, dict) and 'Error Message' in data:
                print(f'  [FMP] {ticker}: {data["Error Message"]}')
        else:
            print(f'  [FMP] {ticker}: HTTP {r.status_code} — {r.text[:200]}')
    except Exception as e:
        print(f'  [FMP] Ratios fetch failed for {ticker}: {e}')

    try:
        json.dump(result, open(cache_path, 'w'))
    except Exception:
        pass
    return result


def get_fmp_profile(ticker):
    """Fetch AUM, beta, day change, 52-week range, and last dividend from
    FMP's /stable/profile endpoint — confirmed working on the free tier via
    direct testing (unlike ratios-ttm and etf/info, which both returned 402
    Restricted Endpoint). For an ETF, FMP's 'marketCap' field is effectively
    AUM: price * shares outstanding IS the fund's net assets by definition
    of how ETF arbitrage keeps price pinned to NAV — verified against SCHD's
    known real AUM (~$94.9B), matched to within $31 (pure rounding).

    Does NOT include expense ratio — confirmed absent from the complete
    response for SCHD, so this fills the AUM/beta/52-week/dividend gaps but
    not the expense ratio one. Disk-cached 24h."""
    ticker = ticker.upper().strip()
    os.makedirs(FMP_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(FMP_CACHE_DIR, f'{ticker}_profile.json')
    if os.path.exists(cache_path):
        age_hrs = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hrs < 24:
            try:
                return json.load(open(cache_path))
            except Exception:
                pass

    result = {
        'aum': None, 'beta': None, 'price': None, 'day_change': None,
        'day_change_pct': None, 'fifty_two_week_high': None,
        'fifty_two_week_low': None, 'dividend_yield': None,
    }
    try:
        r = requests.get(
            'https://financialmodelingprep.com/stable/profile',
            params={'symbol': ticker, 'apikey': FMP_API_KEY},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                row = data[0]
                print(f'  [FMP-Profile] {ticker} raw response: {row}')
                result['aum']            = row.get('marketCap')
                result['beta']           = row.get('beta')
                result['price']          = row.get('price')
                result['day_change']     = row.get('change')
                result['day_change_pct'] = row.get('changePercentage')

                # "range" comes back as a string like "26.21-32.92"
                range_str = row.get('range', '')
                if range_str and '-' in range_str:
                    try:
                        lo, hi = range_str.split('-')
                        result['fifty_two_week_low']  = float(lo)
                        result['fifty_two_week_high'] = float(hi)
                    except (ValueError, TypeError):
                        pass

                # lastDividend appears to already be a trailing annual
                # total (confirmed: SCHD's 1.048/32.39 ≈ 3.2%, matching its
                # known real yield — a single quarterly payment would imply
                # an implausible ~13% yield instead)
                last_div = row.get('lastDividend')
                price = row.get('price')
                if last_div is not None and price:
                    result['dividend_yield'] = round(last_div / price * 100, 2)
        else:
            print(f'  [FMP-Profile] {ticker}: HTTP {r.status_code} — {r.text[:200]}')
    except Exception as e:
        print(f'  [FMP-Profile] Fetch failed for {ticker}: {e}')

    try:
        json.dump(result, open(cache_path, 'w'))
    except Exception:
        pass
    return result


# ── yfinance history cache ──
_yf_hist_cache = {}
_yf_hist_time  = {}
YF_HIST_TTL = 600  # 10 minutes

def yf_hist_cached(ticker, period='1y'):
    """Get yfinance history with in-memory cache."""
    key = f'{ticker}_{period}'
    now = time.time()
    if key in _yf_hist_cache and now - _yf_hist_time.get(key, 0) < YF_HIST_TTL:
        return _yf_hist_cache[key]
    try:
        hist = yf.Ticker(ticker).history(period=period)
        _yf_hist_cache[key] = hist
        _yf_hist_time[key] = now
        return hist
    except:
        return None

def yf_latest(ticker):
    try:
        hist = yf_hist_cached(ticker, '5d')
        if hist is None or hist.empty:
            return None
        return safe_float(hist['Close'].iloc[-1])
    except:
        return None

def yf_series(ticker, period='1y', field='Close'):
    try:
        hist = yf_hist_cached(ticker, period)
        if hist is None or hist.empty:
            return []
        return [{'date': str(d.date()), 'value': safe_float(v)} for d, v in zip(hist.index, hist[field])]
    except:
        return []


# ════════════════════════════════════════════════════════════════
#  ETF HOLDINGS STORE — loaded from uploaded files
#  Kurt drops xlsx/csv files; server parses and stores in memory
#  Store lives at BASE_DIR/holdings/*.json (auto-loaded on startup)
# ════════════════════════════════════════════════════════════════

import os
_etf_holdings_store = {}  # ticker -> list of {t, n, w}

def load_holdings_from_disk():
    """Load all pre-processed holdings JSON files from the holdings/ folder."""
    global _etf_holdings_store
    holdings_dir = os.path.join(BASE_DIR, 'holdings')
    if not os.path.exists(holdings_dir):
        os.makedirs(holdings_dir)
        print(f'  [Holdings] Created holdings dir: {holdings_dir}')
    count = 0
    for fn in os.listdir(holdings_dir):
        if fn.endswith('.json'):
            ticker = fn.replace('.json','').upper()
            try:
                data = json.load(open(os.path.join(holdings_dir, fn)))
                _etf_holdings_store[ticker] = data
                count += 1
            except Exception as e:
                print(f'  [Holdings] Failed to load {fn}: {e}')
    print(f'  [Holdings] Loaded {count} ETF holdings from disk')

load_holdings_from_disk()


# ════════════════════════════════════════════════════════════════
#  SEC EDGAR HOLDINGS PIPELINE (Phase 2)
#  Ticker -> CIK -> latest N-PORT filing -> full holdings (CUSIP-keyed)
#  -> OpenFIGI batch resolution (CUSIP -> ticker + clean name)
#  -> cached to disk, refreshed monthly (N-PORT itself only updates monthly)
#
#  This REPLACES the old top-10-only Yahoo/yfinance holdings as the primary
#  source. yfinance/Yahoo remain as fallback for tickers EDGAR can't resolve
#  (e.g. very new funds with no N-PORT filed yet).
# ════════════════════════════════════════════════════════════════

import re
import xml.etree.ElementTree as ET

EDGAR_HEADERS = {'User-Agent': 'AlphaScout research kurth-personal-use@example.com'}
EDGAR_CACHE_DIR = os.path.join(BASE_DIR, 'edgar_cache')
INDUSTRY_CACHE_DIR = os.path.join(BASE_DIR, 'industry_cache')
DIVIDEND_HISTORY_CACHE_DIR = os.path.join(BASE_DIR, 'dividend_history_cache')
DIVIDEND_HISTORY_CACHE_TTL_HOURS = 24  # dividend history doesn't change retroactively; daily refresh is plenty
EDGAR_CACHE_TTL_DAYS = 30  # N-PORT is monthly — no need to refetch more often than this

# Free OpenFIGI API key — get one at openfigi.com/api (instant, no payment info).
# Raises batch size from 10->100 and removes the slow rate limit.
# Leave blank to fall back to the slower unauthenticated tier.
OPENFIGI_API_KEY = ''

# Free Business Quant API key — get one at businessquant.com (no credit card).
# Used specifically for the Dividends endpoint, which gives us ex_date AND
# payment_date together for both stocks and ETFs — a gap yfinance can't
# reliably fill (it usually only has ex-date, rarely payment date). We still
# use yfinance for price history/CAGR; this is just for dividend mechanics.
BUSINESSQUANT_API_KEY = ''
BUSINESSQUANT_BASE_URL = 'https://data.businessquant.com'

# ── Ticker -> CIK/seriesId resolution via sec-cik-mapper ──
# This package solves the exact problem we were hitting: most ETF issuers
# (iShares, NEOS, SPDR, Vanguard, etc.) register MANY funds under one shared
# trust CIK. A plain ticker->CIK lookup isn't enough to find the right fund's
# N-PORT filing among dozens/hundreds of siblings under that CIK — we need the
# specific seriesId. sec-cik-mapper maintains a daily-updated ticker->seriesId
# crosswalk built from SEC's own data, so we don't have to discover seriesIds
# by hand one ETF at a time.
#
# pip install sec-cik-mapper --break-system-packages  (if not already installed)
try:
    from sec_cik_mapper import MutualFundMapper, StockMapper
    _fund_mapper = None
    _stock_mapper = None
    SEC_CIK_MAPPER_AVAILABLE = True
except ImportError:
    SEC_CIK_MAPPER_AVAILABLE = False
    print('  [EDGAR] sec-cik-mapper not installed — run: pip install sec-cik-mapper --break-system-packages')
    print('  [EDGAR] Falling back to manual overrides only for multi-fund-trust ETFs')

def _get_fund_mapper():
    """Lazily initialize the mutual fund/ETF mapper (covers series/class-based
    funds — this is what we need for nearly all ETFs, since ETFs are legally
    organized the same way as mutual fund series)."""
    global _fund_mapper
    if _fund_mapper is None and SEC_CIK_MAPPER_AVAILABLE:
        try:
            _fund_mapper = MutualFundMapper()
            print(f'  [EDGAR] Loaded sec-cik-mapper fund mappings ({len(_fund_mapper.ticker_to_series_id)} tickers)')
        except Exception as e:
            print(f'  [EDGAR] Failed to initialize MutualFundMapper: {e}')
            _fund_mapper = False  # sentinel for "tried and failed", don't retry every call
    return _fund_mapper if _fund_mapper else None

def _get_stock_mapper():
    """Lazily initialize the plain stock mapper (covers regular equities —
    rarely needed for our ETF-focused use case, but kept as a fallback)."""
    global _stock_mapper
    if _stock_mapper is None and SEC_CIK_MAPPER_AVAILABLE:
        try:
            _stock_mapper = StockMapper()
        except Exception as e:
            print(f'  [EDGAR] Failed to initialize StockMapper: {e}')
            _stock_mapper = False
    return _stock_mapper if _stock_mapper else None

# Manual CIK + series overrides — only needed now as a fallback for tickers
# sec-cik-mapper doesn't have yet (e.g. very recently launched funds not in
# its daily snapshot, or edge cases). Format: 'TICKER': ('CIK', 'seriesId')
EDGAR_CIK_OVERRIDES = {
    # 'TICKER': ('0001234567', 'S000123456'),  # add only if sec-cik-mapper misses it
}


def get_cik_for_ticker(ticker):
    """Resolve a ticker to (cik, series_id) using, in order:
    1. Manual override table (only needed for tickers the mapper doesn't have yet)
    2. sec-cik-mapper's MutualFundMapper (covers virtually all ETFs — this is
       the primary path now, since ETFs are legally structured as fund series)
    3. sec-cik-mapper's StockMapper (plain equities, rarely hit for our use case)
    4. SEC's raw company_tickers.json as a last resort (no series disambiguation —
       only reliable for genuinely single-fund trusts)
    """
    ticker = ticker.upper()

    if ticker in EDGAR_CIK_OVERRIDES:
        return EDGAR_CIK_OVERRIDES[ticker]

    fund_mapper = _get_fund_mapper()
    if fund_mapper:
        series_id = fund_mapper.ticker_to_series_id.get(ticker)
        if series_id:
            cik = fund_mapper.series_id_to_cik.get(series_id)
            if cik:
                return (cik, series_id)

    stock_mapper = _get_stock_mapper()
    if stock_mapper:
        cik = stock_mapper.ticker_to_cik.get(ticker)
        if cik:
            return (cik, None)

    # Last resort: raw company_tickers.json (no series info — risky for
    # multi-fund trusts, but better than nothing for edge cases)
    mapping = _load_ticker_cik_map()
    cik = mapping.get(ticker)
    return (cik, None) if cik else None


_ticker_cik_map = None

def _load_ticker_cik_map():
    """Fallback only — download SEC's raw ticker->CIK map (no series info).
    Used only when sec-cik-mapper is unavailable or doesn't have a ticker."""
    global _ticker_cik_map
    if _ticker_cik_map is not None:
        return _ticker_cik_map
    try:
        r = requests.get('https://www.sec.gov/files/company_tickers.json', headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        _ticker_cik_map = {entry['ticker'].upper(): str(entry['cik_str']).zfill(10) for entry in data.values()}
        print(f'  [EDGAR] Loaded {len(_ticker_cik_map)} raw ticker->CIK mappings (fallback, no series info)')
    except Exception as e:
        print(f'  [EDGAR] Failed to load ticker->CIK map: {e}')
        _ticker_cik_map = {}
    return _ticker_cik_map


def get_latest_nport_filing(cik, series_id=None, target_ticker=None):
    """Given a 10-digit CIK, find the most recent NPORT-P filing.

    IMPORTANT: Many ETF issuers (NEOS, Kurv, YieldMax, etc.) register ALL of
    their funds under one shared trust CIK. Each individual fund ("series")
    files its OWN separate N-PORT each month — so a single CIK can have many
    NPORT-P filings, one per fund. Three ways we resolve to the right one:

    1. series_id given (best — exact match via EDGAR_CIK_OVERRIDES once you've
       looked it up manually): check each candidate filing's genInfo.seriesId
       until we find the match. Fast, deterministic.
    2. No series_id but it's a single-fund trust (the common case for most
       ETFs): just return the most recent NPORT-P filing — no ambiguity.
    3. Multi-fund trust with no series_id yet (target_ticker given instead):
       best-effort — peek at each filing's seriesName and try to fuzzy-match
       against known fund name patterns. This is slower (multiple requests)
       and less reliable; only used until a real seriesId override is added.
    """
    try:
        url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        dates = recent.get('filingDate', [])

        candidates = [
            {'accession': accessions[i], 'date': dates[i], 'cik': cik}
            for i, form in enumerate(forms) if form in ('NPORT-P', 'NPORT-P/A')
        ]
        if not candidates:
            return None

        # Case 2: no series disambiguation needed/possible — most recent is correct
        if not series_id and not target_ticker:
            return candidates[0]

        # Case 1: exact seriesId match. Large issuers (iShares, Vanguard, SPDR)
        # file NPORT-P for hundreds of funds under one CIK each month, so the
        # target fund's filing can be buried far down the "recent" list. We
        # search up to 150 candidates in parallel (each check is a lightweight
        # peek at just the filing's genInfo block, not the full holdings XML)
        # so this stays fast even though the search space is much wider now.
        if series_id:
            import concurrent.futures as _cf
            MAX_CANDIDATES_TO_CHECK = 150
            to_check = candidates[:MAX_CANDIDATES_TO_CHECK]

            def check_candidate(cand):
                found = _get_filing_series_id(cand['cik'], cand['accession'])
                return (cand, found == series_id)

            with _cf.ThreadPoolExecutor(max_workers=15) as ex:
                futures = {ex.submit(check_candidate, c): c for c in to_check}
                for future in _cf.as_completed(futures):
                    cand, matched = future.result()
                    if matched:
                        print(f'  [EDGAR] Matched seriesId {series_id} -> accession {cand["accession"]} (filed {cand["date"]})')
                        # Cancel remaining lookups since we found our answer
                        for f in futures:
                            f.cancel()
                        return cand

            print(f'  [EDGAR] seriesId {series_id} not found after checking {len(to_check)} of {len(candidates)} filings for CIK {cik}')
            return None

        # Case 3: best-effort name peek (no series_id on file yet)
        for cand in candidates[:12]:
            series_name = _get_filing_series_name(cand['cik'], cand['accession'])
            if series_name and target_ticker.upper() in series_name.upper().replace(' ', ''):
                return cand
        print(f'  [EDGAR] Could not match {target_ticker} by name among {len(candidates)} filings under CIK {cik} — add a seriesId override once found')
        return None
    except Exception as e:
        print(f'  [EDGAR] get_latest_nport_filing error for CIK {cik}: {e}')
        return None


def get_nport_filing_history(cik, series_id=None, target_ticker=None, limit=6):
    """Like get_latest_nport_filing, but returns up to `limit` matching
    filings (newest first) instead of stopping at the first match. SEC only
    discloses one N-PORT filing per quarter publicly, so `limit=6` covers
    roughly the last 1.5 years — enough history to empirically detect a
    fund's actual rebalance cadence rather than guessing from its index
    family name."""
    try:
        url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        dates = recent.get('filingDate', [])

        candidates = [
            {'accession': accessions[i], 'date': dates[i], 'cik': cik}
            for i, form in enumerate(forms) if form in ('NPORT-P', 'NPORT-P/A')
        ]
        if not candidates:
            return []

        if not series_id and not target_ticker:
            return candidates[:limit]

        if series_id:
            import concurrent.futures as _cf
            MAX_CANDIDATES_TO_CHECK = 150
            to_check = candidates[:MAX_CANDIDATES_TO_CHECK]
            matches = []

            def check_candidate(cand):
                found = _get_filing_series_id(cand['cik'], cand['accession'])
                return (cand, found == series_id)

            with _cf.ThreadPoolExecutor(max_workers=15) as ex:
                futures = {ex.submit(check_candidate, c): c for c in to_check}
                for future in _cf.as_completed(futures):
                    cand, matched = future.result()
                    if matched:
                        matches.append(cand)
                        if len(matches) >= limit:
                            for f in futures:
                                f.cancel()
                            break
            matches.sort(key=lambda c: c['date'], reverse=True)
            print(f'  [EDGAR] Filing history: found {len(matches)} matches for seriesId {series_id}')
            return matches[:limit]

        # target_ticker fallback (name-peek) — check more candidates since we
        # need several matches this time, not just the first one
        matches = []
        for cand in candidates[:30]:
            series_name = _get_filing_series_name(cand['cik'], cand['accession'])
            if series_name and target_ticker.upper() in series_name.upper().replace(' ', ''):
                matches.append(cand)
                if len(matches) >= limit:
                    break
        return matches
    except Exception as e:
        print(f'  [EDGAR] get_nport_filing_history error for CIK {cik}: {e}')
        return []


def compute_market_cycle():
    """Determine the current market cycle stage from already-cached macro
    data (yield curve, labor market, inflation trend) — reuses the macro
    dashboard's existing precomputed data rather than adding new fetches,
    so this stays fast inside the ETF Lab. Returns a transparent, evidence-
    based assessment (every signal and its reasoning is shown, not a
    black-box label) since reasonable people can read the same data
    differently — this is a framework, not a certainty."""
    macro = _macro_cache.get('data') or {}
    if not macro:
        return None
    rates = macro.get('rates', {}) or {}
    labor = macro.get('labor', {}) or {}
    inflation = macro.get('inflation', {}) or {}

    signals = []  # each: {factor, reading, lean, weight, note}

    # ── Yield curve (10yr-2yr spread) ──
    spread = rates.get('two_ten_spread')
    if spread is not None:
        if spread < 0:
            signals.append({'factor': 'Yield Curve', 'reading': f'{spread:+.2f}% (inverted)', 'lean': 'late', 'weight': 1.2,
                             'note': 'An inverted curve has historically preceded economic slowdowns'})
        elif spread < 0.5:
            signals.append({'factor': 'Yield Curve', 'reading': f'{spread:+.2f}% (mildly positive)', 'lean': 'late', 'weight': 1.0,
                             'note': 'A mild positive spread following a period of inversion — historically the re-steepening, not the inversion itself, has tracked closest to recessions'})
        else:
            signals.append({'factor': 'Yield Curve', 'reading': f'{spread:+.2f}% (normal)', 'lean': 'mid', 'weight': 0.8,
                             'note': 'A healthy positive slope, consistent with mid-cycle expansion'})

    # ── Labor market trend (unemployment vs. a year ago) ──
    unemployment = labor.get('unemployment')
    unemployment_series = labor.get('unemployment_series') or []
    if unemployment is not None and len(unemployment_series) >= 12:
        idx = -13 if len(unemployment_series) >= 13 else 0
        year_ago = unemployment_series[idx]['value']
        change = unemployment - year_ago
        if change > 0.3:
            signals.append({'factor': 'Labor Market Trend', 'reading': f'{unemployment:.1f}% ({change:+.1f}pt YoY)', 'lean': 'late', 'weight': 1.1,
                             'note': 'Rising unemployment is a classic late-cycle signal'})
        elif change < -0.2:
            signals.append({'factor': 'Labor Market Trend', 'reading': f'{unemployment:.1f}% ({change:+.1f}pt YoY)', 'lean': 'early', 'weight': 0.9,
                             'note': 'Falling unemployment is consistent with early/mid-cycle expansion'})
        else:
            signals.append({'factor': 'Labor Market Trend', 'reading': f'{unemployment:.1f}% (roughly flat YoY)', 'lean': 'mid', 'weight': 0.7,
                             'note': 'A stable labor market, consistent with mid-cycle conditions'})

    # ── Inflation trend ──
    cpi_yoy = inflation.get('cpi_yoy')
    cpi_prior = inflation.get('cpi_prior_yoy')
    if cpi_yoy is not None:
        accelerating = cpi_prior is not None and cpi_yoy > cpi_prior + 0.2
        elevated = cpi_yoy > 3.0
        if accelerating and elevated:
            signals.append({'factor': 'Inflation Trend', 'reading': f'{cpi_yoy:.1f}% YoY (accelerating)', 'lean': 'late', 'weight': 1.2,
                             'note': "Reaccelerating, above-target inflation alongside slowing growth is a stagflation-tilted late-cycle signature"})
        elif elevated:
            signals.append({'factor': 'Inflation Trend', 'reading': f'{cpi_yoy:.1f}% YoY (elevated)', 'lean': 'late', 'weight': 0.9,
                             'note': "Inflation running well above the Fed's 2% target"})
        elif cpi_yoy < 2.5 and (cpi_prior is None or cpi_yoy <= cpi_prior):
            signals.append({'factor': 'Inflation Trend', 'reading': f'{cpi_yoy:.1f}% YoY (near target)', 'lean': 'mid', 'weight': 0.8,
                             'note': "Inflation near the Fed's target, consistent with mid-cycle equilibrium"})
        else:
            signals.append({'factor': 'Inflation Trend', 'reading': f'{cpi_yoy:.1f}% YoY', 'lean': 'mid', 'weight': 0.6, 'note': ''})

    # ── Fed policy stance (short rate vs. inflation, as a real-rate proxy) ──
    fed_funds_proxy = rates.get('fed_funds')
    if fed_funds_proxy is not None and cpi_yoy is not None:
        real_rate = fed_funds_proxy - cpi_yoy
        if real_rate > 1.5:
            signals.append({'factor': 'Fed Policy Stance', 'reading': f'~{fed_funds_proxy:.1f}% short rate (restrictive)', 'lean': 'late', 'weight': 1.0,
                             'note': 'Real policy rate well above inflation — historically a late-cycle tightening posture'})
        elif real_rate < 0:
            signals.append({'factor': 'Fed Policy Stance', 'reading': f'~{fed_funds_proxy:.1f}% short rate (accommodative)', 'lean': 'early', 'weight': 0.9,
                             'note': 'Real policy rate below inflation — historically an early-cycle accommodative posture'})
        else:
            signals.append({'factor': 'Fed Policy Stance', 'reading': f'~{fed_funds_proxy:.1f}% short rate (roughly neutral)', 'lean': 'mid', 'weight': 0.7, 'note': ''})

    if not signals:
        return None

    stage_scores = {'early': 0.0, 'mid': 0.0, 'late': 0.0, 'contraction': 0.0}
    for s in signals:
        stage_scores[s['lean']] += s['weight']
    total_weight = sum(stage_scores.values())
    stage_pcts = {k: round(v / total_weight * 100, 1) for k, v in stage_scores.items()} if total_weight > 0 else {}
    dominant_stage = max(stage_scores, key=stage_scores.get) if total_weight > 0 else 'mid'

    stage_labels = {'early': 'Early Cycle', 'mid': 'Mid Cycle', 'late': 'Late Cycle', 'contraction': 'Contraction / Recession'}
    stage_descriptions = {
        'early': 'Recovery phase — accelerating growth, easy policy, falling unemployment.',
        'mid': 'Steady expansion — moderate growth, stable inflation, roughly neutral policy.',
        'late': 'Decelerating growth alongside tightening or restrictive policy and rising inflation/labor risk.',
        'contraction': 'Outright economic contraction.',
    }
    favored_tags = {
        'early': ['Small Cap', 'Value', 'Cyclical', 'Consumer Discretionary', 'High Yield'],
        'mid': ['Large Cap', 'Growth', 'Technology', 'Industrials'],
        'late': ['Value', 'Energy', 'Health Care', 'Consumer Staples', 'Quality', 'Low Volatility'],
        'contraction': ['Consumer Staples', 'Utilities', 'Health Care', 'Treasury', 'Low Volatility'],
    }

    return {
        'stage': stage_labels[dominant_stage],
        'stage_key': dominant_stage,
        'description': stage_descriptions[dominant_stage],
        'favored_tags': favored_tags[dominant_stage],
        'signals': signals,
        'stage_distribution': stage_pcts,
        'as_of': datetime.now().strftime('%Y-%m-%d'),
    }


def compute_cycle_fit(cycle, category, style_bucket, size_bucket, sector_weights):
    """Assess how well THIS specific ETF's profile aligns with the current
    market cycle stage's historically-favored characteristics. Transparent
    match-by-match, not a single opaque score."""
    if not cycle:
        return None
    favored = [t.lower() for t in cycle.get('favored_tags', [])]
    matches = []

    text_fields = ' '.join(filter(None, [category, style_bucket, size_bucket])).lower()
    for tag in cycle.get('favored_tags', []):
        if tag.lower() in text_fields:
            matches.append(f'{tag} — matches this fund\'s stated category/style')

    if sector_weights:
        top_sectors = sorted(sector_weights, key=lambda s: s.get('weight', 0), reverse=True)[:3]
        for sec in top_sectors:
            sec_name = sec.get('sector', '')
            for tag in cycle.get('favored_tags', []):
                if tag.lower().replace(' ', '') in sec_name.lower().replace(' ', '') or sec_name.lower().replace(' ', '') in tag.lower().replace(' ', ''):
                    matches.append(f'{tag} — {sec_name} is a top holding sector ({sec.get("weight",0):.1f}%)')

    match_count = len(matches)
    if match_count >= 2:
        fit_label, fit_color = 'Strong Fit', 'green'
    elif match_count == 1:
        fit_label, fit_color = 'Moderate Fit', 'amber'
    else:
        fit_label, fit_color = 'Weak Fit', 'red'

    return {
        'fit_label': fit_label,
        'fit_color': fit_color,
        'matches': matches,
        'match_count': match_count,
    }


def detect_empirical_rebalance(ticker):
    """Detect a fund's ACTUAL rebalance/reconstitution cadence by comparing
    holdings composition across several historical N-PORT filings, rather
    than guessing from the fund's stated index family name. SEC discloses
    N-PORT publicly once per quarter, so consecutive filings are naturally
    ~3 months apart — this lines up well with standard rebalance schedules
    and reveals genuine fund-specific behavior. Disk-cached 30 days (same
    TTL as the regular holdings cache) since this doesn't change often and
    involves several EDGAR fetches. Returns None if there isn't enough
    filing history to say anything reliable (caller should fall back to
    the index-family heuristic in that case)."""
    ticker = ticker.upper()
    os.makedirs(EDGAR_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(EDGAR_CACHE_DIR, f'{ticker}_rebalance.json')

    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days < EDGAR_CACHE_TTL_DAYS:
            try:
                cached = json.load(open(cache_path))
                print(f'  [EDGAR-Rebalance] {ticker}: serving from cache (age {age_days:.1f}d)')
                return cached
            except Exception:
                pass

    cik_result = get_cik_for_ticker(ticker)
    if not cik_result or not cik_result[0]:
        return None
    cik, series_id = cik_result
    target_ticker = ticker if (ticker in EDGAR_CIK_OVERRIDES and not series_id) else None

    filings = get_nport_filing_history(cik, series_id=series_id, target_ticker=target_ticker, limit=6)
    if not filings or len(filings) < 3:
        print(f'  [EDGAR-Rebalance] {ticker}: only {len(filings) if filings else 0} filings found — not enough history')
        return None

    MAX_HOLDINGS_FOR_EMPIRICAL = 1200  # funds larger than this (e.g. VXUS at
    # ~8000 holdings) take too long to fetch+compare across 6 historical
    # filings to fit in a single request — fall back to the heuristic instead

    # Fetch the first filing alone so we can size-check before committing to
    # the rest — then fetch the remaining filings concurrently instead of
    # sequentially, since they're independent SEC requests. This was one of
    # the biggest sources of avoidable latency on a cold (uncached) lookup.
    snapshots = []
    first = filings[0]
    first_holdings = fetch_and_parse_nport(first['cik'], first['accession'], lightweight=True)
    if first_holdings:
        if len(first_holdings) > MAX_HOLDINGS_FOR_EMPIRICAL:
            print(f'  [EDGAR-Rebalance] {ticker}: {len(first_holdings)} holdings — too large for '
                  f'empirical detection within one request, falling back to heuristic')
            return None
        snapshots.append({'date': first['date'], 'holdings': first_holdings})

    remaining = filings[1:]
    if remaining:
        import concurrent.futures as _cf2
        with _cf2.ThreadPoolExecutor(max_workers=min(5, len(remaining))) as ex:
            futures = {ex.submit(fetch_and_parse_nport, f['cik'], f['accession'], lightweight=True): f for f in remaining}
            for future in _cf2.as_completed(futures):
                f = futures[future]
                try:
                    holdings = future.result()
                    if holdings:
                        snapshots.append({'date': f['date'], 'holdings': holdings})
                except Exception as e:
                    print(f'  [EDGAR-Rebalance] {ticker}: filing fetch failed for {f["accession"]}: {e}')

    if len(snapshots) < 3:
        print(f'  [EDGAR-Rebalance] {ticker}: only {len(snapshots)} filings parsed successfully — not enough history')
        return None

    snapshots.sort(key=lambda s: s['date'])  # oldest first, for chronological comparison

    def turnover_between(snap_a, snap_b):
        """Total % of fund weight that changed between two snapshots —
        new positions + dropped positions + net weight shifts on positions
        held in both, all summed as one turnover figure."""
        a_map = {h['cusip']: h['pct_of_fund'] for h in snap_a['holdings'] if h.get('cusip')}
        b_map = {h['cusip']: h['pct_of_fund'] for h in snap_b['holdings'] if h.get('cusip')}
        all_cusips = set(a_map) | set(b_map)
        if not all_cusips:
            return 0.0
        return sum(abs(b_map.get(c, 0) - a_map.get(c, 0)) for c in all_cusips)

    changes = []
    for i in range(1, len(snapshots)):
        delta = turnover_between(snapshots[i-1], snapshots[i])
        month = datetime.strptime(snapshots[i]['date'][:10], '%Y-%m-%d').strftime('%B')
        changes.append({'date': snapshots[i]['date'], 'month': month, 'turnover_pct': round(delta, 1)})

    if not changes:
        return None

    avg_change = statistics.mean(c['turnover_pct'] for c in changes)
    stdev_change = statistics.pstdev(c['turnover_pct'] for c in changes)
    cv = stdev_change / avg_change if avg_change > 0 else 0
    total_periods = len(changes)

    # ── Branch A: UNIFORM pattern (every filing shows similar turnover) ──
    # A fund that shows consistently similar turnover at every quarterly
    # filing IS, by definition, trading on a regular cadence — the
    # magnitude itself (not spikiness) tells us how often. This was a
    # critical gap in the earlier version: a relative "spike above own
    # average" test can mathematically never fire when every period looks
    # similar, so the clearest, most common case (steady quarterly
    # rebalancing) was always misclassified as "Low Turnover." Verified via
    # synthetic testing at 10%, 25%, and 50% uniform turnover before this
    # fix — all three incorrectly returned "Low Turnover" under the old logic.
    UNIFORM_CV_THRESHOLD = 0.35
    LOW_TURNOVER_MAX = 8.0
    MONTHLY_MIN = 30.0

    if cv < UNIFORM_CV_THRESHOLD:
        if avg_change < LOW_TURNOVER_MAX:
            label = 'Low Turnover'
            note = (f'Holdings have stayed consistently stable across {total_periods} quarterly filings '
                    f'(avg {avg_change:.1f}% turnover per filing) — no clear periodic rebalance pattern detected')
        elif avg_change < MONTHLY_MIN:
            label = 'Quarterly'
            note = (f'Consistent turnover found in every one of {total_periods} quarterly filings '
                    f'(avg {avg_change:.1f}% per filing) — consistent with quarterly rebalancing')
        else:
            label = 'Monthly (High Turnover)'
            note = (f'Turnover is consistently very high across every quarterly filing (avg {avg_change:.1f}% '
                    f'per filing) — this level of change is larger than a single quarterly rebalance typically '
                    f'produces, suggesting the fund rebalances monthly or more frequently. Flagged as high '
                    f'turnover: expect materially higher trading costs and taxable distributions than a '
                    f'standard quarterly-rebalanced fund.')

    else:
        # ── Branch B: SPIKY pattern (some filings much higher than others) ──
        # Look for periodic spikes, but require a genuine GAP between the
        # "significant" and "quiet" groups before committing to a label —
        # otherwise noisy or steadily-trending data (verified via testing:
        # a gradually increasing 3%->15% trend with no real periodicity)
        # gets mistaken for a clean annual/semi-annual pattern.
        threshold = max(avg_change * 1.4, 3.0)
        significant = [c for c in changes if c['turnover_pct'] >= threshold]
        quiet = [c for c in changes if c['turnover_pct'] < threshold]
        sig_count = len(significant)

        MIN_GAP_RATIO = 1.8  # smallest "significant" value must clear the largest "quiet" value by this multiple
        gap_ratio = (min(c['turnover_pct'] for c in significant) / max(c['turnover_pct'] for c in quiet)
                     if significant and quiet and max(c['turnover_pct'] for c in quiet) > 0 else None)

        if sig_count == 0 or sig_count == total_periods or gap_ratio is None or gap_ratio < MIN_GAP_RATIO:
            label = 'Irregular / No Clear Pattern'
            note = (f'Holdings changes across {total_periods} quarterly filings don\'t show a clean periodic '
                    f'pattern (avg {avg_change:.1f}% turnover, but no consistent cadence) — this fund may '
                    f'rebalance opportunistically, or there isn\'t enough filing history yet for a confident read')
        elif sig_count >= total_periods - 1:
            label = 'Quarterly'
            note = (f'Meaningful holdings changes found in {sig_count} of {total_periods} quarterly filings — '
                    f'consistent with quarterly rebalancing')
        elif sig_count >= max(2, total_periods // 2):
            months = ', '.join(sorted(set(c['month'] for c in significant)))
            label = 'Semi-Annual'
            note = f'Meaningful holdings changes concentrated around {months} — consistent with semi-annual rebalancing'
        else:
            months = ', '.join(sorted(set(c['month'] for c in significant)))
            label = 'Annual'
            note = f'Meaningful holdings changes concentrated around {months} — consistent with annual reconstitution'

    result = {
        'frequency': label,
        'note': note,
        'evidence': changes,
        'filings_analyzed': total_periods + 1,
        'source': 'sec_edgar_empirical',
    }
    try:
        json.dump(result, open(cache_path, 'w'))
    except Exception:
        pass
    print(f'  [EDGAR-Rebalance] {ticker}: detected "{label}" from {total_periods+1} filings — {note}')
    return result


def _peek_filing_geninfo(cik, accession):
    """Fetch just enough of a filing to read its genInfo block (seriesId, seriesName).
    Shared helper for both seriesId matching and name-based fallback matching."""
    try:
        accession_nodash = accession.replace('-', '')
        index_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/'
        r = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        xml_files = re.findall(r'href="([^"]+\.xml)"', r.text)
        if not xml_files:
            return None
        target = next((f for f in xml_files if 'primary_doc' in f.lower()), xml_files[0])
        xml_url = index_url + target.split('/')[-1]

        r2 = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=20)
        if r2.status_code != 200:
            return None

        root = ET.fromstring(r2.text)

        def find_local(elem, tag_name):
            for child in elem:
                if _local_tag(child) == tag_name:
                    return child
            return None

        for elem in root.iter():
            if _local_tag(elem) == 'genInfo':
                series_id_el = find_local(elem, 'seriesId')
                series_name_el = find_local(elem, 'seriesName')
                return {
                    'series_id': series_id_el.text.strip() if series_id_el is not None and series_id_el.text else None,
                    'series_name': series_name_el.text.strip() if series_name_el is not None and series_name_el.text else None,
                }
        return None
    except Exception:
        return None


def _get_filing_series_id(cik, accession):
    info = _peek_filing_geninfo(cik, accession)
    return info['series_id'] if info else None


def _get_filing_series_name(cik, accession):
    info = _peek_filing_geninfo(cik, accession)
    return info['series_name'] if info else None


def _local_tag(elem):
    """Strip XML namespace prefix from a tag, e.g. '{http://...}invstOrSec' -> 'invstOrSec'"""
    return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag


def fetch_and_parse_nport(cik, accession, lightweight=False, fund_info_out=None):
    """Fetch the N-PORT primary XML document and extract all holdings
    (CUSIP-keyed — ticker resolution happens separately via OpenFIGI).

    lightweight=True skips per-holding debug logging and non-essential
    field extraction (ISIN lookup, derivative/option detail) — used by the
    historical rebalance comparison, which only needs cusip+weight per
    holding. Funds like VXUS carry ~8,000 holdings; printing every single
    one across the 6 historical filings we compare was a severe I/O
    bottleneck that could push a single Lab request past the client-side
    timeout entirely. This mode cuts that overhead dramatically.

    fund_info_out: optional dict — if provided, gets populated with
    fund-level data from the filing's <genInfo> block (currently just
    net_assets). This is an opt-in side-channel rather than changing the
    function's return type, so every existing caller is unaffected. Added
    because AUM, expense ratio, and dividend yield had NO fallback when
    Yahoo's .info call gets rate-limited — unlike P/B/PEG/P/S/Beta, which
    already fall back to a weighted-average-of-holdings calculation. N-PORT
    filings report total net assets directly, giving AUM a real fallback
    that's independent of Yahoo entirely."""
    try:
        accession_nodash = accession.replace('-', '')
        index_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/'
        r = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            print(f'  [EDGAR] fetch_and_parse_nport: index fetch failed ({r.status_code}) for {index_url}')
            return None

        xml_files = re.findall(r'href="([^"]+\.xml)"', r.text)
        if not lightweight: print(f'  [EDGAR] Filing index XML files found: {xml_files}')
        if not xml_files:
            print(f'  [EDGAR] No XML files found in filing index at {index_url}')
            return None
        target = next((f for f in xml_files if 'primary_doc' in f.lower()), xml_files[0])
        xml_url = index_url + target.split('/')[-1]
        if not lightweight: print(f'  [EDGAR] Using XML file: {xml_url}')

        r2 = requests.get(xml_url, headers=EDGAR_HEADERS, timeout=25)
        if r2.status_code != 200:
            print(f'  [EDGAR] XML fetch failed ({r2.status_code}) for {xml_url}')
            return None

        root = ET.fromstring(r2.text)

        def find_local(elem, tag_name):
            for child in elem:
                if _local_tag(child) == tag_name:
                    return child
            return None

        def findtext_local(elem, tag_name, default=''):
            found = find_local(elem, tag_name)
            return found.text if found is not None and found.text else default

        if fund_info_out is not None:
            try:
                gen_info_elem = next((e for e in root.iter() if _local_tag(e) == 'genInfo'), None)
                if gen_info_elem is not None:
                    net_assets_txt = findtext_local(gen_info_elem, 'netAssets', '')
                    if net_assets_txt:
                        fund_info_out['net_assets'] = float(net_assets_txt)
                        print(f'  [EDGAR] Net assets from filing genInfo: ${fund_info_out["net_assets"]:,.0f}')
            except Exception as fe:
                print(f'  [EDGAR] Could not extract net assets from genInfo: {fe}')

        # Diagnostic: count all distinct top-level-ish tag names we see, so we
        # Diagnostic tag-counting moved to only run on genuine failures (see
        # below, after the main parse) — this used to run unconditionally
        # on every request via two full root.iter() tree-walks, which was
        # useful early on for debugging but adds real overhead that scales
        # with fund size on every request now that the parser is proven
        # working (confirmed slow specifically for large filings like IWM's
        # ~2000 holdings, on top of VXUS's already-documented ~8000).

        def find_nested(elem, *tag_path):
            """Walk down a chain of nested tag names, e.g.
            find_nested(elem, 'derivativeInfo', 'optionSwaptionWarrantDeriv')"""
            current = elem
            for tag in tag_path:
                current = find_local(current, tag)
                if current is None:
                    return None
            return current

        holdings = []
        for elem in root.iter():
            if _local_tag(elem) == 'invstOrSec':
                name = findtext_local(elem, 'name', '')
                ticker = findtext_local(elem, 'ticker', '')
                cusip = findtext_local(elem, 'cusip', '')
                pct_val = findtext_local(elem, 'pctVal', '0')

                if lightweight:
                    # Only cusip+weight needed for the historical rebalance
                    # comparison — skip name/ticker cleanup, ISIN lookup,
                    # asset category, and derivative/option parsing entirely
                    try:
                        holdings.append({'cusip': cusip.strip(), 'pct_of_fund': float(pct_val)})
                    except ValueError:
                        continue
                    continue

                # Pass 1 (cheap, for every holding): just enough to sort by
                # weight and decide which holdings are even worth the
                # expensive per-holding work below. A fund like IWM (tracks
                # the Russell 2000, ~2000 holdings) would otherwise pay full
                # ISIN-lookup + derivative-parsing cost on every single one,
                # even though only the top ~150 are ever displayed anywhere
                # in this app — this was a real, verifiable slowdown
                # independent of any Yahoo rate-limiting.
                try:
                    holdings.append({
                        'name': name.strip(), 'ticker': ticker.strip(),
                        'cusip': cusip.strip(), 'pct_of_fund': float(pct_val),
                        '_elem': elem,  # kept only for pass 2 below, stripped before return
                    })
                except ValueError:
                    continue

        if not holdings:
            # Something's genuinely wrong — now it's worth paying for the
            # full diagnostic tree-walk to understand why, since this only
            # runs on actual failures rather than every request
            all_tags_seen = set()
            for elem in root.iter():
                all_tags_seen.add(_local_tag(elem))
            print(f'  [EDGAR] No holdings parsed. genInfo present: {"genInfo" in all_tags_seen}. '
                  f'Total distinct tags: {len(all_tags_seen)}. Sample: {sorted(all_tags_seen)[:30]}')

        if lightweight:
            total_pct = sum(h['pct_of_fund'] for h in holdings)
            print(f'  [NPORT-Parse] TOTAL holdings parsed (lightweight): {len(holdings)}, SUM: {total_pct:.2f}%')
            holdings.sort(key=lambda h: -h['pct_of_fund'])
            return holdings

        # Sort by weight and keep a buffer above the 150-holding display cap
        # (fetch_edgar_holdings_live) so pass 2 below only does expensive
        # work on holdings that could plausibly end up displayed
        holdings.sort(key=lambda h: -h['pct_of_fund'])
        PASS2_BUFFER = 200
        top_holdings = holdings[:PASS2_BUFFER]
        remainder = holdings[PASS2_BUFFER:]
        for h in remainder:
            h.pop('_elem', None)

        # Pass 2 (expensive, only for the top ~200 by weight): ISIN lookup,
        # asset category, derivative/option detail
        for h in top_holdings:
            elem = h.pop('_elem')
            asset_cat = findtext_local(elem, 'assetCat', '')
            maturity_dt = findtext_local(elem, 'maturityDt', '')
            title_of_class = findtext_local(elem, 'titleOfClass', '')

            # ISIN fallback — foreign holdings (e.g. SK Hynix, a Korean
            # stock) routinely have NO CUSIP in the filing at all, since
            # CUSIPs are primarily a North American identifier. Per SEC's
            # own N-PORT schema docs, the ISIN lives in a nested
            # <identifiers><isin value="..."/></identifiers> block — an
            # attribute, not text content, so it needs its own lookup
            # rather than findtext_local. Without this, any holding that
            # lacks a CUSIP gets silently dropped before OpenFIGI resolution
            # ever runs, even when its actual fund weight is the largest
            # in the entire portfolio (this was happening to AIQ's SK Hynix).
            isin = ''
            identifiers_elem = find_local(elem, 'identifiers')
            if identifiers_elem is not None:
                isin_elem = find_local(identifiers_elem, 'isin')
                if isin_elem is not None:
                    isin = isin_elem.get('value', '').strip()

            # Derivative/option detail — only present on option/swaption/
            # warrant holdings (item C.11 of the filing). Most equity
            # holdings won't have this block at all.
            option_info = None
            opt_elem = find_nested(elem, 'derivativeInfo', 'optionSwaptionWarrantDeriv')
            if opt_elem is not None:
                put_or_call = findtext_local(opt_elem, 'putOrCall', '')
                strike = findtext_local(opt_elem, 'exercisePrice', '')
                exp_dt = findtext_local(opt_elem, 'expDt', '')
                underlying_elem = find_local(opt_elem, 'putOrCallUnderlying') or find_local(opt_elem, 'descRefInstrmnt')
                underlying_name = findtext_local(underlying_elem, 'name', '') if underlying_elem is not None else ''
                if put_or_call or strike or exp_dt:
                    option_info = {
                        'put_or_call': put_or_call.strip(),
                        'strike': strike.strip(),
                        'expiration': exp_dt.strip(),
                        'underlying_name': underlying_name.strip(),
                    }

            h['isin'] = isin
            h['asset_cat'] = asset_cat.strip()
            h['maturity_dt'] = maturity_dt.strip()
            h['title_of_class'] = title_of_class.strip()
            if option_info:
                h['option_info'] = option_info

        holdings = top_holdings + remainder
        total_pct = sum(h['pct_of_fund'] for h in holdings)
        print(f'  [NPORT-Parse] TOTAL holdings parsed: {len(holdings)}, SUM of pct_of_fund: {total_pct:.2f}% '
              f'(should be ~100%) — expensive per-holding fields only extracted for top {len(top_holdings)}')

        return holdings
    except Exception as e:
        print(f'  [EDGAR] fetch_and_parse_nport error: {e}')
        return None


def resolve_cusips_via_openfigi(holdings, api_key=None):
    """Batch-resolve CUSIPs (or ISINs, as a fallback) to tickers + clean
    company names via OpenFIGI. Free, no key required (10/batch, slower);
    with a free key: 100/batch, faster. Mutates and returns the holdings
    list with 'ticker'/'name' filled where found.

    Foreign holdings (e.g. SK Hynix, a Korean stock) routinely have NO
    CUSIP in N-PORT filings at all — CUSIPs are a North American identifier.
    Those holdings carry an ISIN instead. Without this fallback, any
    no-CUSIP holding was previously skipped entirely before reaching
    OpenFIGI, even when it was the fund's single largest position by weight."""
    url = 'https://api.openfigi.com/v3/mapping'
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key

    batch_size = 100 if api_key else 10
    sleep_between = 0.3 if api_key else 2.5
    resolved = 0

    for i in range(0, len(holdings), batch_size):
        batch = holdings[i:i+batch_size]
        # Prefer CUSIP when present; fall back to ISIN for holdings that
        # have no CUSIP at all (typically foreign/non-US-listed securities).
        identifiable_holdings = [h for h in batch if h.get('cusip') or h.get('isin')]
        jobs = []
        for h in identifiable_holdings:
            if h.get('cusip'):
                jobs.append({'idType': 'ID_CUSIP', 'idValue': h['cusip']})
            else:
                jobs.append({'idType': 'ID_ISIN', 'idValue': h['isin']})
        if not jobs:
            continue
        try:
            r = requests.post(url, headers=headers, json=jobs, timeout=20)
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code != 200:
                continue
            results = r.json()
            for h, result in zip(identifiable_holdings, results):
                if 'data' in result and result['data']:
                    candidates = result['data']
                    # OpenFIGI can return MULTIPLE distinct instruments for one
                    # CUSIP — e.g. the common stock AND an options contract tied
                    # to the same underlying (their own docs show this exact
                    # case for IBM). Blindly trusting candidates[0] risks
                    # picking an option/warrant/foreign-listing ticker instead
                    # of the plain equity, producing garbled-looking tickers.
                    # Prefer: Common Stock, listed on a US exchange.
                    match = next(
                        (c for c in candidates if c.get('securityType') == 'Common Stock' and c.get('exchCode') == 'US'),
                        None
                    )
                    if not match:
                        # Relax to Common Stock on ANY exchange (covers foreign
                        # ADRs/listings that aren't exchCode='US' but are still
                        # genuinely the equity, not an option/warrant)
                        match = next((c for c in candidates if c.get('securityType') == 'Common Stock'), None)
                    if not match:
                        # Last resort: original behavior, but log it clearly
                        # since this means we genuinely have no clean equity match
                        match = candidates[0]
                        id_used = f"cusip={h.get('cusip')!r}" if h.get('cusip') else f"isin={h.get('isin')!r}"
                        print(f'  [OpenFIGI] WARNING: no Common Stock match for {id_used}, '
                              f'falling back to first candidate (securityType={match.get("securityType")!r})')

                    if match.get('ticker'):
                        h['ticker'] = match['ticker']
                        resolved += 1
                    if match.get('name'):
                        h['name'] = match['name']
                    id_used = f"cusip={h.get('cusip')!r}" if h.get('cusip') else f"isin={h.get('isin')!r}"
                    print(f'  [OpenFIGI] {id_used} -> ticker={match.get("ticker")!r} '
                          f'name={match.get("name")!r} securityType={match.get("securityType")!r} '
                          f'exchCode={match.get("exchCode")!r} all_matches_count={len(candidates)}')
        except Exception as e:
            print(f'  [EDGAR] OpenFIGI batch error: {e}')
        time.sleep(sleep_between)

    print(f'  [EDGAR] OpenFIGI resolved {resolved}/{len(holdings)} holdings')
    return holdings


def fetch_edgar_holdings_live(ticker, api_key=None, fund_info_out=None):
    """Full pipeline: ticker -> CIK -> N-PORT -> holdings -> OpenFIGI resolution.
    Returns a list of {t, n, w} dicts in our standard format, or None on failure
    at any stage. This is the slow path — only called when the disk cache is
    missing or stale; results get cached to disk by the caller."""
    import re as re_module

    cik_result = get_cik_for_ticker(ticker)
    if not cik_result or not cik_result[0]:
        print(f'  [EDGAR] No CIK found for {ticker} (may need a manual override — see EDGAR_CIK_OVERRIDES)')
        return None
    cik, series_id = cik_result

    # If we have a CIK override but no seriesId yet, fall back to best-effort
    # name matching using the ticker itself (works sometimes, not guaranteed —
    # encourages finding the real seriesId and adding it to the override table)
    target_ticker = ticker if (ticker in EDGAR_CIK_OVERRIDES and not series_id) else None

    filing = get_latest_nport_filing(cik, series_id=series_id, target_ticker=target_ticker)
    if not filing:
        print(f'  [EDGAR] No N-PORT filing found for {ticker} (CIK {cik})')
        return None

    raw_holdings = fetch_and_parse_nport(filing['cik'], filing['accession'], fund_info_out=fund_info_out)
    if not raw_holdings:
        print(f'  [EDGAR] Could not parse N-PORT XML for {ticker}')
        return None

    # Cap before OpenFIGI resolution — raw_holdings is already sorted by
    # weight descending (fetch_and_parse_nport does this), and nothing in
    # this app ever displays more than ~100 holdings, so resolving a fund's
    # entire long tail (VXUS alone carries ~8000 holdings) through OpenFIGI's
    # slow no-key rate limit would take 30+ minutes for no practical benefit.
    HOLDINGS_RESOLVE_CAP = 150
    if len(raw_holdings) > HOLDINGS_RESOLVE_CAP:
        print(f'  [EDGAR] {ticker}: {len(raw_holdings)} total holdings in filing — '
              f'resolving only the top {HOLDINGS_RESOLVE_CAP} by weight via OpenFIGI')
        raw_holdings = raw_holdings[:HOLDINGS_RESOLVE_CAP]

    resolved = resolve_cusips_via_openfigi(raw_holdings, api_key=api_key)

    # Asset categories that are NOT regular equities — these don't have real
    # trading tickers, so OpenFIGI sometimes returns an awkward Bloomberg-style
    # placeholder (e.g. "B 0 04/21/26" for a T-bill) instead of a clean symbol.
    # For these, build our own clean label from N-PORT's own data directly
    # rather than trusting OpenFIGI's ticker field.
    NON_EQUITY_CATEGORIES = {'STIV', 'DBT', 'RA', 'DCO', 'COMM'}  # short-term invest., debt, repo agreement, etc.

    def clean_label_for_non_equity(h):
        """Build a readable label like 'U.S. Treasury Bills (matures 4/21/26)'
        from N-PORT's own name/maturity fields, used when the holding isn't
        a normal equity with a real trading ticker."""
        base_name = h.get('name') or 'Unknown Security'
        # Shorten verbose official names a bit for display (plural, since a
        # single CUSIP position still represents many individual bills/notes)
        if 'TREASURY' in base_name.upper():
            base_name = 'U.S. Treasury Bills' if 'BILL' in base_name.upper() else \
                        'U.S. Treasury Notes' if 'NOTE' in base_name.upper() else \
                        'U.S. Treasury Bonds' if 'BOND' in base_name.upper() else \
                        'U.S. Treasuries'

        print(f'  [LabelDebug] name={h.get("name")!r} ticker={h.get("ticker")!r} '
              f'maturity_dt={h.get("maturity_dt")!r} cusip={h.get("cusip")!r}')

        maturity = h.get('maturity_dt', '')
        if maturity:
            try:
                m = datetime.strptime(maturity[:10], '%Y-%m-%d')
                return f'{base_name} (matures {m.month}/{m.day}/{str(m.year)[2:]})'
            except Exception as e:
                print(f'  [LabelDebug] maturity_dt parse failed: {e}')

        # Fallback: our own maturity_dt field came up empty (N-PORT field
        # path or formatting varies by filer) — but OpenFIGI's placeholder
        # ticker for Treasuries reliably embeds the date (e.g. "B 0 04.21.26"),
        # so extract it from there instead of giving up.
        ticker_str = h.get('ticker', '')
        date_match = re_module.search(r'(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})', ticker_str)
        if date_match:
            mm, dd, yy = date_match.groups()
            return f'{base_name} (matures {int(mm)}/{int(dd)}/{yy})'

        print(f'  [LabelDebug] No date found anywhere, returning base_name only: {base_name!r}')
        return base_name

    # Convert to our standard {t, n, w} format.

    def looks_like_bloomberg_treasury_code(ticker_str):
        """OpenFIGI sometimes returns a Bloomberg-style placeholder for
        Treasuries instead of a real ticker, e.g. 'B 0 04.21.26' (Bill, 0%
        coupon, matures 4/21/26) or similar patterns with embedded dates.
        Real equity tickers don't look like this — they're short, mostly
        letters, no date-like number patterns with periods/slashes."""
        if not ticker_str:
            return False
        # Matches patterns like "04.21.26", "04/21/26", "4-21-26" embedded in the string
        has_date_pattern = bool(re_module.search(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}', ticker_str))
        # Real tickers are typically short and alphabetic; these placeholders
        # tend to be longer with spaces and numbers mixed in
        looks_odd = ' ' in ticker_str and any(c.isdigit() for c in ticker_str)
        return has_date_pattern or (looks_odd and len(ticker_str) > 8)

    def looks_like_treasury_by_name(name_str):
        """Check the security's own NAME field (from N-PORT directly, not
        OpenFIGI) for Treasury/government-debt keywords — more reliable than
        guessing the exact assetCat code SEC uses, since filers' assetCat
        usage varies and isn't always the code we'd expect."""
        if not name_str:
            return False
        upper = name_str.upper()
        return any(kw in upper for kw in [
            'UNITED STATES TREASURY', 'US TREASURY', 'U.S. TREASURY',
            'TREASURY BILL', 'TREASURY NOTE', 'TREASURY BOND',
            'T-BILL', 'TREASURY INFLATION',
        ])

    out = []
    for idx, h in enumerate(resolved):
        asset_cat = h.get('asset_cat', '')
        option_info = h.get('option_info')

        # Layered detection: assetCat match OR name pattern OR OpenFIGI
        # returned an odd Bloomberg-style code instead of a real ticker.
        # Any one of these signals is enough to treat it as a cash-equivalent.
        is_non_equity = (
            asset_cat in NON_EQUITY_CATEGORIES
            or looks_like_treasury_by_name(h.get('name'))
            or looks_like_bloomberg_treasury_code(h.get('ticker'))
        )

        if option_info:
            # Option/derivative position — these don't have a normal equity
            # ticker, they're identified by the contract itself. Build a
            # readable label like "GLD 07/17/2026 425 C" from the underlying
            # name + expiration + strike + put/call, matching how option
            # contracts are conventionally displayed.
            underlying = option_info.get('underlying_name') or h.get('name') or 'Option'
            strike = option_info.get('strike', '')
            exp = option_info.get('expiration', '')
            pc = option_info.get('put_or_call', '')
            pc_letter = 'C' if pc.lower() == 'call' else 'P' if pc.lower() == 'put' else ''
            exp_display = exp
            try:
                exp_display = datetime.strptime(exp[:10], '%Y-%m-%d').strftime('%m/%d/%Y')
            except Exception:
                pass
            label_parts = [p for p in [underlying, exp_display, strike, pc_letter] if p]
            label = ' '.join(label_parts) if label_parts else (h.get('name') or 'Option Position')
            # Always include the array index so multiple option holdings with
            # identical/empty CUSIP+label (e.g. several small hedge legs with
            # no disclosed underlying) never collide on the same synthetic
            # ticker — distinct holdings must never share a key.
            synthetic_ticker = 'OPT-' + (h.get('cusip') or label[:12]).replace(' ', '') + f'-{idx}'
            out.append({
                't': synthetic_ticker, 'n': label, 'w': round(h['pct_of_fund'], 4),
                'is_option': True,
                'option_detail': {
                    'underlying': underlying, 'strike': strike, 'expiration': exp_display,
                    'put_or_call': pc.capitalize() if pc else None,
                    'security_name': h.get('name'), 'cusip': h.get('cusip'),
                },
            })
        elif is_non_equity:
            # Use a synthetic, stable "ticker" so the frontend still has a
            # unique key to render/sort by, but show a clean readable label
            # as the name instead of OpenFIGI's awkward placeholder ticker.
            label = clean_label_for_non_equity(h)
            synthetic_ticker = 'CASH-' + (h.get('cusip') or label[:10]).replace(' ', '')
            out.append({'t': synthetic_ticker, 'n': label, 'w': round(h['pct_of_fund'], 4), 'is_cash_equiv': True})
        elif h.get('ticker'):
            # Sanitize tickers with slashes (e.g. BRK/B) for safe use as JSON keys / URL segments
            clean_ticker = h['ticker'].replace('/', '.')
            out.append({'t': clean_ticker, 'n': h['name'] or h['ticker'], 'w': round(h['pct_of_fund'], 4)})
        elif h.get('name') and h.get('pct_of_fund') is not None:
            # No ticker resolved at all — typically a foreign holding whose
            # ISIN OpenFIGI doesn't map to a US-tradeable ticker (e.g. many
            # Korean/Japanese ordinary shares like SK Hynix, Samsung Electronics).
            # Previously these were dropped entirely, which silently removed a
            # fund's largest position from the treemap. Show them by name and
            # weight using a synthetic, stable key so the frontend can still
            # render/sort them like any other tile, just without a clickable
            # US ticker.
            label = h['name']
            synthetic_ticker = 'NOTICKER-' + (h.get('isin') or h.get('cusip') or label[:12]).replace(' ', '')
            out.append({
                't': synthetic_ticker, 'n': label, 'w': round(h['pct_of_fund'], 4),
                'no_ticker': True,
            })

    print(f'  [EDGAR] {ticker}: {len(out)}/{len(raw_holdings)} holdings have a usable ticker')
    return out if out else None


def get_edgar_holdings_cached(ticker):
    """Check disk cache first (valid for EDGAR_CACHE_TTL_DAYS since N-PORT is
    monthly anyway); if missing/stale, fetch live and cache the result.
    Returns None if EDGAR has no usable data for this ticker (caller should
    fall back to yfinance/Yahoo)."""
    ticker = ticker.upper()
    os.makedirs(EDGAR_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(EDGAR_CACHE_DIR, f'{ticker}.json')
    net_assets_cache_path = os.path.join(EDGAR_CACHE_DIR, f'{ticker}_netassets.json')

    if os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        print(f'  [EDGAR-Cache] {ticker}: cache file EXISTS at {cache_path}, age={age_days:.2f} days (TTL={EDGAR_CACHE_TTL_DAYS})')
        if age_days < EDGAR_CACHE_TTL_DAYS:
            try:
                print(f'  [EDGAR-Cache] {ticker}: serving from CACHE (not re-fetching)')
                return json.load(open(cache_path))
            except Exception:
                print(f'  [EDGAR-Cache] {ticker}: cache file corrupted, falling through to live fetch')
                pass  # corrupted cache, fall through to refetch
    else:
        print(f'  [EDGAR-Cache] {ticker}: NO cache file found at {cache_path} — fetching live')

    print(f'  [EDGAR-Cache] {ticker}: calling fetch_edgar_holdings_live() now...')
    fund_info = {}
    holdings = fetch_edgar_holdings_live(ticker, api_key=OPENFIGI_API_KEY or None, fund_info_out=fund_info)
    if holdings:
        try:
            json.dump(holdings, open(cache_path, 'w'))
        except Exception as e:
            print(f'  [EDGAR] Failed to write cache for {ticker}: {e}')
        if 'net_assets' in fund_info:
            try:
                json.dump({'net_assets': fund_info['net_assets']}, open(net_assets_cache_path, 'w'))
            except Exception as e:
                print(f'  [EDGAR] Failed to write net-assets cache for {ticker}: {e}')
        return holdings

    return None


def get_edgar_net_assets_cached(ticker):
    """Read the AUM fallback captured from a fund's N-PORT filing, if we've
    ever done a live EDGAR fetch for this ticker since this feature was
    added. Returns None if not yet captured — the holdings cache and this
    companion cache are written together during a live fetch, but an
    EXISTING holdings cache entry (written before this feature existed)
    won't have a companion file until its holdings cache naturally expires
    and gets refreshed (up to EDGAR_CACHE_TTL_DAYS days)."""
    ticker = ticker.upper()
    path = os.path.join(EDGAR_CACHE_DIR, f'{ticker}_netassets.json')
    if os.path.exists(path):
        try:
            return json.load(open(path)).get('net_assets')
        except Exception:
            return None
    return None


def get_dividend_history_cached(ticker):
    """Per-ticker dividend payment history (real ex-dividend dates and real
    per-share amounts actually paid, from Yahoo via yfinance), disk-cached
    for DIVIDEND_HISTORY_CACHE_TTL_HOURS. This is what powers "Monthly
    Distributions" — without a disk cache, recomputing this for ~150+
    holdings on every portfolio page load would hammer Yahoo's API and
    reliably trigger the same rate-limiting seen elsewhere in this app.
    Returns a list of {date: 'YYYY-MM-DD', amount: per-share $} or None on
    failure (caller should treat that ticker as contributing $0, not crash
    the whole calculation)."""
    ticker = ticker.upper()
    os.makedirs(DIVIDEND_HISTORY_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(DIVIDEND_HISTORY_CACHE_DIR, f'{ticker}.json')

    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < DIVIDEND_HISTORY_CACHE_TTL_HOURS:
            try:
                return json.load(open(cache_path))
            except Exception:
                pass  # corrupted cache, fall through to refetch

    try:
        t = yf.Ticker(ticker)
        divs = t.dividends  # pandas Series, index=ex-div date, value=$/share
        if divs is None:
            json.dump([], open(cache_path, 'w'))
            return []
        history = [{'date': d.strftime('%Y-%m-%d'), 'amount': float(v)} for d, v in divs.items()]
        json.dump(history, open(cache_path, 'w'))
        return history
    except Exception as e:
        # Certain money-market / cash-equivalent funds (e.g. SPAXX) reliably
        # crash yfinance's .dividends property with a "Period 'max' is
        # invalid" error — this is a known yfinance quirk for that fund
        # type, not a real failure worth retrying or treating as missing
        # data. Cache an empty result so it doesn't get refetched (and
        # re-logged) every time.
        print(f'  [DivHistory] Failed for {ticker}: {e}')
        json.dump([], open(cache_path, 'w'))
        return []


@app.route('/api/etf/holdings/edgar-clear-all-cache', methods=['POST'])
def edgar_clear_all_cache():
    """Delete ALL cached EDGAR holdings files at once. Use this after any
    server-side parsing/labeling fix, so every fund picks up the corrected
    logic on its next view rather than continuing to serve pre-fix cached
    data one ticker at a time. Does not re-fetch anything itself — each
    fund's holdings simply get fetched fresh the next time its treemap or
    top25 endpoint is requested."""
    if not os.path.exists(EDGAR_CACHE_DIR):
        return jsonify({'cleared': 0, 'message': 'Cache directory does not exist — nothing to clear'})
    files = [f for f in os.listdir(EDGAR_CACHE_DIR) if f.endswith('.json')]
    for f in files:
        try:
            os.remove(os.path.join(EDGAR_CACHE_DIR, f))
        except Exception as e:
            print(f'  [EDGAR-Cache] Failed to remove {f}: {e}')
    print(f'  [EDGAR-Cache] Cleared {len(files)} cached holdings files')
    return jsonify({'cleared': len(files), 'tickers': [f.replace('.json','') for f in files]})


@app.route('/api/etf/holdings/edgar-refresh/<ticker>', methods=['POST'])
def edgar_refresh_holdings(ticker):
    """Force a fresh EDGAR fetch for one ticker, bypassing the disk cache,
    and report what actually changed vs. the previous cached holdings —
    useful for spotting reconstitution/rebalance events (new entrants,
    dropped holdings, meaningful weight shifts)."""
    ticker = ticker.upper()
    cache_path = os.path.join(EDGAR_CACHE_DIR, f'{ticker}.json')

    # Snapshot the OLD holdings before we wipe the cache, so we can diff
    old_holdings = None
    if os.path.exists(cache_path):
        try:
            old_holdings = json.load(open(cache_path))
        except Exception:
            pass
        os.remove(cache_path)

    new_holdings = get_edgar_holdings_cached(ticker)
    if not new_holdings:
        return jsonify({'ticker': ticker, 'error': 'EDGAR fetch failed — check CIK override or N-PORT availability'}), 404

    if not old_holdings:
        # First-ever fetch for this ticker — nothing to diff against
        return jsonify({
            'ticker': ticker, 'count': len(new_holdings), 'source': 'sec_edgar',
            'refreshed': True, 'has_previous': False,
        })

    # Build ticker->weight maps for comparison
    old_map = {h['t']: h['w'] for h in old_holdings}
    new_map = {h['t']: h['w'] for h in new_holdings}

    added = sorted(set(new_map) - set(old_map), key=lambda t: -new_map[t])
    removed = sorted(set(old_map) - set(new_map), key=lambda t: -old_map[t])

    # Meaningful weight changes — only flag shifts of 0.5 percentage points
    # or more, so tiny rounding noise doesn't look like a "change"
    WEIGHT_CHANGE_THRESHOLD = 0.5
    weight_changes = []
    for t in set(old_map) & set(new_map):
        diff = new_map[t] - old_map[t]
        if abs(diff) >= WEIGHT_CHANGE_THRESHOLD:
            weight_changes.append({'ticker': t, 'old_weight': old_map[t], 'new_weight': new_map[t], 'change': round(diff, 2)})
    weight_changes.sort(key=lambda x: -abs(x['change']))

    has_changes = bool(added or removed or weight_changes)

    return jsonify({
        'ticker': ticker, 'count': len(new_holdings), 'source': 'sec_edgar',
        'refreshed': True, 'has_previous': True, 'has_changes': has_changes,
        'added': [{'ticker': t, 'weight': new_map[t]} for t in added[:10]],
        'removed': [{'ticker': t, 'weight': old_map[t]} for t in removed[:10]],
        'weight_changes': weight_changes[:15],
        'old_count': len(old_holdings), 'new_count': len(new_holdings),
    })



# ── Shared Fred instance + series cache ──
_fred_instance = None
_fred_series_cache = {}

def get_fred():
    global _fred_instance
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return None
    if _fred_instance is None:
        _fred_instance = Fred(api_key=FRED_API_KEY)
    return _fred_instance

def fred_get_series_cached(series_id):
    """Get a FRED series, caching in memory for the session."""
    if series_id in _fred_series_cache:
        return _fred_series_cache[series_id]
    f = get_fred()
    if not f:
        return None
    try:
        s = f.get_series(series_id).dropna()
        _fred_series_cache[series_id] = s
        return s
    except Exception as e:
        print(f'  FRED series error [{series_id}]: {e}')
        return None

def fred_latest(series_id):
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return None
    try:
        s = fred_get_series_cached(series_id)
        if s is None or len(s) == 0:
            return None
        return safe_float(s.iloc[-1])
    except:
        return None

def fred_series(series_id, periods=24):
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return []
    try:
        s = fred_get_series_cached(series_id)
        if s is None:
            return []
        s = s.tail(periods)
        return [{'date': str(d.date()), 'value': safe_float(v)} for d, v in s.items()]
    except:
        return []

def fred_yoy(series_id):
    """Calculate year-over-year change for a FRED series"""
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return None
    try:
        s = fred_get_series_cached(series_id)
        if s is None or len(s) < 13:
            return None
        latest = s.iloc[-1]
        year_ago = s.iloc[-13]
        return safe_float(((latest - year_ago) / year_ago) * 100)
    except:
        return None

# ── Data fetchers ──

def get_rates():
    return {
        'fed_funds': yf_latest('^IRX'),  # 13-week T-bill proxy
        'two_year': yf_latest('^IRX'),
        'ten_year': yf_latest('^TNX'),
        'thirty_year': yf_latest('^TYX'),
        'two_ten_spread': None,  # calculated below
        'mortgage_30': fred_latest('MORTGAGE30US'),
        'series': {
            'ten_year': yf_series('^TNX', '1y'),
            'two_year': yf_series('^IRX', '1y'),
        }
    }

def get_rates_full():
    r = get_rates()
    try:
        ten = r.get('ten_year') or 0
        two = r.get('two_year') or 0  # using 13W as proxy
        # Get actual 2yr via yfinance
        two_yr = yf_latest('^IRX')
        r['two_ten_spread'] = safe_float(ten - two_yr) if ten and two_yr else None
    except:
        pass
    return r

def get_vix():
    vix_now = yf_latest('^VIX')
    series = yf_series('^VIX', '6mo')
    avg_30d = None
    if series:
        last_30 = [x['value'] for x in series[-21:] if x['value']]
        avg_30d = safe_float(sum(last_30)/len(last_30)) if last_30 else None
    return {
        'current': vix_now,
        'avg_30d': avg_30d,
        'regime': 'low' if vix_now and vix_now < 15 else 'normal' if vix_now and vix_now < 20 else 'elevated' if vix_now and vix_now < 30 else 'fear',
        'series': series
    }

def get_market_internals():
    # Use sector ETFs as proxy for internals
    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Cons. Discretionary',
        'XLP': 'Cons. Staples',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLB': 'Materials',
        'XLC': 'Comm. Services',
    }
    sector_data = []
    for ticker, name in sectors.items():
        try:
            hist = yf_hist_cached(ticker, '3mo')
            if hist is None or hist.empty:
                continue
            closes = hist['Close'].tolist()
            price = closes[-1]
            ret_1m = safe_float(((closes[-1] - closes[-21]) / closes[-21]) * 100) if len(closes) >= 21 else None
            ret_3m = safe_float(((closes[-1] - closes[0]) / closes[0]) * 100) if len(closes) > 5 else None
            sma50 = safe_float(sum(closes[-50:]) / min(50, len(closes)))
            above_50sma = price > sma50 if sma50 else None
            sector_data.append({
                'ticker': ticker,
                'name': name,
                'price': safe_float(price),
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'above_50sma': above_50sma,
            })
        except Exception as e:
            print(f'  Sector error {ticker}: {e}')
    return sector_data

def get_breadth():
    # SPY advance/decline proxy using % above 200SMA
    try:
        spy_hist = yf_hist_cached('SPY', '1y')
        if spy_hist is None or spy_hist.empty:
            return {}
        closes = spy_hist['Close'].tolist()
        price = closes[-1]
        sma200 = sum(closes[-200:]) / min(200, len(closes))
        sma50 = sum(closes[-50:]) / min(50, len(closes))
        spy_ret_ytd = safe_float(((closes[-1] - closes[0]) / closes[0]) * 100)

        # AD line proxy: use equal weight vs market cap weight
        rsp_hist = yf_hist_cached('RSP', '3mo')
        rsp_closes = rsp_hist['Close'].tolist() if rsp_hist is not None and not rsp_hist.empty else []
        rsp_ret = safe_float(((rsp_closes[-1] - rsp_closes[0]) / rsp_closes[0]) * 100) if rsp_closes else None
        spy_3m = safe_float(((closes[-1] - closes[-63]) / closes[-63]) * 100) if len(closes) >= 63 else None
        breadth_signal = 'strong' if rsp_ret and spy_3m and rsp_ret >= spy_3m * 0.8 else 'diverging'

        return {
            'spy_price': safe_float(price),
            'spy_vs_200sma': safe_float(((price - sma200) / sma200) * 100),
            'spy_vs_50sma': safe_float(((price - sma50) / sma50) * 100),
            'spy_ytd': spy_ret_ytd,
            'rsp_3m': rsp_ret,
            'spy_3m': spy_3m,
            'breadth_signal': breadth_signal,
        }
    except Exception as e:
        print(f'  Breadth error: {e}')
        return {}

def get_liquidity():
    # M2 via FRED, proxy others via yfinance
    try:
        # DXY (dollar strength = liquidity drain globally)
        dxy = yf_latest('DX-Y.NYB')
        dxy_series = yf_series('DX-Y.NYB', '1y')

        # High yield credit spread proxy (HYG vs LQD)
        hyg_hist_df = yf_hist_cached('HYG', '1y')
        lqd_hist_df = yf_hist_cached('LQD', '1y')
        hyg_hist = hyg_hist_df['Close'].tolist() if hyg_hist_df is not None and not hyg_hist_df.empty else []
        lqd_hist = lqd_hist_df['Close'].tolist() if lqd_hist_df is not None and not lqd_hist_df.empty else []
        hyg_price = safe_float(hyg_hist[-1]) if hyg_hist else None
        hyg_1m = safe_float(((hyg_hist[-1]-hyg_hist[-21])/hyg_hist[-21])*100) if len(hyg_hist)>=21 else None

        # TLT (long bond) as rates/liquidity proxy
        tlt = yf_latest('TLT')

        # M2 from FRED if available
        m2 = fred_latest('M2SL')
        m2_yoy = fred_yoy('M2SL')

        # Fed balance sheet
        fed_bs = fred_latest('WALCL')

        return {
            'dxy': dxy,
            'dxy_series': dxy_series[-52:] if dxy_series else [],
            'hyg_price': hyg_price,
            'hyg_1m_ret': hyg_1m,
            'credit_signal': 'tightening' if hyg_1m and hyg_1m < -1 else 'stable' if hyg_1m and hyg_1m > -1 else 'unknown',
            'tlt': tlt,
            'm2': m2,
            'm2_yoy': m2_yoy,
            'fed_balance_sheet': fed_bs,
        }
    except Exception as e:
        print(f'  Liquidity error: {e}')
        return {}

def fred_yoy_series(series_id, periods=6):
    """Return last N yoy values for sparkline/history display"""
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return []
    try:
        s = fred_get_series_cached(series_id)
        if s is None:
            return []
        results = []
        for i in range(periods, 0, -1):
            try:
                cur = s.iloc[-i]
                prev = s.iloc[-i-12]
                yoy = safe_float(((cur - prev) / prev) * 100)
                results.append({'date': str(s.index[-i].date()), 'value': yoy})
            except:
                pass
        return results
    except:
        return []

def get_inflation():
    try:
        # CPI from FRED
        cpi_latest = fred_latest('CPIAUCSL')
        cpi_yoy = fred_yoy('CPIAUCSL')
        cpi_core_yoy = fred_yoy('CPILFESL')
        cpi_series = fred_series('CPIAUCSL', 24)
        # Prior year values — use cached series already fetched above
        cpi_prior = None
        cpi_core_prior = None
        pce_prior = None
        core_pce_prior = None
        try:
            s = fred_get_series_cached('CPIAUCSL')
            if s is not None and len(s) >= 25:
                cpi_prior = safe_float(((s.iloc[-13] - s.iloc[-25]) / s.iloc[-25]) * 100)
            s2 = fred_get_series_cached('CPILFESL')
            if s2 is not None and len(s2) >= 25:
                cpi_core_prior = safe_float(((s2.iloc[-13] - s2.iloc[-25]) / s2.iloc[-25]) * 100)
            s3 = fred_get_series_cached('PCEPI')
            if s3 is not None and len(s3) >= 25:
                pce_prior = safe_float(((s3.iloc[-13] - s3.iloc[-25]) / s3.iloc[-25]) * 100)
            s4 = fred_get_series_cached('PCEPILFE')
            if s4 is not None and len(s4) >= 25:
                core_pce_prior = safe_float(((s4.iloc[-13] - s4.iloc[-25]) / s4.iloc[-25]) * 100)
        except:
            pass

        # PCE (Fed's preferred measure)
        pce_yoy = fred_yoy('PCEPI')
        core_pce_yoy = fred_yoy('PCEPILFE')
        # Historical yoy series for sparklines (last 6 months)
        cpi_history = fred_yoy_series('CPIAUCSL', 6)
        core_cpi_history = fred_yoy_series('CPILFESL', 6)
        pce_history = fred_yoy_series('PCEPI', 6)
        core_pce_history = fred_yoy_series('PCEPILFE', 6)
        truflation_note = 'Check truflation.com for real-time reading'
        return {
            'cpi_latest': cpi_latest,
            'cpi_yoy': cpi_yoy,
            'cpi_prior_yoy': cpi_prior,
            'cpi_core_yoy': cpi_core_yoy,
            'cpi_core_prior_yoy': cpi_core_prior,
            'pce_yoy': pce_yoy,
            'pce_prior_yoy': pce_prior,
            'core_pce_yoy': core_pce_yoy,
            'core_pce_prior_yoy': core_pce_prior,
            'breakeven_5yr': fred_latest('T5YIE'),
            'truflation_note': truflation_note,
            'cpi_series': cpi_series,
            'cpi_history': cpi_history,
            'core_cpi_history': core_cpi_history,
            'pce_history': pce_history,
            'core_pce_history': core_pce_history,
        }
    except Exception as e:
        print(f'  Inflation error: {e}')
        return {}

def get_labor():
    try:
        # Unemployment rate
        unemployment = fred_latest('UNRATE')
        unemployment_series = fred_series('UNRATE', 24)

        # Non-farm payrolls MoM change (thousands)
        nfp = fred_latest('PAYEMS')
        nfp_series = fred_series('PAYEMS', 13)
        nfp_mom = None
        if len(nfp_series) >= 2:
            nfp_mom = safe_float(
                (nfp_series[-1]['value'] - nfp_series[-2]['value']) * 1000
            )

        # Initial jobless claims
        claims = fred_latest('ICSA')
        claims_series = fred_series('ICSA', 26)

        # Labor force participation rate
        lfpr = fred_latest('CIVPART')

        # JOLTS job openings (millions)
        jolts = fred_latest('JTSJOL')

        # Quits rate
        quits = fred_latest('JTSQUR')

        # Build quarterly snapshots for labor metrics (last 4 quarters)
        def quarterly_series(series_id, count=5):
            if not FRED_AVAILABLE or not FRED_API_KEY:
                return []
            try:
                from fredapi import Fred as FredApi
                f = FredApi(api_key=FRED_API_KEY)
                s = f.get_series(series_id).dropna()
                # Sample approx quarterly (every 3 months)
                result = []
                indices = range(min(count*3, len(s)), 0, -3)
                for i in indices:
                    try:
                        result.append({'date': str(s.index[-i].date()), 'value': safe_float(s.iloc[-i])})
                    except:
                        pass
                return list(reversed(result))
            except:
                return []

        def nfp_quarterly(count=5):
            """NFP MoM change for last N quarters (sampled monthly)"""
            if not nfp_series or len(nfp_series) < 4:
                return []
            result = []
            step = max(1, len(nfp_series) // count)
            for i in range(0, min(len(nfp_series)-1, count*step), step):
                try:
                    idx = min(i+1, len(nfp_series)-1)
                    chg = safe_float((nfp_series[idx]['value'] - nfp_series[i]['value']) * 1000)
                    result.append({'date': nfp_series[idx]['date'], 'value': chg})
                except:
                    pass
            return result[-4:]

        # Quarterly series — use already-cached FRED data
        try:
            lfpr_series     = quarterly_series('CIVPART', 5) if 'CIVPART' in _fred_series_cache else []
            unemp_quarterly = quarterly_series('UNRATE', 5)  if 'UNRATE'  in _fred_series_cache else []
            claims_quarterly= quarterly_series('ICSA', 5)    if 'ICSA'    in _fred_series_cache else []
            nfp_quarterly_data = nfp_quarterly()
        except:
            lfpr_series = []; unemp_quarterly = []; claims_quarterly = []; nfp_quarterly_data = []

        return {
            'unemployment': unemployment,
            'unemployment_series': unemployment_series,
            'unemp_quarterly': unemp_quarterly[-4:],
            'nfp_level': nfp,
            'nfp_mom': nfp_mom,
            'nfp_series': nfp_series,
            'nfp_quarterly': nfp_quarterly_data,
            'initial_claims': claims,
            'claims_series': claims_series,
            'claims_quarterly': claims_quarterly[-4:],
            'lfpr': lfpr,
            'lfpr_series': lfpr_series[-4:],
            'jolts_openings': jolts,
            'quits_rate': quits,
        }
    except Exception as e:
        print(f'  Labor error: {e}')
        return {}

def get_insider_activity():
    """Search for recent notable insider selling via web"""
    try:
        # We'll return a placeholder that gets populated by the narrative search
        return {
            'note': 'Insider activity sourced via web search during narrative generation',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    except:
        return {}

def get_all_macro():
    import time as _time
    t0 = _time.time()
    print(f'  [Macro] Starting parallel fetch...')

    # Pre-warm yfinance cache
    try:
        macro_tickers = ['^IRX','^TNX','^TYX','^VIX','DX-Y.NYB','HYG','LQD','TLT','SPY','RSP',
                         'XLK','XLF','XLV','XLE','XLI','XLY','XLP','XLU','XLRE','XLB','XLC']
        yf.download(macro_tickers, period='3mo', auto_adjust=True, progress=False, threads=True, group_by='ticker')
        print(f'  [Macro] yf prefetch done in {_time.time()-t0:.1f}s')
    except Exception as e:
        print(f'  [Macro] yf prefetch error: {e}')

    results = {}
    def run(key, fn):
        ts = _time.time()
        try:
            results[key] = cached(key, fn)
            print(f'  [Macro] {key} done in {_time.time()-ts:.1f}s')
        except Exception as e:
            print(f'  [Macro] {key} ERROR in {_time.time()-ts:.1f}s: {e}')
            results[key] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [
            ex.submit(run, 'rates',     get_rates_full),
            ex.submit(run, 'vix',       get_vix),
            ex.submit(run, 'sectors',   get_market_internals),
            ex.submit(run, 'breadth',   get_breadth),
            ex.submit(run, 'liquidity', get_liquidity),
            ex.submit(run, 'inflation', get_inflation),
        ]
        concurrent.futures.wait(futs, timeout=30)

    print(f'  [Macro] parallel done in {_time.time()-t0:.1f}s — starting labor...')
    ts = _time.time()
    results['labor'] = cached('labor', get_labor)
    print(f'  [Macro] labor done in {_time.time()-ts:.1f}s')
    print(f'  [Macro] TOTAL: {_time.time()-t0:.1f}s')

    return {
        'rates':     results.get('rates', {}),
        'vix':       results.get('vix', {}),
        'sectors':   results.get('sectors', {}),
        'breadth':   results.get('breadth', {}),
        'liquidity': results.get('liquidity', {}),
        'inflation': results.get('inflation', {}),
        'labor':     results.get('labor', {}),
        'insider':   get_insider_activity(),
        'timestamp': datetime.now().strftime('%B %d, %Y %I:%M %p'),
        'fred_connected': FRED_AVAILABLE and bool(FRED_API_KEY),
    }

# ── Flask routes ──

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'entry-point-scanner-v12.html')

@app.route('/macro')
def macro_page():
    return send_from_directory(BASE_DIR, 'macro_dashboard.html')

@app.route('/api/macro')
def api_macro():
    # Serve full pre-computed cache if available
    if _macro_cache['data']:
        return jsonify(_macro_cache['data'])
    # Serve partial section caches if available (background compute still running)
    partial = {k: _cache[k] for k in ['rates','vix','sectors','breadth','liquidity','inflation','labor'] if k in _cache}
    if partial:
        partial.update({'timestamp': datetime.now().strftime('%B %d, %Y %I:%M %p'),
                        'fred_connected': FRED_AVAILABLE and bool(FRED_API_KEY),
                        '_partial': True})
        return jsonify(partial)
    # First-ever load — run with 45s hard limit
    try:
        data = get_all_macro()
        _macro_cache['data'] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

@app.route('/api/macro/sectors')
def api_sectors():
    return jsonify(cached('sectors', get_market_internals))

@app.route('/api/macro/labor')
def api_labor():
    return jsonify(cached('labor', get_labor))

@app.route('/api/macro/inflation')
def api_inflation():
    return jsonify(cached('inflation', get_inflation))

@app.route('/api/macro/rates')
def api_rates():
    return jsonify(cached('rates', get_rates_full))

@app.route('/api/macro/liquidity')
def api_liquidity():
    return jsonify(cached('liquidity', get_liquidity))

@app.route('/api/macro/vix')
def api_vix():
    return jsonify(cached('vix', get_vix))

def get_ticker_performance(ticker, name):
    """Real Monthly, 3-Month, and 52-Week price performance for a single
    ticker — used to power the live discretionary-spending proxy cards
    (restaurants/DRI, cruise/CCL, hotels/MAR) instead of a static blurb."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='1y', interval='1d')['Close'].dropna()
        if hist.empty or len(hist) < 2:
            return {'ticker': ticker, 'name': name, 'error': 'No price data available'}

        def pct_change_over(days):
            if len(hist) <= days:
                sub = hist
            else:
                sub = hist.iloc[-days-1:]
            if len(sub) < 2 or sub.iloc[0] == 0:
                return None
            return round(((sub.iloc[-1] - sub.iloc[0]) / sub.iloc[0]) * 100, 2)

        current_price = round(float(hist.iloc[-1]), 2)
        return {
            'ticker': ticker,
            'name': name,
            'price': current_price,
            'change_1mo': pct_change_over(21),   # ~21 trading days/month
            'change_3mo': pct_change_over(63),    # ~63 trading days/quarter
            'change_52wk': pct_change_over(252),  # ~252 trading days/year
        }
    except Exception as e:
        print(f'  [TickerPerf] Failed to fetch {ticker}: {e}')
        return {'ticker': ticker, 'name': name, 'error': str(e)}

@app.route('/api/macro/restaurants')
def api_restaurants():
    return jsonify(cached('restaurants', lambda: get_ticker_performance('DRI', 'Darden Restaurants')))

@app.route('/api/macro/cruise')
def api_cruise():
    return jsonify(cached('cruise', lambda: get_ticker_performance('CCL', 'Carnival Corporation')))

@app.route('/api/macro/hotels')
def api_hotels():
    return jsonify(cached('hotels', lambda: get_ticker_performance('MAR', 'Marriott International')))

@app.route('/api/macro/narrative')
def api_narrative():
    """Generate narrative using available data"""
    data = get_all_macro()
    bullets = build_narrative(data)
    return jsonify({'narrative': bullets, 'timestamp': data['timestamp']})

# High-yield ETF tickers not in yfinance but available via Yahoo API
HY_ETF_FALLBACK = {
    'NVDY': {'name':'YieldMax NVDA Option Income','yield':38.0,'er':0.99},
    'TSLY': {'name':'YieldMax TSLA Option Income','yield':45.0,'er':0.99},
    'MSFO': {'name':'YieldMax MSFT Option Income','yield':22.0,'er':0.99},
    'AMZY': {'name':'YieldMax AMZN Option Income','yield':25.0,'er':0.99},
    'GOOY': {'name':'YieldMax GOOG Option Income','yield':24.0,'er':0.99},
    'FEBY': {'name':'YieldMax META Option Income','yield':28.0,'er':0.99},
    'APLY': {'name':'YieldMax AAPL Option Income','yield':21.0,'er':0.99},
    'CONY': {'name':'YieldMax COIN Option Income','yield':49.0,'er':0.99},
    'PLTY': {'name':'YieldMax PLTR Option Income','yield':44.0,'er':0.99},
    'AMDY': {'name':'YieldMax AMD Option Income','yield':35.0,'er':0.99},
    'JEPY': {'name':'Defiance S&P 500 Enhanced Option','yield':20.0,'er':0.99},
    'WDTE': {'name':'Roundhill S&P 500 0DTE CC','yield':22.0,'er':0.95},
    'XDTE': {'name':'Roundhill S&P 500 0DTE CC','yield':24.0,'er':0.95},
    'SPYT': {'name':'Defiance S&P 500 Target Inc','yield':21.0,'er':0.99},
    'ULTY': {'name':'YieldMax Ultra Option Inc','yield':46.0,'er':1.19},
    'YMAG': {'name':'YieldMax Mag 7 Fund','yield':30.0,'er':1.29},
    'FEAT': {'name':'Amplify Inflation Fighter','yield':20.5,'er':0.85},
}

def fetch_yahoo_quote_direct(ticker):
    """Fetch quote via Yahoo Finance v8 API directly — works for ETFs yfinance misses"""
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1y'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            # Try v7 endpoint
            url2 = f'https://query2.finance.yahoo.com/v7/finance/quote?symbols={ticker}'
            r2 = requests.get(url2, headers=headers, timeout=10)
            if r2.status_code == 200:
                data = r2.json()
                result = data.get('quoteResponse',{}).get('result',[])
                if result:
                    q = result[0]
                    price = q.get('regularMarketPrice')
                    prev  = q.get('regularMarketPreviousClose')
                    return {
                        'price': price, 'prev': prev,
                        'closes': [], 'volumes': [], 'highs': [], 'lows': [],
                        'dayHigh': q.get('regularMarketDayHigh'),
                        'dayLow': q.get('regularMarketDayLow'),
                        'open': q.get('regularMarketOpen'),
                    }
            return None
        data = r.json()
        chart = data.get('chart',{}).get('result',[])
        if not chart:
            return None
        meta = chart[0].get('meta',{})
        timestamps = chart[0].get('timestamp',[])
        quote_data = chart[0].get('indicators',{}).get('quote',[{}])[0]
        closes  = [v for v in (quote_data.get('close') or []) if v is not None]
        volumes = [v for v in (quote_data.get('volume') or []) if v is not None]
        highs   = [v for v in (quote_data.get('high') or []) if v is not None]
        lows    = [v for v in (quote_data.get('low') or []) if v is not None]
        price = meta.get('regularMarketPrice') or (closes[-1] if closes else None)
        prev  = meta.get('chartPreviousClose') or (closes[-2] if len(closes)>1 else price)
        if not price:
            return None
        return {
            'price': safe_float(price),
            'prev': safe_float(prev),
            'closes': [safe_float(c) for c in closes],
            'volumes': [safe_float(v) for v in volumes],
            'highs': [safe_float(h) for h in highs],
            'lows': [safe_float(l) for l in lows],
            'dayHigh': safe_float(meta.get('regularMarketDayHigh',price)),
            'dayLow': safe_float(meta.get('regularMarketDayLow',price)),
            'open': safe_float(closes[-2] if len(closes)>1 else price),
        }
    except Exception as e:
        print(f'  Direct quote error {ticker}: {e}')
        return None

@app.route('/api/etf/meta/<ticker>')
def etf_meta(ticker):
    """Return ETF full name and metadata from yfinance."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = (info.get('longName') or info.get('shortName') or ticker).strip()
        div_yield = safe_float(info.get('yield') or info.get('dividendYield'))
        if div_yield and div_yield < 1: div_yield = round(div_yield * 100, 2)
        aum = safe_float(info.get('totalAssets'))
        er  = normalize_expense_ratio(info.get('annualReportExpenseRatio') or info.get('expenseRatio'))
        return jsonify({'ticker': ticker, 'name': name, 'yield': div_yield, 'aum': aum, 'er': er})
    except Exception as e:
        return jsonify({'ticker': ticker, 'name': ticker, 'error': str(e)})

# ── Fund Structure lookup (manual, on-demand — see ETF detail panel) ──
# This is NOT auto-derived from EDGAR/yfinance. The physical-vs-synthetic-vs-ETN
# distinction lives in prose prospectus language, not structured filing data,
# so determining it requires actually reading the fund's own description.
# This endpoint just stores/serves whatever has been manually looked up and
# confirmed so far — it persists across sessions rather than re-researching
# the same ticker every time. Add entries via /api/etf/structure/save.
FUND_STRUCTURE_OPTIONS = ['underlying_assets', 'synthetic', 'physical_eln', 'etn', 'cef']
FUND_STRUCTURE_LABELS = {
    'underlying_assets': 'Underlying Assets/Index',
    'synthetic':         'Synthetic',
    'physical_eln':      'Underlying Assets/Index + ELN Overlay',
    'etn':               'ETN (Exchange-Traded Note)',
    'cef':               'CEF (Closed-End Fund)',
}

_fund_structure_file = None
def get_fund_structure_file():
    global _fund_structure_file
    if _fund_structure_file is None:
        _fund_structure_file = os.path.join(BASE_DIR, 'fund_structures.json')
    return _fund_structure_file

def load_fund_structures():
    f = get_fund_structure_file()
    if os.path.exists(f):
        try: return json.load(open(f))
        except: pass
    return {}

def save_fund_structures(data):
    json.dump(data, open(get_fund_structure_file(), 'w'), indent=2)

@app.route('/api/etf/structure/<ticker>')
def get_fund_structure(ticker):
    """Return the manually-confirmed structure for a ticker, if we have one."""
    ticker = ticker.upper()
    stored = load_fund_structures()
    entry = stored.get(ticker)
    if entry:
        return jsonify({'ticker': ticker, **entry, 'found': True})
    return jsonify({'ticker': ticker, 'found': False, 'options': FUND_STRUCTURE_OPTIONS, 'labels': FUND_STRUCTURE_LABELS})

@app.route('/api/etf/structure/save', methods=['POST'])
def save_fund_structure():
    """Save a manually-confirmed structure determination for a ticker.
    Expects JSON: {ticker, structure, note (optional), source_url (optional)}"""
    from flask import request as freq
    data = freq.json
    ticker = (data.get('ticker') or '').upper()
    structure = data.get('structure')
    if not ticker or structure not in FUND_STRUCTURE_OPTIONS:
        return jsonify({'error': f'ticker required and structure must be one of {FUND_STRUCTURE_OPTIONS}'}), 400

    stored = load_fund_structures()
    stored[ticker] = {
        'structure': structure,
        'label': FUND_STRUCTURE_LABELS[structure],
        'note': data.get('note', ''),
        'source_url': data.get('source_url', ''),
        'confirmed_date': datetime.now().strftime('%Y-%m-%d'),
    }
    save_fund_structures(stored)
    print(f'  [Structure] Saved {ticker} -> {structure}')
    return jsonify({'ok': True, 'ticker': ticker, **stored[ticker]})

@app.route('/api/etf/structure/list')
def list_fund_structures():
    """Return all manually-confirmed structures (for a reference table view)."""
    return jsonify(load_fund_structures())

# ── Bond Fund Data (credit quality, duration, maturity, asset mix) ──
# Replaces the equity treemap concept for bond funds — there's no meaningful
# way to "size tiles" for a bond portfolio the way we do for stock holdings,
# so this is an informational breakdown panel instead. Sourced from yfinance's
# funds_data.bond_ratings / bond_holdings / asset_classes, which are real,
# documented fields — but coverage is inconsistent fund-to-fund (e.g. some
# funds have Duration but not Maturity, or neither). Missing fields are
# reported honestly as null so the frontend can show "N/A" rather than a
# fabricated number.
def safe_round_pct(val, decimals=1):
    """Round a fraction to a percentage, clamping tiny negative/over-100 noise
    from Yahoo's own data (e.g. -0.0006 or 1.0008) to a clean 0-100 range."""
    if val is None:
        return None
    try:
        pct = float(val) * 100
        return round(max(0.0, min(100.0, pct)), decimals)
    except (TypeError, ValueError):
        return None

@app.route('/api/etf/bond-data/<ticker>')
def etf_bond_data(ticker):
    """Return bond-specific fund data: asset mix, credit quality breakdown,
    duration, maturity, and yield. Used for the bond fund info panel in place
    of the equity treemap."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        fd = t.funds_data

        # Asset mix (cash/stock/bond/preferred/convertible/other) as percentages
        asset_mix = {}
        try:
            ac = fd.asset_classes or {}
            asset_mix = {
                'cash': safe_round_pct(ac.get('cashPosition')),
                'stock': safe_round_pct(ac.get('stockPosition')),
                'bond': safe_round_pct(ac.get('bondPosition')),
                'preferred': safe_round_pct(ac.get('preferredPosition')),
                'convertible': safe_round_pct(ac.get('convertiblePosition')),
                'other': safe_round_pct(ac.get('otherPosition')),
            }
        except Exception as e:
            print(f'  [BondData] asset_classes failed for {ticker}: {e}')

        # Credit quality breakdown as percentages
        credit_quality = {}
        try:
            br = fd.bond_ratings or {}
            credit_quality = {
                'us_government': safe_round_pct(br.get('us_government')),
                'aaa':    safe_round_pct(br.get('aaa')),
                'aa':     safe_round_pct(br.get('aa')),
                'a':      safe_round_pct(br.get('a')),
                'bbb':    safe_round_pct(br.get('bbb')),
                'bb':     safe_round_pct(br.get('bb')),
                'b':      safe_round_pct(br.get('b')),
                'below_b': safe_round_pct(br.get('below_b')),
                'other':  safe_round_pct(br.get('other')),
                'nr':     safe_round_pct(br.get('not_rated') or br.get('nr')),
            }
        except Exception as e:
            print(f'  [BondData] bond_ratings failed for {ticker}: {e}')

        # Duration / Maturity — coverage is inconsistent; report null honestly
        # when missing rather than guessing, so the frontend shows "N/A".
        duration, maturity, credit_quality_summary = None, None, None
        try:
            bh = fd.bond_holdings
            if bh is not None and ticker in bh.columns:
                col = bh[ticker]
                dur_val = col.get('Duration')
                mat_val = col.get('Maturity')
                cq_val = col.get('Credit Quality')
                duration = float(dur_val) if dur_val is not None and str(dur_val) != '<NA>' else None
                maturity = float(mat_val) if mat_val is not None and str(mat_val) != '<NA>' else None
                credit_quality_summary = str(cq_val) if cq_val is not None and str(cq_val) != '<NA>' else None
        except Exception as e:
            print(f'  [BondData] bond_holdings failed for {ticker}: {e}')

        # Yield + category (category is a Morningstar-style label, e.g.
        # "Long Government" — the closest thing to a clean "bond type" tag
        # available; not as granular as Gov/Corp/Muni but a real signal)
        category, fund_family = None, None
        div_yield = None
        try:
            info = t.info
            div_yield = safe_float(info.get('yield') or info.get('dividendYield'))
            if div_yield and div_yield > 1:  # already a percent (e.g. 4.55 not 0.0455)
                pass
            elif div_yield:
                div_yield = round(div_yield * 100, 2)
            category = info.get('category')
            fund_family = info.get('fundFamily')
        except Exception as e:
            print(f'  [BondData] info/yield failed for {ticker}: {e}')

        if not category:
            try:
                category = fd.fund_overview.get('categoryName')
                fund_family = fund_family or fd.fund_overview.get('family')
            except Exception:
                pass

        return jsonify({
            'ticker': ticker,
            'category': category,
            'fund_family': fund_family,
            'yield': div_yield,
            'asset_mix': asset_mix,
            'credit_quality': credit_quality,
            'duration': duration,
            'maturity': maturity,
            'credit_quality_summary': credit_quality_summary,
        })
    except Exception as e:
        print(f'  [BondData] Overall failure for {ticker}: {e}')
        return jsonify({'ticker': ticker, 'error': str(e)}), 500

# ── Dividend Growth Streak calculation ──
# Used to auto-classify "Dividend Growth" funds: 3+ consecutive years of the
# FUND'S OWN total annual distribution rising. This uses the fund's own
# payout history (not the underlying holdings' individual streaks, which
# would require checking every holding — a much bigger, less practical lift).
def calculate_dividend_growth_streak(dividends_series, as_of=None):
    """
    Resample a yfinance dividends Series to annual totals, then count
    consecutive years (most recent COMPLETE year first) where each year's
    total >= the prior year's total. Excludes the current in-progress
    calendar year from the comparison since a partial year would unfairly
    look like a decline.
    Returns: (streak_years:int, annual_totals:dict of {year: total})
    """
    if dividends_series is None or len(dividends_series) == 0:
        return 0, {}

    # yfinance's dividends Series index is typically timezone-AWARE (localized
    # to the exchange's timezone). Comparing it against a timezone-naive
    # pd.Timestamp throws "Cannot compare tz-naive and tz-aware timestamps".
    # Match the series' own tz (or lack thereof) for every comparison timestamp
    # we construct, rather than assuming naive.
    series_tz = dividends_series.index.tz

    if as_of is None:
        as_of = pd.Timestamp.now(tz=series_tz) if series_tz else pd.Timestamp.now()
    elif series_tz and as_of.tzinfo is None:
        as_of = as_of.tz_localize(series_tz)
    elif not series_tz and as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)

    annual = dividends_series.resample('YE').sum()
    annual = annual[annual > 0]

    def make_ts(year, month, day):
        ts = pd.Timestamp(year=year, month=month, day=day)
        return ts.tz_localize(series_tz) if series_tz else ts

    current_year_end = make_ts(as_of.year, 12, 31)
    current_year_start = make_ts(as_of.year, 1, 1)
    if len(annual) > 0 and annual.index[-1] >= current_year_start and as_of < current_year_end:
        annual = annual.iloc[:-1]  # drop incomplete current year

    if len(annual) == 0:
        return 0, {}
    if len(annual) == 1:
        return 1, {annual.index[0].year: round(float(annual.iloc[0]), 4)}

    streak = 1
    for i in range(len(annual) - 1, 0, -1):
        if annual.iloc[i] >= annual.iloc[i-1]:
            streak += 1
        else:
            break

    annual_dict = {d.year: round(float(v), 4) for d, v in annual.items()}
    return streak, annual_dict


@app.route('/api/etf/dividend-streak/<ticker>')
def etf_dividend_streak(ticker):
    """Return the fund's own dividend growth streak (consecutive years of
    rising total annual distributions), used to auto-classify Dividend
    Growth funds (3+ year streak threshold)."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        streak, annual = calculate_dividend_growth_streak(divs)
        return jsonify({
            'ticker': ticker,
            'streak_years': streak,
            'annual_totals': annual,
            'qualifies_dividend_growth': streak >= 3,
        })
    except Exception as e:
        print(f'  [DivStreak] Failed for {ticker}: {e}')
        return jsonify({'ticker': ticker, 'error': str(e), 'streak_years': 0, 'qualifies_dividend_growth': False}), 500

# ── Dividend Growth & Income panel data ──
# Total-return CAGR (3/5/10yr), dividend frequency, dividend rate, inception
# date, and 1/3/5-year dividend history for the chart. Total return uses
# auto_adjust=True (dividends reinvested into the price series) per the
# methodology we verified earlier for the returns endpoint.
def calculate_cagr(total_return_series, years):
    """CAGR % over the given lookback, or None if insufficient history
    (returns N/A on the frontend rather than a fabricated number)."""
    if total_return_series.empty:
        return None
    latest_date = total_return_series.index[-1]
    target_date = latest_date - pd.DateOffset(years=years)
    valid = total_return_series[total_return_series.index <= target_date]
    if valid.empty:
        return None
    start_price = valid.iloc[-1]
    end_price = total_return_series.iloc[-1]
    actual_years = (latest_date - valid.index[-1]).days / 365.25
    if start_price <= 0 or actual_years <= 0:
        return None
    return round(((end_price / start_price) ** (1/actual_years) - 1) * 100, 2)


def detect_dividend_frequency(dividend_dates):
    """Classify payment cadence from the median gap between the most recent
    payments. Using the last 12 payments (not just 2) makes this robust to
    one-off special/irregular distributions skewing the result."""
    if len(dividend_dates) < 2:
        return 'Unknown'
    recent = sorted(dividend_dates)[-12:]
    gaps_days = [(recent[i] - recent[i-1]).days for i in range(1, len(recent))]
    median_gap = statistics.median(gaps_days)
    if median_gap <= 9:    return 'Weekly'
    elif median_gap <= 40: return 'Monthly'
    elif median_gap <= 100: return 'Quarterly'
    elif median_gap <= 200: return 'Semi-Annual'
    elif median_gap <= 400: return 'Annual'
    else: return 'Irregular'


def fetch_businessquant_dividends(ticker):
    """Fetch dividend history from Business Quant's free Dividends API —
    gives us ex_date AND payment_date together for both stocks and ETFs,
    which yfinance can't reliably provide (it usually only has ex-date).
    Returns None if the key isn't configured or the request fails, so
    callers can fall back to yfinance gracefully."""
    if not BUSINESSQUANT_API_KEY:
        print(f'  [BusinessQuant] {ticker}: no API key configured, skipping')
        return None
    try:
        r = requests.get(f'{BUSINESSQUANT_BASE_URL}/dividends',
                         params={'ticker': ticker, 'mode': 'dps', 'api_key': BUSINESSQUANT_API_KEY},
                         timeout=15)
        if r.status_code != 200:
            print(f'  [BusinessQuant] {ticker}: dividends fetch returned {r.status_code}: {r.text[:200]}')
            return None
        data = r.json()
        record_count = len(data.get('data', []))
        print(f'  [BusinessQuant] {ticker}: SUCCESS — {record_count} dividend records, '
              f'metadata={data.get("metadata", {})}')
        return data
    except Exception as e:
        print(f'  [BusinessQuant] {ticker}: dividends fetch failed: {e}')
        return None


@app.route('/api/etf/dividend-growth-income/<ticker>')
def etf_dividend_growth_income(ticker):
    """Comprehensive data for the 'Dividend Growth & Income' detail panel:
    CAGR (3/5/10yr total return), consecutive dividend growth years, dividend
    frequency, current dividend rate, ex-dividend AND payment date, inception
    date, and 1/3/5-year dividend payment history for the chart.

    Dividend mechanics (ex-date, payment date, per-payment amounts) come from
    Business Quant's free Dividends API when configured — it reliably has
    BOTH dates for stocks and ETFs, unlike yfinance which usually only has
    ex-date. CAGR/price-return calculations still use yfinance's price history,
    since that's working well and doesn't need replacing."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)

        # Total-return price series (dividends reinvested) for CAGR — still yfinance
        total_hist = t.history(period='max', interval='1d', auto_adjust=True)['Close']
        cagr_3 = calculate_cagr(total_hist, 3) if not total_hist.empty else None
        cagr_5 = calculate_cagr(total_hist, 5) if not total_hist.empty else None
        cagr_10 = calculate_cagr(total_hist, 10) if not total_hist.empty else None

        # ── Dividend mechanics: try Business Quant first, fall back to yfinance ──
        bq_data = fetch_businessquant_dividends(ticker)
        div_source = None
        div_history = []          # [{date, amount, ex_date, payment_date}]
        div_dates_for_freq = []   # used for frequency detection / streak calc
        next_ex_div_date, last_ex_div_date, last_payment_date = None, None, None
        dividend_rate = None

        if bq_data and bq_data.get('data'):
            div_source = 'businessquant'
            records = bq_data['data']  # newest first, per their docs example
            for rec in records:
                div_history.append({
                    'date': rec['ex_date'],  # keep 'date' key for chart compatibility
                    'amount': round(float(rec['dividend']), 4),
                    'ex_date': rec['ex_date'],
                    'payment_date': rec.get('payment_date'),
                })
                div_dates_for_freq.append(datetime.strptime(rec['ex_date'], '%Y-%m-%d'))
            if records:
                last_ex_div_date = records[0]['ex_date']
                last_payment_date = records[0].get('payment_date')
            next_ex_div_date = bq_data.get('metadata', {}).get('nextdividend')
            ttm = bq_data.get('metadata', {}).get('ttmdividend')
            if ttm:
                dividend_rate = round(float(ttm), 4)

        if not div_history:
            # Fallback: yfinance dividends (ex-date only, no payment date)
            div_source = 'yfinance'
            divs = t.dividends
            if len(divs) > 0:
                for d, v in divs.items():
                    div_history.append({'date': d.strftime('%Y-%m-%d'), 'amount': round(float(v), 4),
                                        'ex_date': d.strftime('%Y-%m-%d'), 'payment_date': None})
                    div_dates_for_freq.append(d)
                last_ex_div_date = div_history[-1]['ex_date']  # yfinance order is oldest-first

        frequency = detect_dividend_frequency(div_dates_for_freq) if len(div_dates_for_freq) >= 2 else 'Unknown'
        streak, _ = calculate_dividend_growth_streak(pd.Series(
            [h['amount'] for h in div_history],
            index=pd.to_datetime([h['date'] for h in div_history])
        ).sort_index()) if div_history else (0, {})  # .sort_index() — Business Quant returns newest-first, but resample('YE') needs chronological order

        # If Business Quant didn't give us a TTM-based rate, fall back to the
        # most-recent-payment × frequency-multiplier estimate
        if dividend_rate is None and div_history:
            freq_multiplier = {'Weekly': 52, 'Monthly': 12, 'Quarterly': 4, 'Semi-Annual': 2, 'Annual': 1}.get(frequency)
            if freq_multiplier:
                most_recent_amount = div_history[0]['amount'] if div_source == 'businessquant' else div_history[-1]['amount']
                dividend_rate = round(most_recent_amount * freq_multiplier, 4)

        ex_div_date = next_ex_div_date or last_ex_div_date
        ex_div_is_upcoming = bool(next_ex_div_date)
        payout_date = last_payment_date  # genuinely available now via Business Quant

        # Inception date — still yfinance
        inception = None
        try:
            first_trade_ts = t.info.get('fundInceptionDate') or t.info.get('firstTradeDateEpochUtc')
            if first_trade_ts:
                inception = datetime.fromtimestamp(first_trade_ts).strftime('%Y-%m-%d')
        except Exception:
            pass
        if not inception and not total_hist.empty:
            inception = total_hist.index[0].strftime('%Y-%m-%d')

        print(f'  [DivGrowthIncome] {ticker}: FINAL VALUES — cagr_3y={cagr_3}, cagr_5y={cagr_5}, '
              f'cagr_10y={cagr_10}, div_source={div_source}, payout_date={payout_date}, '
              f'ex_div_date={ex_div_date}, div_history_len={len(div_history)}, '
              f'total_hist_len={len(total_hist) if not total_hist.empty else 0}')

        return jsonify({
            'ticker': ticker,
            'cagr_3y': cagr_3, 'cagr_5y': cagr_5, 'cagr_10y': cagr_10,
            'consecutive_div_years': streak,
            'dividend_frequency': frequency,
            'dividend_rate': dividend_rate,
            'ex_dividend_date': ex_div_date,
            'ex_dividend_is_upcoming': ex_div_is_upcoming,
            'payout_date': payout_date,
            'inception_date': inception,
            'dividend_history': div_history,
            'dividend_data_source': div_source,
        })
    except Exception as e:
        print(f'  [DivGrowthIncome] Failed for {ticker}: {e}')
        return jsonify({'ticker': ticker, 'error': str(e)}), 500

# ── Upside/Downside Capture Ratio ──
# Standard fund-performance metric: how much of the benchmark's gains a fund
# captures during up-months, vs. how much of its losses it captures during
# down-months. Uses the industry-standard compounded (geometric) method, not
# a simple average — e.g. monthly returns of +3%, +2%, +4% compound to 9.24%,
# not 3%. Verified against a known-construction test case before building this.
def calculate_capture_ratios(fund_returns, benchmark_returns):
    """
    fund_returns, benchmark_returns: pandas Series of monthly returns
    (decimals, e.g. 0.03 for 3%), aligned on the same date index.
    Returns: (upside_capture, downside_capture, up_months_count, down_months_count)
    as percentages, or None for either ratio if there were zero up/down months
    in the lookback window (rare, but possible for a short window).
    """
    df = pd.DataFrame({'fund': fund_returns, 'bench': benchmark_returns}).dropna()
    up_months = df[df['bench'] > 0]
    down_months = df[df['bench'] < 0]

    def compound(returns):
        if len(returns) == 0:
            return None
        return (1 + returns).prod() - 1

    fund_up, bench_up = compound(up_months['fund']), compound(up_months['bench'])
    fund_down, bench_down = compound(down_months['fund']), compound(down_months['bench'])

    upside = round(fund_up / bench_up * 100, 1) if bench_up else None
    downside = round(fund_down / bench_down * 100, 1) if bench_down else None

    return upside, downside, len(up_months), len(down_months)


@app.route('/api/etf/capture-ratio/<ticker>')
def etf_capture_ratio(ticker):
    """Return upside/downside capture ratios vs SPY over the available
    history (up to 5 years of monthly data, industry-standard lookback).
    A fund capturing >100% upside and <100% downside (like CGDV's real
    103/83 profile) has a favorable risk/return asymmetry."""
    ticker = ticker.upper()
    try:
        # Pull ~5 years of daily data, then resample to monthly — gives us
        # the same monthly-return convention used by Morningstar/industry standard
        fund_hist = yf.Ticker(ticker).history(period='5y', interval='1d')['Close']
        bench_hist = yf.Ticker('SPY').history(period='5y', interval='1d')['Close']

        if fund_hist.empty or bench_hist.empty:
            return jsonify({'ticker': ticker, 'error': 'No price history available'}), 404

        fund_monthly = fund_hist.resample('ME').last().pct_change().dropna()
        bench_monthly = bench_hist.resample('ME').last().pct_change().dropna()

        print(f'  [CaptureRatio] {ticker}: fund_monthly has {len(fund_monthly)} points, '
              f'index sample: {list(fund_monthly.index[:3])}')
        print(f'  [CaptureRatio] {ticker}: bench_monthly has {len(bench_monthly)} points, '
              f'index sample: {list(bench_monthly.index[:3])}')
        print(f'  [CaptureRatio] {ticker}: fund tz={fund_monthly.index.tz}, bench tz={bench_monthly.index.tz}')

        upside, downside, n_up, n_down = calculate_capture_ratios(fund_monthly, bench_monthly)
        print(f'  [CaptureRatio] {ticker}: upside={upside}, downside={downside}, n_up={n_up}, n_down={n_down}')

        return jsonify({
            'ticker': ticker,
            'benchmark': 'SPY',
            'upside_capture': upside,
            'downside_capture': downside,
            'capture_spread': round(upside - downside, 1) if (upside is not None and downside is not None) else None,
            'up_months_analyzed': n_up,
            'down_months_analyzed': n_down,
            'lookback': '5y (or less if fund history is shorter)',
        })
    except Exception as e:
        print(f'  [CaptureRatio] Failed for {ticker}: {e}')
        return jsonify({'ticker': ticker, 'error': str(e)}), 500

# ── 1/3/6-Month Price Return AND Total Return ──
# Price return = pure price change (no dividends). Total return = price +
# reinvested dividends. These genuinely differ for income-focused funds —
# a high-yield ETF can show a flat/negative price return while still having
# a solidly positive total return once distributions are counted, so showing
# both side by side is meaningfully more informative than either alone.
#
# Methodology: auto_adjust=True gives total-return-adjusted prices (dividends
# + splits baked into the series); auto_adjust=False gives raw traded price.
# Verified this is yfinance's documented, correct convention before building.
@app.route('/api/etf/returns/<ticker>')
def etf_returns(ticker):
    """Return 1/3/6-month price return and total return for a ticker."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        # 7 months of daily history gives enough buffer to find a trading
        # day on/near each lookback boundary even across weekends/holidays
        price_hist = t.history(period='7mo', interval='1d', auto_adjust=False)['Close']
        total_hist = t.history(period='7mo', interval='1d', auto_adjust=True)['Close']

        if price_hist.empty or total_hist.empty:
            return jsonify({'ticker': ticker, 'error': 'No price history available'}), 404

        def return_over_months(series, months):
            """Find the closing value ~N months back and compute % change
            to the most recent close. Uses the closest available trading day
            on/before the target date, since exact calendar dates often land
            on weekends/holidays with no trading data."""
            if len(series) < 2:
                return None
            latest_date = series.index[-1]
            target_date = latest_date - pd.DateOffset(months=months)
            valid = series[series.index <= target_date]
            if valid.empty:
                return None  # not enough history for this lookback
            start_price = valid.iloc[-1]
            end_price = series.iloc[-1]
            if start_price == 0:
                return None
            return round((end_price / start_price - 1) * 100, 2)

        result = {
            'ticker': ticker,
            'price_return_1m': return_over_months(price_hist, 1),
            'price_return_3m': return_over_months(price_hist, 3),
            'price_return_6m': return_over_months(price_hist, 6),
            'total_return_1m': return_over_months(total_hist, 1),
            'total_return_3m': return_over_months(total_hist, 3),
            'total_return_6m': return_over_months(total_hist, 6),
        }
        print(f'  [Returns] {ticker}: 1m={result["total_return_1m"]}, 3m={result["total_return_3m"]}, 6m={result["total_return_6m"]}')
        return jsonify(result)
    except Exception as e:
        print(f'  [Returns] Failed for {ticker}: {e}')
        return jsonify({'ticker': ticker, 'error': str(e)}), 500

# ── Fund Designation overrides (Passive/Active/Thematic/Dividend Growth/etc) ──
# The default designation is computed client-side (yield-based rules + the
# dividend streak check above); this just persists any manual override the
# person picks from the dropdown so it survives across sessions.
_fund_designations_file = None
def get_fund_designations_file():
    global _fund_designations_file
    if _fund_designations_file is None:
        _fund_designations_file = os.path.join(BASE_DIR, 'fund_designations.json')
    return _fund_designations_file

def load_fund_designations():
    f = get_fund_designations_file()
    if os.path.exists(f):
        try: return json.load(open(f))
        except: pass
    return {}

def save_fund_designations_to_disk(data):
    json.dump(data, open(get_fund_designations_file(), 'w'), indent=2)

@app.route('/api/etf/designation/list')
def list_fund_designations():
    return jsonify(load_fund_designations())

@app.route('/api/etf/designation/save', methods=['POST'])
def save_fund_designation():
    from flask import request as freq
    data = freq.json
    ticker = (data.get('ticker') or '').upper()
    designation = data.get('designation')
    valid_designations = ['Passive','Active','Thematic','Dividend Growth','Dividend',
                          'Mid Yield Dividend','High Yield Dividend','Ultra High Yield Dividend',
                          'Bond','Fixed Income']
    if not ticker or designation not in valid_designations:
        return jsonify({'error': f'ticker required and designation must be one of {valid_designations}'}), 400

    stored = load_fund_designations()
    stored[ticker] = designation
    save_fund_designations_to_disk(stored)
    print(f'  [Designation] {ticker} -> {designation}')
    return jsonify({'ok': True, 'ticker': ticker, 'designation': designation})

# ── Custom ETF persistence ──
_custom_etfs_file = None

def get_custom_etfs_file():
    global _custom_etfs_file
    if _custom_etfs_file is None:
        _custom_etfs_file = os.path.join(BASE_DIR, 'custom_etfs.json')
    return _custom_etfs_file

def load_custom_etfs():
    """Returns a flat list of custom ETF entries. The old format nested
    these by category ({cat: [entries]}); that distinction no longer
    matters now that taxonomy is derived automatically, so old-format files
    are flattened transparently here."""
    f = get_custom_etfs_file()
    if os.path.exists(f):
        try:
            data = json.load(open(f))
            if isinstance(data, dict):
                # Old nested-by-category format — flatten it
                flat = []
                for entries in data.values():
                    flat.extend(entries)
                return flat
            return data  # already a flat list
        except: pass
    return []

def save_custom_etfs(data):
    json.dump(data, open(get_custom_etfs_file(), 'w'), indent=2)

@app.route('/api/etf/custom/list')
def list_custom_etfs():
    return jsonify(load_custom_etfs())

@app.route('/api/etf/custom/save', methods=['POST'])
def save_custom_etf():
    from flask import request as freq
    data = freq.json
    entry = data.get('entry', {})
    if not entry.get('t'):
        return jsonify({'error': 'entry.t required'}), 400
    stored = load_custom_etfs()
    existing = next((i for i,e in enumerate(stored) if e['t']==entry['t']), None)
    if existing is not None: stored[existing] = entry
    else: stored.append(entry)
    save_custom_etfs(stored)
    print(f'  [Custom ETF] Saved {entry["t"]} ({entry.get("n","")})')
    return jsonify({'ok': True, 'ticker': entry['t']})

@app.route('/api/etf/custom/delete', methods=['POST'])
def delete_custom_etf():
    from flask import request as freq
    data = freq.json
    ticker = data.get('ticker','').upper()
    stored = load_custom_etfs()
    stored = [e for e in stored if e['t'] != ticker]
    save_custom_etfs(stored)
    return jsonify({'ok': True})

@app.route('/api/etf/holdings/<ticker>')
def etf_holdings(ticker):
    """Return ETF holdings — from uploaded file store first, then EDGAR (full holdings,
    refreshed monthly), then yfinance/Yahoo as fallback if EDGAR has no data for this ticker."""
    ticker = ticker.upper()

    # Check pre-loaded store first (uploaded xlsx/csv files) — user override always wins
    if ticker in _etf_holdings_store:
        print(f'  [Holdings] {ticker}: serving {len(_etf_holdings_store[ticker])} holdings from file store')
        return jsonify({'ticker': ticker, 'holdings': _etf_holdings_store[ticker], 'source': 'uploaded_file'})

    # Try EDGAR (full holdings, cached to disk, refreshed monthly)
    edgar_holdings = get_edgar_holdings_cached(ticker)
    if edgar_holdings:
        print(f'  [Holdings] {ticker}: serving {len(edgar_holdings)} holdings from EDGAR cache')
        return jsonify({'ticker': ticker, 'holdings': edgar_holdings, 'source': 'sec_edgar'})

    holdings = []

    # Method 1: yfinance funds_data.top_holdings (works for ETFs)
    try:
        t = yf.Ticker(ticker)
        fd = t.funds_data
        th = fd.top_holdings
        if th is not None and not th.empty:
            print(f'  [Holdings] {ticker} top_holdings columns: {list(th.columns)}')
            for sym, row in th.iterrows():
                # Column name varies — try common ones
                pct = None
                for col in ['Holding Percent', 'holdingPercent', 'percent', row.index[0] if len(row) else None]:
                    if col and col in row.index:
                        pct = float(row[col])
                        break
                if pct is None and len(row) > 0:
                    pct = float(row.iloc[0])
                name = sym  # use ticker as name fallback
                # Try to get a better name from NAMES dict on client side
                if pct is not None and pct > 0:
                    # yfinance returns as decimal (0.07) or percent (7.0) — normalize
                    w = pct * 100 if pct < 1 else pct
                    holdings.append({'t': str(sym), 'n': str(sym), 'w': round(float(w), 2)})
            if holdings:
                print(f'  [Holdings] {ticker}: {len(holdings)} holdings via yfinance')
                return jsonify({'ticker': ticker, 'holdings': holdings[:30], 'source': 'yfinance'})
    except Exception as e:
        print(f'  [Holdings] yfinance failed for {ticker}: {e}')

    # Method 2: Yahoo Finance quoteSummary v10
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=topHoldings'
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code == 200:
            data = r.json()
            top = data.get('quoteSummary',{}).get('result',[{}])[0].get('topHoldings',{})
            raw = top.get('holdings', [])
            print(f'  [Holdings] {ticker} Yahoo v10: {len(raw)} raw holdings')
            for h in raw:
                sym  = h.get('symbol','')
                name = h.get('holdingName', sym)
                pct  = h.get('holdingPercent', 0)
                if sym and pct:
                    w = float(pct)*100 if float(pct) < 1 else float(pct)
                    holdings.append({'t': sym, 'n': name, 'w': round(w, 2)})
            if holdings:
                return jsonify({'ticker': ticker, 'holdings': holdings[:30], 'source': 'yahoo_v10'})
    except Exception as e:
        print(f'  [Holdings] Yahoo v10 failed for {ticker}: {e}')

    print(f'  [Holdings] {ticker}: all methods failed, returning 404')
    return jsonify({'ticker': ticker, 'holdings': [],
                    'error': 'Live holdings unavailable'}), 404

def fetch_stooq_quote(ticker):
    """Fallback price source when yfinance fails (e.g. Yahoo's 'Invalid Crumb'
    anti-bot errors, which recur periodically and have no permanent fix).
    Stooq is free, no API key, no rate limiting drama — just a plain CSV
    download. Returns the same shape as our normal quote dict, or None if
    Stooq also has nothing for this ticker.

    Stooq tickers use a .us suffix for US equities/ETFs (e.g. 'spy.us').
    """
    try:
        stooq_ticker = f'{ticker.lower()}.us'
        url = f'https://stooq.com/q/d/l/?s={stooq_ticker}&i=d'
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or 'Date' not in r.text[:50]:
            return None  # Stooq returns a plain "N/D" or similar when ticker not found

        import csv as csv_module
        from io import StringIO
        reader = csv_module.DictReader(StringIO(r.text))
        rows = list(reader)
        if not rows or len(rows) < 2:
            return None

        closes  = [float(row['Close'])  for row in rows if row.get('Close')]
        highs   = [float(row['High'])   for row in rows if row.get('High')]
        lows    = [float(row['Low'])    for row in rows if row.get('Low')]
        volumes = [float(row['Volume']) for row in rows if row.get('Volume')]
        opens   = [float(row['Open'])   for row in rows if row.get('Open')]

        if len(closes) < 2:
            return None

        return {
            'ticker': ticker.upper(),
            'price': closes[-1], 'prev': closes[-2],
            'dayHigh': highs[-1], 'dayLow': lows[-1], 'open': opens[-1],
            'closes': closes, 'volumes': volumes, 'highs': highs, 'lows': lows,
            'yield': None, 'aum': None, 'er': None,  # Stooq doesn't provide fund metadata
            'source': 'stooq_fallback',
        }
    except Exception as e:
        print(f'  [Stooq] Fallback fetch failed for {ticker}: {e}')
        return None


@app.route('/api/etf/holdings/batch', methods=['POST'])
def etf_holdings_batch():
    """Fetch price+prev for all holding tickers using yf.download batch — fast and complete."""
    from flask import request as freq
    tickers = freq.json.get('tickers', [])
    if not tickers:
        return jsonify({})

    results = {}

    # Use yf.download for batch — handles 500+ tickers efficiently
    try:
        df = yf.download(
            tickers,
            period='5d',
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by='ticker'
        )
        for t in tickers:
            try:
                if len(tickers) == 1:
                    closes = df['Close'].dropna().tolist()
                else:
                    closes = df[t]['Close'].dropna().tolist() if t in df.columns.get_level_values(0) else []
                if len(closes) >= 2:
                    results[t] = {'price': round(float(closes[-1]),4), 'prev': round(float(closes[-2]),4)}
                elif len(closes) == 1:
                    results[t] = {'price': round(float(closes[-1]),4), 'prev': round(float(closes[-1]),4)}
                else:
                    results[t] = {'price': None, 'prev': None}
            except:
                results[t] = {'price': None, 'prev': None}
        print(f'  [Batch] Got prices for {sum(1 for v in results.values() if v["price"])} / {len(tickers)} tickers')
        return jsonify(results)
    except Exception as e:
        print(f'  [Batch] yf.download failed: {e}, falling back to fast_info')

    # Fallback: parallel fast_info for all tickers
    def fetch_one(t):
        try:
            info = yf.Ticker(t).fast_info
            price = float(info.last_price)     if info.last_price     else None
            prev  = float(info.previous_close) if info.previous_close else None
            return t, {'price': price, 'prev': prev}
        except:
            return t, {'price': None, 'prev': None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for t, r in ex.map(fetch_one, tickers):
            results[t] = r

    print(f'  [Batch] Fallback: got {sum(1 for v in results.values() if v["price"])} / {len(tickers)}')

    # Third tier: for any tickers still missing a price (often a Yahoo "Invalid
    # Crumb" anti-bot block affecting the whole batch), try Stooq individually.
    missing = [t for t, v in results.items() if not v.get('price')]
    if missing:
        print(f'  [Batch] {len(missing)} tickers still missing prices, trying Stooq fallback...')
        def fetch_stooq_one(t):
            sq = fetch_stooq_quote(t)
            if sq:
                return t, {'price': sq['price'], 'prev': sq['prev']}
            return t, results[t]  # keep the None/None we already had

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for t, r in ex.map(fetch_stooq_one, missing):
                results[t] = r
        recovered = sum(1 for t in missing if results[t].get('price'))
        print(f'  [Batch] Stooq recovered {recovered}/{len(missing)} tickers')

    return jsonify(results)


@app.route('/api/etf/holdings/top25/<ticker>')
def etf_holdings_top25(ticker):
    """Return the top 25 holdings (or fewer if the ETF has less) by weight,
    WITH live price + previous close for each — everything the treemap needs
    in a single call. Full holdings list is still available unfiltered via
    /api/etf/holdings/<ticker> for the 'view all' link."""
    ticker = ticker.upper()

    # Reuse the same source-priority logic as the full holdings endpoint:
    # uploaded file > EDGAR cache > yfinance/Yahoo fallback
    full_holdings = None
    source = None

    if ticker in _etf_holdings_store:
        full_holdings = _etf_holdings_store[ticker]
        source = 'uploaded_file'
    else:
        edgar_holdings = get_edgar_holdings_cached(ticker)
        if edgar_holdings:
            full_holdings = edgar_holdings
            source = 'sec_edgar'

    if not full_holdings:
        return jsonify({'ticker': ticker, 'holdings': [], 'error': 'No holdings data available'}), 404

    # Already sorted by weight descending from upstream; take top 25 (or fewer)
    top25 = sorted(full_holdings, key=lambda h: -h.get('w', 0))[:25]

    # Cash-equivalents (Treasuries, repo agreements, etc.) and option/derivative
    # positions both get synthetic tickers with no real trading symbol — skip
    # both from the price lookup rather than sending garbage symbols to yfinance.
    real_tickers = [h for h in top25 if not h.get('is_cash_equiv') and not h.get('is_option')]
    tickers_needed = [h['t'] for h in real_tickers]

    # Fetch live prices for just these tickers
    price_map = {}
    if tickers_needed:
        try:
            df = yf.download(tickers_needed, period='5d', auto_adjust=True,
                             progress=False, threads=True, group_by='ticker')
            for t in tickers_needed:
                try:
                    if len(tickers_needed) == 1:
                        closes = df['Close'].dropna().tolist()
                    else:
                        closes = df[t]['Close'].dropna().tolist() if t in df.columns.get_level_values(0) else []
                    if len(closes) >= 2:
                        price_map[t] = {'price': round(float(closes[-1]),4), 'prev': round(float(closes[-2]),4)}
                    elif len(closes) == 1:
                        price_map[t] = {'price': round(float(closes[-1]),4), 'prev': round(float(closes[-1]),4)}
                except Exception:
                    pass
        except Exception as e:
            print(f'  [Top25] Batch price fetch failed for {ticker}: {e}')

    # Merge weight + price + day change into one clean record per holding.
    # Cash-equivalents and option positions both get chg_pct=None (no price
    # to compare) and a flag so the frontend can render them distinctly.
    out = []
    for h in top25:
        if h.get('is_cash_equiv'):
            out.append({
                't': h['t'], 'n': h['n'], 'w': h['w'],
                'price': None, 'prev': None, 'chg_pct': None, 'is_cash_equiv': True,
            })
            continue
        if h.get('is_option'):
            out.append({
                't': h['t'], 'n': h['n'], 'w': h['w'],
                'price': None, 'prev': None, 'chg_pct': None, 'is_option': True,
                'option_detail': h.get('option_detail'),
            })
            continue
        if h.get('no_ticker'):
            # Foreign holding with no resolvable US ticker (e.g. SK Hynix,
            # Samsung Electronics) — there's nothing to look up a price for,
            # so skip straight to passing through the name/weight, same as
            # the cash-equivalent and option branches above.
            out.append({
                't': h['t'], 'n': h['n'], 'w': h['w'],
                'price': None, 'prev': None, 'chg_pct': None, 'no_ticker': True,
            })
            continue
        p = price_map.get(h['t'], {})
        price, prev = p.get('price'), p.get('prev')
        chg_pct = round(((price - prev) / prev) * 100, 2) if price and prev else None
        out.append({
            't': h['t'], 'n': h['n'], 'w': h['w'],
            'price': price, 'prev': prev, 'chg_pct': chg_pct,
        })

    print(f'  [Top25] {ticker}: top {len(out)} of {len(full_holdings)} total holdings, source={source}')
    return jsonify({
        'ticker': ticker, 'holdings': out, 'source': source,
        'total_holdings_available': len(full_holdings),
    })


@app.route('/api/etf/holdings/upload', methods=['POST'])
def upload_holdings():
    """Accept xlsx or csv holdings file, parse, and store persistently."""
    from flask import request as freq
    import io, csv

    ticker = freq.form.get('ticker','').upper().strip()
    if not ticker:
        return jsonify({'error': 'ticker required'}), 400

    file = freq.files.get('file')
    if not file:
        return jsonify({'error': 'file required'}), 400

    filename = file.filename.lower()
    holdings = []

    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            import pandas as pd
            # Try multiple header row positions
            content = file.read()
            for skip in range(0, 8):
                df = pd.read_excel(io.BytesIO(content), skiprows=skip)
                cols = [str(c).lower() for c in df.columns]
                if any('weight' in c for c in cols) and any('ticker' in c or 'symbol' in c for c in cols):
                    # Find column indices
                    ticker_col = next((df.columns[i] for i,c in enumerate(cols) if 'ticker' in c or 'symbol' in c), None)
                    weight_col = next((df.columns[i] for i,c in enumerate(cols) if 'weight' in c), None)
                    name_col   = next((df.columns[i] for i,c in enumerate(cols) if 'name' in c), None)
                    if ticker_col and weight_col:
                        df = df.dropna(subset=[ticker_col, weight_col])
                        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
                        df = df.dropna(subset=[weight_col])
                        df = df[df[weight_col] > 0]
                        df = df.sort_values(weight_col, ascending=False)
                        for _, row in df.iterrows():
                            t = str(row[ticker_col]).strip().upper()
                            n = str(row[name_col]).strip().title() if name_col else t
                            w = round(float(row[weight_col]), 4)
                            if t and t != 'NAN' and w > 0:
                                holdings.append({'t': t, 'n': n, 'w': w})
                        break

        elif filename.endswith('.csv'):
            content = file.read().decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                keys = {k.lower().strip(): v for k,v in row.items()}
                t = (keys.get('ticker') or keys.get('symbol') or '').strip().upper()
                n = (keys.get('name') or keys.get('security name') or t).strip().title()
                w_str = keys.get('weight') or keys.get('weight (%)') or keys.get('portfolio weight') or '0'
                try:
                    w = float(str(w_str).replace('%','').strip())
                    if t and w > 0:
                        holdings.append({'t': t, 'n': n, 'w': round(w, 4)})
                except:
                    pass
            holdings.sort(key=lambda x: x['w'], reverse=True)

    except Exception as e:
        return jsonify({'error': f'Parse error: {str(e)}'}), 500

    if not holdings:
        return jsonify({'error': 'No valid holdings found — check column names (need Ticker/Symbol and Weight)'}), 400

    # Persist to disk
    holdings_dir = os.path.join(BASE_DIR, 'holdings')
    os.makedirs(holdings_dir, exist_ok=True)
    out_path = os.path.join(holdings_dir, f'{ticker}.json')
    json.dump(holdings, open(out_path, 'w'))

    # Update in-memory store
    _etf_holdings_store[ticker] = holdings
    print(f'  [Holdings] Saved {len(holdings)} holdings for {ticker} → {out_path}')

    return jsonify({'ticker': ticker, 'count': len(holdings), 'source': 'uploaded_file', 'saved': True})

@app.route('/api/etf/holdings/list')
def list_holdings():
    """Return list of ETFs with uploaded holdings."""
    return jsonify({t: len(h) for t,h in _etf_holdings_store.items()})

@app.route('/quote/fast/<ticker>')
def quote_fast(ticker):
    """Lightweight quote — just price + prev close. Used by treemap for speed."""
    ticker = ticker.upper()
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.last_price)      if info.last_price      else None
        prev  = float(info.previous_close)  if info.previous_close  else None
        if not price:
            hist = t.history(period='5d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                prev  = float(hist['Close'].iloc[-2]) if len(hist) > 1 else price
        return jsonify({'ticker': ticker, 'price': price, 'prev': prev})
    except Exception as e:
        return jsonify({'ticker': ticker, 'price': None, 'prev': None, 'error': str(e)})


@app.route('/quote/<ticker>')
def quote(ticker):
    ticker = ticker.upper()
    # Try direct Yahoo API first for known HY ETFs
    if ticker in HY_ETF_FALLBACK:
        result = fetch_yahoo_quote_direct(ticker)
        if result:
            return jsonify({**result, 'ticker': ticker})
        # Return stub with metadata if live data unavailable
        meta = HY_ETF_FALLBACK[ticker]
        return jsonify({'ticker': ticker, 'price': None, 'prev': None,
                       'closes':[], 'volumes':[], 'highs':[], 'lows':[],
                       'error': False, 'note': 'Live data unavailable via yfinance; check fund page'})
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period='1y', interval='1d')
        if hist.empty:
            raise ValueError('No data from yfinance')
        closes  = [float(x) for x in hist['Close'].tolist()]
        volumes = [float(x) for x in hist['Volume'].tolist()]
        highs   = [float(x) for x in hist['High'].tolist()]
        lows    = [float(x) for x in hist['Low'].tolist()]
        price    = float(info.last_price)     if info.last_price    else closes[-1]
        prev     = float(info.previous_close) if info.previous_close else closes[-2]
        day_high = float(info.day_high)       if info.day_high      else highs[-1]
        day_low  = float(info.day_low)        if info.day_low       else lows[-1]
        open_p   = float(hist['Open'].iloc[-1])
        # Extra ETF metadata — yield, AUM, expense ratio
        try:
            full_info = t.info
            div_yield = safe_float(full_info.get('yield') or full_info.get('dividendYield') or full_info.get('trailingAnnualDividendYield'))
            if div_yield and div_yield < 1: div_yield = round(div_yield * 100, 2)  # convert decimal to %
            aum       = safe_float(full_info.get('totalAssets'))
            er        = normalize_expense_ratio(full_info.get('annualReportExpenseRatio') or full_info.get('expenseRatio') or full_info.get('netExpenseRatio'))
            # Sanity check: no real ETF/fund expense ratio is anywhere near
            # this high (even the most expensive leveraged/alt funds top out
            # well under 5%). A value like 45% means yfinance returned bad or
            # mismatched data for this field — discard rather than display
            # something nonsensical.
            if er is not None and er > 5:
                print(f'  [Quote] {ticker}: discarding implausible expense ratio {er}% (likely bad yfinance data)')
                er = None
            if div_yield is None or er is None:
                print(f'  [Quote] {ticker}: metadata gaps — yield={div_yield}, er={er}, aum={aum}. '
                      f'Available info keys sample: {list(full_info.keys())[:20]}')
        except Exception as meta_e:
            print(f'  [Quote] {ticker}: metadata extraction failed entirely: {meta_e}')
            div_yield, aum, er = None, None, None
        return jsonify({
            'ticker': ticker, 'price': price, 'prev': prev,
            'dayHigh': day_high, 'dayLow': day_low, 'open': open_p,
            'closes': closes, 'volumes': volumes, 'highs': highs, 'lows': lows,
            'yield': div_yield, 'aum': aum, 'er': er,
        })
    except Exception as e:
        # yfinance failed (often Yahoo's "Invalid Crumb" anti-bot block, which
        # recurs periodically with no permanent fix) — try Stooq as fallback
        print(f'  [Quote] yfinance failed for {ticker} ({e}), trying Stooq fallback...')
        stooq_result = fetch_stooq_quote(ticker)
        if stooq_result:
            print(f'  [Quote] {ticker}: served via Stooq fallback')
            # Stooq has no fund metadata (yield/aum/er) at all — but the
            # specific yfinance call that failed was price/history, which is
            # a DIFFERENT Yahoo endpoint than .info (fund metadata). Try that
            # one separately since it sometimes succeeds even when history
            # is blocked, rather than silently leaving yield/aum/er empty.
            try:
                full_info = yf.Ticker(ticker).info
                div_yield = safe_float(full_info.get('yield') or full_info.get('dividendYield') or full_info.get('trailingAnnualDividendYield'))
                if div_yield and div_yield < 1: div_yield = round(div_yield * 100, 2)
                aum = safe_float(full_info.get('totalAssets'))
                er  = normalize_expense_ratio(full_info.get('annualReportExpenseRatio') or full_info.get('expenseRatio'))
                if div_yield: stooq_result['yield'] = div_yield
                if aum: stooq_result['aum'] = aum
                if er: stooq_result['er'] = er
                print(f'  [Quote] {ticker}: recovered metadata (yield={div_yield}, aum={aum}, er={er}) despite price falling back to Stooq')
            except Exception as meta_e:
                print(f'  [Quote] {ticker}: metadata fetch also failed ({meta_e}) — yield/aum/er will be empty')
            return jsonify(stooq_result)
        return jsonify({'error': str(e)}), 500

@app.route('/alerts')
def get_alerts():
    alert_file = os.path.join(BASE_DIR, 'scanner_alerts.json')
    try:
        if os.path.exists(alert_file):
            with open(alert_file, 'r') as f:
                return jsonify(json.load(f))
    except:
        pass
    return jsonify([])

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'fred': FRED_AVAILABLE and bool(FRED_API_KEY)})

def build_narrative(data):
    """Build bullet-point macro narrative from live data — watch/warning items only"""
    bullets = []

    # ── Rates ──
    r = data.get('rates', {})
    ten_yr = r.get('ten_year')
    spread = r.get('two_ten_spread')
    if ten_yr:
        if spread is not None and spread < 0:
            bullets.append(f'[RATES] Yield curve INVERTED at {spread:+.2f}% (2s vs 10s). Historically this has preceded recessions by 12–18 months. 10-yr Treasury at {ten_yr:.2f}%.')
        elif spread is not None and spread < 0.3:
            bullets.append(f'[RATES] Yield curve nearly flat at {spread:+.2f}%. Markets expect limited growth. 10-yr at {ten_yr:.2f}%.')
        elif ten_yr > 5.0:
            bullets.append(f'[RATES] 10-yr Treasury elevated at {ten_yr:.2f}%. High rates pressure equity valuations and refinancing costs. Watch.')
        # else: green (normal rates) — suppressed from Market Pulse

    # ── VIX ──
    v = data.get('vix', {})
    vix = v.get('current')
    if vix:
        if vix >= 30:
            bullets.append(f'[VIX] Fear reading at {vix:.1f} — FEAR territory. Elevated uncertainty. Patient buyers may find opportunities but caution warranted.')
        elif vix >= 20:
            bullets.append(f'[VIX] Volatility ELEVATED at {vix:.1f}. Stress above normal. Monitor closely for escalation.')
        elif vix < 13:
            bullets.append(f'[VIX] Fear gauge at {vix:.1f} — Complacency risk. Very low fear often precedes sharp selloffs. Stay alert.')
        # 13–20: normal — suppressed

    # ── Liquidity ──
    liq = data.get('liquidity', {})
    dxy = liq.get('dxy')
    m2_yoy = liq.get('m2_yoy')
    hyg_ret = liq.get('hyg_1m_ret')
    liq_issues = []
    if dxy and dxy > 104:
        liq_issues.append(f'Strong dollar (DXY {dxy:.1f}) is tightening global liquidity and pressuring commodities and emerging markets.')
    if m2_yoy is not None and m2_yoy < 0:
        liq_issues.append(f'Money supply CONTRACTING (M2 {m2_yoy:+.1f}% YoY) — less fuel for risk assets.')
    if hyg_ret is not None and hyg_ret < -1.5:
        liq_issues.append(f'High-yield credit under pressure (HYG {hyg_ret:+.2f}% past month) — risk appetite warning signal.')
    if liq_issues:
        bullets.append('[LIQUIDITY] ' + ' | '.join(liq_issues))

    # ── Inflation ──
    inf = data.get('inflation', {})
    cpi = inf.get('cpi_yoy')
    core_cpi = inf.get('cpi_core_yoy')
    pce = inf.get('core_pce_yoy')
    cpi_prior = inf.get('cpi_prior_yoy')
    if cpi:
        arrow = ''
        if cpi_prior:
            arrow = ' (trending down ↓)' if cpi < cpi_prior else ' (trending up ↑)' if cpi > cpi_prior else ''
        if cpi > 4:
            bullets.append(f'[INFLATION] CPI HOT at {cpi:.1f}% YoY{arrow}. Core CPI {core_cpi:.1f}% | Core PCE {pce:.1f}% vs Fed 2.0% target. Rate cuts off the table.')
        elif cpi > 2.5:
            bullets.append(f'[INFLATION] CPI {cpi:.1f}% YoY{arrow} — still above target. Core CPI {core_cpi:.1f}% | Core PCE {pce:.1f}%. Fed in watch mode, easing not imminent.')
        # else near target — green, suppressed

    # ── Labor ──
    lab = data.get('labor', {})
    unemp = lab.get('unemployment')
    nfp = lab.get('nfp_mom')
    claims = lab.get('initial_claims')
    lab_issues = []
    if unemp and unemp > 5:
        lab_issues.append(f'Unemployment rising at {unemp:.1f}% — above healthy range.')
    if nfp is not None and nfp < 75000:
        lab_issues.append(f'Payrolls slowing at {int(nfp/1000)}K MoM — below trend.')
    if claims and claims > 260000:
        lab_issues.append(f'Jobless claims ELEVATED at {int(claims/1000)}K — watch for further deterioration.')
    if lab_issues:
        bullets.append('[LABOR] ' + ' | '.join(lab_issues) + f' Unemployment: {unemp:.1f}%' if unemp else '')
    elif unemp and unemp > 4.2:
        bullets.append(f'[LABOR] Softening at the edges. Unemployment at {unemp:.1f}%, payrolls {("+" if nfp and nfp>0 else "") + str(int(nfp/1000))+"K" if nfp else "—"}. Watch claims trend closely.')

    # ── Breadth ──
    b = data.get('breadth', {})
    spy_ytd = b.get('spy_ytd')
    rsp_3m  = b.get('rsp_3m')
    spy_3m  = b.get('spy_3m')
    if rsp_3m is not None and spy_3m is not None:
        gap = spy_3m - rsp_3m
        if gap > 3:
            bullets.append(
                f'[BREADTH] Narrow market rally — the S&P 500 is up {spy_3m:+.1f}% over 3 months, '
                f'but the average stock (equal-weight index) is only up {rsp_3m:+.1f}%. '
                f'This {gap:.1f}% gap means a handful of mega-cap stocks are doing the heavy lifting. '
                f'When breadth narrows like this, the rally is fragile — if those leaders stumble, there\'s less support underneath.'
            )
        # else broad participation — green, suppressed

    # ── Sector Rotation signal ──
    sectors = data.get('sectors', [])
    if sectors:
        gainers = sorted([s for s in sectors if s.get('ret_1m') is not None], key=lambda x: x['ret_1m'], reverse=True)
        losers  = sorted([s for s in sectors if s.get('ret_1m') is not None], key=lambda x: x['ret_1m'])
        if gainers and losers:
            top_names = [s['name'] for s in gainers[:3]]
            defensive = ['Utilities','Consumer Staples','Health Care']
            if any(d in top_names for d in defensive):
                top2 = ' | '.join([f"{s['name']} ({s['ret_1m']:+.1f}%)" for s in gainers[:2]])
                bot2 = ' | '.join([f"{s['name']} ({s['ret_1m']:.1f}%)" for s in losers[:2]])
                bullets.append(
                    f'[ROTATION] Defensive sectors leading: {top2}. '
                    f'Defensive leadership — Utilities, Staples, Healthcare outperforming — is a warning sign. '
                    f'It typically means money is rotating away from risk and into safety. '
                    f'Lagging: {bot2}.'
                )
            # risk-on rotation: green — suppressed

    if not bullets:
        return ['__ALL_GREEN__']

    return bullets


@app.route('/home')
@app.route('/home.html')
def home_page():
    return send_from_directory(BASE_DIR, 'home.html')

@app.route('/etf')
def etf_scanner():
    return send_from_directory(BASE_DIR, 'etf-scanner.html')

@app.route('/api/channel-episodes')
def channel_episodes():
    """Fetch latest YouTube episode for each financial channel via RSS."""
    import xml.etree.ElementTree as ET
    import re

    channels = {
        '@TheMoneyPrinter':    'UC54e-Ng4LwR_2tHKEsJt80A',
        '@TheCompoundNews':    'UCBRpqrzuuqE8TZcWw75JSdw',
        '@animalspirits':      'UCfhW84xfAu3A5N1lHgZMIxQ',
        '@RealClearPolitics':  'UCaFhHkIdFWcMprQDFgIzgxA',
    }

    results = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for handle, channel_id in channels.items():
        try:
            url = f'https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}'
            r = requests.get(url, headers=headers, timeout=8)
            print(f'  [ChannelRSS] {handle}: HTTP {r.status_code}, {len(r.content)} bytes')
            if r.status_code != 200:
                continue
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015',
                  'media': 'http://search.yahoo.com/mrss/'}
            root = ET.fromstring(r.content)
            entries = root.findall('atom:entry', ns)
            print(f'  [ChannelRSS] {handle}: found {len(entries)} entries')
            if not entries:
                continue
            entry = entries[0]
            title = entry.findtext('atom:title', '', ns)
            link  = entry.find('atom:link', ns)
            href  = link.get('href','') if link is not None else ''
            pub   = entry.findtext('atom:published', '', ns)
            # Thumbnail lives inside <media:group><media:thumbnail url="...">
            thumbnail = ''
            media_group = entry.find('media:group', ns)
            if media_group is not None:
                thumb_el = media_group.find('media:thumbnail', ns)
                if thumb_el is not None:
                    thumbnail = thumb_el.get('url', '')
            if not thumbnail:
                # Fallback: regex the raw XML for this entry's thumbnail URL
                # directly, in case the namespaced ElementTree lookup missed
                # it due to a structural quirk in YouTube's feed.
                entry_xml = ET.tostring(entry, encoding='unicode')
                m = re.search(r'<media:thumbnail[^>]*url="([^"]+)"', entry_xml)
                if m:
                    thumbnail = m.group(1)
            print(f'  [ChannelRSS] {handle}: title={title[:50] if title else None!r}, thumbnail_found={bool(thumbnail)}')
            # Format date
            date_str = ''
            if pub:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(pub.replace('Z','+00:00'))
                    date_str = dt.strftime('%b %d, %Y')
                except:
                    date_str = pub[:10]
            results[handle] = {'title': title, 'url': href, 'date': date_str, 'thumbnail': thumbnail}
        except Exception as e:
            print(f'  [ChannelRSS] error ({handle}): {e}')
            continue

    return jsonify(results)

@app.route('/portfolio')
def portfolio_page():
    return send_from_directory(BASE_DIR, 'portfolio.html')

# ════════════════════════════════════════════════════════════════
#  PORTFOLIO ANALYSIS — CSV upload, returns, risk metrics, benchmarks
# ════════════════════════════════════════════════════════════════

_portfolio_store_file = None
def get_portfolio_file():
    global _portfolio_store_file
    if _portfolio_store_file is None:
        _portfolio_store_file = os.path.join(BASE_DIR, 'portfolio_data.json')
    return _portfolio_store_file

def load_portfolio():
    """Load the combined, merged view across all uploaded sources (e.g.
    Snowball, Fidelity, Robinhood). Internally, holdings are stored per
    source in 'sources': {source_name: {holdings: [...], margin, uploaded_at,
    filename}} so uploading a new source never wipes out previously uploaded
    sources — multiple brokerages can coexist."""
    f = get_portfolio_file()
    if not os.path.exists(f):
        return None
    try:
        raw = json.load(open(f, encoding='utf-8'))
    except Exception:
        return None

    # Backward compatibility: older files saved a single flat
    # {'holdings': [...], 'margin': ...} with no 'sources' wrapper. Treat
    # that as a single 'snowball' source so existing saved data keeps
    # working without requiring a re-upload.
    if 'sources' not in raw:
        raw = {
            'sources': {
                'snowball': {
                    'holdings': raw.get('holdings', []),
                    'margin': raw.get('margin', 0),
                    'uploaded_at': raw.get('uploaded_at'),
                    'filename': raw.get('filename'),
                }
            }
        }

    combined_holdings = []
    total_margin = 0
    latest_upload = None
    for source_name, source_data in raw['sources'].items():
        for h in source_data.get('holdings', []):
            h = dict(h)  # don't mutate the stored copy
            h.setdefault('source', source_name)
            combined_holdings.append(h)
        total_margin += source_data.get('margin', 0) or 0
        ts = source_data.get('uploaded_at')
        if ts and (latest_upload is None or ts > latest_upload):
            latest_upload = ts

    return {
        'holdings': combined_holdings,
        'margin': total_margin,
        'uploaded_at': latest_upload,
        'sources': raw['sources'],  # exposed for the future accounts-management UI
    }

def save_portfolio_source(source_name, holdings, margin, filename):
    """Save one brokerage source's holdings without disturbing any other
    already-uploaded source. This is what makes multi-brokerage portfolios
    possible — uploading a Robinhood export, for instance, only touches the
    'robinhood' bucket and leaves 'snowball' or 'fidelity' untouched."""
    f = get_portfolio_file()
    existing = {}
    if os.path.exists(f):
        try:
            existing = json.load(open(f, encoding='utf-8'))
        except Exception:
            existing = {}
    if 'sources' not in existing:
        # Migrate old flat format into the new sources wrapper before adding
        # the new source, so we don't silently lose previously uploaded data.
        if existing.get('holdings'):
            existing = {'sources': {'snowball': {
                'holdings': existing.get('holdings', []),
                'margin': existing.get('margin', 0),
                'uploaded_at': existing.get('uploaded_at'),
                'filename': existing.get('filename'),
            }}}
        else:
            existing = {'sources': {}}

    existing['sources'][source_name] = {
        'holdings': holdings,
        'margin': margin,
        'uploaded_at': datetime.now().isoformat(),
        'filename': filename,
    }
    json.dump(existing, open(f, 'w', encoding='utf-8'), indent=2)

def remove_portfolio_source(source_name):
    """Remove one brokerage source entirely (e.g. when cancelling Snowball
    and fully switching to direct Fidelity/Robinhood uploads)."""
    f = get_portfolio_file()
    if not os.path.exists(f):
        return False
    try:
        existing = json.load(open(f, encoding='utf-8'))
    except Exception:
        return False
    if 'sources' not in existing or source_name not in existing['sources']:
        return False
    del existing['sources'][source_name]
    json.dump(existing, open(f, 'w', encoding='utf-8'), indent=2)
    return True

def save_portfolio(data):
    """Legacy single-blob save, kept for any code path that still calls it
    directly. New code should prefer save_portfolio_source() so multiple
    brokerages can coexist."""
    json.dump(data, open(get_portfolio_file(), 'w', encoding='utf-8'), indent=2)

@app.route('/api/portfolio/margin-rate', methods=['GET', 'POST'])
def margin_rate():
    """Get or set the annualized margin interest rate (default 5%)."""
    from flask import request as freq
    rate_file = os.path.join(BASE_DIR, 'margin_rate.json')

    if freq.method == 'POST':
        rate = freq.json.get('rate', 5.0)
        json.dump({'rate': rate}, open(rate_file, 'w'))
        return jsonify({'ok': True, 'rate': rate})

    if os.path.exists(rate_file):
        try:
            return jsonify(json.load(open(rate_file)))
        except: pass
    return jsonify({'rate': 5.0})  # default 5%

@app.route('/api/portfolio/margin-amount', methods=['GET', 'POST'])
def margin_amount():
    """Get or set the current margin balance directly. This is a manual
    override stored separately from the uploaded CSV — it takes precedence
    over any 'Total margin' row that might also be present in the upload, so
    the person can update their margin balance any time without needing to
    edit and re-upload the CSV every time it changes."""
    from flask import request as freq
    amount_file = os.path.join(BASE_DIR, 'margin_amount.json')

    if freq.method == 'POST':
        amount = freq.json.get('amount')
        if amount is None:
            # Explicit clear — fall back to whatever the CSV says (or 0)
            if os.path.exists(amount_file):
                os.remove(amount_file)
            return jsonify({'ok': True, 'amount': None})
        json.dump({'amount': amount}, open(amount_file, 'w'))
        return jsonify({'ok': True, 'amount': amount})

    if os.path.exists(amount_file):
        try:
            return jsonify(json.load(open(amount_file)))
        except: pass
    return jsonify({'amount': None})  # no override set

# Maps Snowball's verbose account names to short display labels. Matching is
# substring-based and case-insensitive so small naming variations in future
# exports (e.g. "Fidelity Roth IRA - Kurt" vs "Fidelity - Roth IRA - Kurt")
# still resolve correctly.
ACCOUNT_SHORT_LABELS = [
    ('roth ira - kurt', 'Roth Kurt'),
    ('roth ira - gaby', 'Roth Gaby'),
    ('traditional ira', 'Traditional IRA'),
    ('robinhood', 'Robinhood Brokerage'),
]

def account_short_label(raw_name):
    """Map one raw Snowball account name to its short label. Falls back to
    the original string (trimmed) if nothing matches, so a brokerage added
    later still shows up rather than disappearing."""
    if not raw_name:
        return 'Unassigned'
    lower = raw_name.lower()
    for needle, label in ACCOUNT_SHORT_LABELS:
        if needle in lower:
            return label
    return raw_name.strip()

@app.route('/api/portfolio/upload', methods=['POST'])
def portfolio_upload():
    """Parse Snowball-style holdings CSV and persist."""
    from flask import request as freq
    import io, csv as csv_mod

    file = freq.files.get('file')
    if not file:
        return jsonify({'error': 'file required'}), 400

    try:
        raw = file.read()
        # Try utf-8 first, fall back to latin1 (Snowball exports use latin1)
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('latin1')

        reader = csv_mod.reader(io.StringIO(text))
        all_rows = list(reader)
    except Exception as e:
        return jsonify({'error': f'Could not read file: {e}'}), 400

    if not all_rows:
        return jsonify({'error': 'File is empty'}), 400

    header = [h.strip() for h in all_rows[0]]
    data_rows = all_rows[1:]

    # Snowball's export has a few duplicate-named columns (two "Capital gain"
    # columns, two "Total profit" columns — one $ and one %). A name-based
    # dict lookup silently collapses to whichever one appears last, so we
    # find column positions explicitly instead and pick the first occurrence
    # of each pair, which is consistently the dollar-value version in
    # Snowball's column order (confirmed against a real export: the first
    # "Total profit" holds a $ amount like 280.20, the second holds a daily
    # % figure like 2.07).
    def col_index(name, occurrence=0):
        matches = [i for i, h in enumerate(header) if h == name]
        return matches[occurrence] if len(matches) > occurrence else None

    idx = {
        'ticker': col_index('Holding'),
        'name': col_index("Holdings' name"),
        'shares': col_index('Shares'),
        'cost_basis': col_index('Cost basis'),
        'current_value': col_index('Current value'),
        'share_price': col_index('Share price'),
        'cost_per_share': col_index('Cost per share'),
        'country': col_index('Country'),
        'note': col_index('Note'),
        'sector': col_index('Sector'),
        'beta': col_index('Beta'),
        'yield': col_index('Dividend yield'),
        'expense_ratio': col_index('Expense ratio'),
        'account': (
            col_index('Portfolios') if col_index('Portfolios') is not None
            else col_index('Potfolio') if col_index('Potfolio') is not None
            else col_index('Portfolio')
        ),
        'category': col_index('Category'),
        'div_received': col_index('Div. received'),
        'realized_pl': col_index('Realized P&L'),
        'total_profit': col_index('Total profit', 0),
    }

    def get(row, key):
        i = idx.get(key)
        if i is None or i >= len(row):
            return None
        return row[i]

    def clean_money(s):
        if s is None: return None
        s = str(s).replace('$','').replace(',','').replace('(','-').replace(')','').strip()
        if s in ('', '-', 'nan', 'NaN'): return None
        try: return float(s)
        except: return None

    def clean_num(s):
        if s is None: return None
        s = str(s).strip()
        if s in ('', '-', 'nan', 'NaN'): return None
        try: return float(s)
        except: return None

    def looks_like_cusip(s):
        """A CUSIP is exactly 9 alphanumeric characters and (unlike a real
        ticker symbol) always contains at least one digit. Real US equity
        tickers are virtually never 9 characters with embedded digits, so
        this is a safe, simple heuristic — not a full CUSIP checksum
        validation, but accurate enough to catch the cases that matter."""
        return len(s) == 9 and s.isalnum() and any(c.isdigit() for c in s)

    holdings = []
    margin = 0
    skipped_cusip_rows = []
    for row in data_rows:
        if not row or all(not c.strip() for c in row):
            continue
        ticker = (get(row, 'ticker') or '').strip()
        if not ticker:
            continue
        if ticker.lower().startswith('total margin'):
            margin = abs(clean_money(get(row, 'current_value')) or 0)
            continue
        if looks_like_cusip(ticker):
            # Snowball occasionally exports a raw CUSIP instead of a ticker
            # for certain obscure or inactive positions it can't map. Yahoo
            # will always fail on a CUSIP, and feeding it through corrupts
            # downstream price-history calculations (inception-date
            # truncation, growth charts, etc.) — better to skip the row
            # and flag it than silently break those features.
            skipped_cusip_rows.append({'cusip': ticker, 'name': (get(row, 'name') or '').strip()})
            continue
        cv = clean_money(get(row, 'current_value'))
        cb = clean_money(get(row, 'cost_basis'))
        if cv is None:
            continue
        holdings.append({
            't': ticker,
            'n': (get(row, 'name') or ticker).strip(),
            'source': 'snowball',
            'shares': clean_num(get(row, 'shares')),
            'cv': cv,
            'cb': cb,
            'price': clean_money(get(row, 'share_price')),
            'cost_per_share': clean_money(get(row, 'cost_per_share')),
            'country': (get(row, 'country') or 'Unknown').strip(),
            'investment_type': (get(row, 'note') or '').strip(),
            'sector': (get(row, 'sector') or 'Unknown').strip(),
            'beta': clean_num(get(row, 'beta')),
            'yield': clean_num(get(row, 'yield')),
            'expense_ratio': clean_num(get(row, 'expense_ratio')),
            'account': (get(row, 'account') or 'Unassigned').strip(),
            'category': (get(row, 'category') or '').strip(),
            'div_received': clean_money(get(row, 'div_received')),
            'realized_pl': clean_money(get(row, 'realized_pl')),
            'total_profit': clean_money(get(row, 'total_profit')),
        })

    if not holdings:
        return jsonify({'error': 'No valid holdings found in file'}), 400

    save_portfolio_source('snowball', holdings, margin, file.filename)
    print(f'  [Portfolio] Saved {len(holdings)} holdings from Snowball, margin=${margin:,.0f}')
    if skipped_cusip_rows:
        skipped_summary = ', '.join(f"{r['cusip']} ({r['name']})" for r in skipped_cusip_rows)
        print(f'  [Portfolio] WARNING: skipped {len(skipped_cusip_rows)} row(s) where the Holding '
              f'column contained a CUSIP instead of a ticker symbol: {skipped_summary}')
    return jsonify({
        'ok': True, 'count': len(holdings), 'margin': margin,
        'skipped_cusip_rows': skipped_cusip_rows,
    })

@app.route('/api/portfolio/data')
def portfolio_data():
    data = load_portfolio()
    if not data:
        return jsonify({'error': 'No portfolio uploaded yet'}), 404
    return jsonify(data)

@app.route('/api/portfolio/sources')
def portfolio_sources():
    """List currently-uploaded brokerage sources (e.g. snowball, fidelity,
    robinhood) with basic stats per source, for an accounts-management UI."""
    data = load_portfolio()
    if not data or not data.get('sources'):
        return jsonify({'sources': []})
    summary = []
    for name, s in data['sources'].items():
        holdings = s.get('holdings', [])
        summary.append({
            'source': name,
            'filename': s.get('filename'),
            'uploaded_at': s.get('uploaded_at'),
            'holding_count': len(holdings),
            'total_value': round(sum(h.get('cv', 0) or 0 for h in holdings), 2),
            'margin': s.get('margin', 0),
        })
    return jsonify({'sources': summary})

@app.route('/api/portfolio/sources/<source_name>', methods=['DELETE'])
def portfolio_remove_source(source_name):
    """Remove one brokerage source — e.g. once Snowball is cancelled and
    Fidelity/Robinhood are uploaded directly, this drops the stale Snowball
    data without touching anything else."""
    removed = remove_portfolio_source(source_name)
    if not removed:
        return jsonify({'error': f'No source named "{source_name}" found'}), 404
    return jsonify({'ok': True, 'removed': source_name})

@app.route('/api/portfolio/analysis')
def portfolio_analysis():
    """Compute returns, risk metrics, sector breakdown, and benchmark comparison."""
    data = load_portfolio()
    if not data:
        return jsonify({'error': 'No portfolio uploaded yet'}), 404

    holdings = data['holdings']
    margin = data.get('margin', 0)
    # A manually-set margin amount (via the popup editor) overrides whatever
    # the CSV says, since the CSV's "Total margin" row goes stale the moment
    # the actual balance changes and re-uploading the whole CSV just to
    # update one number is needless friction.
    margin_amount_file = os.path.join(BASE_DIR, 'margin_amount.json')
    if os.path.exists(margin_amount_file):
        try:
            override = json.load(open(margin_amount_file)).get('amount')
            if override is not None:
                margin = override
        except: pass
    tickers = list(set(h['t'] for h in holdings))
    total_value = sum(h['cv'] for h in holdings)

    # Fetch 5y history for all tickers + benchmarks in one batch
    benchmarks = {'SPY':'S&P 500', 'QQQ':'Nasdaq 100', 'DIA':'Dow 30'}
    all_tickers = list(set(tickers + list(benchmarks.keys())))

    try:
        hist_df = yf.download(all_tickers, period='5y', auto_adjust=True,
                              progress=False, threads=True, group_by='ticker')
    except Exception as e:
        print(f'  [Portfolio] yf.download error: {e}')
        hist_df = None

    def get_closes(ticker):
        try:
            if len(all_tickers) == 1:
                s = hist_df['Close'].dropna()
            else:
                s = hist_df[ticker]['Close'].dropna()
            return s
        except:
            return pd.Series(dtype=float)

    def period_return(series, days):
        if series is None or len(series) < 2:
            return None
        if len(series) <= days:
            sub = series
        else:
            sub = series.iloc[-days:]
        if len(sub) < 2 or sub.iloc[0] == 0:
            return None
        return round(((sub.iloc[-1] - sub.iloc[0]) / sub.iloc[0]) * 100, 2)

    def annualized_return(series, years):
        days = int(years * 252)
        if series is None or len(series) < 2:
            return None
        if len(series) <= days:
            sub = series
            actual_years = len(series) / 252
        else:
            sub = series.iloc[-days:]
            actual_years = years
        if len(sub) < 2 or sub.iloc[0] == 0 or actual_years <= 0:
            return None
        total_ret = (sub.iloc[-1] / sub.iloc[0])
        if total_ret <= 0:
            return None
        ann = (total_ret ** (1/actual_years) - 1) * 100
        return round(ann, 2)

    def max_drawdown(series):
        if series is None or len(series) < 2:
            return None
        cummax = series.cummax()
        dd = (series - cummax) / cummax
        return round(float(dd.min()) * 100, 2)

    def daily_std_annualized(series, days):
        if series is None or len(series) < 10:
            return None
        sub = series.iloc[-days:] if len(series) > days else series
        rets = sub.pct_change().dropna()
        if len(rets) < 5:
            return None
        return round(float(rets.std()) * (252 ** 0.5) * 100, 2)

    RISK_FREE_RATE = 0.045  # ~4.5% — approx current T-bill yield, used for Sharpe/Sortino

    def sharpe_ratio(series, days=252):
        """Annualized Sharpe ratio: (mean return - risk-free) / std dev of returns"""
        if series is None or len(series) < 10:
            return None
        sub = series.iloc[-days:] if len(series) > days else series
        rets = sub.pct_change().dropna()
        if len(rets) < 5:
            return None
        ann_return = float(rets.mean()) * 252
        ann_std = float(rets.std()) * (252 ** 0.5)
        if ann_std == 0:
            return None
        return round((ann_return - RISK_FREE_RATE) / ann_std, 2)

    def sortino_ratio(series, days=252):
        """Like Sharpe but only penalizes downside volatility (negative returns)"""
        if series is None or len(series) < 10:
            return None
        sub = series.iloc[-days:] if len(series) > days else series
        rets = sub.pct_change().dropna()
        if len(rets) < 5:
            return None
        downside = rets[rets < 0]
        if len(downside) < 2:
            return None
        ann_return = float(rets.mean()) * 252
        downside_std = float(downside.std()) * (252 ** 0.5)
        if downside_std == 0:
            return None
        return round((ann_return - RISK_FREE_RATE) / downside_std, 2)

    def beta_vs_benchmark(series, bench_series, days=252):
        """Beta = covariance(portfolio, benchmark) / variance(benchmark)"""
        if series is None or bench_series is None:
            return None
        sub_p = series.iloc[-days:] if len(series) > days else series
        sub_b = bench_series.iloc[-days:] if len(bench_series) > days else bench_series
        common = sub_p.index.intersection(sub_b.index)
        if len(common) < 20:
            return None
        rp = sub_p.reindex(common).pct_change().dropna()
        rb = sub_b.reindex(common).pct_change().dropna()
        common2 = rp.index.intersection(rb.index)
        if len(common2) < 20:
            return None
        rp, rb = rp.reindex(common2), rb.reindex(common2)
        var_b = rb.var()
        if var_b == 0:
            return None
        cov = rp.cov(rb)
        return round(float(cov / var_b), 2)

    def time_weighted_return(series):
        """TWR — geometric link of sub-period returns, removes distortion from cash flows.
        For a price-only series (no external cash flow data), TWR equals the cumulative return."""
        if series is None or len(series) < 2:
            return None
        return round(((series.iloc[-1] / series.iloc[0]) - 1) * 100, 2)

    def monte_carlo_simulation(series, current_value, years=10, n_sims=2000, seed=42):
        """Project the portfolio's future value using a Monte Carlo simulation
        driven by its own historical daily returns (not an assumed market
        model) — each simulated path randomly resamples (bootstraps) daily
        returns drawn from this portfolio's actual 5y history, so the
        simulation reflects the volatility and behavior of the specific mix
        of holdings actually uploaded, rather than a generic index assumption.

        Returns percentile outcomes (10th/25th/50th/75th/90th) at 1, 5, and
        10-year horizons, plus the full set of percentile values at every
        year for charting, and the probability of ending below the starting
        value at the 10-year mark (downside-risk framing).
        """
        if series is None or len(series) < 60 or not current_value:
            return None

        daily_returns = series.pct_change().dropna().values
        if len(daily_returns) < 60:
            return None

        trading_days_per_year = 252
        n_days = years * trading_days_per_year

        rng = np.random.default_rng(seed)
        # Bootstrap resampling: draw n_days returns (with replacement) from the
        # portfolio's own historical daily return distribution, for each of
        # n_sims simulated paths simultaneously (vectorized for speed).
        sampled_returns = rng.choice(daily_returns, size=(n_sims, n_days), replace=True)
        cumulative = np.cumprod(1 + sampled_returns, axis=1)
        paths = current_value * cumulative  # shape (n_sims, n_days)

        def percentiles_at_day(day_idx):
            vals = paths[:, day_idx]
            return {
                'p10': round(float(np.percentile(vals, 10)), 0),
                'p25': round(float(np.percentile(vals, 25)), 0),
                'p50': round(float(np.percentile(vals, 50)), 0),
                'p75': round(float(np.percentile(vals, 75)), 0),
                'p90': round(float(np.percentile(vals, 90)), 0),
            }

        horizons = {}
        for yr in [1, 5, 10]:
            if yr <= years:
                day_idx = min(yr * trading_days_per_year, n_days) - 1
                horizons[f'{yr}y'] = percentiles_at_day(day_idx)

        # Yearly percentile series for charting the full fan of outcomes
        yearly_series = []
        for yr in range(1, years + 1):
            day_idx = min(yr * trading_days_per_year, n_days) - 1
            yearly_series.append({'year': yr, **percentiles_at_day(day_idx)})

        final_vals = paths[:, -1]
        prob_below_start = round(float(np.mean(final_vals < current_value)) * 100, 1)
        prob_double = round(float(np.mean(final_vals >= current_value * 2)) * 100, 1)

        # Annualized return/vol implied by the bootstrap, for context
        mean_daily = float(np.mean(daily_returns))
        std_daily = float(np.std(daily_returns))
        implied_annual_return = round(((1 + mean_daily) ** trading_days_per_year - 1) * 100, 2)
        implied_annual_vol = round(std_daily * (trading_days_per_year ** 0.5) * 100, 2)

        return {
            'starting_value': round(current_value, 0),
            'years': years,
            'n_simulations': n_sims,
            'horizons': horizons,
            'yearly_series': yearly_series,
            'prob_below_start_10y': prob_below_start,
            'prob_double_10y': prob_double,
            'implied_annual_return': implied_annual_return,
            'implied_annual_vol': implied_annual_vol,
            'historical_days_used': len(daily_returns),
        }

    # ── Per-holding closes ──
    closes_cache = {t: get_closes(t) for t in all_tickers}

    # ── Build portfolio composite series (weighted by current value) ──
    # IMPORTANT: use UNION of all dates, not intersection — many holdings (especially
    # newer option-income ETFs like KSLV, BLOX, NVIT) only have 1-2 years of history.
    # Using intersection would truncate the ENTIRE portfolio lookback to the youngest holding.
    # Instead: build on the longest available index (a benchmark like SPY/QQQ/DIA, which have
    # full 5y history), forward-fill each holding, and renormalize weights among holdings
    # that actually have data at each point in time.
    weights = {h['t']: h['cv']/total_value for h in holdings if h['cv']}

    # Use the benchmark with the longest history as the master index (SPY has full history)
    master_index = None
    for bk in benchmarks:
        s = closes_cache.get(bk)
        if s is not None and len(s) > 0:
            if master_index is None or len(s) > len(master_index):
                master_index = s.index
    if master_index is None or len(master_index) < 2:
        # Fallback: longest holding series
        for t in tickers:
            s = closes_cache.get(t)
            if s is not None and len(s) > 0:
                if master_index is None or len(s) > len(master_index):
                    master_index = s.index

    if master_index is None:
        master_index = pd.DatetimeIndex([])

    # Build normalized return series per holding (% change from ITS OWN first available date)
    # then composite using weights, renormalizing across holdings that have data at each date
    holding_norm = {}  # ticker -> normalized series (rebased to 1.0 at its own inception) aligned to master_index
    holding_has_data = {}  # ticker -> boolean mask of where data exists
    for t in tickers:
        s = closes_cache.get(t)
        if s is None or len(s) == 0 or t not in weights:
            continue
        aligned = s.reindex(master_index).ffill()
        # Find first valid value to normalize from
        first_valid = aligned.first_valid_index()
        if first_valid is None:
            continue
        base_val = aligned.loc[first_valid]
        if not base_val:
            continue
        normalized = aligned / base_val
        holding_norm[t] = normalized
        holding_has_data[t] = aligned.notna()

    # Composite: at each date, sum(weight_i * normalized_i) / sum(weight_i for holdings with data)
    # Vectorized using pandas DataFrames instead of per-date Python loop (much faster for 1300+ days)
    if holding_norm:
        norm_df = pd.DataFrame(holding_norm)  # columns = tickers, index = master_index
        mask_df = pd.DataFrame(holding_has_data)
        weight_arr = pd.Series({t: weights[t] for t in norm_df.columns})

        weighted_vals = norm_df.fillna(0) * weight_arr  # broadcast weights across columns
        weighted_mask = mask_df.astype(float) * weight_arr

        numerator = weighted_vals.sum(axis=1)
        denominator = weighted_mask.sum(axis=1)
        portfolio_series = (numerator / denominator).dropna()
        portfolio_series = portfolio_series[denominator > 0]

        # ── Truncate to the latest inception date among currently-held
        # positions ──
        # Without this, a historical period (e.g. the 2022 bear market)
        # would be computed from whichever OLDER holdings happened to exist
        # back then, silently excluding newer ones that now make up a large
        # share of the portfolio's actual dollar weight (e.g. funds that
        # launched in 2023-2024). That understates the portfolio's true
        # historical exposure rather than honestly reflecting "what would
        # have happened if I'd held my current full mix the whole time" —
        # which is the entire point of a backtest. Truncating to start only
        # once every currently-held position has data means every plotted
        # point reflects the genuine current allocation, even though it
        # shortens the chart's overall timeline.
        inception_dates_with_ticker = [
            (t, holding_norm[t].first_valid_index()) for t in holding_norm
            if t in weights and weights[t] > 0
        ]
        # Cash-equivalents (money market funds like SPAXX) are excluded from
        # driving the truncation date — their NAV is pegged flat at $1.00,
        # which often produces sparse or unreliable historical price data on
        # Yahoo that looks like a "recent inception" but isn't a real signal
        # about when the position started, and a cash holding shouldn't be
        # able to truncate the whole backtest based on that data quirk.
        cash_tickers = {h['t'] for h in holdings if (h.get('category') or '').strip().lower() == 'cash'}
        inception_dates_with_ticker = [(t, dt) for t, dt in inception_dates_with_ticker
                                        if dt is not None and t not in cash_tickers]
        inception_dates = [dt for t, dt in inception_dates_with_ticker]
        if inception_dates:
            newest_inception = max(inception_dates)
            pre_truncate_len = len(portfolio_series)
            portfolio_series = portfolio_series[portfolio_series.index >= newest_inception]
            newest_inception_ticker = max(inception_dates_with_ticker, key=lambda x: x[1])[0]
            print(f'  [GrowthChart] Truncating portfolio series at {newest_inception.date()} '
                  f'(driven by {newest_inception_ticker}\'s inception) — '
                  f'{pre_truncate_len} days -> {len(portfolio_series)} days remaining')
            # Guard: if truncation leaves an unusably short series (e.g. a
            # data glitch on one ticker made its "first_valid_index" look
            # like it's almost at the end of the window, even though that
            # ticker actually has plenty of real history), fall back to the
            # untruncated series rather than silently showing an empty or
            # near-empty chart. 30 days is a reasonable floor for a growth
            # chart to be meaningful at all.
            if len(portfolio_series) < 30:
                print(f'  [GrowthChart] WARNING: truncation left only {len(portfolio_series)} days '
                      f'(ticker {newest_inception_ticker}, date {newest_inception.date()}) — '
                      f'this is almost certainly a data issue with that ticker, not a real recent '
                      f'inception. Falling back to untruncated series.')
                portfolio_series = (numerator / denominator).dropna()
                portfolio_series = portfolio_series[denominator > 0]
                newest_inception = None
                newest_inception_ticker = None
        else:
            newest_inception = None
            newest_inception_ticker = None
    else:
        portfolio_series = pd.Series(dtype=float)
        newest_inception = None
        newest_inception_ticker = None

    # ── Returns table: portfolio + each benchmark ──
    periods = {'1M': 21, '6M': 126, '1Y': 252, '3Y': 756, '5Y': 1260}
    returns_table = {}
    for label, series, key in [('Portfolio', portfolio_series, 'portfolio')] + \
                               [(name, closes_cache.get(tk), tk) for tk,name in benchmarks.items()]:
        row = {}
        for plabel, pdays in periods.items():
            row[plabel] = period_return(series, pdays)
        row['TTM'] = period_return(series, 252)
        row['3Y_Ann'] = annualized_return(series, 3)
        row['5Y_Ann'] = annualized_return(series, 5)
        returns_table[key] = row

    # ── Risk metrics ──
    risk_table = {}
    for label, series, key in [('Portfolio', portfolio_series, 'portfolio')] + \
                               [(name, closes_cache.get(tk), tk) for tk,name in benchmarks.items()]:
        risk_table[key] = {
            'max_dd_1y': max_drawdown(series.iloc[-252:]) if series is not None and len(series)>252 else max_drawdown(series),
            'max_dd_3y': max_drawdown(series.iloc[-756:]) if series is not None and len(series)>756 else None,
            'max_dd_5y': max_drawdown(series),
            'std_1y': daily_std_annualized(series, 252),
            'std_3y': daily_std_annualized(series, 756),
            'std_5y': daily_std_annualized(series, 1260),
        }

    # ── Sector breakdown ──
    sector_totals = {}
    for h in holdings:
        sec = h.get('sector') or 'Unknown'
        if sec == 'Funds':  # Generic ETF sector — try category instead
            sec = h.get('category') or 'Funds (Mixed)'
        sector_totals[sec] = sector_totals.get(sec, 0) + h['cv']
    sector_breakdown = [{'sector': k, 'value': round(v,2), 'pct': round(v/total_value*100,2)}
                        for k,v in sorted(sector_totals.items(), key=lambda x:-x[1])]

    # ── Account breakdown ──
    # Snowball's "Portfolios" column lists comma-separated account names when
    # one holding spans multiple accounts (e.g. shared in a joint sleeve).
    # The CSV doesn't say how much goes to each, so split evenly across the
    # named accounts rather than showing it as one confusing combined slice.
    account_totals = {}
    for h in holdings:
        acct = h.get('account') or 'Unassigned'
        if ',' in acct:
            parts = [account_short_label(p.strip()) for p in acct.split(',')]
            split_value = h['cv'] / len(parts)
            for p in parts:
                account_totals[p] = account_totals.get(p, 0) + split_value
        else:
            label = account_short_label(acct)
            account_totals[label] = account_totals.get(label, 0) + h['cv']
    account_breakdown = [{'account': k, 'value': round(v,2), 'pct': round(v/total_value*100,2)}
                         for k,v in sorted(account_totals.items(), key=lambda x:-x[1])]
    # Add margin as its own slice so the Portfolios tab's percentages sum to
    # 100% of total capital deployed (holdings + borrowed margin), not just
    # 100% across holdings alone with margin invisible.
    if margin:
        denom = total_value + margin
        account_breakdown = [{'account': a['account'], 'value': a['value'], 'pct': round(a['value']/denom*100,2)}
                             for a in account_breakdown]
        account_breakdown.append({'account': 'Margin', 'value': round(margin,2), 'pct': round(margin/denom*100,2)})

    # ── Country breakdown — simplified to United States vs Other ──
    # Snowball's "Country" column only resolves to a real country for
    # individual stocks; ETFs/funds come through as "Other" since the CSV
    # has no look-through into a fund's underlying country exposure. Rather
    # than show a long tail of mostly-"Other" rows, collapse this to the
    # two-bucket split that's actually meaningful given the data available.
    us_value = sum(h['cv'] for h in holdings if (h.get('country') or '') == 'United States of America')
    other_value = total_value - us_value
    country_breakdown = [
        {'country': 'United States', 'value': round(us_value,2), 'pct': round(us_value/total_value*100,2)},
        {'country': 'Other', 'value': round(other_value,2), 'pct': round(other_value/total_value*100,2)},
    ]

    # ── Investment Type breakdown ──
    # Sourced from the "Note" column, which the person fills in manually
    # with labels like "Dividend Income", "Growth", "Cash Equiv", etc.
    # Holdings with no Note yet are grouped as "Unassigned" so the chart
    # still accounts for 100% of the portfolio rather than silently
    # dropping unlabeled holdings.
    invtype_totals = {}
    for h in holdings:
        itype = h.get('investment_type') or 'Unassigned'
        invtype_totals[itype] = invtype_totals.get(itype, 0) + h['cv']
    investment_type_breakdown = [{'type': k, 'value': round(v,2), 'pct': round(v/total_value*100,2)}
                         for k,v in sorted(invtype_totals.items(), key=lambda x:-x[1])]

    # ── Weighted beta & yield ──
    beta_holdings = [h for h in holdings if h.get('beta') is not None]
    weighted_beta = None
    if beta_holdings:
        bv = sum(h['cv'] for h in beta_holdings)
        weighted_beta = round(sum(h['cv']*h['beta'] for h in beta_holdings) / bv, 2) if bv else None

    yield_holdings = [h for h in holdings if h.get('yield') is not None]
    weighted_yield = None
    if yield_holdings:
        yv = sum(h['cv'] for h in yield_holdings)
        weighted_yield = round(sum(h['cv']*h['yield'] for h in yield_holdings) / yv, 2) if yv else None

    # ── Monthly Distributions ("Passive Income" card) + Dividends tab data ──
    # Sums what was *actually paid* across every holding, using each
    # ticker's real dividend payment history (ex-div dates + per-share
    # amounts from Yahoo) rather than estimating off a static yield figure.
    # Uses current share count as the multiplier since the CSV doesn't
    # track historical share counts over time — for anyone not actively
    # trading in/out of positions intra-month, this is exact; for anyone
    # who changed position size mid-month, it's a close approximation using
    # today's share count instead of the actual count held on each specific
    # payment date.
    today = datetime.now()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    # Trailing 12 calendar months, oldest first, e.g. ['2025-07', ..., '2026-06']
    trailing_12_months = []
    cursor = first_of_this_month
    for _ in range(12):
        cursor = (cursor - timedelta(days=1)).replace(day=1)
        trailing_12_months.append(cursor.strftime('%Y-%m'))
    trailing_12_months.reverse()
    monthly_payout_totals = {m: 0.0 for m in trailing_12_months}

    monthly_distributions = 0.0
    monthly_distributions_by_holding = []
    trailing_12mo_total = 0.0
    holding_yields = []
    for h in holdings:
        shares = h.get('shares')
        if h.get('yield') is not None and h['yield'] <= 175:
            # Yields above 175% are virtually always a data error (a stale
            # or miscalculated figure from the source), not a real
            # distribution rate — exclude them rather than show a number
            # that would mislead more than inform.
            holding_yields.append({
                't': h['t'], 'n': h.get('n') or h['t'], 'yield': h['yield'],
                'is_active': bool(shares),
            })
        if not shares:
            continue
        history = get_dividend_history_cached(h['t'])
        if not history:
            continue
        paid_last_month = 0.0
        for entry in history:
            month_key = entry['date'][:7]  # 'YYYY-MM'
            amt = entry['amount'] * shares
            if month_key in monthly_payout_totals:
                monthly_payout_totals[month_key] += amt
                trailing_12mo_total += amt
            if last_month_start.strftime('%Y-%m-%d') <= entry['date'] <= last_month_end.strftime('%Y-%m-%d'):
                paid_last_month += entry['amount']
        if paid_last_month > 0:
            holding_total = round(paid_last_month * shares, 2)
            monthly_distributions += holding_total
            monthly_distributions_by_holding.append({'t': h['t'], 'amount': holding_total})
    monthly_distributions = round(monthly_distributions, 2)
    monthly_distributions_by_holding.sort(key=lambda x: -x['amount'])
    holding_yields.sort(key=lambda x: -x['yield'])

    monthly_payout_history = [{'month': m, 'amount': round(v, 2)} for m, v in
                               sorted(monthly_payout_totals.items())]
    trailing_12mo_total = round(trailing_12mo_total, 2)
    # Annual income: prefer the real trailing-12-month actual total (captures
    # true payment cadence — monthly/quarterly/weekly funds all net out
    # correctly over a full year); fall back to annualizing last month if a
    # holding's history doesn't go back far enough yet.
    annual_income = trailing_12mo_total if trailing_12mo_total > 0 else round(monthly_distributions * 12, 2)
    daily_income = round(annual_income / 365, 2)

    # ── Weighted Portfolio P/E (fetch trailing P/E per holding via yfinance, value-weighted) ──
    # Note: P/E lives in the slower .info dict, not fast_info — and many ETFs don't have one
    # (e.g. bond funds, commodity ETFs). Only equity-like holdings will contribute.
    # Fetched in parallel since .info is a slow call and there can be 20-30 tickers.
    weighted_pe = None
    pe_values = {}
    try:
        import concurrent.futures as _cf
        def fetch_pe(t):
            try:
                info = yf.Ticker(t).info
                pe = info.get('trailingPE')
                return (t, pe) if pe and pe > 0 else (t, None)
            except:
                return (t, None)
        with _cf.ThreadPoolExecutor(max_workers=10) as ex:
            for t, pe in ex.map(fetch_pe, tickers):
                if pe:
                    pe_values[t] = pe
        if pe_values:
            pe_holdings = [h for h in holdings if h['t'] in pe_values]
            pv = sum(h['cv'] for h in pe_holdings)
            if pv:
                weighted_pe = round(sum(h['cv']*pe_values[h['t']] for h in pe_holdings) / pv, 1)
    except Exception as e:
        print(f'  [Portfolio] P/E calc error: {e}')

    # SPY P/E for comparison (approx market P/E)
    spy_pe = None
    try:
        spy_info = yf.Ticker('SPY').info
        spy_pe = spy_info.get('trailingPE')
    except: pass

    # ── Risk-adjusted metrics: TWR, Sharpe, Sortino, Beta — portfolio vs each benchmark ──
    risk_adjusted = {
        'portfolio_pe': weighted_pe,
        'spy_pe': round(spy_pe,1) if spy_pe else None,
        'twr_1y': time_weighted_return(portfolio_series.iloc[-252:] if len(portfolio_series)>252 else portfolio_series),
        'sharpe_1y': sharpe_ratio(portfolio_series, 252),
        'sortino_1y': sortino_ratio(portfolio_series, 252),
        'beta_vs_spy': beta_vs_benchmark(portfolio_series, closes_cache.get('SPY'), 252),
        'spy_sharpe_1y': sharpe_ratio(closes_cache.get('SPY'), 252),
        'spy_sortino_1y': sortino_ratio(closes_cache.get('SPY'), 252),
        'spy_twr_1y': time_weighted_return(closes_cache.get('SPY').iloc[-252:] if closes_cache.get('SPY') is not None and len(closes_cache.get('SPY'))>252 else closes_cache.get('SPY')),
    }

    # ── Monte Carlo simulation: 10-year forward projection bootstrapped from
    # the portfolio's own historical daily returns ──
    monte_carlo = None
    try:
        starting_value = total_value - (margin or 0)  # net of margin, same basis as "Portfolio Value" card
        monte_carlo = monte_carlo_simulation(portfolio_series, starting_value, years=10, n_sims=2000)
    except Exception as e:
        print(f'  [Portfolio] Monte Carlo error: {e}')

    # ── Growth chart series (normalized to 100 at start) ──
    # Portfolio series also includes actual dollar value (scaled by current total_value)
    # Benchmarks are truncated to the SAME start date as portfolio_series
    # (the latest inception date among currently-held positions) so the
    # comparison is apples-to-apples over an identical window — otherwise
    # the portfolio line would only span its truncated period while the
    # benchmark kept showing the full untruncated 5y history, making the
    # two impossible to compare visually.
    growth_chart = {}
    portfolio_start = portfolio_series.index.min() if len(portfolio_series) > 0 else None
    for label, series, key in [('portfolio', portfolio_series, 'portfolio')] + \
                               [(tk, closes_cache.get(tk), tk) for tk in benchmarks]:
        if series is not None and len(series) > 0:
            if key != 'portfolio' and portfolio_start is not None:
                series = series[series.index >= portfolio_start]
            if len(series) == 0:
                continue
            norm = (series / series.iloc[0] * 100).round(2)
            step = max(1, len(norm) // 260)
            sampled = norm.iloc[::step]
            if key == 'portfolio':
                # Scale normalized index to actual dollar value at each point in time
                # (approximation: assumes current total_value represents the end of the series)
                dollar_scale = total_value / float(norm.iloc[-1]) * 100
                growth_chart[key] = [{'date': str(d.date()) if hasattr(d,'date') else str(d),
                                      'value': float(v), 'dollars': round(float(v) * dollar_scale / 100, 2)}
                                     for d,v in sampled.items()]
            else:
                growth_chart[key] = [{'date': str(d.date()) if hasattr(d,'date') else str(d), 'value': float(v)}
                                     for d,v in sampled.items()]

    # ── Margin interest cost ──
    margin_rate_data = {}
    rate_file = os.path.join(BASE_DIR, 'margin_rate.json')
    if os.path.exists(rate_file):
        try: margin_rate_data = json.load(open(rate_file))
        except: pass
    margin_annual_rate = margin_rate_data.get('rate', 5.0)
    margin_annual_cost = round(margin * margin_annual_rate / 100, 2) if margin else 0
    margin_monthly_cost = round(margin_annual_cost / 12, 2) if margin else 0

    # ── Holdings detail list (for the readout under the pie charts) ──
    # One row per holding — no value-splitting across accounts here, since
    # the frontend now shows all accounts a position resides in as a single
    # combined column rather than separate per-account rows.
    holdings_detail = []
    for h in holdings:
        acct = h.get('account') or 'Unassigned'
        account_labels = [account_short_label(p.strip()) for p in acct.split(',')] if acct else ['Unassigned']
        # For holdings split across multiple accounts (e.g. CGDV held in both
        # "Roth Gaby" and "Traditional IRA"), the combined Holdings Detail
        # table shows the full value — but the per-account popup needs the
        # actual split portion for that specific account, or it would show
        # (and double-count) the entire position under each account it's
        # split into. account_values maps each account label to its share.
        n_accounts = len(account_labels) or 1
        split_value = round(h['cv'] / n_accounts, 2)
        account_values = {label: split_value for label in account_labels}
        holdings_detail.append({
            't': h['t'], 'n': h['n'], 'accounts': account_labels,
            'account_values': account_values,
            'value': round(h['cv'], 2),
            'shares': h.get('shares'),
            'cost_basis': h.get('cb'),
            'share_price': h.get('price'),
            'cost_per_share': h.get('cost_per_share'),
            'realized_pl': h.get('realized_pl'),
            'div_received': h.get('div_received'),
            'total_profit': h.get('total_profit'),
        })
    holdings_detail.sort(key=lambda x: (-x['value']))

    return jsonify({
        'total_value': round(total_value, 2),
        'margin': margin,
        'margin_annual_rate': margin_annual_rate,
        'margin_annual_cost': margin_annual_cost,
        'margin_monthly_cost': margin_monthly_cost,
        'holdings_count': len(holdings),
        'holdings_detail': holdings_detail,
        'weighted_beta': weighted_beta,
        'weighted_yield': weighted_yield,
        'monthly_distributions': monthly_distributions,
        'annual_income': annual_income,
        'daily_income': daily_income,
        'monthly_payout_history': monthly_payout_history,
        'holding_yields': holding_yields,
        'monthly_distributions_month': last_month_start.strftime('%B %Y'),
        'monthly_distributions_by_holding': monthly_distributions_by_holding,
        'returns': returns_table,
        'risk': risk_table,
        'sectors': sector_breakdown,
        'accounts': account_breakdown,
        'countries': country_breakdown,
        'investment_types': investment_type_breakdown,
        'risk_adjusted': risk_adjusted,
        'monte_carlo': monte_carlo,
        'growth_chart': growth_chart,
        'growth_chart_truncated_by': newest_inception_ticker,
        'growth_chart_start_date': str(portfolio_start.date()) if portfolio_start is not None else None,
        'benchmarks': benchmarks,
        'timestamp': datetime.now().strftime('%B %d, %Y %I:%M %p'),
    })


def home_redirect():
    return send_from_directory(BASE_DIR, 'home.html')

@app.route('/api/earnings/today')
def earnings_today():
    """Fetch today earnings using yfinance calendar"""
    try:
        import datetime
        import yfinance as yf
        import pandas as pd

        today = datetime.date.today()
        earnings_list = []

        # Pull earnings calendar for a date range around today
        try:
            # yfinance earnings calendar - check key tickers
            watchlist = [
                'AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','COST','NFLX',
                'AMD','PLTR','MU','INTC','CRWD','DDOG','ZS','PANW','CRM','ORCL',
                'JPM','GS','BAC','WMT','HD','NKE','DIS','SBUX','MCD','KO',
                'CLS','NBIS','MRVL','ALAB','LITE','ROK','IBM','BE','APLD'
            ]
            for sym in watchlist:
                try:
                    t = yf.Ticker(sym)
                    cal = t.calendar
                    if cal is not None and not cal.empty:
                        # Check earnings date columns
                        for col in cal.columns:
                            val = cal[col].iloc[0] if len(cal) > 0 else None
                            if val is not None:
                                try:
                                    earn_date = pd.Timestamp(val).date()
                                    if earn_date == today:
                                        info = t.info
                                        earnings_list.append({
                                            'symbol': sym,
                                            'name': info.get('shortName', sym),
                                            'time': 'BMO' if 'Before' in str(col) else 'AMC' if 'After' in str(col) else 'TBD'
                                        })
                                        break
                                except:
                                    pass
                except:
                    continue
        except Exception as e:
            print(f'  Earnings calendar error: {e}')

        if not earnings_list:
            earnings_list = [
                {'symbol': '📅', 'name': f'No confirmed earnings today ({today.strftime("%b %d")})', 'time': 'UNK'},
                {'symbol': '🔗', 'name': 'Check earningswhispers.com for full calendar', 'time': 'UNK'},
            ]

        return jsonify(earnings_list)
    except Exception as e:
        return jsonify([{'symbol': 'ERR', 'name': str(e)[:60], 'time': 'UNK'}])

def extract_headline_keyword(headline):
    """Extract the single most visually meaningful search term from a
    headline for use as an Unsplash photo query. Strips common financial/news
    filler words and picks the most concrete noun or named entity that would
    produce an interesting background image — e.g. 'Federal Reserve rate cut'
    → 'federal reserve', 'Oil prices surge' → 'oil', 'Tech stocks rally'
    → 'technology stock market'."""
    import re

    headline_lower = headline.lower()

    # Ordered keyword-to-search-query mappings — first match wins.
    # Tuned for financial/macro news headlines specifically.
    keyword_map = [
        # Macroeconomic
        (['federal reserve', 'fed rate', 'fomc', 'powell', 'interest rate'], 'federal reserve building'),
        (['inflation', 'cpi', 'pce', 'price index', 'consumer price'], 'inflation economy prices'),
        (['recession', 'gdp', 'economic growth', 'contraction'], 'economy recession'),
        (['jobs', 'unemployment', 'payroll', 'labor market', 'hiring'], 'jobs employment workers'),
        (['housing', 'mortgage', 'home price', 'real estate'], 'housing real estate'),
        # Markets & assets
        (['stock market', 's&p 500', 'dow jones', 'nasdaq', 'wall street', 'equities', 'rally', 'selloff', 'bull', 'bear'], 'stock market wall street'),
        (['bond', 'treasury', 'yield curve', 'fixed income'], 'us treasury bonds'),
        (['bitcoin', 'crypto', 'ethereum', 'digital asset'], 'bitcoin cryptocurrency'),
        (['gold', 'silver', 'precious metal'], 'gold bars precious metal'),
        (['oil', 'crude', 'opec', 'energy price', 'gasoline'], 'oil energy crude'),
        (['natural gas', 'lng'], 'natural gas energy'),
        # Companies & sectors
        (['nvidia', 'ai chip', 'semiconductor', 'chips act'], 'nvidia semiconductor chip'),
        (['apple', 'iphone', 'ipad', 'mac'], 'apple technology'),
        (['amazon', 'aws'], 'amazon headquarters'),
        (['microsoft', 'azure'], 'microsoft technology'),
        (['google', 'alphabet', 'alphabet inc'], 'google headquarters'),
        (['tesla', 'electric vehicle', 'ev '], 'electric vehicle tesla'),
        (['bank', 'jpmorgan', 'goldman', 'morgan stanley', 'citigroup', 'wells fargo', 'finance'], 'bank finance wall street'),
        (['tech', 'technology', 'software', 'silicon valley'], 'technology silicon valley'),
        (['healthcare', 'pharma', 'drug', 'fda', 'biotech'], 'healthcare pharmaceutical'),
        (['retail', 'consumer spending', 'walmart', 'amazon'], 'retail shopping consumer'),
        # Geopolitical
        (['china', 'beijing', 'trade war', 'tariff'], 'china trade'),
        (['ukraine', 'russia', 'war', 'conflict'], 'geopolitical conflict'),
        (['europe', 'ecb', 'eurozone', 'european'], 'europe european union'),
        (['middle east', 'israel', 'iran', 'opec'], 'middle east'),
        # Policy & politics
        (['congress', 'senate', 'white house', 'biden', 'trump', 'administration'], 'us capitol congress'),
        (['debt ceiling', 'deficit', 'budget', 'fiscal'], 'us government budget'),
        (['regulation', 'sec', 'antitrust'], 'government regulation'),
        # Misc business
        (['merger', 'acquisition', 'takeover', 'deal'], 'business merger deal'),
        (['ipo', 'public offering', 'listing'], 'stock market ipo'),
        (['earnings', 'revenue', 'profit', 'quarterly results'], 'corporate earnings business'),
        (['supply chain', 'manufacturing', 'factory'], 'supply chain manufacturing'),
    ]

    for keywords, query in keyword_map:
        if any(kw in headline_lower for kw in keywords):
            return query

    # Fallback: extract the first meaningful word(s) from the headline,
    # stripping common filler words, and append 'finance' for context
    stop_words = {'the','a','an','is','are','was','were','has','have','had',
                  'in','on','at','to','for','of','and','or','but','with',
                  'as','by','from','that','this','it','its','be','do','did',
                  'will','would','could','should','may','might','new','says',
                  'said','set','get','got','up','down','over','after','before',
                  'how','why','what','when','where','who','report','data',
                  'amid','amid','while','amid','despite','after','since'}
    words = re.findall(r"[a-z']+", headline_lower)
    meaningful = [w for w in words if w not in stop_words and len(w) > 3]
    if meaningful:
        return ' '.join(meaningful[:2]) + ' finance'
    return 'financial news economy'


def fetch_unsplash_image(query):
    """Fetch a relevant background photo URL from Unsplash for a given search
    query. Results are cached on disk per query (24h TTL) so repeated loads
    of the same headline topic don't burn through the free 50 req/hour limit.
    Returns the photo URL (landscape, medium size) or None on failure."""
    import hashlib
    os.makedirs(UNSPLASH_CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(query.lower().encode()).hexdigest()
    cache_path = os.path.join(UNSPLASH_CACHE_DIR, f'{cache_key}.json')

    # Serve from disk cache if fresh (24h — Unsplash photos don't change)
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 24:
            try:
                cached = json.load(open(cache_path))
                return cached.get('url')
            except Exception:
                pass

    try:
        resp = requests.get(
            'https://api.unsplash.com/search/photos',
            params={
                'query': query,
                'per_page': 5,
                'orientation': 'landscape',
                'content_filter': 'high',
            },
            headers={'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'},
            timeout=6,
        )
        if resp.status_code != 200:
            print(f'  [Unsplash] HTTP {resp.status_code} for query "{query}"')
            return None
        results = resp.json().get('results', [])
        if not results:
            print(f'  [Unsplash] No results for query "{query}"')
            return None
        # Pick a random one from the top 5 so the same topic doesn't
        # always show the exact same photo every single load
        import random
        photo = random.choice(results[:min(5, len(results))])
        url = photo['urls'].get('regular')  # ~1080px wide, good for backgrounds
        json.dump({'url': url, 'query': query}, open(cache_path, 'w'))
        print(f'  [Unsplash] Fetched image for "{query}": {url[:60]}...')
        return url
    except Exception as e:
        print(f'  [Unsplash] Error for "{query}": {e}')
        return None


@app.route('/api/unsplash-hero')
def unsplash_hero():
    """Given a headline, extract a search keyword and return a matching
    Unsplash background image URL for the hero card on the home page."""
    from flask import request as freq
    headline = freq.args.get('headline', '').strip()
    if not headline:
        return jsonify({'error': 'headline param required'}), 400
    query = extract_headline_keyword(headline)
    url = fetch_unsplash_image(query)
    if not url:
        return jsonify({'url': None, 'query': query})
    return jsonify({'url': url, 'query': query})


@app.route('/api/headlines')
def macro_headlines():
    """Fetch real macro headlines from RSS feeds — economic, headline news, market"""
    try:
        import requests as req
        import xml.etree.ElementTree as ET
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        headlines = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        }

        feeds = [
            # Economic / macro data
            ('https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines', 'MarketWatch'),
            ('https://finance.yahoo.com/news/rssindex', 'Yahoo Finance'),
            ('https://www.cnbc.com/id/100003114/device/rss/rss.html', 'CNBC'),
            ('https://www.cnbc.com/id/15839135/device/rss/rss.html', 'CNBC Economy'),
            ('https://news.google.com/rss/search?q=economy+inflation+GDP+jobs&hl=en-US&gl=US&ceid=US:en', 'Google News'),
            ('https://news.google.com/rss/search?q=stock+market+S%26P+500+nasdaq&hl=en-US&gl=US&ceid=US:en', 'Google News'),
            ('https://news.google.com/rss/search?q=breaking+news+business+finance&hl=en-US&gl=US&ceid=US:en', 'Google News'),
            ('https://www.investing.com/rss/news.rss', 'Investing.com'),
            ('https://www.ft.com/?format=rss', 'Financial Times'),
            ('https://feeds.bbci.co.uk/news/business/rss.xml', 'BBC Business'),
        ]

        def parse_pub_date(raw):
            """Parse RSS pubDate to ISO string, return None if unparseable"""
            if not raw:
                return None
            try:
                dt = parsedate_to_datetime(raw)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                try:
                    # Try common formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S%z','%Y-%m-%d %H:%M:%S','%a, %d %b %Y %H:%M:%S %Z']:
                        try:
                            dt = datetime.strptime(raw.strip()[:25], fmt[:len(raw.strip()[:25])])
                            return dt.isoformat()
                        except:
                            continue
                except:
                    pass
            return None

        def categorize(title):
            tl = title.lower()
            if any(w in tl for w in ['inflation','cpi','pce','gdp','jobs report','unemployment','payroll','nonfarm','jobless','claims','economic data','retail sales','pmi','ism','consumer confidence','housing','federal budget','deficit','debt ceiling','treasury yield','yield curve','rate cut','rate hike','interest rate','federal reserve','fomc','powell','fed meeting','basis points','monetary policy','tightening','easing']):
                return 'economic'
            elif any(w in tl for w in ['war','conflict','russia','ukraine','china','taiwan','middle east','nato','military','troops','sanctions','geopolit','election','president','congress','tariff','trade war','biden','trump','white house','g7','g20','imf','world bank']):
                return 'headline'
            else:
                return 'market'

        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=24)

        for url, source in feeds:
            try:
                r = req.get(url, headers=headers, timeout=8)
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.content)
                items = root.findall('.//item')[:8]
                for item in items:
                    title = item.findtext('title','').strip()
                    link  = item.findtext('link','').strip()
                    pub   = item.findtext('pubDate','') or item.findtext('{http://purl.org/dc/elements/1.1/}date','')
                    if not title or len(title) < 10:
                        continue
                    iso_time = parse_pub_date(pub)
                    # Filter to last 24 hours (skip if older, keep if time unparseable)
                    if iso_time:
                        try:
                            item_dt = datetime.fromisoformat(iso_time)
                            if item_dt < cutoff_dt:
                                continue
                        except:
                            pass
                    cat = categorize(title)
                    headlines.append({
                        'headline': title,
                        'category': cat,
                        'source': source,
                        'time': iso_time or pub[:16] if pub else '',
                        'url': link
                    })
            except Exception as e:
                print(f'  RSS feed error ({source}): {e}')
                continue

        # Deduplicate by headline similarity
        seen = []
        deduped = []
        for h in headlines:
            words = set(h['headline'].lower().split())
            if not any(len(words & set(s.lower().split())) > 5 for s in seen):
                deduped.append(h)
                seen.append(h['headline'])
        headlines = deduped

        # Sort by time descending within each category
        headlines.sort(key=lambda x: x.get('time',''), reverse=True)

        # Return up to 5 per category (economic, headline, market)
        economic = [h for h in headlines if h['category']=='economic'][:5]
        headline = [h for h in headlines if h['category']=='headline'][:5]
        market   = [h for h in headlines if h['category']=='market'][:5]
        result   = economic + headline + market

        if not result:
            result = [
                {'headline': 'Economic data — check BLS.gov for latest CPI and jobs figures', 'category': 'economic', 'source': 'BLS', 'time': '', 'url': 'https://www.bls.gov'},
                {'headline': 'Markets update available on Yahoo Finance', 'category': 'market', 'source': 'Yahoo Finance', 'time': '', 'url': 'https://finance.yahoo.com'},
            ]

        return jsonify(result)
    except Exception as e:
        print(f'  Headlines endpoint error: {e}')
        return jsonify([{'headline': f'Headlines unavailable: {str(e)}', 'category': 'market', 'source': '', 'time': '', 'url': ''}])

def claude_call(prompt):
    """Make a call to the Anthropic API and return parsed JSON list."""
    import re
    resp = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        json={
            'model': 'claude-sonnet-4-6',
            'max_tokens': 1000,
            'messages': [{'role': 'user', 'content': prompt}]
        },
        timeout=30
    )
    resp.raise_for_status()
    raw = resp.json()['content'][0]['text']
    # Strip markdown code fences if present
    clean = re.sub(r'```(?:json)?', '', raw).strip()
    return json.loads(clean)

@app.route('/api/market-movers')
def market_movers():
    """Fetch true market-wide top 5 gainers and losers from Yahoo Finance screeners."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36',
            'Accept': 'application/json',
        }

        def fetch_screener(scrtype):
            url = f'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&scrIds={scrtype}&count=6&start=0'
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            quotes = data['finance']['result'][0]['quotes']
            out = []
            for q in quotes:
                sym   = q.get('symbol','')
                name  = q.get('shortName') or q.get('longName') or sym
                price = q.get('regularMarketPrice')
                chg   = q.get('regularMarketChangePercent')
                pts   = q.get('regularMarketChange')
                vol   = q.get('regularMarketVolume')
                if not sym or price is None or chg is None:
                    continue
                # Shorten long names
                name = name.replace(' Inc.','').replace(' Corp.','').replace(' Corporation','').replace(' Holdings','').replace(', Inc','').strip()
                out.append({
                    'sym': sym,
                    'name': name[:22],
                    'price': round(price, 2),
                    'chg': round(chg, 2),
                    'pts': round(pts, 2) if pts is not None else None,
                    'vol': vol,
                })
            return out[:5]

        gainers = fetch_screener('day_gainers')
        losers  = fetch_screener('day_losers')
        return jsonify({'gainers': gainers, 'losers': losers})
    except Exception as e:
        print(f'  Market movers error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-beneficiaries')
def ai_beneficiaries():
    """Top 5 AI-benefiting sectors and companies."""
    try:
        today = datetime.now().strftime('%b %d, %Y')
        prompt = (
            f"Today is {today}. You are a senior equity strategist. "
            "Identify the TOP 5 sectors and specific company names that are currently benefiting most from AI adoption and spending. "
            "For each, give: sector name, 2-3 specific publicly traded company tickers, and 1 concise sentence on why AI is a tailwind for them right now. "
            "Focus on forward-looking structural advantage, not just recent price action. "
            "Respond ONLY as a JSON array like: "
            '[{"rank":1,"sector":"Sector Name","tickers":["TICK1","TICK2"],"thesis":"one sentence why"}]. '
            "No preamble, no markdown, pure JSON array only."
        )
        result = claude_call(prompt)
        return jsonify(result)
    except Exception as e:
        print(f'  AI beneficiaries error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-bottlenecks')
def ai_bottlenecks():
    """NDX 100 companies with backlog + margin expansion = pricing power."""
    try:
        today = datetime.now().strftime('%b %d, %Y')
        prompt = (
            f"Today is {today}. You are a fundamental equity analyst. "
            "Analyze the Nasdaq 100 universe to identify companies that are current BOTTLENECKS in the AI infrastructure buildout — "
            "meaning their products or services are in tight demand. "
            "Use this two-factor framework: "
            "(1) Rising or elevated backlog — demand is outpacing their ability to deliver, "
            "(2) Expanding gross or operating margins — pricing power confirmed, they can charge more because supply is constrained. "
            "Identify 4-5 specific NDX 100 companies where BOTH signals are present or trending. "
            "For each give: ticker, company name, what the bottleneck is (what AI-critical product/service they provide), "
            "and one line on backlog + margin evidence. "
            "Respond ONLY as a JSON array: "
            '[{"ticker":"TICK","name":"Full Name","bottleneck":"what they make that AI needs","evidence":"backlog + margin signal in one sentence"}]. '
            "No preamble, no markdown, pure JSON only."
        )
        result = claude_call(prompt)
        return jsonify(result)
    except Exception as e:
        print(f'  AI bottlenecks error: {e}')
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
#  SCANNER PRE-COMPUTE ENGINE
#  Batch downloads all tickers every 15 min on a background thread.
#  Browser gets instant results from cache — no per-ticker fetches.
# ═══════════════════════════════════════════════════════════════

import math
import concurrent.futures

# ── Scanner ticker universe ──
SCANNER_TICKERS = [
    # ── Custom AI / Semis / Tech (original watchlist) ──
    'CLS','BE','NBIS','MRVL','ALAB','APLD','PLTR','LITE','IREN','MXL','CIEN','TSEM',
    'CRDO','ICHR','FORM','SILC','BB','SANM','INTT','FSLY','OUST','RBRK','S',
    'AVT','KEYS','HPE','SNDK','APP','TEAM','ANET','SNPS','OKTA','MCHP',

    # ── S&P 500 — Information Technology ──
    'AAPL','MSFT','NVDA','AVGO','ORCL','CRM','ACN','AMD','CSCO','ADBE','TXN','QCOM',
    'INTU','IBM','AMAT','NOW','LRCX','KLAC','ADI','MU','SNPS','CDNS','FTNT','PANW',
    'INTC','ROP','CTSH','ANSS','KEYS','GLW','GDDY','FFIV','NTAP','ZBRA','STX','WDC',
    'JNPR','TER','EPAM','AKAM','SWKS','NXPI','MPWR','ENPH','FSLR','SEDG','HPQ',

    # ── S&P 500 — Communication Services ──
    'GOOGL','GOOG','META','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR','WBD',
    'PARA','FOX','FOXA','IPG','OMC','NWS','NWSA','LYV','EA','TTWO','MTCH',

    # ── S&P 500 — Consumer Discretionary ──
    'AMZN','TSLA','HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','ORLY','AZO',
    'ABNB','MAR','HLT','YUM','DPZ','EXPE','ULTA','LULU','RL','PVH','TPR','HAS',
    'MHK','ETSY','EBAY','BBY','DG','DLTR','TGT','KR','ROST','APTV','BWA',
    'LEA','F','GM','LVS','MGM','WYNN','CZR','CCL','RCL','NCLH','VFC',

    # ── S&P 500 — Consumer Staples ──
    'WMT','PG','COST','KO','PEP','PM','MO','MDLZ','CL','MNST','KHC','GIS',
    'STZ','MKC','SJM','CPB','HRL','CAG','K','CHD','CLX','EL','KMB',

    # ── S&P 500 — Financials ──
    'JPM','BAC','WFC','MS','GS','BLK','SCHW','AXP','SPGI','MCO','COF','USB',
    'PNC','TFC','BK','STT','MTB','FITB','RF','HBAN','CFG','ALLY','SYF','DFS',
    'AIG','MET','PRU','AFL','ALL','PGR','TRV','CB','MMC','AON','WTW','BRO',
    'ICE','CME','CBOE','NDAQ','MSCI','V','MA','PYPL','FIS','FI','GPN','WEX',
    'BRK-B','TROW','IVZ','BEN','AMG',

    # ── S&P 500 — Health Care ──
    'UNH','LLY','JNJ','ABBV','MRK','TMO','ABT','DHR','BMY','PFE','AMGN','ISRG',
    'VRTX','REGN','CVS','CI','HUM','CNC','MOH','ELV','HCA','IQV','MDT','BSX',
    'EW','BDX','SYK','ZBH','HOLX','IDXX','GEHC','A','ALGN','DXCM','PODD',
    'BIIB','GILD','MRNA','ILMN','CTLT','DVA','HSIC','MTD','PKI','RMD','VAR',
    'BAX','BIO','CRL','TECH','ZTS','VTRS','NBIX','INCY','EXAS',

    # ── S&P 500 — Industrials ──
    'GE','HON','CAT','DE','RTX','LMT','NOC','GD','BA','MMM','UPS','FDX','UNP',
    'CSX','NSC','EMR','ETN','PH','ROK','ITW','CMI','CARR','OTIS','IR','FAST',
    'GWW','PCAR','ODFL','JBHT','XPO','SAIA','RSG','WM','ECL','CTAS','VRSK',
    'CPRT','EXPD','CH','CHRW','SWK','RRX','AME','AXON','HWM','TDG','TXT',
    'LHX','LDOS','BAH','SAIC','J','PWR','HUBB','FTV','MAS','SNA','FLR','GGG',
    'GNRC','ALLE','XYL','IEX','PNR','AAON','AIRC',

    # ── S&P 500 — Energy ──
    'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','HAL','BKR','HES',
    'DVN','FANG','MRO','APA','EQT','CTRA','KMI','WMB','OKE','LNG','TRGP',

    # ── S&P 500 — Materials ──
    'LIN','SHW','APD','ECL','DD','DOW','NEM','FCX','NUE','STLD','CF','MOS',
    'ALB','PPG','EMN','FMC','IFF','RPM','VMC','MLM','PKG','IP','WRK','CCK',
    'AVY','SEE','BLL','AMCR',

    # ── S&P 500 — Real Estate ──
    'PLD','AMT','EQIX','CCI','SPG','PSA','O','VICI','WELL','EQR','AVB','MAA',
    'UDR','CPT','ESS','NXR','ARE','BXP','VNO','KIM','REG','FRT','NRZ',
    'INVH','AMH','SUI','ELS','IRM','SBAC','DLR','QTS',

    # ── S&P 500 — Utilities ──
    'NEE','DUK','SO','D','AEP','EXC','SRE','PEG','XEL','WEC','ES','ETR',
    'FE','PPL','CMS','NI','AES','LNT','EVRG','PNW','AWK','CNP','NRG',

    # ── S&P 500 — Energy / Midstream extras ──
    'PSX','VLO','MPC',

    # ── S&P 500 — Remaining / Mixed sectors ──
    'AOS','AEE','AAL','AMP','APH','ACGL','ADM','AJG','AIZ','ATO','ADP',
    'BALL','BBWI','WRB','BX','BR','BLDR','BG','CAH','KMX','CBRE','CDW',
    'CE','COR','CINF','C','ED','COO','CPAY','CTVA','CSGP','DRI','DELL','DAL',
    'XRAY','DOV','DHI','DTE','EIX','EFX','EG','EXR','FDS','FICO','FLT','GRMN',
    'IT','GEN','GL','DOC','HSY','HST','HII','PODD','IFF','IP','JKHY','JCI',
    'KVUE','KEY','LH','LW','L','LYB','MKTX','TAP','MSI','NOV','NVR','OGN',
    'PKG','PCG','POOL','PFG','PTC','QRVO','DGX','RJF','RVTY','ROL',
    'SOLV','LUV','STE','SYY','TDY','TFX','TSCO','TT','TRMB','TSN',
    'UBER','UAL','URI','VTR','VLTO','VRSN','WBA','WAT','WST','WHR','WY',
    'GWW','ZTS','PAYC','LEN','ELV','BAX','AMCR','SEE','NRG','BF-B',
    'NTRS','RF','HBAN','MTB','FITB','CFG','USB','STT','ALLY','SYF','DFS',
    'WRB','GL','MHK','TAP','BBWI','VFC',
]

# Deduplicate while preserving order
seen = set()
SCANNER_TICKERS_DEDUPED = []
for t in SCANNER_TICKERS:
    if t not in seen:
        seen.add(t)
        SCANNER_TICKERS_DEDUPED.append(t)
SCANNER_TICKERS = SCANNER_TICKERS_DEDUPED

# ── In-memory scanner cache ──
_scanner_cache = {
    'data': {},          # ticker -> computed dict
    'last_run': None,    # datetime of last completed run
    'running': False,    # is a compute currently in progress
    'progress': 0,       # 0-100
    'total': 0,
    'done': 0,
    'errors': [],
}
SCANNER_TTL = 900  # 15 minutes

# ── Math helpers (server-side mirrors of JS functions) ──
def _sma(arr, period):
    if len(arr) < period:
        return None
    return sum(arr[-period:]) / period

def _ema(arr, period):
    k = 2.0 / (period + 1)
    e = arr[0]
    for v in arr[1:]:
        e = v * k + e * (1 - k)
    return e

def _rsi14(closes):
    if len(closes) < 15:
        return None
    g = l = 0.0
    for i in range(len(closes) - 14, len(closes)):
        d = closes[i] - closes[i-1]
        if d >= 0: g += d
        else: l += abs(d)
    avg_g = g / 14
    avg_l = l / 14
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)

def _bollinger(closes, period=20, mult=2.0):
    if len(closes) < period:
        return None
    sl = closes[-period:]
    mu = sum(sl) / period
    variance = sum((x - mu) ** 2 for x in sl) / period
    std = math.sqrt(variance)
    return {'u': mu + mult * std, 'l': mu - mult * std, 'm': mu}

def _macd(closes):
    if len(closes) < 35:
        return None
    recent = closes[-60:] if len(closes) >= 60 else closes
    line = _ema(recent, 12) - _ema(recent, 26)
    sigs = []
    for i in range(9, len(recent) + 1):
        sigs.append(_ema(recent[:i], 12) - _ema(recent[:i], 26))
    sig = _ema(sigs, 9) if len(sigs) >= 9 else sigs[-1] if sigs else 0
    return {'line': line, 'sig': sig, 'hist': line - sig}

BB_SIGMA_SERVER = {
    'ALAB': 3.0, 'APLD': 3.0, 'IREN': 3.0, 'NBIS': 2.8, 'BE': 2.5,
    'PLTR': 2.5, 'AMD': 2.2, 'MRVL': 2.2, 'NVDA': 2.2, 'TSLA': 2.5,
    'META': 2.2, 'CRDO': 2.5, 'FSLY': 3.0, 'OUST': 3.0, 'MXL': 2.5,
    'RBRK': 2.5, 'S': 2.5, 'APP': 2.2, 'CRWD': 2.5, 'MELI': 2.5,
    'DDOG': 2.5, 'ZS': 2.5, 'TTD': 2.5, 'MRNA': 3.0, 'ENPH': 3.0,
    'ZM': 2.5, 'ALGN': 2.5, 'WBD': 2.5, 'DXCM': 2.5, 'ABNB': 2.2,
}

def compute_ticker_data(ticker, closes, volumes, highs, lows, price, prev, day_high, day_low, open_p):
    """Compute all scanner indicators for one ticker from raw OHLCV data."""
    try:
        std  = BB_SIGMA_SERVER.get(ticker, 2.0)
        chg  = ((price - prev) / prev * 100) if prev else 0
        bb   = _bollinger(closes, 20, std)
        rsi  = _rsi14(closes)
        s50  = _sma(closes, 50)
        s100 = _sma(closes, 100)
        s200 = _sma(closes, 200)
        mc   = _macd(closes)
        vol  = volumes[-1] if volumes else 0
        avg_v = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else vol or 1
        vr   = vol / avg_v if avg_v else 1
        recent_h = highs[-252:] if len(highs) >= 252 else highs
        recent_l = lows[-252:]  if len(lows)  >= 252 else lows
        h52  = max(recent_h) if recent_h else price
        l52  = min(recent_l) if recent_l else price
        rp   = (price - l52) / (h52 - l52) if (h52 - l52) > 0 else 0.5
        bp   = (price - bb['l']) / (bb['u'] - bb['l']) if bb and (bb['u'] - bb['l']) > 0 else 0.5
        bp   = max(0, min(1, bp))

        # Value score
        score = 5.0
        if rsi is not None:
            if rsi < 30: score += 2.5
            elif rsi < 38: score += 1.5
            elif rsi < 45: score += 0.5
            elif rsi > 70: score -= 2.0
            elif rsi > 60: score -= 1.0
        if bp < 0.1: score += 2.0
        elif bp < 0.25: score += 1.0
        elif bp > 0.85: score -= 1.5
        above = sum(1 for sma_v in [s50, s100, s200] if sma_v and price > sma_v)
        if above == 0: score += 1.5
        elif above == 1: score += 0.5
        elif above == 3: score += 0.3
        if mc and mc['hist'] is not None:
            score += 0.4 if mc['hist'] > 0 else -0.2
        if vr > 1.5: score += 0.3
        score = max(1.0, min(10.0, round(score, 1)))

        return {
            'ticker': ticker,
            'price': round(price, 4),
            'prev': round(prev, 4),
            'chg': round(chg, 4),
            'dayHigh': round(day_high, 4),
            'dayLow': round(day_low, 4),
            'open': round(open_p, 4),
            'rsi': round(rsi, 2) if rsi is not None else None,
            'bb': {'u': round(bb['u'],4), 'l': round(bb['l'],4), 'm': round(bb['m'],4)} if bb else None,
            'bp': round(bp, 4),
            'std': std,
            's50': round(s50, 4) if s50 else None,
            's100': round(s100, 4) if s100 else None,
            's200': round(s200, 4) if s200 else None,
            'macd': {'line': round(mc['line'],5), 'sig': round(mc['sig'],5), 'hist': round(mc['hist'],5)} if mc else None,
            'vr': round(vr, 3),
            'rp': round(rp, 4),
            'h52': round(h52, 4),
            'l52': round(l52, 4),
            'score': score,
            'closes': [round(c, 4) for c in closes[-60:]],   # last 60 days only (enough for all indicators)
            'volumes': [int(v) for v in volumes[-60:]],
            'highs': [round(h, 4) for h in highs[-252:]],
            'lows': [round(l, 4) for l in lows[-252:]],
        }
    except Exception as e:
        return None

def run_scanner_compute():
    """
    Background compute: batch download all tickers, compute indicators, store in cache.
    Uses yf.download() — one network round-trip for all tickers instead of N individual calls.
    """
    global _scanner_cache
    if _scanner_cache['running']:
        print('  Scanner compute already running — skipping')
        return

    _scanner_cache['running'] = True
    _scanner_cache['progress'] = 0
    _scanner_cache['errors'] = []
    tickers = SCANNER_TICKERS
    _scanner_cache['total'] = len(tickers)
    _scanner_cache['done'] = 0

    print(f'  [Scanner] Starting batch compute for {len(tickers)} tickers...')
    t_start = time.time()

    try:
        # ── Batch download — one call for all tickers ──
        # Download in chunks of 100 to avoid timeouts
        CHUNK = 100
        all_hist = {}
        for i in range(0, len(tickers), CHUNK):
            chunk = tickers[i:i+CHUNK]
            try:
                df = yf.download(
                    chunk,
                    period='1y',
                    interval='1d',
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                for tk in chunk:
                    try:
                        if len(chunk) == 1:
                            tkdf = df
                        else:
                            tkdf = df[tk] if tk in df.columns.get_level_values(0) else None
                        if tkdf is not None and not tkdf.empty:
                            all_hist[tk] = tkdf
                    except:
                        pass
            except Exception as e:
                print(f'  [Scanner] Chunk {i//CHUNK+1} error: {e}')
            _scanner_cache['done'] = min(i + CHUNK, len(tickers))
            _scanner_cache['progress'] = int(_scanner_cache['done'] / len(tickers) * 70)

        # ── Fetch current prices in parallel using fast_info ──
        def get_fast_info(tk):
            try:
                t = yf.Ticker(tk)
                fi = t.fast_info
                return tk, {
                    'last': float(fi.last_price) if fi.last_price else None,
                    'prev': float(fi.previous_close) if fi.previous_close else None,
                    'day_high': float(fi.day_high) if fi.day_high else None,
                    'day_low': float(fi.day_low) if fi.day_low else None,
                }
            except:
                return tk, {}

        fast_infos = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(get_fast_info, tk): tk for tk in tickers}
            done_count = 0
            for fut in concurrent.futures.as_completed(futs):
                tk, info = fut.result()
                fast_infos[tk] = info
                done_count += 1
                _scanner_cache['progress'] = 70 + int(done_count / len(tickers) * 25)

        # ── Compute indicators for each ticker ──
        new_data = {}
        for tk in tickers:
            try:
                hist = all_hist.get(tk)
                if hist is None or hist.empty:
                    _scanner_cache['errors'].append(tk)
                    continue
                closes  = [float(x) for x in hist['Close'].tolist() if not math.isnan(x)]
                volumes = [float(x) for x in hist['Volume'].tolist() if not math.isnan(x)]
                highs   = [float(x) for x in hist['High'].tolist() if not math.isnan(x)]
                lows    = [float(x) for x in hist['Low'].tolist() if not math.isnan(x)]
                if len(closes) < 20:
                    continue
                fi = fast_infos.get(tk, {})
                price    = fi.get('last')    or (closes[-1] if closes else None)
                prev     = fi.get('prev')    or (closes[-2] if len(closes) > 1 else price)
                day_high = fi.get('day_high') or (highs[-1] if highs else price)
                day_low  = fi.get('day_low')  or (lows[-1] if lows else price)
                open_p   = float(hist['Open'].iloc[-1]) if not math.isnan(hist['Open'].iloc[-1]) else price
                if price is None:
                    continue
                result = compute_ticker_data(tk, closes, volumes, highs, lows, price, prev, day_high, day_low, open_p)
                if result:
                    new_data[tk] = result
            except Exception as e:
                _scanner_cache['errors'].append(tk)

        _scanner_cache['data'] = new_data
        _scanner_cache['last_run'] = datetime.now().isoformat()
        _scanner_cache['progress'] = 100
        elapsed = round(time.time() - t_start, 1)
        print(f'  [Scanner] Compute complete: {len(new_data)} tickers in {elapsed}s ({len(_scanner_cache["errors"])} errors)')

    except Exception as e:
        print(f'  [Scanner] Fatal compute error: {e}')
    finally:
        _scanner_cache['running'] = False

def scanner_background_loop():
    """Run compute on startup, then every 15 minutes."""
    time.sleep(3)  # brief delay to let Flask start
    while True:
        run_scanner_compute()
        time.sleep(SCANNER_TTL)


# ═══════════════════════════════════════════════════════════════
#  MACRO DASHBOARD BACKGROUND PRE-COMPUTE
#  Warms all macro data caches on startup so first page load is fast.
# ═══════════════════════════════════════════════════════════════

MACRO_TTL = 900  # 15 minutes
_macro_cache = {'data': None, 'last_run': None, 'running': False}

def run_macro_precompute():
    global _fred_series_cache
    if _macro_cache['running']:
        return
    _macro_cache['running'] = True
    print(f'  [Macro] Starting background pre-compute...')
    t0 = time.time()
    try:
        # Clear FRED series cache so we get fresh data
        _fred_series_cache = {}
        data = get_all_macro()
        _macro_cache['data'] = data
        _macro_cache['last_run'] = datetime.now().isoformat()
        print(f'  [Macro] Pre-compute done in {round(time.time()-t0,1)}s')
    except Exception as e:
        print(f'  [Macro] Pre-compute error: {e}')
    finally:
        _macro_cache['running'] = False

def macro_background_loop():
    time.sleep(5)  # Let Flask start first
    while True:
        run_macro_precompute()
        time.sleep(MACRO_TTL)

_macro_thread = threading.Thread(target=macro_background_loop, daemon=True)
_macro_thread.start()

# Start background thread
_scanner_thread = threading.Thread(target=scanner_background_loop, daemon=True)
_scanner_thread.start()

@app.route('/api/scanner/data')
def scanner_data():
    """Return all pre-computed scanner data as one JSON payload."""
    return jsonify({
        'data': list(_scanner_cache['data'].values()),
        'last_run': _scanner_cache['last_run'],
        'running': _scanner_cache['running'],
        'progress': _scanner_cache['progress'],
        'total': _scanner_cache['total'],
        'done': _scanner_cache['done'],
        'count': len(_scanner_cache['data']),
    })


def classify_etf(ticker, name=''):
    """Classify an ETF into the standardized taxonomy:
      size_bucket  : Large Cap | Mid Cap | Small Cap | International | Other
      style_bucket : Blend | Growth | Value | Other
      anchor_tag   : the most recognizable benchmark/theme
      display_label: "[Size] [Style] — [Anchor]"

    Priority order: explicit name/ticker rules → yfinance category fallback.
    Prefer widely recognized index names over issuer marketing terms.
    Prefer consistency over cleverness."""

    t = ticker.upper()
    n = (name or '').lower()

    # ── 1. EXACT TICKER OVERRIDES ──────────────────────────────────────────
    # These are the most common ETFs; hardcoding them is faster and more
    # reliable than trying to parse marketing copy in the name field.
    exact = {
        # S&P 500 Large Cap Blend
        'SPY':('Large Cap','Blend','S&P 500','Large Cap Blend — S&P 500'),
        'IVV':('Large Cap','Blend','S&P 500','Large Cap Blend — S&P 500'),
        'VOO':('Large Cap','Blend','S&P 500','Large Cap Blend — S&P 500'),
        'SPLG':('Large Cap','Blend','S&P 500','Large Cap Blend — S&P 500'),
        'RSP':('Large Cap','Blend','Equal Weight S&P 500','Large Cap Blend — Equal Weight S&P 500'),
        # Total market
        'VTI':('Large Cap','Blend','Total US Market','Large Cap Blend — Total US Market'),
        'ITOT':('Large Cap','Blend','Total US Market','Large Cap Blend — Total US Market'),
        'SCHB':('Large Cap','Blend','Total US Market','Large Cap Blend — Total US Market'),
        'FSKAX':('Large Cap','Blend','Total US Market','Large Cap Blend — Total US Market'),
        # Nasdaq / Growth
        'QQQ':('Large Cap','Growth','Nasdaq-100','Large Cap Growth — Nasdaq-100'),
        'QQQM':('Large Cap','Growth','Nasdaq-100','Large Cap Growth — Nasdaq-100'),
        'TQQQ':('Other','Growth','Nasdaq-100 3x Leveraged','Leveraged — Nasdaq-100 3x'),
        'SQQQ':('Other','Other','Nasdaq-100 3x Inverse','Inverse — Nasdaq-100 3x'),
        # Large Cap Growth / Value
        'VUG':('Large Cap','Growth','Large Growth','Large Cap Growth — Large Growth'),
        'IVW':('Large Cap','Growth','S&P 500 Growth','Large Cap Growth — S&P 500 Growth'),
        'SCHG':('Large Cap','Growth','Large Growth','Large Cap Growth — Large Growth'),
        'MGK':('Large Cap','Growth','Mega Cap Growth','Large Cap Growth — Mega Cap Growth'),
        'VTV':('Large Cap','Value','Large Value','Large Cap Value — Large Value'),
        'IVE':('Large Cap','Value','S&P 500 Value','Large Cap Value — S&P 500 Value'),
        'SCHV':('Large Cap','Value','Large Value','Large Cap Value — Large Value'),
        # Mid Cap
        'MDY':('Mid Cap','Blend','S&P MidCap 400','Mid Cap Blend — S&P MidCap 400'),
        'IJH':('Mid Cap','Blend','S&P MidCap 400','Mid Cap Blend — S&P MidCap 400'),
        'VO':('Mid Cap','Blend','Mid Cap','Mid Cap Blend — Mid Cap'),
        'IVOO':('Mid Cap','Growth','S&P MidCap 400 Growth','Mid Cap Growth — S&P MidCap 400 Growth'),
        'IJK':('Mid Cap','Growth','S&P MidCap 400 Growth','Mid Cap Growth — S&P MidCap 400 Growth'),
        'IVOV':('Mid Cap','Value','S&P MidCap 400 Value','Mid Cap Value — S&P MidCap 400 Value'),
        'IJJ':('Mid Cap','Value','S&P MidCap 400 Value','Mid Cap Value — S&P MidCap 400 Value'),
        # Small Cap
        'IWM':('Small Cap','Blend','Russell 2000','Small Cap Blend — Russell 2000'),
        'VTWO':('Small Cap','Blend','Russell 2000','Small Cap Blend — Russell 2000'),
        'SLY':('Small Cap','Blend','S&P SmallCap 600','Small Cap Blend — S&P SmallCap 600'),
        'IJR':('Small Cap','Blend','S&P SmallCap 600','Small Cap Blend — S&P SmallCap 600'),
        'VB':('Small Cap','Blend','Small Cap','Small Cap Blend — Small Cap'),
        'IWO':('Small Cap','Growth','Russell 2000 Growth','Small Cap Growth — Russell 2000 Growth'),
        'VBK':('Small Cap','Growth','Small Cap Growth','Small Cap Growth — Small Cap Growth'),
        'IWN':('Small Cap','Value','Russell 2000 Value','Small Cap Value — Russell 2000 Value'),
        'VBR':('Small Cap','Value','Small Cap Value','Small Cap Value — Small Cap Value'),
        'AVUV':('Small Cap','Value','Small Cap Value (Avantis)','Small Cap Value — Avantis US'),
        'CALF':('Small Cap','Value','S&P 600 Cash Flow','Small Cap Value — Free Cash Flow'),
        # International
        'EFA':('International','Blend','MSCI EAFE','International Blend — MSCI EAFE'),
        'VEA':('International','Blend','FTSE Developed','International Blend — FTSE Developed'),
        'IEFA':('International','Blend','MSCI EAFE','International Blend — MSCI EAFE'),
        'EEM':('International','Blend','MSCI Emerging Markets','International Blend — Emerging Markets'),
        'VWO':('International','Blend','FTSE Emerging Markets','International Blend — Emerging Markets'),
        'IEMG':('International','Blend','MSCI Emerging Markets','International Blend — Emerging Markets'),
        'VXUS':('International','Blend','Total International','International Blend — Total International'),
        'IXUS':('International','Blend','Total International','International Blend — Total International'),
        'ACWI':('International','Blend','MSCI ACWI','International Blend — MSCI ACWI'),
        'VT':('International','Blend','Total World','International Blend — Total World'),
        'AVDV':('International','Value','Developed Small Cap Value','International Value — Developed Small Value'),
        'AVEM':('International','Blend','Emerging Markets (Avantis)','International Blend — Avantis EM'),
        # Dividend
        'VYM':('Large Cap','Blend','High Dividend Yield','Large Cap Blend — High Dividend Yield'),
        'SCHD':('Large Cap','Value','Dividend Quality','Large Cap Value — Dividend Quality'),
        'HDV':('Large Cap','Value','High Dividend','Large Cap Value — High Dividend'),
        'DVY':('Large Cap','Value','High Dividend 100','Large Cap Value — High Dividend 100'),
        'DGRO':('Large Cap','Growth','Dividend Growth','Large Cap Growth — Dividend Growth'),
        'SDY':('Large Cap','Value','Dividend Aristocrats','Large Cap Value — Dividend Aristocrats'),
        'NOBL':('Large Cap','Value','Dividend Aristocrats','Large Cap Value — Dividend Aristocrats'),
        'VYMI':('International','Blend','International High Dividend','International Blend — High Dividend Yield'),
        'IQDF':('International','Value','International Dividend','International Value — Dividend'),
        'SDOG':('Large Cap','Value','Sector Dividend Dogs','Large Cap Value — Sector Dogs'),
        'COWZ':('Large Cap','Value','Free Cash Flow','Large Cap Value — Free Cash Flow'),
        # Factor / Smart Beta
        'QUAL':('Large Cap','Blend','Quality Factor','Large Cap Blend — Quality Factor'),
        'MTUM':('Large Cap','Growth','Momentum Factor','Large Cap Growth — Momentum Factor'),
        'VLUE':('Large Cap','Value','Value Factor','Large Cap Value — Value Factor'),
        'SIZE':('Large Cap','Blend','Size Factor','Large Cap Blend — Size Factor'),
        'USMV':('Large Cap','Blend','Min Volatility','Large Cap Blend — Min Volatility'),
        'SPLV':('Large Cap','Blend','Low Volatility','Large Cap Blend — Low Volatility'),
        'EFAV':('International','Blend','Min Volatility International','International Blend — Min Volatility'),
        'SPMO':('Large Cap','Growth','Momentum','Large Cap Growth — Momentum'),
        'DFAC':('Large Cap','Value','Dimensional Core','Large Cap Value — Dimensional Core'),
        # Fixed Income / Bond
        'AGG':('Other','Other','US Aggregate Bond','Fixed Income — US Aggregate'),
        'BND':('Other','Other','US Total Bond','Fixed Income — US Total Bond'),
        'TLT':('Other','Other','20+ Yr Treasury','Fixed Income — Long Treasury'),
        'IEF':('Other','Other','7-10 Yr Treasury','Fixed Income — Intermediate Treasury'),
        'SHY':('Other','Other','1-3 Yr Treasury','Fixed Income — Short Treasury'),
        'LQD':('Other','Other','IG Corporate Bond','Fixed Income — IG Corporate'),
        'HYG':('Other','Other','High Yield Corporate','Fixed Income — High Yield'),
        'JNK':('Other','Other','High Yield Corporate','Fixed Income — High Yield'),
        'MUB':('Other','Other','Municipal Bond','Fixed Income — Municipal'),
        'VCIT':('Other','Other','Intermediate Corporate','Fixed Income — Corp Intermediate'),
        'VCSH':('Other','Other','Short Corp Bond','Fixed Income — Corp Short'),
        'BIL':('Other','Other','1-3 Month T-Bill','Fixed Income — T-Bill'),
        'SGOV':('Other','Other','0-3 Month T-Bill','Fixed Income — T-Bill'),
        'MINT':('Other','Other','Short Duration Bond','Fixed Income — Ultra Short'),
        'FLOT':('Other','Other','Floating Rate Bond','Fixed Income — Floating Rate'),
        'JAAA':('Other','Other','CLO AAA-Rated','Fixed Income — CLO AAA'),
        'TLTW':('Other','Other','20+ Yr Treasury + Covered Call','Fixed Income — Treasury CovCall'),
        # Commodity / Crypto / Alternatives
        'GLD':('Other','Other','Gold','Commodity — Gold'),
        'IAU':('Other','Other','Gold','Commodity — Gold'),
        'SLV':('Other','Other','Silver','Commodity — Silver'),
        'USO':('Other','Other','Oil','Commodity — Oil'),
        'DJP':('Other','Other','Commodity Index','Commodity — Broad'),
        'PDBC':('Other','Other','Commodity Index','Commodity — Broad'),
        'BITO':('Other','Other','Bitcoin Futures','Crypto — Bitcoin Futures'),
        'IBIT':('Other','Other','Bitcoin Spot','Crypto — Bitcoin Spot'),
        'FBTC':('Other','Other','Bitcoin Spot','Crypto — Bitcoin Spot'),
        # Sector ETFs
        'XLK':('Other','Growth','Technology Sector','Sector — Technology'),
        'XLC':('Other','Blend','Communication Services Sector','Sector — Comm Services'),
        'XLY':('Other','Growth','Consumer Discretionary Sector','Sector — Cons Discretionary'),
        'XLP':('Other','Value','Consumer Staples Sector','Sector — Cons Staples'),
        'XLF':('Other','Blend','Financials Sector','Sector — Financials'),
        'XLV':('Other','Blend','Health Care Sector','Sector — Health Care'),
        'XLI':('Other','Blend','Industrials Sector','Sector — Industrials'),
        'XLE':('Other','Value','Energy Sector','Sector — Energy'),
        'XLB':('Other','Blend','Materials Sector','Sector — Materials'),
        'XLRE':('Other','Blend','Real Estate Sector','Sector — Real Estate'),
        'XLU':('Other','Value','Utilities Sector','Sector — Utilities'),
        # Thematic / ARK
        'ARKK':('Other','Growth','Innovation / Disruptive Tech','Thematic — ARK Innovation'),
        'ARKG':('Other','Growth','Genomic Revolution','Thematic — ARK Genomic'),
        'ARKF':('Other','Growth','FinTech Innovation','Thematic — ARK FinTech'),
        'ARKW':('Other','Growth','Next Gen Internet','Thematic — ARK Next Gen Internet'),
        'ARKX':('Other','Growth','Space Exploration','Thematic — ARK Space'),
        'BOTZ':('Other','Growth','Robotics & AI','Thematic — Robotics & AI'),
        'AIQ':('Other','Growth','AI & Big Data','Thematic — AI & Big Data'),
        'CIBR':('Other','Growth','Cybersecurity','Thematic — Cybersecurity'),
        'BUG':('Other','Growth','Cybersecurity','Thematic — Cybersecurity'),
        'PAVE':('Other','Growth','Infrastructure','Thematic — Infrastructure'),
        'AIRR':('Other','Growth','Infrastructure','Thematic — Infrastructure'),
        'IGV':('Other','Growth','Software','Thematic — Software'),
        'SOXX':('Other','Growth','Semiconductors','Thematic — Semiconductors'),
        'SMH':('Other','Growth','Semiconductors','Thematic — Semiconductors'),
        # High Yield Income (option-writing / covered call)
        'JEPI':('Large Cap','Blend','S&P 500 + Covered Call','Income — Equity CovCall'),
        'JEPQ':('Large Cap','Growth','Nasdaq + Covered Call','Income — Nasdaq CovCall'),
        'XYLD':('Large Cap','Blend','S&P 500 Covered Call','Income — S&P 500 CovCall'),
        'QYLD':('Large Cap','Growth','Nasdaq Covered Call','Income — Nasdaq CovCall'),
        'RYLD':('Small Cap','Blend','Russell 2000 Covered Call','Income — Russell 2000 CovCall'),
        'GPIX':('Large Cap','Blend','S&P 500 + Premium Income','Income — Equity CovCall'),
        'NVDY':('Other','Growth','NVDA Option Income','Income — Single Stock CovCall'),
        'TSLY':('Other','Growth','TSLA Option Income','Income — Single Stock CovCall'),
        'MSFY':('Other','Growth','MSFT Option Income','Income — Single Stock CovCall'),
        'AMZY':('Other','Growth','AMZN Option Income','Income — Single Stock CovCall'),
        'APLY':('Other','Growth','AAPL Option Income','Income — Single Stock CovCall'),
        'GOOY':('Other','Growth','GOOG Option Income','Income — Single Stock CovCall'),
        'CONY':('Other','Growth','COIN Option Income','Income — Single Stock CovCall'),
        'PLTY':('Other','Growth','PLTR Option Income','Income — Single Stock CovCall'),
        'AMDY':('Other','Growth','AMD Option Income','Income — Single Stock CovCall'),
        'FEBY':('Other','Growth','META Option Income','Income — Single Stock CovCall'),
        'CHPY':('Other','Growth','SHOP Option Income','Income — Single Stock CovCall'),
        'MSFO':('Other','Growth','MSFT Option Income (alt)','Income — Single Stock CovCall'),
        'AIPO':('Other','Growth','IPO Option Income','Income — Single Stock CovCall'),
        # Real Asset / REIT
        'VNQ':('Other','Value','US REITs','Real Assets — US REITs'),
        'VNQI':('International','Value','International REITs','Real Assets — Intl REITs'),
        'REM':('Other','Value','Mortgage REITs','Real Assets — Mortgage REITs'),
        'SCHH':('Other','Value','US REITs','Real Assets — US REITs'),
        # Multi-asset / Allocation
        'AOM':('Other','Blend','Moderate Allocation','Multi-Asset — Moderate'),
        'AOA':('Other','Blend','Aggressive Allocation','Multi-Asset — Aggressive'),
        'AOK':('Other','Blend','Conservative Allocation','Multi-Asset — Conservative'),
        'GAL':('Other','Blend','Global Allocation','Multi-Asset — Global'),
        'VBAL':('Other','Blend','Balanced Allocation','Multi-Asset — Balanced'),
    }

    if t in exact:
        size, style, anchor, label = exact[t]
        return {'size_bucket': size, 'style_bucket': style, 'anchor_tag': anchor, 'display_label': label}

    # ── 2. NAME-BASED PATTERN RULES ────────────────────────────────────────
    # Applied in priority order — first match wins.
    n_lower = n
    t_lower = t.lower()

    # Fixed income / bond patterns
    bond_kw = ['bond','treasury','fixed income','aggregate','corporate','municipal',
               'credit','duration','maturity','yield curve','clo','mbs','tips',
               'inflation-protected','short term','ultra short']
    if any(k in n_lower for k in bond_kw) or t in {'BIL','SGOV','MINT','TBIL','CSHI','HYBI','BNDI','TLTI'}:
        anchor = next((k.title() for k in bond_kw if k in n_lower), 'Bond')
        return {'size_bucket':'Other','style_bucket':'Other','anchor_tag':anchor,'display_label':f'Fixed Income — {anchor}'}

    # Sector / Industry patterns
    sector_map = [
        (['technology','software','semiconductor','cyber','cloud','internet','ai ','artificial intel'],   'Technology'),
        (['healthcare','health care','biotech','pharma','medical','genomic','life science'],              'Health Care'),
        (['financial','bank','insurance','fintech'],                                                      'Financials'),
        (['energy','oil','gas','clean energy','solar','wind','utilities'],                               'Energy/Utilities'),
        (['real estate','reit','property'],                                                               'Real Estate'),
        (['consumer','retail','discretionary','staple'],                                                  'Consumer'),
        (['industrial','infrastructure','aerospace','defense'],                                           'Industrials'),
        (['material','gold','silver','commodity','metal','mining'],                                       'Materials/Commodity'),
        (['communication','media','telecom'],                                                             'Comm Services'),
    ]
    for keys, sector_label in sector_map:
        if any(k in n_lower for k in keys):
            # Determine style from name if possible
            style = 'Growth' if 'growth' in n_lower else 'Value' if 'value' in n_lower else 'Other'
            return {'size_bucket':'Other','style_bucket':style,'anchor_tag':sector_label,'display_label':f'Sector — {sector_label}'}

    # International patterns
    intl_kw = ['international','global','world','europe','asia','pacific','emerging','eafe',
               'acwi','vxus','ftse developed','msci em','ex-us','ex us','foreign']
    if any(k in n_lower for k in intl_kw):
        style = 'Growth' if 'growth' in n_lower else 'Value' if 'value' in n_lower else 'Blend'
        anchor = next((k.title() for k in intl_kw if k in n_lower), 'International')
        return {'size_bucket':'International','style_bucket':style,'anchor_tag':anchor,'display_label':f'International {style} — {anchor}'}

    # Thematic patterns
    thematic_kw = ['innovation','disrupt','thematic','megatrend','robotics','autonomous',
                   'clean','esg','carbon','space','genomic','blockchain','metaverse',
                   'video game','esport','cannabis','water','timber']
    if any(k in n_lower for k in thematic_kw):
        theme = next((k.title() for k in thematic_kw if k in n_lower), 'Thematic')
        return {'size_bucket':'Other','style_bucket':'Growth','anchor_tag':theme,'display_label':f'Thematic — {theme}'}

    # Covered call / option income patterns
    if any(k in n_lower for k in ['covered call','option income','premium income','buy-write','buywrite','option strategy']):
        anchor = 'Covered Call Income'
        return {'size_bucket':'Other','style_bucket':'Other','anchor_tag':anchor,'display_label':f'Income — {anchor}'}

    # Size + style from name
    size = 'Large Cap'
    if any(k in n_lower for k in ['small cap','small-cap','smallcap','russell 2000','s&p 600']):
        size = 'Small Cap'
    elif any(k in n_lower for k in ['mid cap','mid-cap','midcap','s&p 400','midcap 400']):
        size = 'Mid Cap'

    style = 'Blend'
    if 'growth' in n_lower:
        style = 'Growth'
    elif 'value' in n_lower:
        style = 'Value'

    # Anchor tag from most recognizable phrase in name
    anchor_candidates = [
        'S&P 500','Russell 1000','Russell 2000','Russell 3000','Nasdaq-100',
        'Total Market','Dividend','Momentum','Quality','Equal Weight',
        'Low Volatility','Min Volatility','Factor','MSCI USA',
    ]
    anchor = next((a for a in anchor_candidates if a.lower() in n_lower), t)
    label = f'{size} {style} — {anchor}'
    return {'size_bucket': size, 'style_bucket': style, 'anchor_tag': anchor, 'display_label': label}


@app.route('/api/scanner/classify')
def scanner_classify():
    """Classify one or more ETFs into the standardized taxonomy.
    Accepts GET ?ticker=SPY or POST {'tickers': ['SPY','VOO','QQQ']}.
    Returns taxonomy fields for each ticker."""
    from flask import request as freq

    if freq.method == 'POST' or freq.content_type == 'application/json':
        body = freq.get_json(silent=True) or {}
        tickers = body.get('tickers', [])
    else:
        ticker = freq.args.get('ticker', '').upper().strip()
        tickers = [ticker] if ticker else []

    if not tickers:
        return jsonify({'error': 'ticker or tickers param required'}), 400

    # Pull display names from scanner cache where available — avoids a
    # separate yfinance call just to get the ETF's full name
    cache = _scanner_cache.get('data', {})
    results = {}
    for t in tickers:
        t = t.upper().strip()
        name = cache.get(t, {}).get('name', '') or ''
        results[t] = classify_etf(t, name)
    return jsonify(results)


@app.route('/api/scanner/classify', methods=['POST'])
def scanner_classify_post():
    return scanner_classify()


# ── ETF Lab ──────────────────────────────────────────────────────────────────

# ── Sector benchmark & rotation helpers for ETF Lab ─────────────────────────

VTI_SECTOR_CACHE_FILE = os.path.join(BASE_DIR, 'vti_sector_cache.json')
VTI_SECTOR_CACHE_TTL_HOURS = 24

SECTOR_ETF_MAP = {
    'XLK': 'Technology', 'XLF': 'Financial Services', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Cyclical',
    'XLP': 'Consumer Defensive', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Basic Materials', 'XLC': 'Communication Services',
}

def get_rebalance_frequency(name, category, description):
    """Infer typical reconstitution/rebalance schedule from the fund's
    underlying index family. Yahoo doesn't expose this as a structured
    field, so this matches against well-documented, publicly known index
    provider conventions. Returns (frequency_label, detail_note)."""
    text = f'{name} {category} {description}'.lower()

    # Order matters — check more specific index families before generic ones
    if 'russell' in text:
        return ('Annual', 'Russell US Indexes reconstitute annually, effective the last Friday of June')
    if 'nasdaq-100' in text or 'nasdaq 100' in text:
        return ('Annual + Quarterly', 'Nasdaq-100 reconstitutes annually each December, with quarterly share rebalancing in Mar/Jun/Sep')
    if 's&p' in text or 'sp 500' in text or 's&p 500' in text or 'sp500' in text:
        return ('Quarterly', 'S&P Dow Jones Indices rebalance quarterly, effective after the 3rd Friday of Mar/Jun/Sep/Dec')
    if 'msci' in text:
        return ('Quarterly + Semi-Annual', 'MSCI applies quarterly index reviews (Feb/Aug) and semi-annual full reviews (May/Nov)')
    if 'crsp' in text:
        return ('Quarterly', 'CRSP US Indexes (common for Vanguard funds) rebalance quarterly, in Mar/Jun/Sep/Dec')
    if 'dow jones industrial' in text:
        return ('As Needed', 'The Dow Jones Industrial Average is changed infrequently, only as needed — no fixed schedule')
    if 'dow jones' in text:
        return ('Semi-Annual', 'Dow Jones dividend/equity indexes typically reconstitute in March and September')
    if 'dividend' in text and ('growth' in text or 'aristocrat' in text):
        return ('Annual', 'Dividend growth/quality indexes typically reconstitute annually, with quarterly weight rebalancing')
    if 'covered call' in text or 'buywrite' in text or 'option income' in text:
        return ('Monthly', 'Options-based income funds roll their option positions monthly')
    if any(k in text for k in ['bond', 'treasury', 'aggregate', 'corporate debt', 'fixed income']):
        return ('Monthly', 'Bond index funds rebalance monthly as holdings mature and new issuance is added')
    if 'equal weight' in text:
        return ('Quarterly', 'Equal-weight indexes rebalance quarterly to reset all positions back to equal weight')
    return (None, None)


def get_vti_sector_benchmark():
    """Fetch VTI's real sector weightings, disk-cached 24h since VTI's sector
    mix barely moves day to day. Returns {sector_name: weight_pct}."""
    if os.path.exists(VTI_SECTOR_CACHE_FILE):
        try:
            cached = json.load(open(VTI_SECTOR_CACHE_FILE))
            age_hrs = (time.time() - cached.get('_ts', 0)) / 3600
            if age_hrs < VTI_SECTOR_CACHE_TTL_HOURS:
                return cached.get('data', {})
        except Exception:
            pass

    result = {}
    try:
        vti = yf.Ticker('VTI')
        if hasattr(vti, 'funds_data') and vti.funds_data:
            sw = vti.funds_data.sector_weightings
            if isinstance(sw, dict):
                for sec, wt in sw.items():
                    if wt is not None:
                        result[sec] = round(float(wt) * 100, 2)
        print(f'  [ETFLab] VTI sector benchmark fetched: {result}')
    except Exception as e:
        print(f'  [ETFLab] VTI sector benchmark fetch failed: {e}')

    if result:
        try:
            json.dump({'_ts': time.time(), 'data': result}, open(VTI_SECTOR_CACHE_FILE, 'w'))
        except Exception:
            pass
    return result


def get_sector_rotation_data(sector_weights):
    """For each sector the ETF holds, compare that sector's proxy ETF
    performance against VTI over 1mo/3mo/6mo/1yr — surfaces relative
    strength and rotation trends (which sectors are leading/lagging the
    broad market right now)."""
    if not sector_weights:
        return []

    # Map each held sector to its proxy ETF ticker
    def match_sector_etf(sector_name):
        sec_lower = sector_name.lower().replace(' ', '').replace('-', '')
        for etf, name in SECTOR_ETF_MAP.items():
            if name.lower().replace(' ', '') in sec_lower or sec_lower in name.lower().replace(' ', ''):
                return etf
        # Fallback keyword matching for common naming variations
        keyword_map = {
            'tech': 'XLK', 'financ': 'XLF', 'health': 'XLV', 'energy': 'XLE',
            'industr': 'XLI', 'cyclical': 'XLY', 'discretionary': 'XLY',
            'defensive': 'XLP', 'staple': 'XLP', 'util': 'XLU',
            'realestate': 'XLRE', 'material': 'XLB', 'communication': 'XLC',
        }
        for kw, etf in keyword_map.items():
            if kw in sec_lower:
                return etf
        return None

    relevant_etfs = set()
    sector_to_etf = {}
    for s in sector_weights:
        etf = match_sector_etf(s['sector'])
        if etf:
            sector_to_etf[s['sector']] = etf
            relevant_etfs.add(etf)
    relevant_etfs.add('VTI')

    if len(relevant_etfs) < 2:
        return []

    try:
        time.sleep(0.4)
        dl = yf.download(' '.join(relevant_etfs), period='13mo', interval='1d',
                          auto_adjust=True, progress=False, group_by='ticker')
    except Exception as e:
        print(f'  [ETFLab] Sector rotation download failed: {e}')
        return []

    def get_close_series(sym):
        try:
            if len(relevant_etfs) == 1:
                return dl['Close'].dropna()
            return dl[sym]['Close'].dropna() if sym in dl.columns.get_level_values(0) else pd.Series()
        except Exception:
            return pd.Series()

    def period_return(series, days):
        if len(series) <= days: return None
        return round((series.iloc[-1] / series.iloc[-days] - 1) * 100, 2)

    vti_series = get_close_series('VTI')
    if vti_series.empty:
        return []

    periods = {'1mo': 21, '3mo': 63, '6mo': 126, '1yr': 252}
    vti_returns = {label: period_return(vti_series, days) for label, days in periods.items()}

    rotation = []
    for sector_name, etf in sector_to_etf.items():
        sec_series = get_close_series(etf)
        if sec_series.empty: continue
        sec_returns = {label: period_return(sec_series, days) for label, days in periods.items()}
        relative = {}
        for label in periods:
            if sec_returns[label] is not None and vti_returns[label] is not None:
                relative[label] = round(sec_returns[label] - vti_returns[label], 2)
            else:
                relative[label] = None
        rotation.append({
            'sector': sector_name, 'proxy_etf': etf,
            'sector_returns': sec_returns, 'vti_returns': vti_returns,
            'relative_strength': relative,
        })
    rotation.sort(key=lambda r: r['relative_strength'].get('3mo') or -999, reverse=True)
    return rotation


@app.route('/api/etf/lab/<ticker>')
def etf_lab(ticker):
    """Comprehensive ETF research endpoint for the ETF Lab page.
    Returns all data needed for the 10-section analysis in one call."""
    ticker = ticker.upper().strip()
    try:
        # ── Retry wrapper for Yahoo rate limiting ─────────────────────────
        def yf_info_with_retry(t, max_retries=4):
            for attempt in range(max_retries):
                try:
                    info = t.info
                    if info and len(info) > 5:
                        return info
                except Exception as e:
                    if 'rate' in str(e).lower() or 'too many' in str(e).lower():
                        wait = min(2 ** (attempt + 1), 15)  # 2s, 4s, 8s, 15s
                        print(f'  [ETFLab] Rate limited on info, waiting {wait}s (attempt {attempt+1})')
                        time.sleep(wait)
                    else:
                        raise
            return {}

        t = yf.Ticker(ticker)
        time.sleep(0.3)  # small initial pause to reduce burst pressure
        info = yf_info_with_retry(t)

        # ── 1. Summary ────────────────────────────────────────────────────
        name = (info.get('longName') or info.get('shortName') or ticker).strip()
        description_raw = (info.get('longBusinessSummary') or '').strip()
        category = info.get('category', '') or info.get('fundFamily', '') or ''

        # Pull fund overview data (manager/family, category, legal structure) —
        # this is more reliably populated than longBusinessSummary and gives
        # the "who runs it, what kind of fund is it" context directly.
        fund_overview = {}
        try:
            if hasattr(t, 'funds_data') and t.funds_data:
                fo = t.funds_data.fund_overview
                if fo: fund_overview = dict(fo)
                print(f'  [ETFLab] {ticker} fund_overview: {fund_overview}')
        except Exception as fe:
            print(f'  [ETFLab] fund_overview fetch failed for {ticker}: {fe}')

        # Build a more informative summary combining fund_overview structured
        # data (issuer, category, legal structure) with a trimmed version of
        # Yahoo's objective/strategy description
        def build_summary(raw_desc, ticker_str, name_str, cat_str, aum_val, er_val, beta_val, sector_list, overview):
            """Create a descriptive summary: who manages the fund and what
            kind of structure it is (from fund_overview), followed by its
            stated objective/strategy (trimmed from Yahoo's description)."""
            top_sectors = ', '.join([s['sector'] for s in (sector_list or [])[:3]]) or 'various sectors'
            aum_str = fmtaum_str(aum_val) if aum_val else 'an undisclosed amount of'
            er_str = f"{er_val:.2f}%" if er_val else 'an undisclosed'

            family = overview.get('family')
            ov_category = overview.get('categoryName')
            legal_type = overview.get('legalType')

            # Opening line: who manages it, what kind of fund it is
            intro_parts = []
            if family:
                intro_parts.append(f"{name_str} ({ticker_str}) is managed by {family}")
            else:
                intro_parts.append(f"{name_str} ({ticker_str}) is an exchange-traded fund")
            if ov_category:
                intro_parts.append(f"categorized as {ov_category}")
            if legal_type and legal_type.lower() not in (ov_category or '').lower():
                intro_parts.append(f"structured as {'an' if legal_type[0].upper() in 'AEIOU' else 'a'} {legal_type}")
            intro = ', '.join(intro_parts) + f'. It holds {aum_str} in assets with a {er_str} expense ratio.'

            # Objective/strategy line from Yahoo, trimmed to the most
            # informative 1-2 sentences (Yahoo descriptions repeat a lot of
            # legal boilerplate after the first couple of sentences)
            objective = ''
            if raw_desc:
                sentences = raw_desc.split('. ')
                objective = '. '.join(sentences[:2]).rstrip('.') + '.'
            elif top_sectors != 'various sectors':
                objective = f'Its largest sector exposures include {top_sectors}.'

            return (intro + ' ' + objective).strip()

        def fmtaum_str(v):
            if v >= 1e12: return f'${v/1e12:.1f}T'
            if v >= 1e9:  return f'${v/1e9:.1f}B'
            if v >= 1e6:  return f'${v/1e6:.1f}M'
            return f'${v:,.0f}'

        # ── 2. Core metrics ───────────────────────────────────────────────
        # Fetch once, reuse for AUM/beta/52-week/dividend-yield fallbacks
        # below — confirmed working via direct testing (200 response with
        # real data on the free tier, unlike ratios-ttm and etf/info which
        # both returned 402). Does NOT cover expense ratio (confirmed
        # absent from the full response) — that gap remains unsolved.
        fmp_profile = get_fmp_profile(ticker)

        aum = info.get('totalAssets')
        aum_source = 'yahoo' if aum is not None else None
        if aum is None and fmp_profile.get('aum') is not None:
            aum = fmp_profile['aum']
            aum_source = 'fmp'
        if aum is None:
            # Fallback: N-PORT filings report total net assets directly —
            # independent of Yahoo, so this covers exactly the case where
            # .info gets rate-limited (which previously left AUM with no
            # fallback at all, unlike P/B/PEG/P/S/Beta)
            edgar_aum = get_edgar_net_assets_cached(ticker)
            if edgar_aum is not None:
                aum = edgar_aum
                aum_source = 'sec_edgar'
        er_raw = (info.get('annualReportExpenseRatio') or info.get('expenseRatio')
                  or info.get('netExpenseRatio') or info.get('grossExpRatio'))
        if not er_raw:
            try:
                if hasattr(t, 'funds_data') and t.funds_data:
                    fo = t.funds_data.fund_overview
                    if fo is not None and 'annualReportExpenseRatio' in fo:
                        er_raw = fo['annualReportExpenseRatio']
            except Exception:
                pass
        er = normalize_expense_ratio(er_raw)
        turnover = info.get('annualHoldingsTurnover')
        if turnover and float(turnover) < 1:
            turnover = round(float(turnover) * 100, 1)
        elif turnover:
            turnover = round(float(turnover), 1)

        dividend_yield = normalize_dividend_yield(info.get('yield') or info.get('dividendYield') or info.get('trailingAnnualDividendYield'))
        dividend_yield_source = 'yahoo' if dividend_yield is not None else None
        if dividend_yield is None and fmp_profile.get('dividend_yield') is not None:
            dividend_yield = fmp_profile['dividend_yield']
            dividend_yield_source = 'fmp'

        # Try empirical detection from actual N-PORT filing history first —
        # this is real evidence of the fund's behavior, not a guess based on
        # its stated index family. Falls back to the heuristic when there
        # isn't enough filing history (new funds, non-SEC-registered, etc.)
        rebalance_source = None
        rebalance_evidence = None
        empirical = detect_empirical_rebalance(ticker)
        if empirical:
            rebalance_freq = empirical['frequency']
            rebalance_note = empirical['note']
            rebalance_source = 'sec_edgar_empirical'
            rebalance_evidence = empirical.get('evidence')
        else:
            rebalance_freq, rebalance_note = get_rebalance_frequency(name, category, description_raw)
            if rebalance_freq:
                rebalance_source = 'index_family_heuristic'

        current_price  = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        prev_close     = info.get('regularMarketPreviousClose') or info.get('previousClose')
        day_change     = None
        day_change_pct = None
        if current_price and prev_close and float(prev_close) != 0:
            day_change     = round(float(current_price) - float(prev_close), 2)
            day_change_pct = round((float(current_price) / float(prev_close) - 1) * 100, 2)

        # ── 3. Sector breakdown vs. VTI ─────────────────────────────────────
        sector_weights = []
        try:
            if hasattr(t, 'funds_data') and t.funds_data:
                sw_data = t.funds_data.sector_weightings
                print(f'  [ETFLab] {ticker} sector_weightings raw: {sw_data}')
                if isinstance(sw_data, dict):
                    for sec, wt in sw_data.items():
                        if wt is not None:
                            sector_weights.append({'sector': sec, 'weight': round(float(wt) * 100, 2)})
        except Exception as se:
            print(f'  [ETFLab] Sector fetch failed for {ticker}: {se}')
        sector_weights.sort(key=lambda x: x['weight'], reverse=True)

        # Real VTI sector weights (disk-cached 24h — VTI's sector mix barely
        # moves day to day, so no need to refetch on every Lab request)
        benchmark_sectors = get_vti_sector_benchmark()

        # ── Sector performance & rotation vs. VTI ───────────────────────────
        # For each sector this ETF holds, compare that sector's ETF proxy
        # performance against VTI over multiple lookback windows. This
        # surfaces relative strength / rotation — which sectors are leading
        # or lagging the broad market, useful context alongside the static
        # weight comparison above.
        sector_rotation = get_sector_rotation_data(sector_weights)

        # ── 4. Valuation ──────────────────────────────────────────────────
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        price_to_book = info.get('priceToBook')
        # Broad market PE benchmark: ~21x trailing (mid-2025 approximation)
        benchmark_pe = 21.0
        pe_signal = None
        if pe_ratio:
            pe_ratio = round(float(pe_ratio), 1)
            ratio = pe_ratio / benchmark_pe
            if ratio > 1.15:   pe_signal = 'above'
            elif ratio < 0.85: pe_signal = 'below'
            else:              pe_signal = 'average'

        # ── 5. Analyst consensus / upside ────────────────────────────────
        target_mean  = info.get('targetMeanPrice')
        target_high  = info.get('targetHighPrice')
        target_low   = info.get('targetLowPrice')
        upside_mean = None
        if target_mean and current_price:
            upside_mean = round((float(target_mean) / float(current_price) - 1) * 100, 1)
        eps_current     = info.get('trailingEps')
        eps_forward     = info.get('forwardEps')
        revenue_growth  = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        # Additional forward metrics
        peg_ratio       = info.get('trailingPegRatio') or info.get('pegRatio')
        forward_pe      = info.get('forwardPE')
        price_to_sales  = info.get('priceToSalesTrailing12Months')

        # Fallback to Financial Modeling Prep for whichever of P/B, PEG, P/S
        # yfinance didn't provide. Note: FMP's ratios are calculated from
        # company financial statements, which ETFs don't file — this is
        # more likely to help for single-stock or thinly-covered tickers
        # than for the ETF wrapper itself, confirmed by the [FMP] log lines.
        if price_to_book is None or peg_ratio is None or price_to_sales is None:
            fmp_ratios = get_fmp_ratios(ticker)
            if price_to_book is None:
                price_to_book = fmp_ratios.get('price_to_book')
            if peg_ratio is None:
                peg_ratio = fmp_ratios.get('peg_ratio')
            if price_to_sales is None:
                price_to_sales = fmp_ratios.get('price_to_sales')

        fifty_two_high  = info.get('fiftyTwoWeekHigh') or fmp_profile.get('fifty_two_week_high')
        fifty_two_low   = info.get('fiftyTwoWeekLow') or fmp_profile.get('fifty_two_week_low')
        pct_from_52h    = None
        pct_from_52l    = None
        if current_price and fifty_two_high:
            pct_from_52h = round((float(current_price)/float(fifty_two_high) - 1)*100, 1)
        if current_price and fifty_two_low:
            pct_from_52l = round((float(current_price)/float(fifty_two_low) - 1)*100, 1)

        # ── 6. Holdings (up to 100) ─────────────────────────────────────────
        # EDGAR N-PORT filings disclose a fund's FULL holdings list (not just
        # the top ~10 that Yahoo/yfinance exposes), so try that first — it's
        # already disk-cached elsewhere in this app and refreshed monthly.
        holdings = []
        holdings_source = None
        edgar_holdings = get_edgar_holdings_cached(ticker)
        if edgar_holdings:
            sorted_edgar = sorted(edgar_holdings, key=lambda h: h.get('w', 0), reverse=True)
            for h in sorted_edgar[:100]:
                holdings.append({
                    'ticker': h.get('t', ''),
                    'name': h.get('n', h.get('t', '')),
                    'weight': round(float(h.get('w', 0)), 2),
                    'day_change_pct': None,
                    'ttm_return': None,
                })
            holdings_source = 'sec_edgar'
            print(f'  [ETFLab] {ticker}: {len(holdings)} holdings from EDGAR (of {len(edgar_holdings)} total in filing)')

        # Fallback: yfinance top_holdings (Yahoo generally caps ~10 regardless
        # of the fund's real holdings count — used only when EDGAR has
        # nothing for this ticker, e.g. very new or non-SEC-registered funds)
        if not holdings:
            time.sleep(0.4)
            try:
                fund_h = None
                try:
                    if hasattr(t, 'funds_data') and t.funds_data:
                        fund_h = t.funds_data.top_holdings
                except Exception:
                    pass

                if fund_h is not None and not fund_h.empty:
                    print(f'  [ETFLab] {ticker} top_holdings columns: {list(fund_h.columns)}, index sample: {list(fund_h.index[:3])}')
                    for sym, row in fund_h.iterrows():
                        tick = str(sym)  # ticker symbol is the DataFrame index
                        nm = None
                        for col in ['Name', 'Holdings Description', 'holdingName']:
                            if col in row.index and row[col]:
                                nm = str(row[col]); break
                        if not nm: nm = tick
                        pct = None
                        for col in ['Holding Percent', '% Net Assets', 'holdingPercent', 'Weight']:
                            if col in row.index and row[col] is not None:
                                pct = float(row[col]); break
                        if pct is None and len(row) > 0:
                            pct = float(row.iloc[0])
                        if pct is None: continue
                        w = pct * 100 if pct < 1 else pct
                        holdings.append({
                            'ticker': tick, 'name': nm, 'weight': round(w, 2),
                            'day_change_pct': None, 'ttm_return': None,
                        })
                    holdings_source = 'yfinance'
            except Exception as he:
                print(f'  [ETFLab] Holdings fetch failed for {ticker}: {he}')

        if not holdings:
            top_h = info.get('holdings') or []
            for h in top_h[:100]:
                holdings.append({
                    'ticker': h.get('symbol', ''),
                    'name': h.get('holdingName', ''),
                    'weight': round(float(h.get('holdingPercent', 0)) * 100, 2),
                    'day_change_pct': None,
                    'ttm_return': None,
                })
            holdings_source = 'yfinance_info'

        holdings_count_note = None
        if holdings_source in ('yfinance', 'yfinance_info') and 0 < len(holdings) < 100:
            holdings_count_note = f"Full N-PORT holdings data isn't available for this fund — showing the top {len(holdings)} from Yahoo Finance"

        # Enrich holdings with day change + TTM return via a single batch download.
        # EDGAR holdings can include non-equity positions (cash, treasuries)
        # with placeholder-style tickers — filter to plausible equity symbols.
        import re as re_lab
        holding_tickers = [h['ticker'] for h in holdings
                            if h.get('ticker') and re_lab.match(r'^[A-Z.]{1,6}$', h['ticker'])]
        if holding_tickers:
            try:
                time.sleep(0.5)
                h_dl = yf.download(
                    ' '.join(holding_tickers),
                    period='13mo', interval='1d',
                    auto_adjust=True, progress=False, group_by='ticker'
                )
                for h in holdings:
                    t_sym = h.get('ticker', '')
                    if not t_sym: continue
                    try:
                        # Navigate multi-ticker download structure
                        if len(holding_tickers) == 1:
                            ts = h_dl['Close'] if 'Close' in h_dl.columns else pd.Series()
                        else:
                            ts = h_dl[t_sym]['Close'] if t_sym in h_dl.columns.get_level_values(0) else pd.Series()
                        ts = ts.dropna()
                        if len(ts) >= 2:
                            h['day_change_pct'] = round((ts.iloc[-1]/ts.iloc[-2] - 1)*100, 2)
                        if len(ts) >= 252:
                            h['ttm_return'] = round((ts.iloc[-1]/ts.iloc[-252] - 1)*100, 1)
                        elif len(ts) >= 20:
                            h['ttm_return'] = round((ts.iloc[-1]/ts.iloc[0] - 1)*100, 1)
                    except Exception:
                        pass
            except Exception as e:
                print(f'  [ETFLab] Holdings enrichment failed: {e}')

        # Industry + valuation ratios per holding — disk-cached indefinitely
        # (see get_ticker_fundamentals), so this is a one-time cost per
        # unique ticker across all future Lab loads. Cap fresh lookups per
        # request, and run them CONCURRENTLY instead of one-by-one — Yahoo
        # has proven to rate-limit hard even on small bursts throughout
        # this app (see the repeated "Too Many Requests" errors in the
        # scanner logs), so this uses a modest worker count rather than
        # firing all 15 at once, but running them in parallel rather than
        # sequentially-with-pacing was one of the largest sources of
        # avoidable latency on a fund with entirely uncached holdings.
        MAX_FRESH_FUNDAMENTALS_LOOKUPS = 15
        to_fetch_fresh = []
        for h in holdings:
            t_sym = h.get('ticker', '')
            if not t_sym:
                h['industry'] = None
                continue
            cache_path = os.path.join(INDUSTRY_CACHE_DIR, f'{t_sym}.json')
            if os.path.exists(cache_path):
                try:
                    fund = get_ticker_fundamentals(t_sym)  # instant disk-cache hit
                    h['industry']       = fund.get('industry')
                    h['price_to_book']  = fund.get('price_to_book')
                    h['peg_ratio']      = fund.get('peg_ratio')
                    h['price_to_sales'] = fund.get('price_to_sales')
                    h['beta']           = fund.get('beta')
                except Exception as ie:
                    print(f'  [ETFLab] Fundamentals cache read failed for {t_sym}: {ie}')
                    h['industry'] = None
            elif len(to_fetch_fresh) < MAX_FRESH_FUNDAMENTALS_LOOKUPS:
                to_fetch_fresh.append((t_sym, h))
            else:
                h['industry'] = None  # will be picked up on a future load

        if to_fetch_fresh:
            import concurrent.futures as _cf3
            with _cf3.ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(get_ticker_fundamentals, t_sym): (t_sym, h) for t_sym, h in to_fetch_fresh}
                for future in _cf3.as_completed(futures):
                    t_sym, h = futures[future]
                    try:
                        fund = future.result()
                        h['industry']       = fund.get('industry')
                        h['price_to_book']  = fund.get('price_to_book')
                        h['peg_ratio']      = fund.get('peg_ratio')
                        h['price_to_sales'] = fund.get('price_to_sales')
                        h['beta']           = fund.get('beta')
                    except Exception as ie:
                        print(f'  [ETFLab] Fundamentals lookup failed for {t_sym}: {ie}')
                        h['industry'] = None

        # Final fallback tier: weighted-average from holdings. An ETF
        # itself doesn't have its own earnings/book value/sales the way a
        # single company does, so when both yfinance and FMP come back
        # empty at the ETF level, compute the weight-normalized average of
        # the underlying holdings' individual ratios instead — the
        # standard approach for fund-level valuation multiples.
        valuation_source = {'price_to_book': None, 'peg_ratio': None, 'price_to_sales': None}
        weighted_ratios = compute_weighted_etf_ratios(holdings)
        if price_to_book is None and weighted_ratios['price_to_book'] is not None:
            price_to_book = weighted_ratios['price_to_book']
            valuation_source['price_to_book'] = {'method': 'weighted_harmonic_avg_holdings', 'coverage_pct': weighted_ratios['price_to_book_coverage']}
        if peg_ratio is None and weighted_ratios['peg_ratio'] is not None:
            peg_ratio = weighted_ratios['peg_ratio']
            valuation_source['peg_ratio'] = {
                'method': 'weighted_harmonic_avg_holdings',
                'coverage_pct': weighted_ratios['peg_ratio_coverage'],
                'excluded_negative_growth_pct': weighted_ratios['peg_ratio_excluded_negative_pct'],
            }
        if price_to_sales is None and weighted_ratios['price_to_sales'] is not None:
            price_to_sales = weighted_ratios['price_to_sales']
            valuation_source['price_to_sales'] = {'method': 'weighted_harmonic_avg_holdings', 'coverage_pct': weighted_ratios['price_to_sales_coverage']}
        print(f'  [ETFLab] {ticker} weighted-avg ratios from holdings: {weighted_ratios}')

        # ── 7+8. Beta & price history for momentum ───────────────────────
        # Yahoo sometimes returns an exact 0.0 placeholder for beta3Year on
        # newer/smaller funds it hasn't fully computed stats for yet. Since
        # a real diversified ETF essentially never has a true beta of
        # exactly 0.000 (that would mean zero correlation with the market),
        # treat an exact zero as missing data rather than a real reading —
        # otherwise Python's `or` chain silently lets a placeholder zero
        # through as if it were a legitimate value (0.0 is falsy, so
        # `info.get('beta') or info.get('beta3Year')` would fall through to
        # a zero beta3Year even when that's just an unset placeholder).
        beta_raw = info.get('beta')
        if beta_raw is None or beta_raw == 0:
            beta_raw = info.get('beta3Year')
        beta = round(float(beta_raw), 2) if beta_raw not in (None, 0) else None

        beta_source = None
        if beta is None and fmp_profile.get('beta') is not None:
            beta = round(float(fmp_profile['beta']), 2)
            beta_source = {'method': 'fmp'}
        if beta is None and weighted_ratios.get('beta') is not None:
            beta = weighted_ratios['beta']
            beta_source = {'method': 'weighted_avg_holdings', 'coverage_pct': weighted_ratios['beta_coverage']}
        valuation_source['beta'] = beta_source

        # Use yf.download() — different Yahoo endpoint, much less rate-limited
        # than t.history(). Small pause before to reduce burst pressure.
        time.sleep(0.5)
        hist = None
        price_series = pd.Series([], dtype=float)
        try:
            raw_dl = yf.download(ticker, period='15mo', interval='1d',
                                 auto_adjust=True, progress=False, multi_level_index=False)
            if raw_dl is not None and not raw_dl.empty:
                hist = raw_dl
                price_series = raw_dl['Close'].dropna()
        except Exception as e:
            print(f'  [ETFLab] history download failed for {ticker}: {e}')

        mom_data = {}
        momentum_score = None
        if len(price_series) >= 60:
            p = price_series.copy()

            # ── Component 1: Trend Duration & Rate of Change (50%) ────────
            def pct_return(series, days):
                if len(series) <= days: return None
                return round((series.iloc[-1] / series.iloc[-days] - 1) * 100, 2)

            r3m  = pct_return(p, 63)   # ~3 months
            r6m  = pct_return(p, 126)  # ~6 months
            r12m_skip1 = None          # 12mo-1mo (skip last month)
            if len(p) >= 252:
                r12m_skip1 = round((p.iloc[-21] / p.iloc[-252] - 1) * 100, 2)

            # Normalize to 0-100 percentile within history
            def percentile_score(val, history_vals):
                if val is None or len(history_vals) == 0: return 50
                below = sum(1 for v in history_vals if v <= val)
                return round(below / len(history_vals) * 100, 1)

            # Rolling 63-day returns for percentile context
            rolls_3m = [((p.iloc[i] / p.iloc[i-63]) - 1) * 100 for i in range(63, len(p))]
            rolls_6m = [((p.iloc[i] / p.iloc[i-126]) - 1) * 100 for i in range(126, len(p))]

            s_3m  = percentile_score(r3m,  rolls_3m)  if r3m  is not None else 50
            s_6m  = percentile_score(r6m,  rolls_6m)  if r6m  is not None else 50
            s_12m = percentile_score(r12m_skip1, rolls_3m) if r12m_skip1 is not None else 50

            trend_score = (s_3m * 0.15 + s_6m * 0.15 + s_12m * 0.20) / 0.50  # normalize to 0-100

            # ── Component 2: Technical Oscillators (30%) ──────────────────
            # 200-day SMA distance
            sma200_score = 50
            if len(p) >= 200:
                sma200 = p.rolling(200).mean().iloc[-1]
                dist = (p.iloc[-1] / sma200 - 1) * 100
                # Map: -20% → 0, 0% → 50, +20% → 100
                sma200_score = min(100, max(0, 50 + dist * 2.5))

            # RSI-14
            delta = p.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi = (100 - 100 / (1 + rs)).iloc[-1]
            # RSI 30-70 is "good momentum zone" → map to 0-100
            # RSI 50 → 60 score (slight bullish lean)
            rsi_score = min(100, max(0, rsi))

            # MACD histogram slope
            ema12 = p.ewm(span=12, adjust=False).mean()
            ema26 = p.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist_macd = macd_line - signal_line
            macd_slope = hist_macd.diff(5).iloc[-1]  # 5-day slope of histogram
            # Normalize: positive slope → bullish, map relative to recent range
            macd_range = hist_macd.rolling(63).std().iloc[-1] or 1
            macd_score = min(100, max(0, 50 + (macd_slope / macd_range) * 25))

            tech_score = (sma200_score * 0.10 + rsi_score * 0.10 + macd_score * 0.10) / 0.30

            # ── Component 3: Volatility & Risk Adjustment (20%) ───────────
            # Trailing 60-day realized volatility (annualized)
            vol60 = p.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252) * 100
            # Low vol = high score; penalize above 25% annualized vol
            vol_score = min(100, max(0, 100 - (vol60 - 5) * 2.5))

            # Bollinger Band Width relative to ATR
            bb_upper = p.rolling(20).mean() + 2 * p.rolling(20).std()
            bb_lower = p.rolling(20).mean() - 2 * p.rolling(20).std()
            bb_width = ((bb_upper - bb_lower) / p).iloc[-1] * 100
            high = hist['High'] if hist is not None and 'High' in hist.columns else price_series
            low  = hist['Low']  if hist is not None and 'Low'  in hist.columns else price_series
            tr   = pd.concat([high - low,
                               (high - p.shift()).abs(),
                               (low  - p.shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr14 / p.iloc[-1] * 100
            # Tight BB relative to ATR = consolidating = good; expanding = overbought risk
            bb_atr_ratio = bb_width / (atr_pct or 1)
            bb_score = min(100, max(0, 100 - (bb_atr_ratio - 1) * 20))

            vol_score_combined = (vol_score * 0.10 + bb_score * 0.10) / 0.20

            # ── Final weighted momentum score ──────────────────────────────
            raw_momentum = trend_score * 0.50 + tech_score * 0.30 + vol_score_combined * 0.20
            momentum_score = round(raw_momentum, 1)

            if momentum_score > 75:   mom_signal = 'Strong Positive Momentum'
            elif momentum_score >= 45: mom_signal = 'Neutral / Consolidating'
            else:                      mom_signal = 'Negative Momentum'

            mom_data = {
                'score': momentum_score,
                'signal': mom_signal,
                'return_3m': r3m, 'return_6m': r6m, 'return_12m_skip1': r12m_skip1,
                'percentile_3m': round(s_3m, 1), 'percentile_6m': round(s_6m, 1),
                'sma200_distance_pct': round((p.iloc[-1] / p.rolling(200).mean().iloc[-1] - 1) * 100, 2) if len(p) >= 200 else None,
                'rsi_14': round(float(rsi), 1),
                'macd_histogram_slope': round(float(macd_slope), 4),
                'vol_60d_annualized': round(float(vol60), 1),
                'bb_width_pct': round(float(bb_width), 2),
                'trend_component': round(trend_score, 1),
                'technical_component': round(tech_score, 1),
                'volatility_component': round(vol_score_combined, 1),
            }

        # ── 8. Fed sensitivity ────────────────────────────────────────────
        # Rule-based from sector weights + known characteristics
        def fed_sensitivity(sectors, category_str, beta_val, er_val):
            tech_w   = next((s['weight'] for s in sectors if 'tech'   in s['sector'].lower()), 0)
            fin_w    = next((s['weight'] for s in sectors if 'financ' in s['sector'].lower()), 0)
            re_w     = next((s['weight'] for s in sectors if 'real'   in s['sector'].lower()), 0)
            util_w   = next((s['weight'] for s in sectors if 'util'   in s['sector'].lower()), 0)
            bond_like = 'bond' in category_str.lower() or 'income' in category_str.lower() or (er_val and er_val < 0.1)

            hike_impact, cut_impact = [], []

            if bond_like:
                hike_impact.append('Negative — rising rates lower bond prices')
                cut_impact.append('Positive — falling rates raise bond prices')
            if re_w > 10:
                hike_impact.append(f'Negative on REIT component ({re_w:.0f}%) — higher borrowing costs compress valuations')
                cut_impact.append(f'Positive on REIT component — lower rates improve property valuations')
            if util_w > 10:
                hike_impact.append(f'Negative on Utilities ({util_w:.0f}%) — rate-sensitive, bond-proxy sector')
                cut_impact.append(f'Positive on Utilities — typically rallies as yields fall')
            if fin_w > 10:
                hike_impact.append(f'Mixed on Financials ({fin_w:.0f}%) — higher rates boost net interest margins but slow loan growth')
                cut_impact.append(f'Mixed on Financials — tighter margins but stronger loan demand')
            if tech_w > 20:
                hike_impact.append(f'Negative on Growth/Tech ({tech_w:.0f}%) — higher discount rates compress long-duration valuations')
                cut_impact.append(f'Positive on Growth/Tech — lower discount rates expand valuations')
            if not hike_impact:
                hike_impact.append('Moderate sensitivity — mixed sector exposure limits direct rate impact')
                cut_impact.append('Moderate benefit — rate cuts broadly supportive of equities')

            return {'rate_hike': hike_impact, 'rate_cut': cut_impact}

        fed = fed_sensitivity(sector_weights, category, beta, er)

        # ── Market Cycle Indicator ───────────────────────────────────────
        market_cycle = compute_market_cycle()
        cycle_fit = compute_cycle_fit(market_cycle, fund_overview.get('categoryName'), None, None, sector_weights) if market_cycle else None

        # ── 9. Portfolio fit ──────────────────────────────────────────────
        portfolio_fit = {}
        try:
            portfolio = load_portfolio()
            if portfolio and portfolio.get('holdings'):
                port_holdings = portfolio['holdings']
                total_cv = sum(h.get('cv', 0) or 0 for h in port_holdings)

                # Sector concentration in existing portfolio
                port_sectors = {}
                for h in port_holdings:
                    sec = h.get('sector', 'Unknown')
                    cv = h.get('cv', 0) or 0
                    port_sectors[sec] = port_sectors.get(sec, 0) + cv
                port_sector_pcts = {k: round(v / total_cv * 100, 1) for k, v in port_sectors.items() if total_cv > 0}

                # Overlap with ETF holdings
                port_tickers = {h['t'].upper() for h in port_holdings}
                lab_tickers  = {h['ticker'].upper() for h in holdings if h.get('ticker')}
                overlap = list(port_tickers & lab_tickers)

                # Sector overlap between ETF and portfolio
                sector_overlap = []
                for sw in sector_weights:
                    sec_name = sw['sector']
                    port_wt  = port_sector_pcts.get(sec_name, 0)
                    etf_wt   = sw['weight']
                    if etf_wt > 5:
                        delta = etf_wt - port_wt
                        sector_overlap.append({
                            'sector': sec_name,
                            'etf_weight': etf_wt,
                            'portfolio_weight': port_wt,
                            'delta': round(delta, 1),
                        })

                # Beta of existing portfolio (weighted average)
                port_beta = None
                beta_sum = sum((h.get('beta', 0) or 0) * (h.get('cv', 0) or 0) for h in port_holdings if h.get('beta'))
                beta_wt  = sum((h.get('cv', 0) or 0) for h in port_holdings if h.get('beta'))
                if beta_wt > 0: port_beta = round(beta_sum / beta_wt, 2)

                aggressiveness = None
                if beta and port_beta:
                    if beta > port_beta * 1.2:   aggressiveness = 'More aggressive than your current portfolio'
                    elif beta < port_beta * 0.8: aggressiveness = 'More conservative than your current portfolio'
                    else:                         aggressiveness = 'Aligned with your current portfolio risk level'

                portfolio_fit = {
                    'overlapping_holdings': overlap,
                    'overlap_count': len(overlap),
                    'sector_comparison': sector_overlap,
                    'portfolio_beta': port_beta,
                    'etf_beta': beta,
                    'aggressiveness': aggressiveness,
                    'portfolio_sector_weights': port_sector_pcts,
                }
        except Exception as e:
            print(f'  [ETFLab] Portfolio fit error: {e}')

        description = build_summary(description_raw, ticker, name, category, aum, er, beta, sector_weights, fund_overview)

        print(f'  [ETFLab] {ticker}: {name}, AUM=${aum}, er={er}%, PE={pe_ratio}, beta={beta}, mom={momentum_score}, holdings={len(holdings)}')

        return jsonify({
            'ticker': ticker,
            'name': name,
            'description': description,
            'category': category,
            'price': current_price,
            'prev_close': prev_close,
            'day_change': day_change,
            'day_change_pct': day_change_pct,
            'fifty_two_week_high': fifty_two_high,
            'fifty_two_week_low': fifty_two_low,
            'pct_from_52h': pct_from_52h,
            'pct_from_52l': pct_from_52l,
            'aum': aum,
            'aum_source': aum_source,
            'expense_ratio': er,
            'turnover': turnover,
            'dividend_yield': dividend_yield,
            'dividend_yield_source': dividend_yield_source,
            'rebalance_frequency': rebalance_freq,
            'rebalance_note': rebalance_note,
            'rebalance_source': rebalance_source,
            'rebalance_evidence': rebalance_evidence,
            'pe_ratio': pe_ratio,
            'pe_signal': pe_signal,
            'benchmark_pe': benchmark_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'price_to_sales': price_to_sales,
            'valuation_source': valuation_source,
            'beta': beta,
            'target_mean_price': target_mean,
            'target_high_price': target_high,
            'target_low_price': target_low,
            'upside_mean_pct': upside_mean,
            'eps_trailing': eps_current,
            'eps_forward': eps_forward,
            'earnings_growth': earnings_growth,
            'revenue_growth': revenue_growth,
            'sector_weights': sector_weights,
            'benchmark_sectors': benchmark_sectors,
            'benchmark_name': 'VTI',
            'sector_rotation': sector_rotation,
            'holdings': holdings,
            'holdings_source': holdings_source,
            'holdings_count_note': holdings_count_note,
            'holdings_yahoo_url': f'https://finance.yahoo.com/quote/{ticker}/holdings/',
            'momentum': mom_data,
            'fed_sensitivity': fed,
            'market_cycle': market_cycle,
            'cycle_fit': cycle_fit,
            'portfolio_fit': portfolio_fit,
        })

    except Exception as e:
        import traceback
        print(f'  [ETFLab] Error for {ticker}: {e}\n{traceback.format_exc()}')
        return jsonify({'ticker': ticker, 'error': str(e)}), 500


@app.route('/api/etf/lab/chart')
def etf_lab_chart():
    """Return normalized % return series for one or more tickers over a
    selected period, for the ETF Lab's comparison chart. Series are indexed
    to 0% at the start of the window so tickers at very different price
    levels can be compared directly on one chart."""
    from flask import request as freq
    tickers_param = freq.args.get('tickers', '')
    period = freq.args.get('period', '1y')
    mode = freq.args.get('mode', 'total')  # 'total' (dividend-adjusted) or 'price' (raw price only)

    tickers = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
    if not tickers:
        return jsonify({'error': 'tickers required'}), 400
    if len(tickers) > 8:
        tickers = tickers[:8]  # keep the chart readable and the request bounded

    period_map = {
        '1d':  ('1d', '5m'),
        '1w':  ('5d', '30m'),
        '1mo': ('1mo', '1d'),
        '3mo': ('3mo', '1d'),
        '6mo': ('6mo', '1d'),
        'ytd': ('ytd', '1d'),
        '1y':  ('1y', '1d'),
        '3y':  ('3y', '1d'),
        '5y':  ('5y', '1wk'),
    }
    yf_period, interval = period_map.get(period, ('1y', '1d'))
    auto_adjust = (mode == 'total')  # total return = dividend-reinvested; price return = raw price only

    try:
        dl = yf.download(' '.join(tickers), period=yf_period, interval=interval,
                          auto_adjust=auto_adjust, progress=False, group_by='ticker')
    except Exception as e:
        print(f'  [ETFLab-Chart] Download failed for {tickers}: {e}')
        return jsonify({'error': str(e)}), 500

    if dl is None or dl.empty:
        return jsonify({'error': 'No data returned for these tickers/period'}), 404

    series_out = {}
    for tk in tickers:
        try:
            if len(tickers) == 1:
                closes = dl['Close'].dropna() if 'Close' in dl.columns else None
            else:
                closes = dl[tk]['Close'].dropna() if tk in dl.columns.get_level_values(0) else None
            if closes is None or closes.empty:
                series_out[tk] = None
                continue
            base = float(closes.iloc[0])
            if base == 0:
                series_out[tk] = None
                continue
            pct_series = ((closes / base - 1) * 100).round(2)
            series_out[tk] = [
                {'date': (idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)), 'value': float(v)}
                for idx, v in pct_series.items()
            ]
        except Exception as e:
            print(f'  [ETFLab-Chart] Series build failed for {tk}: {e}')
            series_out[tk] = None

    return jsonify({'series': series_out, 'period': period, 'mode': mode})


@app.route('/etf-lab')
def etf_lab_page():
    return send_from_directory(BASE_DIR, 'etf-lab.html')




@app.route('/api/scanner/status')
def scanner_status():
    """Lightweight status endpoint — browser polls this during compute."""
    return jsonify({
        'running': _scanner_cache['running'],
        'progress': _scanner_cache['progress'],
        'total': _scanner_cache['total'],
        'done': _scanner_cache['done'],
        'last_run': _scanner_cache['last_run'],
        'count': len(_scanner_cache['data']),
    })

@app.route('/api/scanner/refresh')
def scanner_refresh():
    """Trigger a manual re-compute in the background."""
    if _scanner_cache['running']:
        return jsonify({'status': 'already_running', 'progress': _scanner_cache['progress']})
    t = threading.Thread(target=run_scanner_compute, daemon=True)
    t.start()
    return jsonify({'status': 'started'})

if __name__ == '__main__':
    print('')
    print('  ================================================')
    print('  ⬡  Entry Point Scanner + Macro Dashboard')
    print('  ================================================')
    print('  Entry Scanner : http://localhost:5000')
    print('  Macro Dashboard: http://localhost:5000/macro')
    print('')
    print('  NOTE: For full macro data (CPI, M2, labor,')
    print('  yield curve) get a FREE FRED API key at:')
    print('  https://fred.stlouisfed.org/docs/api/api_key.html')
    print('  FRED API key is configured.')
    print('')
    print('  Keep this window open. Ctrl+C to stop.')
    print('  ================================================')
    print('')
    app.run(host='127.0.0.1', port=5000, debug=False)
