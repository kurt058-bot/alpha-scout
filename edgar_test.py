"""
Phase 1 — EDGAR N-PORT Holdings Fetcher (standalone test script)
==================================================================
Goal: prove out the pipeline ticker -> CIK -> latest N-PORT filing -> full holdings list
before wiring anything into macro_server.py.

Run this directly on your machine (not in the sandbox) since data.sec.gov
needs real internet access:

    python edgar_test.py SPY
    python edgar_test.py QQQI
    python edgar_test.py SPYI

Requires: pip install requests --break-system-packages  (if not already installed)
"""

import requests
import json
import re
import sys
import time

HEADERS = {'User-Agent': 'AlphaScout research kurth-personal-use@example.com'}

# ── Step 1: ticker -> CIK ──
# SEC publishes a master ticker->CIK mapping file. For ETFs, this usually maps
# to the TRUST's CIK (e.g. SPY -> SPDR S&P 500 ETF Trust), not a "company" CIK.
_TICKER_CIK_CACHE = None

def load_ticker_cik_map():
    global _TICKER_CIK_CACHE
    if _TICKER_CIK_CACHE is not None:
        return _TICKER_CIK_CACHE
    print("Downloading SEC ticker->CIK map (one-time, ~600KB)...")
    r = requests.get('https://www.sec.gov/files/company_tickers.json', headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    # data is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {}
    for entry in data.values():
        mapping[entry['ticker'].upper()] = entry['cik_str']
    _TICKER_CIK_CACHE = mapping
    print(f"  Loaded {len(mapping)} ticker mappings")
    return mapping


def get_cik_for_ticker(ticker):
    """Look up CIK for a ticker. Note: this maps EQUITY tickers well, but many
    ETFs are NOT in company_tickers.json under their trading ticker — they're
    filed under the TRUST name instead. We try the direct map first, then fall
    back to EDGAR full text search if that fails."""
    mapping = load_ticker_cik_map()
    cik = mapping.get(ticker.upper())
    if cik:
        return str(cik).zfill(10)
    return None


def search_cik_by_company_name(name_query):
    """Fallback: use EDGAR full text search to find a CIK by company/trust name."""
    url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={name_query}&type=NPORT-P&dateb=&owner=include&count=10&output=atom'
    r = requests.get(url, headers=HEADERS, timeout=15)
    print(f"  Fallback search status: {r.status_code}")
    return r.text


# ── Step 2: CIK -> latest N-PORT filing ──
def get_latest_nport_filing(cik):
    """Given a 10-digit CIK, find the most recent NPORT-P filing accession number."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        print(f"  Submissions lookup failed: {r.status_code}")
        return None
    data = r.json()

    recent = data.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    accessions = recent.get('accessionNumber', [])
    dates = recent.get('filingDate', [])
    primary_docs = recent.get('primaryDocument', [])

    for i, form in enumerate(forms):
        if form in ('NPORT-P', 'NPORT-P/A'):
            return {
                'accession': accessions[i],
                'date': dates[i],
                'primary_doc': primary_docs[i],
                'cik': cik,
            }
    return None


# ── Step 3: fetch and parse the N-PORT XML for holdings ──
def fetch_nport_holdings(cik, accession):
    """Fetch the N-PORT primary XML document and extract holdings."""
    accession_nodash = accession.replace('-', '')
    # The primary N-PORT data is usually in a file like primary_doc.xml
    index_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/'
    r = requests.get(index_url, headers=HEADERS, timeout=15)
    print(f"  Filing index status: {r.status_code} -- {index_url}")

    # Find the XML file in the directory listing
    xml_files = re.findall(r'href="([^"]+\.xml)"', r.text)
    print(f"  Found XML files: {xml_files}")

    if not xml_files:
        return None

    # Usually "primary_doc.xml" is the one with full holdings
    target = next((f for f in xml_files if 'primary_doc' in f.lower()), xml_files[0])
    xml_url = index_url + target.split('/')[-1]
    print(f"  Fetching: {xml_url}")

    r2 = requests.get(xml_url, headers=HEADERS, timeout=20)
    if r2.status_code != 200:
        print(f"  XML fetch failed: {r2.status_code}")
        return None

    return parse_nport_xml(r2.text)


def parse_nport_xml(xml_text):
    """Parse N-PORT XML for holdings. Uses regex for a quick test —
    production version should use a real XML parser (lxml/ElementTree).

    IMPORTANT: N-PORT filings report CUSIP as the primary security identifier.
    The <ticker> field is frequently blank/missing — issuers aren't required to
    populate it. So 'ticker' below is often actually a CUSIP, not a trading symbol.
    Phase 2 will need to resolve company NAME -> trading ticker (e.g. via the same
    name dictionary approach used in the Equity Scanner) since CUSIP -> ticker
    mapping requires either a paid reference data service or built-in known list.
    """
    import xml.etree.ElementTree as ET

    holdings = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return None

    def local_tag(elem):
        """Strip namespace prefix from a tag, e.g. '{http://...}invstOrSec' -> 'invstOrSec'"""
        return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

    def find_local(elem, tag_name):
        """Find a child element by local tag name, ignoring namespace."""
        for child in elem:
            if local_tag(child) == tag_name:
                return child
        return None

    def findtext_local(elem, tag_name, default=''):
        found = find_local(elem, tag_name)
        return found.text if found is not None and found.text else default

    # Holdings live under invstOrSecs -> invstOrSec (repeated), regardless of namespace
    for elem in root.iter():
        if local_tag(elem) == 'invstOrSec':
            name = findtext_local(elem, 'name', '')
            ticker = findtext_local(elem, 'ticker', '')  # often blank in practice
            cusip = findtext_local(elem, 'cusip', '')
            value_usd = findtext_local(elem, 'valUSD', '0')
            pct_val = findtext_local(elem, 'pctVal', '0')
            try:
                holdings.append({
                    'name': name.strip(),
                    'ticker': ticker.strip(),      # may be empty
                    'cusip': cusip.strip(),         # usually populated
                    'value_usd': float(value_usd),
                    'pct_of_fund': float(pct_val),
                })
            except ValueError:
                continue

    holdings.sort(key=lambda h: -h['pct_of_fund'])
    return holdings


def batch_precheck(tickers):
    """
    Quick pre-flight check across many tickers at once — sorts them into:
    - DIRECT: ticker maps cleanly to a CIK via company_tickers.json (script will likely work)
    - NEEDS_LOOKUP: ticker not found in the direct map — usually means it's filed
      under the issuer's TRUST name (e.g. NEOS ETF Trust, Kurv ETF Trust, YieldMax Trust)
      rather than the trading ticker itself. These need a manual CIK lookup before
      the full pipeline can run.

    This lets you scan your whole ETF list in one shot instead of discovering
    failures one ticker at a time.
    """
    mapping = load_ticker_cik_map()
    direct = []
    needs_lookup = []

    for t in tickers:
        cik = mapping.get(t.upper())
        if cik:
            direct.append((t, str(cik).zfill(10)))
        else:
            needs_lookup.append(t)

    print(f"\n{'='*60}")
    print(f"BATCH PRE-CHECK RESULTS — {len(tickers)} tickers")
    print(f"{'='*60}\n")

    print(f"✅ DIRECT MAPPING FOUND ({len(direct)}) — pipeline should work as-is:")
    for t, cik in direct:
        print(f"   {t:8s} -> CIK {cik}")

    print(f"\n⚠️  NEEDS MANUAL CIK LOOKUP ({len(needs_lookup)}) — likely filed under issuer trust name:")
    for t in needs_lookup:
        print(f"   {t:8s} -> https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={t}&type=NPORT-P&dateb=&owner=include&count=10")

    print(f"\nSummary: {len(direct)}/{len(tickers)} ready to test directly, {len(needs_lookup)}/{len(tickers)} need manual lookup first.")
    return direct, needs_lookup


def resolve_cusips_via_openfigi(holdings, api_key=None):
    """
    Batch-resolve CUSIPs to tickers + company names using OpenFIGI.
    OpenFIGI is free, maintained by Bloomberg as a public good, no key required
    for modest volume (though a free key raises the rate limit — get one at
    https://www.openfigi.com/api if we hit limits with larger ETFs).

    Batches of 100 CUSIPs per request (OpenFIGI's max per call).
    Returns the same holdings list with 'ticker' and 'name' filled in from
    OpenFIGI where it found a match (OpenFIGI's name is often cleaner/more
    standard than what's in the N-PORT filing, so we prefer it when available).
    """
    url = 'https://api.openfigi.com/v3/mapping'
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key

    resolved = 0
    failed = 0

    # OpenFIGI batch limit: 10 jobs/request WITHOUT an API key, 100 WITH one.
    # Without a key we're also limited to ~25 requests/minute, so 503 holdings
    # takes a while (~51 batches x ~2.5s spacing ≈ 2 minutes). Get a free key
    # at openfigi.com/api to raise both limits to 100/job and faster pacing.
    BATCH_SIZE = 100 if api_key else 10
    SLEEP_BETWEEN = 0.3 if api_key else 2.5
    total_batches = (len(holdings) + BATCH_SIZE - 1) // BATCH_SIZE
    est_seconds = total_batches * SLEEP_BETWEEN
    print(f"  {total_batches} batches needed (batch size {BATCH_SIZE}) — est. ~{est_seconds:.0f}s total"
          + ("" if api_key else " (no API key — get one free at openfigi.com/api to go ~8x faster)"))

    for batch_num, i in enumerate(range(0, len(holdings), BATCH_SIZE), 1):
        batch = holdings[i:i+BATCH_SIZE]
        jobs = [{'idType': 'ID_CUSIP', 'idValue': h['cusip']} for h in batch if h.get('cusip')]
        if not jobs:
            continue
        print(f"  Batch {batch_num}/{total_batches}...", end='\r')

        try:
            r = requests.post(url, headers=headers, json=jobs, timeout=20)
        except Exception as e:
            print(f"  OpenFIGI request error: {e}")
            failed += len(jobs)
            continue

        if r.status_code == 429:
            print("  ⚠️  Rate limited by OpenFIGI — consider getting a free API key at openfigi.com/api")
            time.sleep(2)
            continue
        if r.status_code != 200:
            print(f"  OpenFIGI batch failed: {r.status_code} — {r.text[:200]}")
            failed += len(jobs)
            continue

        results = r.json()
        for h, result in zip([h for h in batch if h.get('cusip')], results):
            if 'data' in result and result['data']:
                match = result['data'][0]  # take best match
                ticker = match.get('ticker', '')
                name = match.get('name', '')
                if ticker:
                    h['ticker'] = ticker
                    resolved += 1
                if name:
                    h['name'] = name  # prefer OpenFIGI's cleaner name
            else:
                failed += 1

        time.sleep(SLEEP_BETWEEN)  # be polite — OpenFIGI free tier is rate-limited

    print(f"\n  OpenFIGI resolution: {resolved} resolved, {failed} unresolved (out of {len(holdings)})")
    return holdings


def test_ticker(ticker, api_key=None):
    print(f"\n{'='*60}\nTesting: {ticker}\n{'='*60}")

    cik = get_cik_for_ticker(ticker)
    if not cik:
        print(f"  ❌ No CIK found for {ticker} in company_tickers.json")
        print(f"  This likely means {ticker} is filed under its TRUST name, not the ticker.")
        print(f"  Manual lookup needed at: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=NPORT-P")
        return

    print(f"  ✅ CIK: {cik}")

    filing = get_latest_nport_filing(cik)
    if not filing:
        print(f"  ❌ No NPORT-P filing found for CIK {cik}")
        return

    print(f"  ✅ Latest N-PORT filing: {filing['date']} (accession {filing['accession']})")

    holdings = fetch_nport_holdings(filing['cik'], filing['accession'])
    if not holdings:
        print(f"  ❌ Could not parse holdings from filing")
        return

    print(f"  ✅ Parsed {len(holdings)} holdings")

    tickers_before = sum(1 for h in holdings if h['ticker'])
    print(f"  {tickers_before}/{len(holdings)} holdings had a populated <ticker> field directly from the filing.")
    print(f"  Resolving CUSIPs -> tickers + clean names via OpenFIGI...")

    holdings = resolve_cusips_via_openfigi(holdings, api_key=api_key)

    print(f"\n  Top 10 by weight (post-OpenFIGI resolution):")
    for h in holdings[:10]:
        ticker_display = h['ticker'] if h['ticker'] else f"[{h['cusip']}]"
        print(f"    {ticker_display:8s} {h['pct_of_fund']:6.2f}%  {h['name'][:50]}")


if __name__ == '__main__':
    # Known Alpha Scout ETF universe — covers your custom additions plus common ones.
    # Edit this list any time you add a new ETF to the scanner; or pass tickers
    # directly on the command line to override.
    DEFAULT_UNIVERSE = [
        'SPY', 'QQQ', 'DIA', 'VTI', 'IWM',  # broad market
        'QQQM', 'SCHD', 'CGGR', 'CGDV', 'CGDG', 'AVGO', 'AIQ', 'AIRR', 'PAVE',  # retirement core
        'JAAA', 'VTIP', 'SGOV', 'FLOT',  # cash-like
        'TSPY', 'SPYI', 'QQQI', 'NVIT', 'MSFY', 'MLPI', 'KSLV', 'KGLD',  # income sleeve
        'GOOP', 'CVNY', 'CHPY', 'BTCI', 'BLOX', 'XSHP', 'XLE',  # income sleeve cont.
        'SPMO',  # momentum
    ]

    args = sys.argv[1:]
    api_key = None
    if '--key' in args:
        idx = args.index('--key')
        api_key = args[idx+1]
        args = args[:idx] + args[idx+2:]  # remove --key and its value from args

    if args and args[0] == '--precheck':
        tickers = args[1:] if len(args) > 1 else DEFAULT_UNIVERSE
        batch_precheck(tickers)
    else:
        tickers = args if args else ['SPY']
        for t in tickers:
            test_ticker(t, api_key=api_key)
            time.sleep(0.5)  # be polite to SEC servers (10 req/sec max)
