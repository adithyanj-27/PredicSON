"""
Master Directory of Indian Equities, ETFs & Assets
Covers 10,680+ stocks across NSE and BSE, merged with curated brand aliases.
Provides high-performance multi-tier search with fuzzy typo-tolerance,
multi-token name matching, and Dalal Street priority ranking.
"""

import json
import sys
import re
import difflib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Set, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

MASTER_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "indian_stocks_master.json"

# Curated High-Priority Brand & Dalal Street Aliases
CURATED_ALIASES: Dict[str, List[str]] = {
    "RELIANCE.NS": ["ril", "reliance", "mukesh ambani", "jio", "reliance retail", "reliance fresh"],
    "TCS.NS": ["tcs", "tata consultancy", "tata consultancy services"],
    "HDFCBANK.NS": ["hdfc bank", "hdfc", "housing development finance"],
    "INFY.NS": ["infy", "infosys", "narayana murthy", "salil parekh"],
    "ICICIBANK.NS": ["icici bank", "icici"],
    "SBIN.NS": ["sbi", "sbin", "state bank", "state bank of india"],
    "BHARTIARTL.NS": ["airtel", "bharti airtel", "sunil mittal"],
    "ITC.NS": ["itc", "itc limited", "itc hotels", "aashirvaad", "sunfeast", "bingo", "yippee", "classmate"],
    "LT.NS": ["l&t", "lt", "larsen", "larsen & toubro", "larsen and toubro", "infra"],
    "BAJFINANCE.NS": ["bajaj finance", "bajfinance", "bajaj finserv"],
    "HINDUNILVR.NS": ["hul", "unilever", "hindustan unilever", "surf excel", "dove", "lux", "lifebuoy", "rin", "horlicks"],
    "MARUTI.NS": ["maruti", "suzuki", "maruti suzuki", "brezza", "swift", "baleno", "dzire", "ertiga"],
    "SUNPHARMA.NS": ["sun pharma", "sun pharmaceutical"],
    "TITAN.NS": ["titan", "tanishq", "fastrack", "eyewear", "skinn", "mia"],
    "TATASTEEL.NS": ["tata steel", "tisco"],
    "M&M.NS": ["m&m", "mahindra", "thar", "scorpio", "xuv", "xuv700", "bolero", "anand mahindra"],
    "TATAMOTORS.NS": ["tata motors", "tatamotors", "tata ev", "harrier", "safari", "nexon", "punch", "curvv", "jlr", "jaguar", "land rover"],
    "ZOMATO.NS": ["zomato", "blinkit", "eternal", "deepinder goyal", "food delivery", "quick commerce"],
    "TRENT.NS": ["trent", "zudio", "westside", "star bazaar", "retail", "noel tata"],
    "JIOFIN.NS": ["jio financial", "jiofin", "jfs", "jio finance"],
    "SUZLON.NS": ["suzlon", "suzlon energy", "wind energy", "green energy", "clean energy"],
    "HAL.NS": ["hal", "tejas", "hindustan aeronautics", "defense", "defence", "fighter jet"],
    "IRFC.NS": ["irfc", "railway finance", "railways", "trains"],
    "PAYTM.NS": ["paytm", "one97", "vijay shekhar sharma"],
    "SWIGGY.NS": ["swiggy", "instamart", "dineout"],
    "OLAELEC.NS": ["ola", "ola electric", "bhavish aggarwal", "ola scooter", "s1 pro"],
    "FEDERALBNK.NS": ["federal bank", "the federal bank"],
    "PNB.NS": ["pnb", "punjab national bank"],
    "BANKBARODA.NS": ["bank of baroda", "bob"],
    "ASIANPAINT.NS": ["asian paints", "asian paint", "royale", "tractor emulsion"],
    "PIDILITIND.NS": ["pidilite", "fevicol", "m-seal", "dr fixit", "fevikwik"],
    "MRF.NS": ["mrf", "mrf tyres"],
    "BSE.NS": ["bse", "bombay stock exchange"],
    "CDSL.NS": ["cdsl", "central depository"],
    "ANGELONE.NS": ["angel one", "angel broking"],
    "MOTILALOFS.NS": ["motilal oswal", "mosl"],
    "NESTLEIND.NS": ["nestle", "maggi", "nescafe", "kitkat"],
    "BRITANNIA.NS": ["britannia", "good day", "marie gold", "bourbon", "treat"],
    "DABUR.NS": ["dabur", "chyawanprash", "real juice", "vatika", "hajmola"],
    "GODREJCP.NS": ["godrej", "godrej consumer", "good knight", "hit", "cinthol"],
    "EICHERMOT.NS": ["eicher", "royal enfield", "bullet", "classic 350", "hunter 350"],
    "BAJAJ-AUTO.NS": ["bajaj auto", "pulsar", "chetak", "ktm", "dominar"],
    "HEROMOTOCO.NS": ["hero", "hero motocorp", "splendor", "hf deluxe"],
    "BEL.NS": ["bel", "bharat electronics", "defense"],
    "RVNL.NS": ["rvnl", "rail vikas nigam", "railways"],
    "COALINDIA.NS": ["coal india", "cil", "coal"],
    "ADANIENT.NS": ["adani enterprises", "adani ent", "gautam adani"],
    "ADANIPORTS.NS": ["adani ports", "mundra port", "ports"],
    "POWERGRID.NS": ["power grid", "powergrid", "pgcil"],
    "NTPC.NS": ["ntpc", "national thermal power"],
    "ONGC.NS": ["ongc", "oil and natural gas"],
    "WIPRO.NS": ["wipro", "azim premji"],
    "TECHM.NS": ["tech mahindra", "techm"],
    "HCLTECH.NS": ["hcl tech", "hcl technologies", "roshni nadar"],
    "CIPLA.NS": ["cipla", "generic pharma"],
    "DRREDDY.NS": ["dr reddy", "dr reddys", "dr reddy's laboratories"],
    "APOLLOHOSP.NS": ["apollo hospitals", "apollo hospital", "apollo pharmacy"]
}

NIFTY_HEAVYWEIGHTS: Set[str] = {
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "TATASTEEL.NS", "M&M.NS", "TATAMOTORS.NS", "ZOMATO.NS",
    "TRENT.NS", "JIOFIN.NS", "SUZLON.NS", "HAL.NS", "IRFC.NS", "PAYTM.NS", "SWIGGY.NS",
    "OLAELEC.NS", "ASIANPAINT.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "ONGC.NS", "WIPRO.NS",
    "HCLTECH.NS", "ULTRACEMCO.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "BEL.NS", "BSE.NS", "CDSL.NS", "ANGELONE.NS",
    "FEDERALBNK.NS", "PNB.NS", "BANKBARODA.NS", "PIDILITIND.NS", "MRF.NS", "TECHM.NS"
}

CURATED_FALLBACK: List[Dict] = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy", "aliases": ["ril", "reliance", "mukesh ambani", "jio"]},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "IT", "aliases": ["tcs", "tata consultancy"]},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking", "aliases": ["hdfc bank", "hdfc"]},
    {"symbol": "INFY.NS", "name": "Infosys Ltd", "sector": "IT", "aliases": ["infy", "infosys"]},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking", "aliases": ["icici bank", "icici"]},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking", "aliases": ["sbi", "sbin", "state bank"]},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecom", "aliases": ["airtel", "bharti airtel"]},
    {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "Consumer", "aliases": ["itc", "itc hotels", "cigarettes"]},
    {"symbol": "LT.NS", "name": "Larsen & Toubro Ltd", "sector": "Infra", "aliases": ["l&t", "lt", "larsen"]},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance Ltd", "sector": "Finance", "aliases": ["bajaj finance", "bajfinance"]},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd", "sector": "Consumer", "aliases": ["hul", "unilever", "hindustan unilever"]},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India Ltd", "sector": "Auto", "aliases": ["maruti", "suzuki", "maruti suzuki"]},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Ind. Ltd", "sector": "Pharma", "aliases": ["sun pharma", "sun"]},
    {"symbol": "TITAN.NS", "name": "Titan Company Ltd", "sector": "Consumer", "aliases": ["titan", "tanishq", "fastrack"]},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Metals", "aliases": ["tata steel", "tisco"]},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra Ltd", "sector": "Auto", "aliases": ["m&m", "mahindra", "thar", "scorpio", "xuv"]},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Auto", "aliases": ["tata motors", "tatamotors", "tata ev", "harrier", "safari", "nexon", "thar"]},
    {"symbol": "ZOMATO.NS", "name": "Zomato Ltd", "sector": "Consumer", "aliases": ["zomato", "blinkit", "eternal", "food delivery"]},
    {"symbol": "TRENT.NS", "name": "Trent Ltd (Westside & Zudio)", "sector": "Consumer", "aliases": ["trent", "zudio", "westside", "retail"]},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services Ltd", "sector": "Finance", "aliases": ["jio financial", "jiofin", "jfs", "jio"]},
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy Ltd", "sector": "Energy", "aliases": ["suzlon", "wind energy", "green energy", "clean energy"]},
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defense", "aliases": ["hal", "tejas", "defense", "defence", "fighter jet"]},
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance Corp", "sector": "Railways", "aliases": ["irfc", "railway finance", "railways", "trains"]},
]


def load_master_directory() -> List[Dict]:
    """Loads all Indian stocks from JSON master file, falling back to the curated list."""
    if MASTER_JSON_PATH.exists():
        try:
            with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read master directory ({e}); using curated fallback.")
    return CURATED_FALLBACK


STOCK_DIRECTORY: List[Dict] = load_master_directory()

# Pre-computed tokenized lookup fields for sub-millisecond multi-token search
_INDEX = []
_FUZZY_LOOKUP: Dict[str, Dict] = {}

for item in STOCK_DIRECTORY:
    sym_full = item.get("symbol", "")
    sym = sym_full.replace(".NS", "").replace(".BO", "").lower()
    name = item.get("name", "").lower()
    name_clean = re.sub(r"\b(ltd|limited|corp|corporation|industries|india|co)\b", "", name, flags=re.IGNORECASE).strip()
    name_words = [w for w in name_clean.split() if w]
    sector = item.get("sector", "").lower()

    # Merge curated aliases if available
    item_aliases = [a.lower() for a in item.get("aliases", [])]
    curated = [a.lower() for a in CURATED_ALIASES.get(sym_full, [])]
    all_aliases = list(dict.fromkeys(item_aliases + curated))

    alias_words = [w for a in all_aliases for w in a.split() if w]
    all_tokens = [sym] + name_words + all_aliases + alias_words

    entry = {
        "item": item,
        "symbol": sym_full,
        "sym": sym,
        "name": name,
        "name_clean": name_clean,
        "name_words": name_words,
        "sector": sector,
        "aliases": all_aliases,
        "all_tokens": all_tokens,
        "is_nse": sym_full.endswith(".NS"),
    }
    _INDEX.append(entry)

    # Build fuzzy lookup dictionary (preferring .NS entries)
    if sym not in _FUZZY_LOOKUP or sym_full.endswith(".NS"):
        _FUZZY_LOOKUP[sym] = item
    for a in all_aliases:
        if a not in _FUZZY_LOOKUP or sym_full.endswith(".NS"):
            _FUZZY_LOOKUP[a] = item
    if name_clean and (name_clean not in _FUZZY_LOOKUP or sym_full.endswith(".NS")):
        _FUZZY_LOOKUP[name_clean] = item

_FUZZY_KEYS: List[str] = list(_FUZZY_LOOKUP.keys())


def search_stock_directory(query: str, limit: int = 12) -> List[Dict]:
    """
    High-precision multi-token ranked search across all Indian stocks (NSE/BSE).
    Features:
    1. Exact symbol / brand / alias matching (TCS, Zudio -> TRENT, Blinkit -> ZOMATO)
    2. Multi-token company name matching ('Federal Bank' -> FEDERALBNK.NS)
    3. Intelligent fuzzy typo-tolerance ('relaince' -> RELIANCE.NS, 'tatamotor' -> TATAMOTORS.NS)
    4. Prioritizes primary NSE (.NS) listings over BSE (.BO) for Dalal Street standards
    5. Clean fallback that never injects fake tickers when genuine matches exist
    """
    if not query:
        return STOCK_DIRECTORY[:limit]

    q = query.strip().lower()
    explicit_bse = q.endswith(".bo") or q.isdigit()
    explicit_nse = q.endswith(".ns")

    q_clean = q.replace(".ns", "").replace(".bo", "").strip()
    q_no_space = q_clean.replace(" ", "").replace("-", "")
    q_words = [w for w in q_clean.split() if w]

    scored_candidates = []

    for entry in _INDEX:
        sym = entry["sym"]
        name = entry["name"]
        name_clean = entry["name_clean"]
        aliases = entry["aliases"]
        sector = entry["sector"]
        all_tokens = entry["all_tokens"]
        is_nse = entry["is_nse"]

        score = 0

        # Tier 1: Exact Symbol Match (Highest Priority)
        if q_clean == sym or q_no_space == sym:
            score = 1500

        # Tier 2: Exact Alias / Brand Match (e.g. 'zudio', 'blinkit', 'fevicol', 'jio')
        elif any(q_clean == a or q_no_space == a.replace(" ", "") for a in aliases):
            score = 1400

        # Tier 3: Exact Company Name Match
        elif q_clean == name or q_no_space == name.replace(" ", "") or q_clean == name_clean:
            score = 1350

        # Tier 4: Symbol Prefix Match (e.g. 'TATA' -> TATAMOTORS, TATASTEEL, TATAPOWER)
        elif sym.startswith(q_clean) or sym.startswith(q_no_space):
            score = 1100 - min(len(sym) - len(q_clean), 30) * 5

        # Tier 5: Company Name / Alias Prefix Match
        elif name.startswith(q_clean) or name_clean.startswith(q_clean) or any(a.startswith(q_clean) for a in aliases):
            score = 1000

        # Tier 6: Multi-Token Order-Independent Match (e.g. 'federal bank', 'state bank of india', 'tata motors')
        elif len(q_words) > 1:
            all_matched = True
            token_pts = 0
            for qw in q_words:
                matched_qw = False
                for t in all_tokens:
                    if t == qw:
                        matched_qw = True
                        token_pts += 50
                        break
                    elif t.startswith(qw):
                        matched_qw = True
                        token_pts += 30
                        break
                    elif qw in t:
                        matched_qw = True
                        token_pts += 15
                        break
                if not matched_qw:
                    all_matched = False
                    break
            if all_matched:
                score = 850 + min(token_pts, 150)

        # Tier 7: Single-Token Substring Match
        elif q_clean in sym:
            score = 650
        elif q_clean in name_clean or q_clean in name:
            score = 600
        elif any(q_clean in a for a in aliases):
            score = 550
        elif sector and q_clean in sector:
            score = 400

        if score > 0:
            # Benchmark Heavyweights vs SME scripts
            sym_full = entry["symbol"]
            if sym_full in NIFTY_HEAVYWEIGHTS:
                score += 260
            elif sym_full.endswith("-SM.NS") or sym_full.endswith("-BE.NS"):
                score -= 160

            # Exchange preference adjustment
            if explicit_bse:
                if not is_nse:
                    score += 150
            else:
                # Default Indian equity preference: NSE is standard
                if is_nse:
                    score += 80
                else:
                    score -= 40

            scored_candidates.append((score, entry["item"]))

    # Tier 8: Fuzzy Typo Matching (difflib)
    # Trigger if fewer than 3 strong candidates were found and query is >= 4 chars
    has_strong_match = any(s >= 1000 for s, _ in scored_candidates)
    if (len(scored_candidates) < 3 or not has_strong_match) and len(q_clean) >= 4:
        fuzzy_matches = difflib.get_close_matches(q_clean, _FUZZY_KEYS, n=4, cutoff=0.72)
        if not fuzzy_matches and q_no_space != q_clean:
            fuzzy_matches = difflib.get_close_matches(q_no_space, _FUZZY_KEYS, n=4, cutoff=0.72)

        for fm in fuzzy_matches:
            item = _FUZZY_LOOKUP.get(fm)
            if item and not any(cand["symbol"] == item["symbol"] for _, cand in scored_candidates):
                ratio = difflib.SequenceMatcher(None, q_clean, fm).ratio()
                fuzzy_score = int(750 * ratio)
                if item.get("symbol", "").endswith(".NS") and not explicit_bse:
                    fuzzy_score += 40
                scored_candidates.append((fuzzy_score, item))

    # Sort descending by relevance score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Intelligent Deduplication: If a company appears as both NSE and BSE,
    # show NSE first and drop the duplicate BSE from top slots unless explicitly requested
    results = []
    seen_canonical_names = set()

    for _, item in scored_candidates:
        sym = item.get("symbol", "")
        # Canonical key: clean ticker name without exchange
        clean_base = sym.replace(".NS", "").replace(".BO", "")
        if not explicit_bse and sym.endswith(".BO") and f"{clean_base}.NS" in [r["symbol"] for r in results]:
            # User already has the NSE listing in results; omit BSE duplicate to keep list clean
            continue
        if sym not in [r["symbol"] for r in results]:
            results.append(item)
            if len(results) >= limit:
                break

    # Live Exchange Fallback: If 0 results, query live Yahoo Finance search
    if not results and query and len(query.strip()) >= 2:
        try:
            live_items = query_yahoo_finance_live_search(query)
            for li in live_items:
                if li["symbol"] not in [r["symbol"] for r in results]:
                    results.append(li)
        except Exception:
            pass

    # Direct Alphanumeric Ticker Fallback
    # ONLY append if results are completely empty OR query explicitly ends with .NS / .BO or is numeric BSE
    if (not results and query) or explicit_nse or explicit_bse:
        single_word = query.strip().upper().replace(" ", "")
        if len(single_word) <= 15:
            if not single_word.endswith(".NS") and not single_word.endswith(".BO") and "." not in single_word:
                custom_sym = f"{single_word}.BO" if single_word.isdigit() else f"{single_word}.NS"
            else:
                custom_sym = single_word
            if not any(r["symbol"] == custom_sym for r in results):
                results.append({
                    "symbol": custom_sym,
                    "name": f"Direct Symbol ({custom_sym})",
                    "sector": "Direct Exchange Query",
                    "aliases": [query.lower()]
                })

    return results[:limit]


def query_yahoo_finance_live_search(query: str) -> List[Dict]:
    """Live real-time search against Yahoo Finance for any unlisted or newly listed stock."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}&quotesCount=8&newsCount=0"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    res = []
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.load(resp)
            quotes = data.get("quotes", [])
            for q in quotes:
                sym = q.get("symbol", "")
                if not sym:
                    continue
                name = q.get("shortname") or q.get("longname") or sym
                exch = q.get("exchange", "")
                if sym.endswith(".NS") or sym.endswith(".BO") or exch in ("NSI", "BSE", "NSE"):
                    clean_sym = sym if (sym.endswith(".NS") or sym.endswith(".BO")) else (f"{sym}.NS" if exch == "NSI" else f"{sym}.BO")
                    res.append({
                        "symbol": clean_sym,
                        "name": name.title(),
                        "sector": "Equities",
                        "exchange": "BSE" if clean_sym.endswith(".BO") else "NSE",
                        "aliases": [clean_sym.lower(), name.lower()]
                    })
    except Exception:
        pass
    return res
