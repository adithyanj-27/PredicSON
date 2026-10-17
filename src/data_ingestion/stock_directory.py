"""
Master Directory of Indian Equities, ETFs & Assets
Covers ~2,600 stocks across NSE and BSE, merged with curated aliases.
Provides multi-tier search across symbol, company name, aliases, brands, and sectors.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

MASTER_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "indian_stocks_master.json"

# Curated fallback list if JSON is ever missing
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

# Pre-computed lowercase lookup fields (built once, not on every keystroke).
_INDEX = [
    {
        "item": item,
        "sym": item.get("symbol", "").replace(".NS", "").replace(".BO", "").lower(),
        "name": item.get("name", "").lower(),
        "sector": item.get("sector", "").lower(),
        "aliases": [a.lower() for a in item.get("aliases", [])],
    }
    for item in STOCK_DIRECTORY
]


def search_stock_directory(query: str, limit: int = 12) -> List[Dict]:
    """
    Smart multi-match search across all NSE & BSE stocks, in priority order:
    1. Exact symbol match (e.g. TCS -> TCS.NS)
    2. Prefix symbol match (e.g. TATA -> TATAMOTORS.NS, TATASTEEL.NS)
    3. Alias / brand match (e.g. zudio -> TRENT.NS, blinkit -> ZOMATO.NS)
    4. Company-name substring match (e.g. State Bank -> SBIN.NS)
    5. Sector match (e.g. railways, defense)
    6. Dynamic fallback for any custom NSE (.NS) / BSE (.BO) ticker.
    """
    if not query:
        return STOCK_DIRECTORY[:limit]

    q = query.strip().lower()
    clean_q = q.replace(".ns", "").replace(".bo", "").replace(" ", "")

    exact, prefix, alias, name, sector = [], [], [], [], []

    for entry in _INDEX:
        sym, nm, sec, aliases = entry["sym"], entry["name"], entry["sector"], entry["aliases"]
        if clean_q == sym:
            exact.append(entry["item"])
        elif sym.startswith(clean_q):
            prefix.append(entry["item"])
        elif any(clean_q == a or a.startswith(clean_q) or clean_q in a for a in aliases):
            alias.append(entry["item"])
        elif clean_q in nm:
            name.append(entry["item"])
        elif clean_q in sec:
            sector.append(entry["item"])

    combined = exact + prefix + alias + name + sector

    # Deduplicate while preserving priority order
    seen = set()
    results = []
    for c in combined:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            results.append(c)

    # Dynamic fallback: an exact custom ticker not present in the directory
    if not results and query:
        custom_sym = query.strip().upper()
        if "." not in custom_sym:
            custom_sym = f"{custom_sym}.BO" if custom_sym.isdigit() else f"{custom_sym}.NS"
        results.append({
            "symbol": custom_sym,
            "name": f"Custom Asset ({custom_sym})",
            "sector": "Indian Equities",
            "aliases": [query.lower()]
        })

    return results[:limit]
