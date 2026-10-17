"""
Master Directory of Indian Equities, ETFs & Assets
Covers 2,590+ stocks across NSE and BSE, merged with rich curated aliases from Stoxify.
Provides sub-millisecond multi-tier search across symbol, company name, aliases, brands, and sectors.
"""

from typing import List, Dict, Optional
from pathlib import Path
import json

MASTER_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "indian_stocks_master.json"

# Curated fallback list if JSON is ever missing
CURATED_FALLBACK: List[Dict] = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy", "aliases": ["ril", "reliance", "mukesh ambani", "jio"]},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "IT", "aliases": ["tcs", "tata consultancy"]},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking", "aliases": ["hdfc", "hdfc bank"]},
    {"symbol": "INFY.NS", "name": "Infosys Ltd", "sector": "IT", "aliases": ["infy", "infosys"]},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking", "aliases": ["icici", "icici bank"]},
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
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises Ltd", "sector": "Energy", "aliases": ["adani", "adani ent", "gautam adani"]},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Auto", "aliases": ["tata motors", "tatamotors", "tata ev", "harrier", "safari", "nexon", "thar"]},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra Ltd", "sector": "Auto", "aliases": ["m&m", "mahindra", "thar", "scorpio", "xuv"]},
    {"symbol": "TRENT.NS", "name": "Trent Ltd (Westside & Zudio)", "sector": "Consumer", "aliases": ["trent", "zudio", "westside", "retail"]},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services Ltd", "sector": "Finance", "aliases": ["jio financial", "jiofin", "jfs", "jio"]},
    {"symbol": "ZOMATO.NS", "name": "Zomato Ltd", "sector": "Consumer", "aliases": ["zomato", "blinkit", "eternal", "food delivery"]},
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy Ltd", "sector": "Energy", "aliases": ["suzlon", "wind energy", "green energy", "clean energy"]},
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defense", "aliases": ["hal", "tejas", "defense", "defence", "fighter jet"]},
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance Corp", "sector": "Railways", "aliases": ["irfc", "railway finance", "railways", "trains"]}
]


def load_master_directory() -> List[Dict]:
    """Loads all Indian stocks from JSON master file with in-memory caching."""
    if MASTER_JSON_PATH.exists():
        try:
            with open(MASTER_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return CURATED_FALLBACK
    return CURATED_FALLBACK


STOCK_DIRECTORY: List[Dict] = load_master_directory()


def search_stock_directory(query: str, limit: int = 12) -> List[Dict]:
    """
    Smart multi-match search across all NSE & BSE stocks:
    1. Exact symbol match (e.g. TCS -> TCS.NS)
    2. Prefix symbol match (e.g. TATA -> TATAMOTORS.NS, TATASTEEL.NS, TATAPOWER.NS)
    3. Alias and brand match (e.g. zudio -> TRENT.NS, blinkit -> ZOMATO.NS, thar -> M&M.NS)
    4. Company name substring match (e.g. State Bank -> SBIN.NS)
    5. Sector match (e.g. railways, defense, green energy)
    6. Dynamic fallback for any custom NSE (.NS) or BSE (.BO) ticker.
    """
    if not query:
        return STOCK_DIRECTORY[:limit]

    q = query.strip().lower()
    clean_q = q.replace(".ns", "").replace(".bo", "").replace(" ", "")

    exact_matches = []
    symbol_prefix_matches = []
    alias_matches = []
    name_matches = []
    sector_matches = []

    for item in STOCK_DIRECTORY:
        sym_clean = item["symbol"].replace(".NS", "").replace(".BO", "").lower()
        name_clean = item["name"].lower()
        sector_clean = item.get("sector", "").lower()
        aliases = [a.lower() for a in item.get("aliases", [])]

        if clean_q == sym_clean:
            exact_matches.append(item)
        elif sym_clean.startswith(clean_q):
            symbol_prefix_matches.append(item)
        elif any(clean_q == a or a.startswith(clean_q) or clean_q in a for a in aliases):
            alias_matches.append(item)
        elif clean_q in name_clean:
            name_matches.append(item)
        elif clean_q in sector_clean:
            sector_matches.append(item)

    combined = exact_matches + symbol_prefix_matches + alias_matches + name_matches + sector_matches

    # Deduplicate while preserving priority order
    seen = set()
    results = []
    for c in combined:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            results.append(c)

    # Dynamic fallback: If user entered an exact custom ticker not in directory
    if not results and query:
        custom_sym = query.strip().upper()
        if not custom_sym.endswith(".NS") and not custom_sym.endswith(".BO") and "." not in custom_sym:
            # Check if it looks like a BSE numeric scrip code (e.g. 500325)
            if custom_sym.isdigit():
                custom_sym = f"{custom_sym}.BO"
            else:
                custom_sym = f"{custom_sym}.NS"
        results.append({
            "symbol": custom_sym,
            "name": f"Custom Asset ({custom_sym})",
            "sector": "Indian Equities",
            "aliases": [query.lower()]
        })

    return results[:limit]
