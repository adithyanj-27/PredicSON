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


def search_stock_directory(query: str, limit: int = 10) -> List[Dict]:
    """
    High-precision multi-token ranked search across all Indian stocks (NSE/BSE).
    Matches company names (e.g. 'Federal Bank' -> FEDERALBNK.NS), brands, aliases, and tickers.
    """
    if not query:
        return STOCK_DIRECTORY[:limit]

    q = query.strip().lower()
    q_clean = q.replace(".ns", "").replace(".bo", "").strip()
    q_words = [w for w in q_clean.split() if w]

    scored_candidates = []

    for item in STOCK_DIRECTORY:
        sym = item["symbol"].upper().replace(".NS", "").replace(".BO", "").lower()
        name = item["name"].lower()
        name_words = [w for w in name.split() if w]
        sector = item.get("sector", "").lower()
        aliases = [a.lower() for a in item.get("aliases", [])]
        all_tokens = [sym] + name_words + [w for a in aliases for w in a.split()]

        score = 0

        # 1. Exact matches (Highest Priority)
        if q_clean == sym:
            score = 1000
        elif q_clean == name:
            score = 950
        elif any(q_clean == a for a in aliases):
            score = 900
        
        # 2. Prefix matches on symbol or company name
        elif sym.startswith(q_clean):
            score = 800
        elif name.startswith(q_clean):
            score = 750

        # 3. Multi-token (order-independent) matching (e.g. 'federal bank', 'tata motors', 'state bank of india')
        elif len(q_words) > 1:
            all_matched = True
            token_pts = 0
            for qw in q_words:
                matched_qw = False
                for t in all_tokens:
                    if t == qw:
                        matched_qw = True
                        token_pts += 150
                        break
                    elif t.startswith(qw):
                        matched_qw = True
                        token_pts += 100
                        break
                    elif qw in t:
                        matched_qw = True
                        token_pts += 50
                        break
                if not matched_qw:
                    all_matched = False
                    break
            if all_matched:
                score = 700 + min(token_pts, 150)

        # 4. Single-token substring matching
        elif q_clean in sym:
            score = 550
        elif q_clean in name:
            score = 500
        elif any(q_clean in a for a in aliases):
            score = 450
        elif sector and q_clean in sector:
            score = 350

        if score > 0:
            scored_candidates.append((score, item))

    # Sort descending by relevance score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    results = [item for _, item in scored_candidates]

    # Only offer a custom fallback if the query looks like a valid alphanumeric ticker with NO spaces!
    if not results and query:
        single_word = query.strip().upper()
        if " " not in single_word and len(single_word) <= 15:
            if not single_word.endswith(".NS") and not single_word.endswith(".BO") and "." not in single_word:
                if single_word.isdigit():
                    custom_sym = f"{single_word}.BO"
                else:
                    custom_sym = f"{single_word}.NS"
            else:
                custom_sym = single_word
            results.append({
                "symbol": custom_sym,
                "name": f"Custom Ticker ({custom_sym})",
                "sector": "Indian Equities",
                "aliases": [query.lower()]
            })

    return results[:limit]
