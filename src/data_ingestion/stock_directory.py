"""
Master Directory of Indian Equities, ETFs & Assets
Combines Stoxify's enriched directory with fuzzy alias search and direct NSE/BSE resolution.
"""

from typing import List, Dict, Optional

STOCK_DIRECTORY: List[Dict] = [
    # --- Heavyweights & NIFTY Titans ---
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
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports & SEZ Ltd", "sector": "Infra", "aliases": ["adani ports", "apsez"]},
    {"symbol": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT", "aliases": ["wipro"]},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corp of India", "sector": "Energy", "aliases": ["powergrid", "power grid"]},
    {"symbol": "NTPC.NS", "name": "NTPC Ltd", "sector": "Energy", "aliases": ["ntpc", "national thermal power"]},
    {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp Ltd", "sector": "Energy", "aliases": ["ongc"]},
    {"symbol": "COALINDIA.NS", "name": "Coal India Ltd", "sector": "Metals", "aliases": ["coal india", "cil"]},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra Ltd", "sector": "Auto", "aliases": ["m&m", "mahindra", "thar", "scorpio"]},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Auto", "aliases": ["tata motors", "tatamotors", "tata ev", "tata cars"]},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Banking", "aliases": ["axis", "axis bank"]},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking", "aliases": ["kotak", "kotak bank"]},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd", "sector": "Infra", "aliases": ["ultratech", "cement"]},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints Ltd", "sector": "Consumer", "aliases": ["asian paints", "paints"]},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto Ltd", "sector": "Auto", "aliases": ["bajaj auto", "pulsar"]},
    {"symbol": "TRENT.NS", "name": "Trent Ltd (Westside & Zudio)", "sector": "Consumer", "aliases": ["trent", "zudio", "westside"]},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services Ltd", "sector": "Finance", "aliases": ["jio financial", "jiofin", "jfs"]},
    {"symbol": "ZOMATO.NS", "name": "Zomato Ltd (Eternal Ltd)", "sector": "Consumer", "aliases": ["zomato", "blinkit", "eternal"]},

    # --- High-Growth Retail & E-Commerce ---
    {"symbol": "SWIGGY.NS", "name": "Swiggy Ltd", "sector": "Consumer", "aliases": ["swiggy", "instamart"]},
    {"symbol": "PAYTM.NS", "name": "One97 Communications (Paytm)", "sector": "IT", "aliases": ["paytm", "one97", "upi"]},
    {"symbol": "NYKAA.NS", "name": "FSN E-Commerce Ventures (Nykaa)", "sector": "Consumer", "aliases": ["nykaa", "beauty"]},
    {"symbol": "DMART.NS", "name": "Avenue Supermarts Ltd (DMart)", "sector": "Consumer", "aliases": ["dmart", "avenue supermarts", "damani"]},
    {"symbol": "VBL.NS", "name": "Varun Beverages Ltd (PepsiCo)", "sector": "Consumer", "aliases": ["vbl", "varun beverages", "pepsi", "sting"]},
    {"symbol": "INDIGO.NS", "name": "InterGlobe Aviation Ltd (IndiGo)", "sector": "Consumer", "aliases": ["indigo", "interglobe", "airlines"]},

    # --- Green Energy, Solar & Power ---
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy Ltd", "sector": "Energy", "aliases": ["suzlon", "wind energy", "green power"]},
    {"symbol": "IREDA.NS", "name": "Indian Renewable Energy Dev Agency", "sector": "Energy", "aliases": ["ireda", "green finance"]},
    {"symbol": "WAAREEENER.NS", "name": "Waaree Energies Ltd", "sector": "Energy", "aliases": ["waaree", "waaree energies", "solar panels"]},
    {"symbol": "PREMIERENE.NS", "name": "Premier Energies Ltd", "sector": "Energy", "aliases": ["premier energies", "premier", "solar cells"]},
    {"symbol": "TATAPOWER.NS", "name": "Tata Power Company Ltd", "sector": "Energy", "aliases": ["tata power", "ev charging", "solar"]},
    {"symbol": "ADANIGREEN.NS", "name": "Adani Green Energy Ltd", "sector": "Energy", "aliases": ["adani green", "solar"]},
    {"symbol": "ADANIPOWER.NS", "name": "Adani Power Ltd", "sector": "Energy", "aliases": ["adani power"]},
    {"symbol": "NHPC.NS", "name": "NHPC Ltd", "sector": "Energy", "aliases": ["nhpc", "hydro power"]},
    {"symbol": "RECLTD.NS", "name": "REC Ltd", "sector": "Finance", "aliases": ["rec", "rural electrification"]},
    {"symbol": "PFC.NS", "name": "Power Finance Corp Ltd", "sector": "Finance", "aliases": ["pfc", "power finance"]},

    # --- Defense, Aerospace & Shipbuilding ---
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics Ltd", "sector": "Defense", "aliases": ["hal", "tejas", "defense", "fighter jet"]},
    {"symbol": "BEL.NS", "name": "Bharat Electronics Ltd", "sector": "Defense", "aliases": ["bel", "defense", "radar"]},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders Ltd", "sector": "Defense", "aliases": ["mazagon", "mazdock", "shipbuilders", "submarines"]},
    {"symbol": "COCHINSHIP.NS", "name": "Cochin Shipyard Ltd", "sector": "Defense", "aliases": ["cochin shipyard", "cochinship"]},
    {"symbol": "GRSE.NS", "name": "Garden Reach Shipbuilders Ltd", "sector": "Defense", "aliases": ["grse", "garden reach"]},
    {"symbol": "BDL.NS", "name": "Bharat Dynamics Ltd", "sector": "Defense", "aliases": ["bdl", "missiles"]},
    {"symbol": "ZENTEC.NS", "name": "Zen Technologies Ltd", "sector": "Defense", "aliases": ["zen", "zen tech", "drones"]},
    {"symbol": "DATAPATTNS.NS", "name": "Data Patterns (India) Ltd", "sector": "Defense", "aliases": ["data patterns", "datapattns"]},

    # --- Railways & PSUs ---
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance Corp", "sector": "Railways", "aliases": ["irfc", "railway finance"]},
    {"symbol": "IRCTC.NS", "name": "Indian Railway Catering & Tourism Corp", "sector": "Railways", "aliases": ["irctc", "railway booking"]},
    {"symbol": "RVNL.NS", "name": "Rail Vikas Nigam Ltd", "sector": "Railways", "aliases": ["rvnl", "rail vikas"]},
    {"symbol": "RAILTEL.NS", "name": "RailTel Corp of India Ltd", "sector": "Railways", "aliases": ["railtel", "railway wifi"]},
    {"symbol": "TITAGARH.NS", "name": "Titagarh Rail Systems Ltd", "sector": "Railways", "aliases": ["titagarh", "wagons", "trains"]},
    {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals Ltd", "sector": "Energy", "aliases": ["bhel", "turbines"]},

    # --- Market Infrastructure & Broking ---
    {"symbol": "BSE.NS", "name": "BSE Ltd (Bombay Stock Exchange)", "sector": "Finance", "aliases": ["bse", "stock exchange"]},
    {"symbol": "CDSL.NS", "name": "Central Depository Services Ltd", "sector": "Finance", "aliases": ["cdsl", "demat"]},
    {"symbol": "MCX.NS", "name": "Multi Commodity Exchange of India", "sector": "Finance", "aliases": ["mcx", "gold futures"]},
    {"symbol": "ANGELONE.NS", "name": "Angel One Ltd", "sector": "Finance", "aliases": ["angel one", "angel broking"]},
    {"symbol": "POLICYBZR.NS", "name": "PB Fintech Ltd (Policybazaar)", "sector": "Finance", "aliases": ["policybazaar", "pb fintech"]},

    # --- Major Pharma ---
    {"symbol": "CIPLA.NS", "name": "Cipla Ltd", "sector": "Pharma", "aliases": ["cipla"]},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharma", "aliases": ["dr reddy", "drreddy"]},
    {"symbol": "MANKIND.NS", "name": "Mankind Pharma Ltd", "sector": "Pharma", "aliases": ["mankind", "manforce"]},
    {"symbol": "ZYDUSLIFE.NS", "name": "Zydus Lifesciences Ltd", "sector": "Pharma", "aliases": ["zydus", "cadila"]},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories Ltd", "sector": "Pharma", "aliases": ["divis", "divi"]},

    # --- Electronics & EMS ---
    {"symbol": "DIXON.NS", "name": "Dixon Technologies (India) Ltd", "sector": "Consumer", "aliases": ["dixon", "ems", "electronics"]},
    {"symbol": "POLYCAB.NS", "name": "Polycab India Ltd", "sector": "Infra", "aliases": ["polycab", "cables", "wires"]},
    {"symbol": "KAYNES.NS", "name": "Kaynes Technology India Ltd", "sector": "IT", "aliases": ["kaynes", "semiconductor"]},

    # --- Commodity & ETFs ---
    {"symbol": "GOLDBEES.NS", "name": "Nippon India ETF Gold BeES", "sector": "Commodity / Gold", "aliases": ["gold", "goldbees", "gold bees", "sona"]},
    {"symbol": "SILVERBEES.NS", "name": "Nippon India ETF Silver BeES", "sector": "Commodity / Silver", "aliases": ["silver", "silverbees", "chandi"]},
    {"symbol": "NIFTYBEES.NS", "name": "Nippon India ETF Nifty BeES", "sector": "Index ETF", "aliases": ["nifty", "niftybees", "nifty 50"]},
    {"symbol": "BANKBEES.NS", "name": "Nippon India ETF Bank BeES", "sector": "Index ETF", "aliases": ["bankbees", "bank nifty"]}
]


def search_stock_directory(query: str, limit: int = 10) -> List[Dict]:
    """
    Smart multi-match search across ticker symbol, full company name, and aliases.
    If no directory matches found, creates a dynamic fallback symbol.
    """
    if not query:
        return STOCK_DIRECTORY[:limit]

    q = query.strip().lower()
    clean_q = q.replace(".ns", "").replace(".bo", "").replace(" ", "")

    exact_matches = []
    symbol_matches = []
    name_matches = []
    alias_matches = []

    for item in STOCK_DIRECTORY:
        sym_clean = item["symbol"].replace(".NS", "").replace(".BO", "").lower()
        name_clean = item["name"].lower()
        aliases = [a.lower() for a in item.get("aliases", [])]

        if clean_q == sym_clean:
            exact_matches.append(item)
        elif sym_clean.startswith(clean_q):
            symbol_matches.append(item)
        elif clean_q in name_clean:
            name_matches.append(item)
        elif any(clean_q in a for a in aliases):
            alias_matches.append(item)

    combined = exact_matches + symbol_matches + name_matches + alias_matches
    # Deduplicate preserving order
    seen = set()
    results = []
    for c in combined:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            results.append(c)

    # If user entered an exact custom ticker not in directory (e.g. TRENT.NS or AAPL)
    if not results and query:
        custom_sym = query.strip().upper()
        if not custom_sym.endswith(".NS") and not custom_sym.endswith(".BO") and "." not in custom_sym:
            custom_sym = f"{custom_sym}.NS"
        results.append({
            "symbol": custom_sym,
            "name": f"Custom Asset ({custom_sym})",
            "sector": "Equities",
            "aliases": [query.lower()]
        })

    return results[:limit]
