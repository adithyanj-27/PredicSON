"""
PredicSON Configuration Settings
Centralized definitions for Stock Universe, News Feeds, and Storage Paths.
"""

from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Root Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"
NEWS_DIR = RAW_DATA_DIR / "news"

# Ensure directories exist
for p in [NEWS_DIR, PREDICTIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Tracked universe: Core NSE large-caps and market leaders across major sectors
PILOT_UNIVERSE = {
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy/Telecom/Retail", "bse_code": "500325"},
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "bse_code": "532540"},
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking & Financial Services", "bse_code": "500180"},
    "INFY.NS": {"name": "Infosys", "sector": "Information Technology", "bse_code": "500209"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking & Financial Services", "bse_code": "532174"},
    "BHARTIARTL.NS": {"name": "Bharti Airtel", "sector": "Telecommunications", "bse_code": "532454"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Public Sector Banking", "bse_code": "500112"},
    "ITC.NS": {"name": "ITC Limited", "sector": "FMCG / Diversified", "bse_code": "500875"},
    "LT.NS": {"name": "Larsen & Toubro", "sector": "Infrastructure / Capital Goods", "bse_code": "500510"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG / Consumer Goods", "bse_code": "500696"},

    # Financial Services & Banking
    "BAJFINANCE.NS": {"name": "Bajaj Finance", "sector": "NBFC / Retail Lending", "bse_code": "500034"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Private Banking", "bse_code": "500247"},
    "AXISBANK.NS": {"name": "Axis Bank", "sector": "Private Banking", "bse_code": "532215"},

    # Automobiles & Mobility
    "MARUTI.NS": {"name": "Maruti Suzuki India", "sector": "Automobile / Passenger Vehicles", "bse_code": "532500"},
    "M&M.NS": {"name": "Mahindra & Mahindra", "sector": "Automobile / Utility & Tractors", "bse_code": "500520"},
    "TATAMOTORS.NS": {"name": "Tata Motors", "sector": "Automobile / EV & Commercial", "bse_code": "500570"},
    "BAJAJ-AUTO.NS": {"name": "Bajaj Auto", "sector": "Automobile / Two-Wheelers", "bse_code": "532977"},

    # Healthcare & Pharmaceuticals
    "SUNPHARMA.NS": {"name": "Sun Pharmaceutical", "sector": "Healthcare / Specialty Pharma", "bse_code": "524715"},
    "CIPLA.NS": {"name": "Cipla", "sector": "Healthcare / Generic Pharma", "bse_code": "500087"},

    # Metals & Materials
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Metals & Mining", "bse_code": "500470"},
    "HINDALCO.NS": {"name": "Hindalco Industries", "sector": "Metals / Aluminum & Copper", "bse_code": "500440"},
    "ULTRACEMCO.NS": {"name": "UltraTech Cement", "sector": "Materials / Cement & Infra", "bse_code": "532538"},

    # Energy, Utilities & Power
    "NTPC.NS": {"name": "NTPC Limited", "sector": "Power Generation / Green Energy", "bse_code": "532555"},
    "POWERGRID.NS": {"name": "Power Grid Corporation", "sector": "Power Transmission", "bse_code": "532898"},
    "ONGC.NS": {"name": "Oil & Natural Gas Corporation", "sector": "Upstream Oil & Gas", "bse_code": "500312"},
    "TATAPOWER.NS": {"name": "Tata Power", "sector": "Power Generation & EV Infra", "bse_code": "500400"},
    "COALINDIA.NS": {"name": "Coal India", "sector": "Energy & Mining", "bse_code": "533278"},
    "SUZLON.NS": {"name": "Suzlon Energy", "sector": "Green Energy / Wind", "bse_code": "532667"},

    # Consumer Discretionary & Retail
    "TITAN.NS": {"name": "Titan Company", "sector": "Consumer Discretionary / Jewelry", "bse_code": "500114"},
    "ASIANPAINT.NS": {"name": "Asian Paints", "sector": "Consumer / Decorative Paints", "bse_code": "500820"},
    "TRENT.NS": {"name": "Trent Ltd (Westside & Zudio)", "sector": "Consumer / Retail", "bse_code": "500251"},
    "ZOMATO.NS": {"name": "Zomato", "sector": "Consumer / Tech Platform", "bse_code": "543320"},

    # Defense & Infrastructure
    "HAL.NS": {"name": "Hindustan Aeronautics Ltd", "sector": "Defense & Aerospace", "bse_code": "541154"},
    "BEL.NS": {"name": "Bharat Electronics Ltd", "sector": "Defense Electronics", "bse_code": "500049"},
    "IRFC.NS": {"name": "Indian Railway Finance Corp", "sector": "Railways & Infra Finance", "bse_code": "543257"},
    "RVNL.NS": {"name": "Rail Vikas Nigam Ltd", "sector": "Railways Infrastructure", "bse_code": "542649"},
    "ADANIENT.NS": {"name": "Adani Enterprises", "sector": "Diversified Conglomerate", "bse_code": "512599"},
    "ADANIPORTS.NS": {"name": "Adani Ports & SEZ", "sector": "Infrastructure / Ports", "bse_code": "532921"},
    "WIPRO.NS": {"name": "Wipro", "sector": "Information Technology", "bse_code": "507685"},
    "BANKBARODA.NS": {"name": "Bank of Baroda", "sector": "Public Sector Banking", "bse_code": "532134"}
}

# Free Indian Financial News RSS Feeds for Dalal Street Wire
NEWS_RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "economictimes_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economictimes_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss"
}
