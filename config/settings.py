"""
PredicSON Configuration Settings
Centralized definitions for Stock Universe, Macro Indicators, News Feeds, and Storage Paths.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Root Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEATURES_DATA_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Raw subdirectories
OHLCV_DIR = RAW_DATA_DIR / "ohlcv"
NEWS_DIR = RAW_DATA_DIR / "news"
MACRO_DIR = RAW_DATA_DIR / "macro"
EXPERTS_DIR = RAW_DATA_DIR / "experts"

# Ensure all directories exist
for p in [OHLCV_DIR, NEWS_DIR, MACRO_DIR, EXPERTS_DIR, FEATURES_DATA_DIR, PREDICTIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Pilot Universe: Top 10 Indian Large-Caps across key sectors
# .NS indicates National Stock Exchange (NSE) on Yahoo Finance
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
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG / Consumer Goods", "bse_code": "500696"}
}

# Macro Tickers (Global & Domestic Drivers)
MACRO_TICKERS = {
    "^NSEI": "NIFTY_50",               # Benchmark index
    "^BSESN": "SENSEX",                # BSE Benchmark index
    "^INDIAVIX": "INDIA_VIX",          # Volatility index
    "BZ=F": "BRENT_CRUDE",             # Crude oil (massive for India import bill)
    "INR=X": "USD_INR",                # Currency exchange rate (USD to INR)
    "^TNX": "US_10Y_YIELD",            # US 10-Year Treasury Yield (FII driver)
    "GC=F": "GOLD_FUTURES",            # Safe haven asset
    "^GSPC": "SP500"                   # Global risk benchmark
}

# Free Indian Financial News RSS Feeds
NEWS_RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "economictimes_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economictimes_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss"
}

# API Keys (Free Tiers)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
