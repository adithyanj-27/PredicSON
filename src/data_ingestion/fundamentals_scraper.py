"""
Fundamental & Balance Sheet Ingestion Engine (Free Tier)
Fetches quarterly earnings growth, valuation multiples (PE, PB), capital efficiency (ROE, ROCE),
debt leverage, and ownership metrics via yfinance.
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, RAW_DATA_DIR

FUNDAMENTALS_DIR = RAW_DATA_DIR / "fundamentals"
FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)


def _to_float(value):
    """Coerce a yfinance field to float, returning None for missing/invalid values."""
    try:
        if value is None:
            return None
        f = float(value)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def fetch_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetch complete balance sheet, valuation, and governance metrics for a stock."""
    print(f"Fetching fundamental balance sheet metrics for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        record = {
            "ticker": ticker,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "valuation": {
                "pe_trailing": _to_float(info.get("trailingPE")),
                "pe_forward": _to_float(info.get("forwardPE")),
                "pb_ratio": _to_float(info.get("priceToBook")),
                "peg_ratio": _to_float(info.get("pegRatio")),
                "ev_to_ebitda": _to_float(info.get("enterpriseToEbitda")),
            },
            "growth": {
                "revenue_growth_yoy": _to_float(info.get("revenueGrowth")),
                "earnings_growth_yoy": _to_float(info.get("earningsGrowth")),
                "profit_margins": _to_float(info.get("profitMargins")),
                "operating_margins": _to_float(info.get("operatingMargins")),
            },
            "solvency_and_quality": {
                "roe": _to_float(info.get("returnOnEquity")),
                "roa": _to_float(info.get("returnOnAssets")),
                "debt_to_equity": _to_float(info.get("debtToEquity")),
                "current_ratio": _to_float(info.get("currentRatio")),
                "free_cash_flow": _to_float(info.get("freeCashflow")),
            },
            "ownership": {
                # yfinance maps promoter/insider stake to heldPercentInsiders for NSE names.
                "insider_pct": _to_float(info.get("heldPercentInsiders")),
                "institutional_pct": _to_float(info.get("heldPercentInstitutions")),
            },
        }
        return record
    except Exception as e:
        print(f"  Error fetching fundamentals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def run_fundamentals_ingestion() -> Path:
    """Fetch fundamentals for all pilot universe stocks and save to JSON."""
    print("=" * 65)
    print("PredicSON: Starting Fundamental & Financial Health Ingestion")
    print("=" * 65)

    all_records = {}
    for ticker in PILOT_UNIVERSE.keys():
        all_records[ticker] = fetch_stock_fundamentals(ticker)

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_file = FUNDAMENTALS_DIR / f"fundamentals_{date_str}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nSaved corporate fundamentals for {len(all_records)} stocks -> {out_file.name}")
    return out_file


if __name__ == "__main__":
    run_fundamentals_ingestion()
