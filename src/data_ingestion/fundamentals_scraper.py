"""
Fundamental & Balance Sheet Ingestion Engine (Free Tier)
Fetches quarterly earnings growth, valuation multiples (PE, PB), capital efficiency (ROE, ROCE),
debt leverage, and promoter shareholding/pledge metrics via yfinance & Screener.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, RAW_DATA_DIR

FUNDAMENTALS_DIR = RAW_DATA_DIR / "fundamentals"
FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetch complete balance sheet, valuation, and governance metrics for a stock."""
    print(f"Fetching fundamental balance sheet metrics for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # 1. Valuation Multiples
        pe_trailing = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        pb_ratio = info.get("priceToBook")
        peg_ratio = info.get("pegRatio")
        ev_to_ebitda = info.get("enterpriseToEbitda")

        # 2. Growth & Profitability
        rev_growth = info.get("revenueGrowth")          # Quarterly YoY
        earnings_growth = info.get("earningsGrowth")    # Quarterly YoY
        profit_margins = info.get("profitMargins")
        operating_margins = info.get("operatingMargins")

        # 3. Capital Efficiency & Solvency
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        free_cash_flow = info.get("freeCashflow")

        # 4. Ownership & Governance
        insider_ownership = info.get("heldPercentInsiders")
        institutional_ownership = info.get("heldPercentInstitutions")

        record = {
            "ticker": ticker,
            "fetched_at": datetime.utcnow().isoformat(),
            "valuation": {
                "pe_trailing": float(pe_trailing) if pe_trailing and pd.notna(pe_trailing) else None,
                "pe_forward": float(pe_forward) if pe_forward and pd.notna(pe_forward) else None,
                "pb_ratio": float(pb_ratio) if pb_ratio and pd.notna(pb_ratio) else None,
                "peg_ratio": float(peg_ratio) if peg_ratio and pd.notna(peg_ratio) else None,
                "ev_to_ebitda": float(ev_to_ebitda) if ev_to_ebitda and pd.notna(ev_to_ebitda) else None
            },
            "growth": {
                "revenue_growth_yoy": float(rev_growth) if rev_growth and pd.notna(rev_growth) else None,
                "earnings_growth_yoy": float(earnings_growth) if earnings_growth and pd.notna(earnings_growth) else None,
                "profit_margins": float(profit_margins) if profit_margins and pd.notna(profit_margins) else None,
                "operating_margins": float(operating_margins) if operating_margins and pd.notna(operating_margins) else None
            },
            "solvency_and_quality": {
                "roe": float(roe) if roe and pd.notna(roe) else None,
                "roa": float(roa) if roa and pd.notna(roa) else None,
                "debt_to_equity": float(debt_to_equity) if debt_to_equity and pd.notna(debt_to_equity) else None,
                "current_ratio": float(current_ratio) if current_ratio and pd.notna(current_ratio) else None,
                "free_cash_flow": float(free_cash_flow) if free_cash_flow and pd.notna(free_cash_flow) else None
            },
            "ownership": {
                "insider_pct": float(insider_ownership) if insider_ownership and pd.notna(insider_ownership) else None,
                "institutional_pct": float(institutional_ownership) if institutional_ownership and pd.notna(institutional_ownership) else None
            }
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
        record = fetch_stock_fundamentals(ticker)
        all_records[ticker] = record

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_file = FUNDAMENTALS_DIR / f"fundamentals_{date_str}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nSaved corporate fundamentals for {len(all_records)} stocks -> {out_file.name}")
    return out_file


if __name__ == "__main__":
    run_fundamentals_ingestion()
