"""
Market Data Ingestion Engine (Free Tier)
Fetches historical OHLCV data for equities and macro indicators using yfinance.
Saves data efficiently in Parquet format.

Prices are fetched with auto_adjust=True so the stored `close` column is already
adjusted for splits and dividends (total-return basis). Return-based modelling
must not mix raw and adjusted prices across sources.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    PILOT_UNIVERSE,
    MACRO_TICKERS,
    OHLCV_DIR,
    MACRO_DIR
)


def clean_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present and standardize column names."""
    if isinstance(df.columns, pd.MultiIndex):
        # Extract first level if yfinance provides ticker level
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure index is datetime with timezone stripped for consistency
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"
    return df


def fetch_and_save_stock_history(ticker: str, start_date: str = "2018-01-01") -> Path:
    """Fetch complete historical OHLCV for a specific NSE/BSE ticker."""
    print(f"Fetching {ticker} from {start_date}...")
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, auto_adjust=True)

        if df.empty:
            print(f"Warning: No data returned for {ticker}")
            return None

        df = clean_yfinance_df(df)

        # Save as parquet
        sanitized_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
        target_path = OHLCV_DIR / f"{sanitized_name}.parquet"
        df.to_parquet(target_path, engine="pyarrow", compression="snappy")
        print(f" Saved {len(df)} rows for {ticker} -> {target_path.name}")
        return target_path
    except Exception as e:
        print(f" Error fetching {ticker}: {e}")
        return None


def fetch_and_save_macro_data(start_date: str = "2018-01-01") -> Path:
    """Fetch macro indicators (Crude, Currency, VIX, US 10Y, Nifty) into unified dataset."""
    print(f"\nFetching Macro Indicators from {start_date}...")
    macro_frames = {}

    for ticker, label in MACRO_TICKERS.items():
        try:
            print(f"  Fetching macro: {label} ({ticker})...")
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, auto_adjust=True)
            if not df.empty:
                df = clean_yfinance_df(df)
                # Keep Close price as the primary macro signal
                macro_frames[label] = df["close"].rename(label)
        except Exception as e:
            print(f"  Error fetching {label}: {e}")

    if not macro_frames:
        print("Warning: No macro data retrieved.")
        return None

    # Merge all macro series on date
    combined_macro = pd.DataFrame(macro_frames)
    combined_macro.sort_index(inplace=True)
    # Forward-fill only: markets close on different holidays. We deliberately do
    # NOT back-fill, which would copy future values backwards into the past.
    combined_macro = combined_macro.ffill()

    target_path = MACRO_DIR / "macro_indicators.parquet"
    combined_macro.to_parquet(target_path, engine="pyarrow", compression="snappy")
    print(f" Saved consolidated macro indicators ({len(combined_macro)} rows) -> {target_path.name}")
    return target_path


def run_full_market_ingestion(start_date: str = "2018-01-01"):
    """Run ingestion for all pilot stocks and macro drivers."""
    print("=" * 60)
    print("PredicSON: Starting Zero-Cost Market & Macro Data Ingestion")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Pilot Universe Stocks
    print(f"\n[1/2] Ingesting {len(PILOT_UNIVERSE)} Pilot Stocks...")
    saved_stocks = []
    for ticker in PILOT_UNIVERSE.keys():
        path = fetch_and_save_stock_history(ticker, start_date=start_date)
        if path:
            saved_stocks.append(path)

    # 2. Macro Drivers
    print(f"\n[2/2] Ingesting Macro Factors...")
    macro_path = fetch_and_save_macro_data(start_date=start_date)

    print("\n" + "=" * 60)
    print(f"Ingestion Completed: {len(saved_stocks)} stocks & macro indicators saved.")
    print("=" * 60)


if __name__ == "__main__":
    run_full_market_ingestion()
