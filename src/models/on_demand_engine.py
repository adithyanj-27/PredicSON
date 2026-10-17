"""
Universal On-Demand Stock Predictor & Caching Engine

Supports any stock on NSE/BSE dynamically on search. If a current feature matrix
exists it is used as-is; otherwise features are rebuilt (fetching live OHLCV only
when the raw price history is absent). Trained models are persisted by the forecaster.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OHLCV_DIR, PREDICTIONS_DIR
from src.feature_engineering.build_dataset import build_stock_feature_matrix
from src.feature_engineering.sentiment_engine import process_daily_news_sentiment
from src.models.fan_chart_generator import generate_fan_chart_data
from src.utils.data_loader import load_daily_news, feature_is_current, feature_paths, sanitize_ticker


def fetch_ohlcv(ticker: str) -> bool:
    """Fetch historical OHLCV from yfinance (adjusted prices). Returns True on success."""
    sanitized = sanitize_ticker(ticker)
    ohlcv_target = OHLCV_DIR / f"{sanitized}.parquet"
    print(f"[OnDemand] Fetching live data for {ticker}...", flush=True)
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            print(f"[OnDemand] Error: insufficient price history for {ticker}", flush=True)
            return False

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).lower() for col in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)
        df.index.name = "date"
        df.to_parquet(ohlcv_target, engine="pyarrow")
        return True
    except Exception as e:
        print(f"[OnDemand] Exception fetching {ticker}: {e}", flush=True)
        return False


def rebuild_features(ticker: str) -> bool:
    """Rebuild the feature matrix from whatever OHLCV is already on disk."""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        df_news = process_daily_news_sentiment(articles=load_daily_news(today_str))
        feat_path = build_stock_feature_matrix(ticker, df_news, today_str)
        return feat_path is not None and feat_path.exists()
    except Exception as e:
        print(f"[OnDemand] Exception building features for {ticker}: {e}", flush=True)
        return False


def get_stock_prediction(ticker: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return (history_df, forecast_df, meta), using a same-day cache when available."""
    sanitized = sanitize_ticker(ticker)
    cache_file = PREDICTIONS_DIR / f"cache_{sanitized}.json"
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Same-day prediction cache
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data.get("as_of_date") == today_str:
                hist_df = pd.DataFrame(cached_data["history"]).set_index("date")
                hist_df.index = pd.to_datetime(hist_df.index)
                fwd_df = pd.DataFrame(cached_data["forecast"]).set_index("date")
                fwd_df.index = pd.to_datetime(fwd_df.index)
                return hist_df, fwd_df, cached_data["meta"]
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            print(f"[Cache] Read error for {ticker}: {e}; regenerating...", flush=True)

    feature_file, _ = feature_paths(ticker)
    ohlcv_target = OHLCV_DIR / f"{sanitized}.parquet"

    # 2. Ensure current features exist.
    if force_refresh:
        fetch_ohlcv(ticker)          # refresh prices, then rebuild
        rebuild_features(ticker)
    elif not feature_file.exists() or not feature_is_current(ticker):
        if not ohlcv_target.exists():
            fetch_ohlcv(ticker)      # only fetch when we have no price history at all
        rebuild_features(ticker)

    if not feature_file.exists():
        raise ValueError(f"Unable to fetch valid market data for ticker '{ticker}'. Please verify the symbol.")

    # 3. Generate the fan chart / probabilistic trajectory
    hist_df, fwd_df, meta = generate_fan_chart_data(ticker)

    # 4. Persist the same-day cache
    try:
        hist_records = hist_df.reset_index().copy()
        date_col_hist = [c for c in hist_records.columns if str(c).lower() == "date"]
        if date_col_hist:
            hist_records.rename(columns={date_col_hist[0]: "date"}, inplace=True)
        else:
            hist_records.rename(columns={hist_records.columns[0]: "date"}, inplace=True)
        hist_records["date"] = hist_records["date"].astype(str)

        fwd_records = fwd_df.reset_index().copy()
        date_col_fwd = [c for c in fwd_records.columns if str(c).lower() == "date"]
        if date_col_fwd:
            fwd_records.rename(columns={date_col_fwd[0]: "date"}, inplace=True)
        else:
            fwd_records.rename(columns={fwd_records.columns[0]: "date"}, inplace=True)
        fwd_records["date"] = fwd_records["date"].astype(str)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker,
                "as_of_date": today_str,
                "history": hist_records.to_dict(orient="records"),
                "forecast": fwd_records.to_dict(orient="records"),
                "meta": meta,
            }, f, indent=2)
        print(f"[Cache] Saved prediction cache for {ticker} -> {cache_file.name}", flush=True)
    except Exception as e:
        print(f"[Cache] Failed to write cache for {ticker}: {e}", flush=True)

    return hist_df, fwd_df, meta
