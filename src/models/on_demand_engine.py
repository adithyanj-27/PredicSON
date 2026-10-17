"""
Universal On-Demand Stock Predictor & Caching Engine
Supports ANY stock on NSE/BSE (or globally) dynamically on search.
If cached today -> Loads instantly in < 0.2s.
If new or outdated -> Automatically fetches live OHLCV, computes features, fits LightGBM, and caches.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OHLCV_DIR, FEATURES_DATA_DIR, PREDICTIONS_DIR, PILOT_UNIVERSE
from src.feature_engineering.build_dataset import build_stock_feature_matrix
from src.feature_engineering.sentiment_engine import process_daily_news_sentiment
from src.models.fan_chart_generator import generate_fan_chart_data
from src.utils.data_loader import load_daily_news


def fetch_and_prepare_stock(ticker: str) -> bool:
    """Fetches historical OHLCV from yfinance for any asset and builds features."""
    sanitized = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    ohlcv_target = OHLCV_DIR / f"{sanitized}.parquet"
    
    print(f"[OnDemand] Fetching live data for {ticker}...", flush=True)
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty or len(df) < 60:
            print(f"[OnDemand] Error: insufficient price history for {ticker}", flush=True)
            return False
            
        # Clean multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
            
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)
        df.to_parquet(ohlcv_target)
        
        # Build features
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_news = load_daily_news(today_str)
        df_news = process_daily_news_sentiment(daily_news)
        
        feat_path = build_stock_feature_matrix(ticker, df_news, today_str)
        return feat_path is not None and feat_path.exists()
    except Exception as e:
        print(f"[OnDemand] Exception fetching {ticker}: {e}", flush=True)
        return False


def get_stock_prediction(ticker: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Universal prediction loader with intelligent permanent caching.
    Loads in < 0.2s if cached; calculates on-demand if new.
    """
    sanitized = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    cache_file = PREDICTIONS_DIR / f"cache_{sanitized}.json"
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Check if cache exists and is fresh
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                
            # If generated today, use instantly
            if cached_data.get("as_of_date") == today_str:
                hist_df = pd.DataFrame(cached_data["history"]).set_index("date")
                hist_df.index = pd.to_datetime(hist_df.index)
                
                fwd_df = pd.DataFrame(cached_data["forecast"]).set_index("date")
                fwd_df.index = pd.to_datetime(fwd_df.index)
                
                meta = cached_data["meta"]
                return hist_df, fwd_df, meta
        except Exception as e:
            print(f"[Cache] Read error for {ticker}: {e}, regenerating...", flush=True)

    # 2. Check if feature parquet exists, else fetch live data
    feat_file = FEATURES_DATA_DIR / f"{sanitized}_features.parquet"
    if not feat_file.exists() or force_refresh:
        success = fetch_and_prepare_stock(ticker)
        if not success and not feat_file.exists():
            raise ValueError(f"Unable to fetch valid market data for ticker '{ticker}'. Please verify the symbol.")

    # 3. Generate Fan Chart & Probabilistic Trajectory
    hist_df, fwd_df, meta = generate_fan_chart_data(ticker)

    # 4. Save permanently to JSON cache
    try:
        hist_records = hist_df.reset_index().copy()
        hist_records["date"] = hist_records["date"].astype(str)
        
        fwd_records = fwd_df.reset_index().copy()
        fwd_records["date"] = fwd_records["date"].astype(str)
        
        cache_payload = {
            "ticker": ticker,
            "as_of_date": today_str,
            "history": hist_records.to_dict(orient="records"),
            "forecast": fwd_records.to_dict(orient="records"),
            "meta": meta
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2)
            
        print(f"[Cache] Successfully saved prediction cache for {ticker} -> {cache_file.name}", flush=True)
    except Exception as e:
        print(f"[Cache] Failed to write cache for {ticker}: {e}", flush=True)

    return hist_df, fwd_df, meta
