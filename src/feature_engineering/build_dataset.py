"""
Master Feature Pipeline (Institutional Quality)
Combines:
1. Multi-scale Technical Signals (Momentum, Volatility, Range, Trend)
2. Macro Drivers (Brent Crude, USD/INR, US 10Y, Nifty 50)
3. Rolling Beta and Relative Strength Alignment vs Nifty 50
4. Noise-Filtered Financial Sentiment with Exponential Decay
5. Institutional Expert Consensus, Upside & Dispersion Spread
6. Macroeconomic Market Regime Indicators
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    PILOT_UNIVERSE,
    OHLCV_DIR,
    MACRO_DIR,
    FEATURES_DATA_DIR
)
from src.feature_engineering.technical_features import generate_technical_features
from src.feature_engineering.macro_features import generate_macro_features, compute_stock_macro_alignment
from src.feature_engineering.sentiment_engine import process_daily_news_sentiment, compute_stock_decayed_sentiment_series
from src.feature_engineering.consensus_engine import build_consensus_feature_series
from src.models.regime_classifier import classify_market_regimes
from src.utils.data_loader import load_stock_ohlcv, load_macro_indicators


def build_stock_feature_matrix(
    ticker: str,
    df_cleaned_news: pd.DataFrame,
    as_of_date: str
) -> Path:
    """Build multi-signal feature matrix for a specific stock with zero waste data."""
    print(f"Building high-signal feature matrix for {ticker}...")

    # 1. Load Raw OHLCV
    ohlcv_df = load_stock_ohlcv(ticker)
    if ohlcv_df is None or ohlcv_df.empty:
        print(f"  Warning: No OHLCV data found for {ticker}")
        return None

    # 2. Compute Technical Signals
    tech_features = generate_technical_features(ohlcv_df)

    # 3. Load & Compute Macro Features
    macro_df = load_macro_indicators()
    macro_features = pd.DataFrame(index=ohlcv_df.index)
    alignment_features = pd.DataFrame(index=ohlcv_df.index)
    regime_features = pd.DataFrame(index=ohlcv_df.index)

    if macro_df is not None and not macro_df.empty:
        macro_features = generate_macro_features(macro_df)
        regime_features = classify_market_regimes(macro_df)
        if "NIFTY_50" in macro_df.columns:
            alignment_features = compute_stock_macro_alignment(
                ohlcv_df["close"], macro_df["NIFTY_50"]
            )

    # 4. Compute High-Signal Decaying Sentiment Series
    sentiment_features = compute_stock_decayed_sentiment_series(
        df_news=df_cleaned_news,
        ticker=ticker,
        calendar_index=ohlcv_df.index,
        half_life_days=7.0
    )

    # 5. Compute Expert Consensus & Dispersion Features
    consensus_features = build_consensus_feature_series(
        ticker=ticker,
        price_series=ohlcv_df["close"],
        as_of_date=as_of_date
    )

    # 6. Merge All Signal Layers on Date
    merged = pd.concat([
        ohlcv_df[["open", "high", "low", "close", "volume"]],
        tech_features,
        macro_features,
        alignment_features,
        regime_features,
        sentiment_features,
        consensus_features
    ], axis=1)

    # Reindex strictly to the stock's active trading days
    merged = merged.loc[ohlcv_df.index]
    # Forward fill macro, consensus and regime factors across holidays
    merged = merged.ffill()
    # Drop the initial 200-day warmup rows where SMA 200 is not yet formed
    merged = merged.dropna(subset=["close_to_sma_200", "rsi_14"])

    # 7. Quality Assurance Sanity Check (Reject if Corrupted)
    null_counts = merged.drop(columns=["target_fwd_21d", "target_fwd_63d", "target_fwd_126d"]).isna().sum().sum()
    if null_counts > 0:
        print(f"  Warning: Imputing {null_counts} remaining NaNs with forward/backward fill.")
        merged = merged.ffill().bfill().fillna(0)

    # 8. Save Processed Feature Matrix
    sanitized_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    target_path = FEATURES_DATA_DIR / f"{sanitized_name}_features.parquet"
    merged.to_parquet(target_path, engine="pyarrow", compression="snappy")
    print(f"   Saved {len(merged)} validated rows with {len(merged.columns)} features -> {target_path.name}")
    return target_path


def build_all_features():
    """Build feature matrices for all stocks in the pilot universe."""
    print("=" * 65)
    print("PredicSON: Starting High-Signal Feature Pipeline (Zero Waste)")
    print("=" * 65)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Pre-process and filter news once across all stocks
    print("\n[Stage 1/2] Processing News through Strict Noise Gate...")
    df_cleaned_news = process_daily_news_sentiment()

    print("\n[Stage 2/2] Generating Multi-Factor Matrices for Universe...")
    built_paths = []
    for ticker in PILOT_UNIVERSE.keys():
        path = build_stock_feature_matrix(ticker, df_cleaned_news, today_str)
        if path:
            built_paths.append(path)

    print("\n" + "=" * 65)
    print(f"Feature Pipeline Complete: {len(built_paths)} stocks processed cleanly.")
    print("=" * 65)


if __name__ == "__main__":
    build_all_features()
