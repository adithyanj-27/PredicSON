"""
Master Feature Pipeline
Combines:
1. Multi-scale Technical Signals (Momentum, Volatility, Range, Trend)
2. Microstructure Alpha Signals (Chaikin Money Flow, Volatility Z-Scores, Trend Sharpe)
3. Macro Drivers (Brent Crude, USD/INR, US 10Y, Nifty 50)
4. Cross-Sectional Alpha & Momentum Acceleration vs Nifty 50
5. Noise-Filtered Financial Sentiment with Exponential Decay
6. Expert Consensus, Upside & Dispersion Spread (static, point-in-time snapshot)
7. Fundamental Balance Sheet Health, Growth, Margins & Valuation Multiples (static snapshot)
8. Macroeconomic Market Regime Indicators

The consensus (6) and fundamental (7) blocks are a single "as-of" snapshot broadcast
flat across history; they carry no historical variation and are therefore excluded
from model fitting (see quantile_forecaster.drop_constant_columns). They remain in
the matrix so the narrative layer and the live consensus overlay can read them.
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, FEATURES_DATA_DIR, FEATURE_SCHEMA_VERSION
from src.feature_engineering.technical_features import generate_technical_features
from src.feature_engineering.macro_features import generate_macro_features, compute_stock_macro_alignment
from src.feature_engineering.alpha_engine import generate_alpha_signals
from src.feature_engineering.sentiment_engine import process_daily_news_sentiment, compute_stock_decayed_sentiment_series
from src.feature_engineering.consensus_engine import build_consensus_feature_series
from src.feature_engineering.fundamental_features import build_fundamental_feature_series
from src.models.regime_classifier import classify_market_regimes
from src.utils.data_loader import (
    load_stock_ohlcv,
    load_macro_indicators,
    sanitize_ticker,
    write_feature_meta,
)

# Columns that are legitimately zero when a stock has no news coverage.
_SENTIMENT_COLS = ["news_sentiment_score", "news_uncertainty_score", "news_volume_count"]


def build_stock_feature_matrix(ticker: str, df_cleaned_news: pd.DataFrame, as_of_date: str) -> Path:
    """Build the multi-signal feature matrix for a specific stock."""
    print(f"Building feature matrix for {ticker}...")

    # 1. Load Raw OHLCV (adjusted prices via the loader)
    ohlcv_df = load_stock_ohlcv(ticker)
    if ohlcv_df is None or ohlcv_df.empty:
        print(f"  Warning: No OHLCV data found for {ticker}")
        return None

    # 2. Technical signals
    tech_features = generate_technical_features(ohlcv_df)

    # 3. Macro features + regime + alignment
    macro_df = load_macro_indicators()
    macro_features = pd.DataFrame(index=ohlcv_df.index)
    alignment_features = pd.DataFrame(index=ohlcv_df.index)
    regime_features = pd.DataFrame(index=ohlcv_df.index)

    nifty_series = None
    if macro_df is not None and not macro_df.empty:
        macro_features = generate_macro_features(macro_df)
        regime_features = classify_market_regimes(macro_df)
        if "NIFTY_50" in macro_df.columns:
            nifty_series = macro_df["NIFTY_50"]
            alignment_features = compute_stock_macro_alignment(ohlcv_df["close"], nifty_series)

    # 4. Cross-sectional alpha & momentum acceleration
    alpha_features = generate_alpha_signals(ohlcv_df, nifty_series)

    # 5. Decaying sentiment series
    sentiment_features = compute_stock_decayed_sentiment_series(
        df_news=df_cleaned_news,
        ticker=ticker,
        calendar_index=ohlcv_df.index,
        half_life_days=7.0,
    )

    # 6. Expert consensus & dispersion (static snapshot)
    consensus_features = build_consensus_feature_series(
        ticker=ticker, price_series=ohlcv_df["close"], as_of_date=as_of_date
    )

    # 7. Fundamental valuation & balance-sheet features (static snapshot)
    fundamental_features = build_fundamental_feature_series(
        ticker=ticker, calendar_index=ohlcv_df.index, as_of_date=as_of_date
    )

    # 8. Merge all signal layers on date
    merged = pd.concat([
        ohlcv_df[["open", "high", "low", "close", "volume"]],
        tech_features,
        macro_features,
        alignment_features,
        alpha_features,
        regime_features,
        sentiment_features,
        consensus_features,
        fundamental_features,
    ], axis=1)

    merged = merged.loc[ohlcv_df.index]
    merged = merged.ffill()

    # Drop the initial warmup rows where long look-backs are not yet formed.
    merged = merged.dropna(subset=["close_to_sma_200", "rsi_14"])

    # Target columns are intentionally NaN for the most recent rows (no future
    # price yet), so they are excluded from the completeness check below.
    target_cols = [c for c in merged.columns if c.startswith("target_fwd_")]
    feature_block = merged.drop(columns=target_cols)

    # Sentiment is genuinely zero when a stock has no qualifying news.
    for col in _SENTIMENT_COLS:
        if col in feature_block.columns:
            feature_block[col] = feature_block[col].fillna(0.0)

    # Drop any residual warmup rows rather than fabricating values with 0.
    valid_index = feature_block.dropna().index
    merged = merged.loc[valid_index]

    # 9. Save processed feature matrix + version metadata
    target_path = FEATURES_DATA_DIR / f"{sanitize_ticker(ticker)}_features.parquet"
    merged.to_parquet(target_path, engine="pyarrow", compression="snappy")
    write_feature_meta(ticker, n_rows=len(merged), n_cols=len(merged.columns))
    print(f"   Saved {len(merged)} validated rows x {len(merged.columns)} cols -> {target_path.name}")
    return target_path


def build_all_features():
    """Build feature matrices for all stocks in the pilot universe."""
    print("=" * 65)
    print(f"PredicSON: Feature Pipeline (schema v{FEATURE_SCHEMA_VERSION})")
    print("=" * 65)

    today_str = datetime.now().strftime("%Y-%m-%d")

    print("\n[Stage 1/2] Processing News through Noise Gate...")
    df_cleaned_news = process_daily_news_sentiment()

    print("\n[Stage 2/2] Generating feature matrices for the universe...")
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
