"""
Technical Feature Engineering Module (Institutional Alpha Edition)
Calculates:
1. Multi-scale Momentum & Returns (5d, 10d, 21d, 63d, 126d)
2. Moving Average Distance & Volatility Z-Score (Distance / ATR)
3. Chaikin Money Flow (CMF) - Institutional volume accumulation vs distribution
4. Wilder's EWM RSI (14) & MACD histogram
5. Realized Volatility & Parkinson High-Low Volatility
6. Rolling 63-day Sharpe Ratio (Quality of Trend)
7. Forward Multi-Horizon Target Labels (1M, 3M, 6M)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using standard Wilder's EWM smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate Moving Average Convergence Divergence."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF): Measures institutional accumulation/distribution pressure.
    Values > +0.10 indicate institutional buying; values < -0.10 indicate selling.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Money Flow Multiplier: [(Close - Low) - (High - Close)] / (High - Low)
    mf_multiplier = ((close - low) - (high - close)) / ((high - low) + 1e-9)
    mf_volume = mf_multiplier * volume

    cmf = mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)
    return cmf


def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive multi-factor feature matrix from raw OHLCV DataFrame.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1. Multi-horizon Returns (Log Returns)
    for w in [5, 10, 21, 63, 126]:
        feat[f"return_{w}d"] = np.log(close / close.shift(w))

    # 2. Moving Average Distance (Trend Ratios)
    for sma_len in [20, 50, 200]:
        sma = close.rolling(sma_len).mean()
        feat[f"close_to_sma_{sma_len}"] = (close - sma) / (sma + 1e-9)

    # 3. Average True Range (ATR) & Volatility Z-Scores
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    feat["atr_14"] = atr_14
    feat["atr_pct"] = atr_14 / (close + 1e-9)

    # Volatility Z-Score: How many ATRs away from the 50-day moving average is price?
    sma_50 = close.rolling(50).mean()
    feat["vol_z_score_50d"] = (close - sma_50) / (atr_14 + 1e-9)

    # 4. Momentum & Oscillators
    feat["rsi_14"] = calculate_rsi(close, 14)
    macd, sig, hist = calculate_macd(close)
    feat["macd"] = macd
    feat["macd_signal"] = sig
    feat["macd_hist"] = hist

    # 5. Volatility (Realized & Parkinson High-Low)
    daily_returns = close.pct_change()
    feat["realized_vol_20d"] = daily_returns.rolling(20).std() * np.sqrt(252)
    feat["realized_vol_60d"] = daily_returns.rolling(60).std() * np.sqrt(252)

    # Parkinson Volatility: ln(High/Low)^2 / (4 * ln(2))
    hl_ratio = np.log(high / (low + 1e-9)) ** 2
    feat["parkinson_vol_20d"] = np.sqrt(hl_ratio.rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    feat["bb_width"] = (upper_band - lower_band) / (sma_20 + 1e-9)
    feat["bb_percent"] = (close - lower_band) / (upper_band - lower_band + 1e-9)

    # 6. Institutional Flow & Volume
    vol_sma_20 = volume.rolling(20).mean()
    feat["volume_ratio_20d"] = volume / (vol_sma_20 + 1e-9)
    feat["chaikin_money_flow_20d"] = calculate_chaikin_money_flow(df, 20)

    # 7. Quality of Trend: Rolling 63-day Sharpe Ratio
    rolling_mean_ret = daily_returns.rolling(63).mean() * 252
    rolling_vol = daily_returns.rolling(63).std() * np.sqrt(252)
    feat["trend_sharpe_63d"] = rolling_mean_ret / (rolling_vol + 1e-9)

    # 8. 52-Week Range Proximity
    high_252 = high.rolling(252, min_periods=60).max()
    low_252 = low.rolling(252, min_periods=60).min()
    feat["dist_52w_high"] = (high_252 - close) / (high_252 + 1e-9)
    feat["dist_52w_low"] = (close - low_252) / (low_252 + 1e-9)

    # 9. Future Targets (for model training)
    feat["target_fwd_21d"] = np.log(close.shift(-21) / close)
    feat["target_fwd_63d"] = np.log(close.shift(-63) / close)
    feat["target_fwd_126d"] = np.log(close.shift(-126) / close)

    return feat
