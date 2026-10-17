"""
Cross-Sectional Alpha & Idiosyncratic Momentum Engine
Calculates:
1. Idiosyncratic Residual Alpha (Stock Return minus Beta * Nifty Return)
2. Relative Strength Rank (Outperformance vs Benchmark over 1M, 3M, 6M)
3. Momentum Acceleration (First derivative of price change: is buying speeding up or slowing down?)
4. Volume Acceleration Ratio (Institutional volume expansion)
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def generate_alpha_signals(
    stock_df: pd.DataFrame,
    nifty_series: Optional[pd.Series]
) -> pd.DataFrame:
    """
    Computes true idiosyncratic alpha and momentum acceleration signals.
    """
    alpha_df = pd.DataFrame(index=stock_df.index)
    close = stock_df["close"]
    volume = stock_df["volume"]

    # 1. Momentum Acceleration
    # Compares 1-month return against normalized 3-month pace
    ret_21d = np.log(close / close.shift(21))
    ret_63d = np.log(close / close.shift(63))
    alpha_df["momentum_acceleration_21d"] = ret_21d - (ret_63d / 3.0)

    # 2. Volume Expansion Acceleration
    vol_sma_10 = volume.rolling(10).mean()
    vol_sma_50 = volume.rolling(50).mean()
    alpha_df["volume_expansion_ratio"] = vol_sma_10 / (vol_sma_50 + 1e-9)

    # 3. Idiosyncratic Alpha vs Nifty 50
    if nifty_series is not None and not nifty_series.empty:
        # Align on trading days
        aligned_nifty = nifty_series.reindex(stock_df.index).ffill()
        nifty_ret_21d = np.log(aligned_nifty / aligned_nifty.shift(21))
        nifty_ret_63d = np.log(aligned_nifty / aligned_nifty.shift(63))

        # Rolling 60d beta
        stock_daily = close.pct_change()
        nifty_daily = aligned_nifty.pct_change()
        cov = stock_daily.rolling(60).cov(nifty_daily)
        var = nifty_daily.rolling(60).var()
        beta = cov / (var + 1e-9)

        # Residual Alpha: Actual Return - Expected CAPM Return
        alpha_df["residual_alpha_21d"] = ret_21d - (beta * nifty_ret_21d)
        alpha_df["residual_alpha_63d"] = ret_63d - (beta * nifty_ret_63d)
        alpha_df["relative_strength_spread"] = ret_63d - nifty_ret_63d
    else:
        alpha_df["residual_alpha_21d"] = 0.0
        alpha_df["residual_alpha_63d"] = 0.0
        alpha_df["relative_strength_spread"] = 0.0

    return alpha_df
