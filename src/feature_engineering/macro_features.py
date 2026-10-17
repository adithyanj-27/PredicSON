"""
Macro & Cross-Asset Feature Engineering Module
Computes macro trend signals (Oil, Currency, Yields, VIX) and calculates
stock-level beta and relative strength against benchmark Nifty 50.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MACRO_DIR, FEATURES_DATA_DIR


def generate_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate macroeconomic feature matrix from consolidated macro series.
    Columns include: NIFTY_50, BRENT_CRUDE, USD_INR, INDIA_VIX, US_10Y_YIELD, GOLD_FUTURES.
    """
    m_feat = pd.DataFrame(index=macro_df.index)

    # 1. Brent Crude Dynamics (Crucial for Indian macro & inflation)
    if "BRENT_CRUDE" in macro_df.columns:
        crude = macro_df["BRENT_CRUDE"]
        m_feat["crude_ret_21d"] = np.log(crude / crude.shift(21))
        m_feat["crude_ret_63d"] = np.log(crude / crude.shift(63))
        m_feat["crude_sma50_dist"] = (crude - crude.rolling(50).mean()) / crude.rolling(50).mean()

    # 2. USD/INR Currency Pressure
    if "USD_INR" in macro_df.columns:
        usdinr = macro_df["USD_INR"]
        m_feat["usdinr_ret_21d"] = np.log(usdinr / usdinr.shift(21))
        m_feat["usdinr_vol_20d"] = usdinr.pct_change().rolling(20).std() * np.sqrt(252)

    # 3. India VIX (Market Sentiment & Panic Indicator)
    if "INDIA_VIX" in macro_df.columns:
        vix = macro_df["INDIA_VIX"]
        m_feat["vix_level"] = vix
        m_feat["vix_change_20d"] = vix - vix.shift(20)
        # Percentile rank of current VIX over the trailing year (vectorised).
        m_feat["vix_pct_rank_252d"] = vix.rolling(252, min_periods=60).apply(
            lambda x: float((x <= x[-1]).mean()), raw=True
        )

    # 4. US 10-Year Treasury Yield (Global Liquidity & FII Driver)
    if "US_10Y_YIELD" in macro_df.columns:
        us10y = macro_df["US_10Y_YIELD"]
        m_feat["us10y_change_63d"] = us10y - us10y.shift(63)

    # 5. Benchmark Nifty 50 Trend
    if "NIFTY_50" in macro_df.columns:
        nifty = macro_df["NIFTY_50"]
        m_feat["nifty_ret_21d"] = np.log(nifty / nifty.shift(21))
        m_feat["nifty_ret_63d"] = np.log(nifty / nifty.shift(63))
        m_feat["nifty_sma200_dist"] = (nifty - nifty.rolling(200).mean()) / nifty.rolling(200).mean()

    return m_feat


def compute_stock_macro_alignment(stock_close: pd.Series, nifty_close: pd.Series) -> pd.DataFrame:
    """Calculate rolling Beta and Relative Strength vs Nifty 50."""
    combined = pd.DataFrame({"stock": stock_close, "nifty": nifty_close}).dropna()
    stock_ret = combined["stock"].pct_change()
    nifty_ret = combined["nifty"].pct_change()

    # Rolling 60-day covariance & variance for Beta
    rolling_cov = stock_ret.rolling(60).cov(nifty_ret)
    rolling_var = nifty_ret.rolling(60).var()
    beta_60d = rolling_cov / (rolling_var + 1e-9)

    # Relative strength (excess return over Nifty)
    rs_21d = np.log(combined["stock"] / combined["stock"].shift(21)) - np.log(combined["nifty"] / combined["nifty"].shift(21))
    rs_63d = np.log(combined["stock"] / combined["stock"].shift(63)) - np.log(combined["nifty"] / combined["nifty"].shift(63))

    alignment = pd.DataFrame(index=combined.index)
    alignment["rolling_beta_60d"] = beta_60d
    alignment["rel_strength_nifty_21d"] = rs_21d
    alignment["rel_strength_nifty_63d"] = rs_63d
    return alignment
