"""
Market Regime Classification Engine
Classifies the overarching macroeconomic environment into distinct regimes:
- BULL_TREND: Nifty above 50 & 200 SMA, moderate/low VIX.
- SIDEWAYS_CONSOLIDATION: Range-bound indices, low VIX, balanced flows.
- ELEVATED_VOLATILITY: Rising VIX (>18), choppy price action.
- CRISIS_BEAR: Sharp drawdown, VIX spike (>22), severe capital flight.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MACRO_DIR
from src.utils.data_loader import load_macro_indicators


REGIME_NAMES = {
    0: "BULL_TREND",
    1: "SIDEWAYS_CONSOLIDATION",
    2: "ELEVATED_VOLATILITY",
    3: "CRISIS_BEAR"
}


def classify_market_regimes(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily regime classification from Nifty 50 and India VIX dynamics.
    Returns DataFrame aligned with macro dates containing regime features.
    """
    regimes = pd.DataFrame(index=macro_df.index)

    nifty = macro_df.get("NIFTY_50")
    vix = macro_df.get("INDIA_VIX")

    if nifty is None or vix is None:
        # Fallback neutral
        regimes["regime_code"] = 1
        regimes["is_bull_regime"] = 0
        regimes["is_crisis_regime"] = 0
        regimes["volatility_multiplier"] = 1.0
        return regimes

    # Technical trend of Nifty
    sma_50 = nifty.rolling(50, min_periods=20).mean()
    sma_200 = nifty.rolling(200, min_periods=50).mean()
    nifty_above_50 = nifty > sma_50
    nifty_above_200 = nifty > sma_200
    
    # 20-day returns and momentum
    nifty_ret_20d = nifty.pct_change(20)

    # VIX conditions
    vix_high = vix > 18.0
    vix_extreme = vix > 23.0
    vix_low = vix < 14.5

    # Determine regime code:
    # 3 = CRISIS_BEAR: VIX extreme or severe 20d drop (< -5%) below 200 SMA
    # 2 = ELEVATED_VOLATILITY: VIX > 18 or negative momentum
    # 0 = BULL_TREND: Above 50 & 200 SMA, VIX not high, positive 20d return
    # 1 = SIDEWAYS_CONSOLIDATION: Everything else

    regime_codes = []
    vol_mults = []

    for i in range(len(macro_df)):
        is_crisis = vix_extreme.iloc[i] or (not nifty_above_200.iloc[i] and nifty_ret_20d.iloc[i] < -0.06)
        is_high_vol = vix_high.iloc[i] or (not nifty_above_50.iloc[i] and nifty_ret_20d.iloc[i] < -0.02)
        is_bull = nifty_above_50.iloc[i] and nifty_above_200.iloc[i] and nifty_ret_20d.iloc[i] > 0 and not vix_high.iloc[i]

        if is_crisis:
            regime_codes.append(3)
            vol_mults.append(1.6)  # Widen prediction intervals by 60%
        elif is_high_vol:
            regime_codes.append(2)
            vol_mults.append(1.25) # Widen intervals by 25%
        elif is_bull:
            regime_codes.append(0)
            vol_mults.append(0.9)  # Tighter intervals in stable bull trends
        else:
            regime_codes.append(1) # Sideways / consolidation
            vol_mults.append(1.0)

    regimes["regime_code"] = regime_codes
    regimes["is_bull_regime"] = [1 if r == 0 else 0 for r in regime_codes]
    regimes["is_crisis_regime"] = [1 if r == 3 else 0 for r in regime_codes]
    regimes["volatility_multiplier"] = vol_mults

    return regimes


def get_current_market_regime() -> Dict[str, Any]:
    """Returns the current market regime diagnosis."""
    macro = load_macro_indicators()
    if macro is None or macro.empty:
        return {"code": 1, "name": "SIDEWAYS_CONSOLIDATION", "multiplier": 1.0}

    regimes = classify_market_regimes(macro)
    latest_row = regimes.iloc[-1]
    code = int(latest_row["regime_code"])
    return {
        "code": code,
        "name": REGIME_NAMES.get(code, "SIDEWAYS_CONSOLIDATION"),
        "is_bull": bool(latest_row["is_bull_regime"]),
        "is_crisis": bool(latest_row["is_crisis_regime"]),
        "volatility_multiplier": float(latest_row["volatility_multiplier"]),
        "as_of_date": macro.index[-1].strftime("%Y-%m-%d")
    }
