"""
Market Regime Classification Engine
Classifies the overarching macroeconomic environment into distinct regimes:
- BULL_TREND: Nifty above 50 & 200 SMA, moderate/low VIX.
- SIDEWAYS_CONSOLIDATION: Range-bound indices, low VIX, balanced flows.
- ELEVATED_VOLATILITY: Rising VIX (>18), choppy price action.
- CRISIS_BEAR: Sharp drawdown, VIX spike (>23), severe capital flight.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.data_loader import load_macro_indicators


REGIME_NAMES = {
    0: "BULL_TREND",
    1: "SIDEWAYS_CONSOLIDATION",
    2: "ELEVATED_VOLATILITY",
    3: "CRISIS_BEAR",
}


def classify_market_regimes(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily regime classification from Nifty 50 and India VIX dynamics (vectorised)."""
    regimes = pd.DataFrame(index=macro_df.index)

    nifty = macro_df.get("NIFTY_50")
    vix = macro_df.get("INDIA_VIX")

    if nifty is None or vix is None:
        regimes["regime_code"] = 1
        regimes["is_bull_regime"] = 0
        regimes["is_crisis_regime"] = 0
        regimes["volatility_multiplier"] = 1.0
        return regimes

    sma_50 = nifty.rolling(50, min_periods=20).mean()
    sma_200 = nifty.rolling(200, min_periods=50).mean()
    nifty_above_50 = (nifty > sma_50).fillna(False)
    nifty_above_200 = (nifty > sma_200).fillna(False)
    nifty_ret_20d = nifty.pct_change(20)

    vix_high = (vix > 18.0).fillna(False)
    vix_extreme = (vix > 23.0).fillna(False)

    # 3 = CRISIS_BEAR, 2 = ELEVATED_VOLATILITY, 0 = BULL_TREND, 1 = SIDEWAYS
    is_crisis = vix_extreme | ((~nifty_above_200) & (nifty_ret_20d < -0.06))
    is_high_vol = vix_high | ((~nifty_above_50) & (nifty_ret_20d < -0.02))
    is_bull = nifty_above_50 & nifty_above_200 & (nifty_ret_20d > 0) & (~vix_high)

    is_crisis = is_crisis.fillna(False)
    is_high_vol = is_high_vol.fillna(False)
    is_bull = is_bull.fillna(False)

    regime_code = np.select([is_crisis, is_high_vol, is_bull], [3, 2, 0], default=1)
    vol_mult = np.select([is_crisis, is_high_vol, is_bull], [1.6, 1.25, 0.9], default=1.0)

    regimes["regime_code"] = regime_code.astype(int)
    regimes["is_bull_regime"] = (regime_code == 0).astype(int)
    regimes["is_crisis_regime"] = (regime_code == 3).astype(int)
    regimes["volatility_multiplier"] = vol_mult.astype(float)

    return regimes


def get_current_market_regime() -> Dict[str, Any]:
    """Returns the current market regime diagnosis."""
    macro = load_macro_indicators()
    if macro is None or macro.empty:
        return {"code": 1, "name": "SIDEWAYS_CONSOLIDATION", "volatility_multiplier": 1.0}

    regimes = classify_market_regimes(macro)
    latest_row = regimes.iloc[-1]
    code = int(latest_row["regime_code"])
    return {
        "code": code,
        "name": REGIME_NAMES.get(code, "SIDEWAYS_CONSOLIDATION"),
        "is_bull": bool(latest_row["is_bull_regime"]),
        "is_crisis": bool(latest_row["is_crisis_regime"]),
        "volatility_multiplier": float(latest_row["volatility_multiplier"]),
        "as_of_date": macro.index[-1].strftime("%Y-%m-%d"),
    }
