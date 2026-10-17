"""
Fundamental Feature Engineering Module
Transforms corporate balance sheet metrics, valuation ratios, growth rates,
and shareholding patterns into standardized modeling features.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import RAW_DATA_DIR

FUNDAMENTALS_DIR = RAW_DATA_DIR / "fundamentals"


def extract_stock_fundamental_features(ticker: str, as_of_date: str) -> Dict[str, float]:
    """Extract and standardize valuation, growth, and quality ratios for a given stock."""
    defaults = {
        "fund_pe_ratio": 20.0,
        "fund_peg_ratio": 1.5,
        "fund_pb_ratio": 3.0,
        "fund_rev_growth_yoy": 0.10,
        "fund_earnings_growth_yoy": 0.10,
        "fund_operating_margin": 0.15,
        "fund_promoter_holding": 0.50,
        "fund_institutional_holding": 0.25
    }

    fund_file = FUNDAMENTALS_DIR / f"fundamentals_{as_of_date}.json"
    if not fund_file.exists():
        # Fallback to most recent file in fundamentals directory
        available = sorted(FUNDAMENTALS_DIR.glob("fundamentals_*.json"))
        if available:
            fund_file = available[-1]
        else:
            return defaults

    try:
        with open(fund_file, "r", encoding="utf-8") as f:
            all_funds = json.load(f)

        data = all_funds.get(ticker, {})
        val = data.get("valuation", {})
        growth = data.get("growth", {})
        ownership = data.get("ownership", {})

        pe = val.get("pe_trailing") or val.get("pe_forward")
        if pe and pe > 0:
            defaults["fund_pe_ratio"] = float(np.clip(pe, 5.0, 150.0))

        peg = val.get("peg_ratio")
        if peg and peg > 0:
            defaults["fund_peg_ratio"] = float(np.clip(peg, 0.1, 10.0))

        pb = val.get("pb_ratio")
        if pb and pb > 0:
            defaults["fund_pb_ratio"] = float(np.clip(pb, 0.5, 30.0))

        rg = growth.get("revenue_growth_yoy")
        if rg is not None:
            defaults["fund_rev_growth_yoy"] = float(np.clip(rg, -0.5, 2.0))

        eg = growth.get("earnings_growth_yoy")
        if eg is not None:
            defaults["fund_earnings_growth_yoy"] = float(np.clip(eg, -0.5, 2.0))

        om = growth.get("operating_margins")
        if om is not None:
            defaults["fund_operating_margin"] = float(np.clip(om, -0.2, 0.6))

        ins = ownership.get("insider_pct")
        if ins is not None:
            defaults["fund_promoter_holding"] = float(np.clip(ins, 0.0, 1.0))

        inst = ownership.get("institutional_pct")
        if inst is not None:
            defaults["fund_institutional_holding"] = float(np.clip(inst, 0.0, 1.0))

    except Exception:
        pass

    return defaults


def build_fundamental_feature_series(
    ticker: str,
    calendar_index: pd.DatetimeIndex,
    as_of_date: str
) -> pd.DataFrame:
    """
    Broadcasts corporate fundamental health metrics across the historical trading index.
    Quarterly metrics provide the baseline valuation anchor for multi-month horizon predictions.
    """
    df_feat = pd.DataFrame(index=calendar_index)
    feats = extract_stock_fundamental_features(ticker, as_of_date)

    for col_name, val in feats.items():
        df_feat[col_name] = val

    return df_feat
