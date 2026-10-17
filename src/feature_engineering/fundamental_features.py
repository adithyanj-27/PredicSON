"""
Fundamental Feature Engineering Module
Transforms corporate balance sheet metrics, valuation ratios, growth rates,
and shareholding patterns into standardized modelling features.

These are a single point-in-time snapshot broadcast across the historical index;
they carry no historical variation and are therefore dropped before model fitting
(see quantile_forecaster.drop_constant_columns) and used for narrative context only.
"""

import sys
import json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

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
        "fund_institutional_holding": 0.25,
    }

    fund_file = FUNDAMENTALS_DIR / f"fundamentals_{as_of_date}.json"
    if not fund_file.exists():
        available = sorted(FUNDAMENTALS_DIR.glob("fundamentals_*.json"))
        if not available:
            return defaults
        fund_file = available[-1]

    try:
        with open(fund_file, "r", encoding="utf-8") as f:
            all_funds = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not read fundamentals for {ticker}: {e}")
        return defaults

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

    return defaults


def build_fundamental_feature_series(ticker: str, calendar_index: pd.DatetimeIndex, as_of_date: str) -> pd.DataFrame:
    """Broadcasts fundamental health metrics across the trading index (flat snapshot)."""
    df_feat = pd.DataFrame(index=calendar_index)
    feats = extract_stock_fundamental_features(ticker, as_of_date)

    for col_name, val in feats.items():
        df_feat[col_name] = val

    return df_feat
