"""
Expert Consensus & Dispersion Feature Engine
Transforms institutional analyst forecasts, target price ranges, and recommendation splits
into calibrated numerical features.

These are a single point-in-time snapshot broadcast across the price index. Because
they carry no historical variation they are dropped before model fitting and used
only as (a) a live overlay on the forecast and (b) narrative context.
"""

import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.data_loader import load_expert_consensus


def extract_stock_consensus_features(ticker: str, current_price: float, as_of_date: str) -> Dict[str, float]:
    """Extracts consensus upside, dispersion (uncertainty), and rating conviction from a snapshot."""
    defaults = {
        "consensus_upside_pct": 0.0,
        "consensus_dispersion_pct": 0.0,
        "analyst_bullish_ratio": 0.50,
        "analyst_total_coverage": 0.0,
    }

    expert_records = load_expert_consensus(as_of_date)
    matched = next((r for r in expert_records if r.get("ticker") == ticker), None)
    if not matched:
        return defaults

    price_targets = matched.get("price_targets", {})
    mean_target = price_targets.get("mean")
    high_target = price_targets.get("high")
    low_target = price_targets.get("low")

    # 1. Consensus Upside / Downside (bounded to prevent extreme outliers)
    if mean_target and current_price > 0:
        upside = (mean_target - current_price) / current_price
        defaults["consensus_upside_pct"] = float(np.clip(upside, -0.5, 1.0))

    # 2. Analyst Dispersion / Disagreement Index
    if high_target and low_target and mean_target and mean_target > 0:
        dispersion = (high_target - low_target) / mean_target
        defaults["consensus_dispersion_pct"] = float(np.clip(dispersion, 0.0, 1.5))

    # 3. Recommendation Conviction (Bullish Ratio)
    recos = matched.get("recommendation_counts", {})
    if recos:
        strong_buy = float(recos.get("strongBuy", recos.get("strong_buy", 0)))
        buy = float(recos.get("buy", 0))
        hold = float(recos.get("hold", 0))
        sell = float(recos.get("sell", 0))
        strong_sell = float(recos.get("strongSell", recos.get("strong_sell", 0)))

        total_analysts = strong_buy + buy + hold + sell + strong_sell
        if total_analysts > 0:
            defaults["analyst_bullish_ratio"] = float((strong_buy + buy) / total_analysts)
            defaults["analyst_total_coverage"] = float(total_analysts)

    return defaults


def build_consensus_feature_series(ticker: str, price_series: pd.Series, as_of_date: str) -> pd.DataFrame:
    """Builds a (flat) consensus feature matrix aligned to the price index."""
    df_feat = pd.DataFrame(index=price_series.index)
    current_price = float(price_series.iloc[-1])
    feats = extract_stock_consensus_features(ticker, current_price, as_of_date)

    for col_name, val in feats.items():
        df_feat[col_name] = val

    return df_feat
