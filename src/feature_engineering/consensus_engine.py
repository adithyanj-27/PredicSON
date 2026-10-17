"""
Expert Consensus & Dispersion Feature Engine
Transforms institutional analyst forecasts, target price ranges, and recommendation splits
into calibrated numerical features.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import EXPERTS_DIR
from src.utils.data_loader import load_expert_consensus


def extract_stock_consensus_features(ticker: str, current_price: float, as_of_date: str) -> Dict[str, float]:
    """
    Extracts consensus upside, dispersion (uncertainty), and rating conviction from expert snapshots.
    """
    # Default neutral features if analyst coverage is not yet filed
    defaults = {
        "consensus_upside_pct": 0.0,
        "consensus_dispersion_pct": 0.0,
        "analyst_bullish_ratio": 0.50,
        "analyst_total_coverage": 0.0
    }

    expert_records = load_expert_consensus(as_of_date)
    matched = next((r for r in expert_records if r.get("ticker") == ticker), None)

    if not matched:
        return defaults

    price_targets = matched.get("price_targets", {})
    mean_target = price_targets.get("mean")
    high_target = price_targets.get("high")
    low_target = price_targets.get("low")

    # 1. Consensus Upside / Downside
    if mean_target and current_price > 0:
        upside = (mean_target - current_price) / current_price
        # Bound between -50% and +100% to prevent extreme outliers
        defaults["consensus_upside_pct"] = float(np.clip(upside, -0.5, 1.0))

    # 2. Analyst Dispersion / Disagreement Index
    # (High Target - Low Target) / Mean Target
    if high_target and low_target and mean_target and mean_target > 0:
        dispersion = (high_target - low_target) / mean_target
        defaults["consensus_dispersion_pct"] = float(np.clip(dispersion, 0.0, 1.5))

    # 3. Recommendation Conviction (Bullish Ratio)
    recos = matched.get("recommendation_counts", {})
    if recos:
        # Standard buckets: strongBuy, buy, hold, sell, strongSell
        strong_buy = float(recos.get("strongBuy", recos.get("strong_buy", 0)))
        buy = float(recos.get("buy", 0))
        hold = float(recos.get("hold", 0))
        sell = float(recos.get("sell", 0))
        strong_sell = float(recos.get("strongSell", recos.get("strong_sell", 0)))

        total_analysts = strong_buy + buy + hold + sell + strong_sell
        if total_analysts > 0:
            bullish_count = strong_buy + buy
            defaults["analyst_bullish_ratio"] = float(bullish_count / total_analysts)
            defaults["analyst_total_coverage"] = float(total_analysts)

    return defaults


def build_consensus_feature_series(
    ticker: str,
    price_series: pd.Series,
    as_of_date: str
) -> pd.DataFrame:
    """
    Builds a aligned time-series feature matrix for expert consensus across the price index.
    """
    df_feat = pd.DataFrame(index=price_series.index)
    
    current_price = float(price_series.iloc[-1])
    feats = extract_stock_consensus_features(ticker, current_price, as_of_date)

    # Broadcast features across recent history (institutional targets typically have 3-6 month validity)
    df_feat["consensus_upside_pct"] = feats["consensus_upside_pct"]
    df_feat["consensus_dispersion_pct"] = feats["consensus_dispersion_pct"]
    df_feat["analyst_bullish_ratio"] = feats["analyst_bullish_ratio"]
    df_feat["analyst_total_coverage"] = feats["analyst_total_coverage"]

    return df_feat
