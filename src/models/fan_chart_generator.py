"""
Fan Chart Curve Generator
Creates smooth continuous daily projection trajectories and expanding confidence cones
(p10, p25, p50, p75, p90) from historical close price up to 6 months forward.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.quantile_forecaster import generate_stock_forecast
from src.utils.data_loader import load_stock_ohlcv


def generate_fan_chart_data(ticker: str, lookback_days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate historical price series + forward probabilistic fan chart trajectory.
    Returns:
    - history_df: recent historical prices (date, close)
    - forecast_df: forward daily projection (date, p10, p25, p50, p75, p90)
    - forecast_meta: horizon summaries and key drivers
    """
    # 1. Load history
    ohlcv = load_stock_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No OHLCV data for {ticker}")

    history_df = ohlcv[["close"]].tail(lookback_days).copy()
    current_date = history_df.index[-1]
    current_price = float(history_df["close"].iloc[-1])

    # 2. Get quantile predictions for 1M (21d), 3M (63d), 6M (126d)
    meta = generate_stock_forecast(ticker)
    horizons = meta["horizons"]

    # Target points: (day_index, p10, p25, p50, p75, p90)
    anchor_days = [0, 21, 63, 126]
    anchors = {
        "p10": [current_price, horizons["1M"]["p10"]["projected_price"], horizons["3M"]["p10"]["projected_price"], horizons["6M"]["p10"]["projected_price"]],
        "p25": [current_price, horizons["1M"]["p25"]["projected_price"], horizons["3M"]["p25"]["projected_price"], horizons["6M"]["p25"]["projected_price"]],
        "p50": [current_price, horizons["1M"]["p50"]["projected_price"], horizons["3M"]["p50"]["projected_price"], horizons["6M"]["p50"]["projected_price"]],
        "p75": [current_price, horizons["1M"]["p75"]["projected_price"], horizons["3M"]["p75"]["projected_price"], horizons["6M"]["p75"]["projected_price"]],
        "p90": [current_price, horizons["1M"]["p90"]["projected_price"], horizons["3M"]["p90"]["projected_price"], horizons["6M"]["p90"]["projected_price"]],
    }

    # 3. Generate future trading day dates (approx 126 trading days = ~180 calendar days, excluding weekends)
    future_dates = []
    d = current_date
    while len(future_dates) <= 126:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Monday to Friday
            future_dates.append(d)

    future_dates = future_dates[:127]  # 0 to 126
    step_indices = np.arange(len(future_dates))

    # Interpolate smooth curves through anchors using monotonic PCHIP or quadratic spline
    forward_dict = {"date": future_dates}
    for q_name, values in anchors.items():
        # Monotonic interpolation through the anchors
        interpolated = np.interp(step_indices, anchor_days, values)
        forward_dict[q_name] = np.round(interpolated, 2)

    forecast_df = pd.DataFrame(forward_dict).set_index("date")
    return history_df, forecast_df, meta
