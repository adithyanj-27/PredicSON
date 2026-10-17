"""
Fan Chart Curve Generator

Creates smooth, monotone daily projection trajectories and an expanding confidence
cone (p10, p25, p50, p75, p90) from the latest close out to six months forward.

Two guarantees the earlier version lacked:
- The cone widens monotonically with horizon: lower quantiles never rise and upper
  quantiles never fall as the horizon lengthens.
- Curves are interpolated with a shape-preserving PCHIP spline, so they are smooth
  and never cross between anchors.
"""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.quantile_forecaster import generate_stock_forecast
from src.utils.data_loader import load_stock_ohlcv

HORIZON_ORDER = ["1M", "3M", "6M"]
HORIZON_DAYS = {"1M": 21, "3M": 63, "6M": 126}
Q_KEYS = ["p10", "p25", "p50", "p75", "p90"]


def _monotone_cone(current_price: float, horizons: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Enforce (a) p10<=p25<=p50<=p75<=p90 within each horizon and (b) widening across
    horizons: lower quantiles non-increasing, upper quantiles non-decreasing.
    """
    per_horizon = {}
    for h in HORIZON_ORDER:
        vals = np.array([horizons[h][q]["projected_price"] for q in Q_KEYS], dtype=float)
        per_horizon[h] = np.maximum.accumulate(vals)  # internal ordering

    for j, q in enumerate(Q_KEYS):
        seq = np.array([per_horizon[h][j] for h in HORIZON_ORDER], dtype=float)
        if j <= 1:      # p10, p25: non-increasing, floored at current price
            seq[0] = min(seq[0], current_price)
            seq = np.minimum.accumulate(seq)
        elif j >= 3:    # p75, p90: non-decreasing, capped from below at current price
            seq[0] = max(seq[0], current_price)
            seq = np.maximum.accumulate(seq)
        for i, h in enumerate(HORIZON_ORDER):
            per_horizon[h][j] = seq[i]

    # Re-assert internal ordering after the cross-horizon pass.
    for h in HORIZON_ORDER:
        per_horizon[h] = np.maximum.accumulate(per_horizon[h])
    return per_horizon


def generate_fan_chart_data(ticker: str, lookback_days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate the historical price series plus a forward probabilistic fan chart.
    Returns (history_df, forecast_df, forecast_meta).
    """
    ohlcv = load_stock_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No OHLCV data for {ticker}")

    history_df = ohlcv[["close"]].tail(lookback_days).copy()
    current_date = history_df.index[-1]
    current_price = float(history_df["close"].iloc[-1])

    meta = generate_stock_forecast(ticker)
    horizons = meta["horizons"]

    cone = _monotone_cone(current_price, horizons)

    # Forward trading-day grid (weekdays), 0..126.
    future_dates = []
    d = current_date
    while len(future_dates) <= 126:
        d += timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)
    future_dates = future_dates[:127]

    anchor_days = np.array([0] + [HORIZON_DAYS[h] for h in HORIZON_ORDER], dtype=float)
    grid = np.arange(len(future_dates), dtype=float)

    forward = {"date": future_dates}
    for j, q in enumerate(Q_KEYS):
        anchor_vals = np.array([current_price] + [cone[h][j] for h in HORIZON_ORDER], dtype=float)
        if HAS_SCIPY:
            interp = PchipInterpolator(anchor_days, anchor_vals)(grid)
        else:
            interp = np.interp(grid, anchor_days, anchor_vals)
        forward[q] = np.round(interp, 2)

    forecast_df = pd.DataFrame(forward).set_index("date")
    return history_df, forecast_df, meta
