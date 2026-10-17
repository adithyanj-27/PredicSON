"""
Modern Portfolio Theory & Optimal Capital Allocator

Solves for the maximum-Sharpe portfolio using the model's forward 3M expected
returns and the empirical covariance of daily returns, subject to per-stock caps.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE
from src.utils.data_loader import load_stock_ohlcv

RISK_FREE_RATE = 0.065  # ~6.5% RBI repo proxy
TRADING_DAYS = 252


def compute_historical_returns_matrix(tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
    """Consolidate daily percentage returns for covariance calculation."""
    series_dict = {}
    for ticker in tickers:
        ohlcv = load_stock_ohlcv(ticker)
        if ohlcv is not None and not ohlcv.empty:
            series_dict[ticker] = ohlcv["close"].pct_change()
    if not series_dict:
        return pd.DataFrame()
    return pd.DataFrame(series_dict).tail(lookback_days).dropna()


def _equal_weight_fallback(valid_tickers: List[str], total_capital: float, note: str) -> Dict[str, Any]:
    """Honest equal-weight allocation; metrics are None when they cannot be computed."""
    if not valid_tickers:
        return {"allocations": [], "total_capital": total_capital, "note": note,
                "expected_3m_portfolio_return_pct": None, "portfolio_volatility_pct": None, "portfolio_sharpe": None}

    w = 1.0 / len(valid_tickers)
    allocations = []
    for t in valid_tickers:
        allocations.append({
            "ticker": t,
            "name": PILOT_UNIVERSE.get(t, {}).get("name", t),
            "sector": PILOT_UNIVERSE.get(t, {}).get("sector", "Diversified").split("/")[0].strip(),
            "weight_pct": round(w * 100, 1),
            "capital_inr": round(total_capital * w, 2),
            "expected_3m_return": None,
        })
    return {
        "allocations": allocations,
        "total_capital": total_capital,
        "note": note,
        "expected_3m_portfolio_return_pct": None,
        "portfolio_volatility_pct": None,
        "portfolio_sharpe": None,
    }


def optimize_portfolio(
    screener_df: pd.DataFrame,
    total_capital: float = 100000.0,
    max_stock_weight: float = 0.25,
    min_stocks: int = 4,
) -> Dict[str, Any]:
    """Compute the optimal allocation from the model's 3M expected returns and historical covariance."""
    if screener_df is None or screener_df.empty:
        return _equal_weight_fallback([], total_capital, "No screener data available.")

    candidates = screener_df[screener_df["expected_return_3m"] > 0].copy()
    if len(candidates) < min_stocks:
        candidates = screener_df.head(min_stocks).copy()

    tickers = list(candidates["ticker"])
    # Annualise the 3M expected return (~4 non-overlapping quarters per year).
    expected_returns = (candidates["expected_return_3m"].values / 100.0) * 4.0

    returns_matrix = compute_historical_returns_matrix(tickers, lookback_days=TRADING_DAYS)
    valid_tickers = [t for t in tickers if t in returns_matrix.columns]

    if len(valid_tickers) < 3 or not HAS_SCIPY:
        note = "Equal-weight fallback: fewer than three assets had usable price history, or scipy is unavailable."
        return _equal_weight_fallback(valid_tickers, total_capital, note)

    idx_map = [tickers.index(t) for t in valid_tickers]
    expected_returns = expected_returns[idx_map]
    cov_matrix = returns_matrix[valid_tickers].cov().values * TRADING_DAYS

    n_assets = len(valid_tickers)
    init_weights = np.ones(n_assets) / n_assets
    bounds = tuple((0.0, max_stock_weight) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(weights):
        port_return = float(np.sum(expected_returns * weights))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))) + 1e-9
        return -(port_return - RISK_FREE_RATE) / port_vol

    res = minimize(neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    # Keep the optimiser's solution as-is: it already respects the caps and sums to 1.
    opt_weights = res.x if res.success else init_weights

    port_return_annual = float(np.sum(expected_returns * opt_weights))
    port_vol_annual = float(np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights))))
    port_sharpe = (port_return_annual - RISK_FREE_RATE) / (port_vol_annual + 1e-9)

    allocations = []
    for idx, t in enumerate(valid_tickers):
        w = float(opt_weights[idx])
        if w > 0.005:
            allocations.append({
                "ticker": t,
                "name": PILOT_UNIVERSE.get(t, {}).get("name", t),
                "sector": PILOT_UNIVERSE.get(t, {}).get("sector", "Diversified").split("/")[0].strip(),
                "weight_pct": round(w * 100, 1),
                "capital_inr": round(total_capital * w, 2),
                "expected_3m_return": round(float(candidates.loc[candidates["ticker"] == t, "expected_return_3m"].values[0]), 1),
            })

    allocations.sort(key=lambda x: x["weight_pct"], reverse=True)

    return {
        "allocations": allocations,
        "total_capital": total_capital,
        "note": "Max-Sharpe allocation (caps respected).",
        "expected_3m_portfolio_return_pct": round((port_return_annual / 4.0) * 100, 2),
        "portfolio_volatility_pct": round(port_vol_annual * 100, 2),
        "portfolio_sharpe": round(port_sharpe, 2),
    }
