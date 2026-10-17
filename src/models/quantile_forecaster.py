"""
Institutional Stacked Quantile Forecaster
Key Improvements:
1. Exponential Recency Weighting: Recent market regimes receive higher sample weights than distant history.
2. Heterogeneous Ensemble: Blends regularized Gradient Boosted Trees with regularized Linear Quantile slopes.
3. Quantile Crossing Protection: Guarantees strict mathematical monotonicity (p10 <= p25 <= p50 <= p75 <= p90).
4. Volatility Normalization: Adjusts spreads dynamically using current regime volatility multipliers.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import FEATURES_DATA_DIR
from src.models.regime_classifier import REGIME_NAMES, get_current_market_regime


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126
}


def compute_sample_weights(dates: pd.DatetimeIndex, half_life_years: float = 3.0) -> np.ndarray:
    """
    Computes exponential recency sample weights:
    Recent market observations receive progressively higher weights.
    Half-life of 3 years means 3-year-old data has 50% weight of today's data.
    """
    max_date = dates.max()
    age_days = (max_date - dates).days.values
    half_life_days = half_life_years * 365.25
    decay_lambda = np.log(2) / half_life_days
    weights = np.exp(-decay_lambda * age_days)
    
    # Normalize so mean weight is 1.0
    weights = weights / np.mean(weights)
    return weights


def prepare_training_data(df: pd.DataFrame, horizon_days: int) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, list]:
    """Extract feature matrix X, target y, and recency weights for a given horizon."""
    target_col = f"target_fwd_{horizon_days}d"

    exclude_cols = [
        "open", "high", "low", "close", "volume",
        "target_fwd_21d", "target_fwd_63d", "target_fwd_126d"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, feature_cols].copy()
    y = df.loc[valid_mask, target_col].copy()

    # Fill any edge NaNs cleanly
    X = X.ffill().bfill().fillna(0)

    # Compute exponential recency weights
    weights = compute_sample_weights(X.index, half_life_years=3.0)

    return X, y, weights, feature_cols


def train_stacked_horizon_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray
) -> Dict[str, Any]:
    """
    Trains a heterogeneous ensemble:
    1. Regularized HistGradientBoosting per quantile with sample weights.
    2. Robust-scaled Ridge regression to capture broad linear factor drifts.
    """
    tree_models = {}
    for q in QUANTILES:
        tree = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=180,
            learning_rate=0.035,
            min_samples_leaf=25,
            l2_regularization=2.0,
            max_leaf_nodes=31,
            random_state=42
        )
        tree.fit(X, y, sample_weight=weights)
        tree_models[q] = tree

    # Linear baseline (Ridge for median drift)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    linear_median = Ridge(alpha=100.0, random_state=42)
    linear_median.fit(X_scaled, y, sample_weight=weights)

    return {
        "trees": tree_models,
        "linear": linear_median,
        "scaler": scaler
    }


def enforce_monotonic_quantiles(quantile_preds: Dict[float, float]) -> Dict[float, float]:
    """
    Enforces strict mathematical monotonicity across quantiles:
    p10 <= p25 <= p50 <= p75 <= p90
    Prevents statistical crossing caused by sample noise.
    """
    sorted_q = sorted(QUANTILES)
    values = np.array([quantile_preds[q] for q in sorted_q])
    
    # Isotonic monotonic adjustment
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]

    return {q: float(values[idx]) for idx, q in enumerate(sorted_q)}


def generate_stock_forecast(ticker: str) -> Dict[str, Any]:
    """
    Generate institutional-grade 1M, 3M, and 6M probabilistic price projections.
    """
    sanitized_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    feature_file = FEATURES_DATA_DIR / f"{sanitized_name}_features.parquet"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature dataset not found for {ticker}. Run feature pipeline first.")

    df = pd.read_parquet(feature_file)
    latest_row = df.iloc[[-1]]
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    current_price = float(latest_row["close"].iloc[0])

    current_regime = get_current_market_regime()
    vol_mult = current_regime["volatility_multiplier"]

    forecasts = {}

    for horizon_name, horizon_days in HORIZONS.items():
        X, y, weights, feature_cols = prepare_training_data(df, horizon_days)
        ensemble = train_stacked_horizon_ensemble(X, y, weights)

        X_latest = latest_row[feature_cols].ffill().bfill().fillna(0)
        X_latest_scaled = ensemble["scaler"].transform(X_latest)

        # 1. Tree quantile predictions
        tree_preds = {q: float(ensemble["trees"][q].predict(X_latest)[0]) for q in QUANTILES}
        
        # 2. Linear factor drift
        linear_pred = float(ensemble["linear"].predict(X_latest_scaled)[0])

        # 3. Stacked blend (80% Non-linear Tree + 20% Regularized Factor Drift)
        blended_preds = {}
        for q in QUANTILES:
            # Shift quantiles with linear factor anchor
            drift_correction = linear_pred - tree_preds[0.50]
            blended_preds[q] = tree_preds[q] + (0.20 * drift_correction)

        # 4. Enforce strict monotonicity (no quantile crossing)
        monotonic_preds = enforce_monotonic_quantiles(blended_preds)
        med_return = monotonic_preds[0.50]

        # 5. Apply regime-dependent volatility scaling to bounds
        horizon_preds = {}
        for q in QUANTILES:
            spread = monotonic_preds[q] - med_return
            adjusted_return = med_return + (spread * vol_mult)
            pred_price = float(current_price * (1.0 + (np.exp(adjusted_return) - 1.0)))

            q_label = f"p{int(q * 100)}"
            horizon_preds[q_label] = {
                "expected_return_pct": round(float(np.exp(adjusted_return) - 1.0) * 100, 2),
                "projected_price": round(pred_price, 2)
            }

        forecasts[horizon_name] = horizon_preds

    # Extract scalar driver values safely
    def get_latest_val(col, default=0.0):
        if col in latest_row.columns:
            val = latest_row[col].iloc[0]
            return float(val) if pd.notna(val) else default
        return default

    regime_code = int(get_latest_val("regime_code", 1))
    regime_name = REGIME_NAMES.get(regime_code, "SIDEWAYS_CONSOLIDATION")

    key_drivers = [
        {"feature": "regime", "label": "Market Macro Regime", "value": regime_name.replace("_", " ")},
        {"feature": "cmf_20d", "label": "Chaikin Money Flow (CMF)", "value": round(get_latest_val("chaikin_money_flow_20d", 0.0), 3)},
        {"feature": "vol_z", "label": "Volatility Z-Score (vs 50-DMA)", "value": f"{round(get_latest_val('vol_z_score_50d', 0.0), 2)} σ"},
        {"feature": "trend_sharpe", "label": "63-Day Trend Sharpe", "value": round(get_latest_val("trend_sharpe_63d", 0.0), 2)},
        {"feature": "rsi_14", "label": "RSI (14-day momentum)", "value": round(get_latest_val("rsi_14", 50.0), 1)},
        {"feature": "close_to_sma_200", "label": "200-DMA Distance", "value": f"{round(get_latest_val('close_to_sma_200', 0.0) * 100, 1)}%"},
        {"feature": "vix_level", "label": "India VIX Volatility", "value": round(get_latest_val("vix_level", 15.0), 1)},
        {"feature": "crude_ret_63d", "label": "Brent Crude 3M Trend", "value": f"{round(get_latest_val('crude_ret_63d', 0.0) * 100, 1)}%"},
        {"feature": "news_sentiment", "label": "News Sentiment Index", "value": round(get_latest_val("news_sentiment_score", 0.0), 3)},
        {"feature": "analyst_upside", "label": "Consensus Target Upside", "value": f"{round(get_latest_val('consensus_upside_pct', 0.0) * 100, 1)}%"},
        {"feature": "analyst_conviction", "label": "Analyst Bullish Conviction", "value": f"{round(get_latest_val('analyst_bullish_ratio', 0.5) * 100, 1)}%"}
    ]

    return {
        "ticker": ticker,
        "as_of_date": latest_date,
        "current_price": round(current_price, 2),
        "regime": regime_name,
        "horizons": forecasts,
        "key_drivers": key_drivers
    }
