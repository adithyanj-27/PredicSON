"""
Institutional Stacked Quantile Forecaster

Key design points:
1. Exponential recency weighting: recent regimes get higher sample weight.
2. Heterogeneous ensemble: regularized quantile GBMs blended with a ridge drift term.
3. Quantile crossing protection: strict monotonicity (p10 <= p25 <= p50 <= p75 <= p90).
4. Volatility normalization: spreads scale with the current regime.
5. Constant columns (static consensus / fundamentals snapshots) are dropped before
   fitting, because a per-stock model cannot split on a zero-variance feature.
6. Trained models are persisted to disk and reused until the underlying data changes,
   so a forecast does not retrain on every request.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:  # ImportError, or OSError when the native lib (libgomp) is missing
    HAS_LIGHTGBM = False

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    MODEL_CACHE_DIR,
    RANDOM_SEED,
    FEATURE_SCHEMA_VERSION,
    MIN_TRAIN_ROWS,
)
from src.models.regime_classifier import REGIME_NAMES, get_current_market_regime
from src.utils.data_loader import load_expert_consensus, feature_paths, sanitize_ticker, feature_is_current


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
HORIZONS = {"1M": 21, "3M": 63, "6M": 126}

EXCLUDE_COLS = ["open", "high", "low", "close", "volume",
                "target_fwd_21d", "target_fwd_63d", "target_fwd_126d"]

# Consensus overlay is deliberately conservative: it is only applied when enough
# analysts cover the name and is capped well below 1.0 so the published forecast
# stays primarily model-driven rather than a restatement of one target price.
MIN_ANALYSTS_FOR_CONSENSUS = 3
MAX_CONSENSUS_WEIGHT = 0.35
DRIFT_WEIGHT = 0.20


def compute_sample_weights(dates: pd.DatetimeIndex, half_life_years: float = 3.0) -> np.ndarray:
    """Exponential recency weights (half-life of 3 years means 3y-old data keeps 50% weight)."""
    max_date = dates.max()
    age_days = (max_date - dates).days
    half_life_days = half_life_years * 365.25
    decay_lambda = np.log(2) / half_life_days
    weights = np.exp(-decay_lambda * np.asarray(age_days, dtype=float))
    return weights / np.mean(weights)


def drop_constant_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove zero-variance columns; return the reduced frame and the retained column list."""
    nunique = X.nunique(dropna=False)
    keep = [c for c in X.columns if nunique[c] > 1]
    return X[keep], keep


def prepare_training_data(df: pd.DataFrame, horizon_days: int):
    """Extract X, y, recency weights, and the (constant-filtered) feature columns."""
    target_col = f"target_fwd_{horizon_days}d"
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, feature_cols].copy()
    y = df.loc[valid_mask, target_col].copy()

    X = X.ffill().fillna(0)
    X, kept_cols = drop_constant_columns(X)

    weights = compute_sample_weights(X.index, half_life_years=3.0)
    return X, y, weights, kept_cols


def train_stacked_horizon_ensemble(X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> Dict[str, Any]:
    """Fit per-quantile GBMs plus a robust-scaled ridge drift model."""
    tree_models = {}
    for q in QUANTILES:
        if HAS_LIGHTGBM:
            tree = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=80, learning_rate=0.04,
                num_leaves=31, min_child_samples=25, reg_lambda=2.0,
                n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
            )
        else:
            tree = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=50, learning_rate=0.05,
                min_samples_leaf=25, l2_regularization=2.0, max_leaf_nodes=31,
                random_state=RANDOM_SEED,
            )
        tree.fit(X, y, sample_weight=weights)
        tree_models[q] = tree

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    linear = Ridge(alpha=100.0, random_state=RANDOM_SEED)
    linear.fit(X_scaled, y, sample_weight=weights)

    return {"trees": tree_models, "linear": linear, "scaler": scaler}


def enforce_monotonic_quantiles(quantile_preds: Dict[float, float]) -> Dict[float, float]:
    """Enforce p10 <= p25 <= p50 <= p75 <= p90 via a cumulative maximum (monotone rearrangement)."""
    sorted_q = sorted(QUANTILES)
    values = np.maximum.accumulate(np.array([quantile_preds[q] for q in sorted_q], dtype=float))
    return {q: float(values[i]) for i, q in enumerate(sorted_q)}


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _fingerprint(df: pd.DataFrame, horizon_days: int, feature_cols: List[str]) -> str:
    payload = {
        "v": FEATURE_SCHEMA_VERSION,
        "h": horizon_days,
        "n": int(len(df)),
        "last": df.index[-1].strftime("%Y-%m-%d"),
        "cols": sorted(feature_cols),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _cache_path(ticker: str, horizon_days: int) -> Path:
    return MODEL_CACHE_DIR / f"{sanitize_ticker(ticker)}_{horizon_days}d.joblib"


def load_or_train_model(ticker: str, horizon_days: int, df: pd.DataFrame, use_cache: bool = True):
    """
    Return (ensemble, feature_cols, from_cache). Loads a persisted model when the
    data fingerprint matches, otherwise fits and (optionally) saves it.
    """
    X, y, weights, feature_cols = prepare_training_data(df, horizon_days)
    fingerprint = _fingerprint(df, horizon_days, feature_cols)
    path = _cache_path(ticker, horizon_days)

    if use_cache and HAS_JOBLIB and path.exists():
        try:
            bundle = joblib.load(path)
            if bundle.get("fingerprint") == fingerprint:
                return bundle["ensemble"], bundle["feature_cols"], True
        except Exception as e:
            print(f"  Model cache unreadable ({e}); retraining.")

    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(f"Insufficient training rows for {ticker} {horizon_days}d: {len(X)} < {MIN_TRAIN_ROWS}")

    ensemble = train_stacked_horizon_ensemble(X, y, weights)

    if use_cache and HAS_JOBLIB:
        try:
            joblib.dump({"fingerprint": fingerprint, "ensemble": ensemble, "feature_cols": feature_cols}, path)
        except Exception as e:
            print(f"  Could not persist model cache: {e}")

    return ensemble, feature_cols, False


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def enforce_cone_across_horizons(forecasts: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """Force the published cone to widen with horizon: lower quantiles non-increasing,
    upper quantiles non-decreasing, and internal ordering preserved. Idempotent."""
    order = [h for h in HORIZONS if h in forecasts]
    qkeys = [f"p{int(q * 100)}" for q in QUANTILES]

    per = {h: np.array([forecasts[h][q]["projected_price"] for q in qkeys], dtype=float) for h in order}
    for h in order:
        per[h] = np.maximum.accumulate(per[h])

    for j in range(len(qkeys)):
        seq = np.array([per[h][j] for h in order], dtype=float)
        if j <= 1:      # p10, p25: non-increasing, floored at current price
            seq[0] = min(seq[0], current_price)
            seq = np.minimum.accumulate(seq)
        elif j >= 3:    # p75, p90: non-decreasing, capped from below at current price
            seq[0] = max(seq[0], current_price)
            seq = np.maximum.accumulate(seq)
        for i, h in enumerate(order):
            per[h][j] = seq[i]

    for h in order:
        per[h] = np.maximum.accumulate(per[h])
        for j, q in enumerate(qkeys):
            price = float(per[h][j])
            forecasts[h][q]["projected_price"] = round(price, 2)
            forecasts[h][q]["expected_return_pct"] = round((price / current_price - 1.0) * 100, 2)
    return forecasts


def consensus_blend_weight(conviction: float, coverage: float, dispersion: float) -> float:
    """Conservative, coverage-gated weight for the analyst-consensus overlay."""
    if coverage < MIN_ANALYSTS_FOR_CONSENSUS:
        return 0.0
    w = 0.10 + 0.25 * float(np.clip(conviction, 0.0, 1.0))
    w *= max(0.0, 1.0 - 0.5 * min(float(dispersion), 1.0))  # shrink when analysts disagree
    return float(np.clip(w, 0.0, MAX_CONSENSUS_WEIGHT))


def consensus_quantiles(current_price: float, mean_t: float, high_t: float, low_t: float, horizon_days: int) -> Dict[float, float]:
    """Map a target-price cone onto log-return quantiles, scaled by horizon."""
    t_h = min(1.0, (horizon_days / 180.0) ** 0.70)
    up_mean = (mean_t - current_price) / current_price
    up_high = (high_t - current_price) / current_price
    up_low = (low_t - current_price) / current_price

    p50 = float(np.log(max(0.1, 1.0 + up_mean * t_h)))
    p90 = float(np.log(max(0.1, 1.0 + up_high * t_h)))
    p10 = float(np.log(max(0.1, 1.0 + up_low * t_h)))
    return {0.10: p10, 0.25: 0.5 * (p10 + p50), 0.50: p50, 0.75: 0.5 * (p50 + p90), 0.90: p90}


def predict_horizon_returns(
    ensemble: Dict[str, Any],
    feature_cols: List[str],
    X_latest: pd.DataFrame,
    consensus_q: Optional[Dict[float, float]] = None,
    consensus_weight: float = 0.0,
    vol_mult: float = 1.0,
) -> Dict[float, float]:
    """
    Produce log-return quantiles for one row. Shared by the live forecast and the
    backtester so both evaluate the same statistical core.
    """
    X = X_latest[feature_cols].ffill().fillna(0)
    X_scaled = ensemble["scaler"].transform(X)

    tree_preds = {q: float(ensemble["trees"][q].predict(X)[0]) for q in QUANTILES}
    linear_pred = float(ensemble["linear"].predict(X_scaled)[0])
    drift = linear_pred - tree_preds[0.50]

    blended = {}
    for q in QUANTILES:
        stat = tree_preds[q] + DRIFT_WEIGHT * drift
        if consensus_q is not None and consensus_weight > 0:
            blended[q] = (1.0 - consensus_weight) * stat + consensus_weight * consensus_q[q]
        else:
            blended[q] = stat

    monotonic = enforce_monotonic_quantiles(blended)
    med = monotonic[0.50]
    return {q: med + (monotonic[q] - med) * vol_mult for q in QUANTILES}


def generate_stock_forecast(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """Generate probabilistic 1M / 3M / 6M price projections for one stock."""
    feature_file, _ = feature_paths(ticker)
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature dataset not found for {ticker}. Run `python main.py features` first.")
    if not feature_is_current(ticker):
        print(f"  Warning: feature matrix for {ticker} is missing a current version stamp; run `python main.py features` to rebuild.")

    df = pd.read_parquet(feature_file)
    if len(df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Insufficient history for {ticker}: {len(df)} rows.")

    latest_row = df.iloc[[-1]]
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    current_price = float(latest_row["close"].iloc[0])

    current_regime = get_current_market_regime()
    vol_mult = current_regime["volatility_multiplier"]

    # --- Consensus overlay (live-only; not available point-in-time historically) ---
    expert_records = load_expert_consensus(latest_date)
    stock_consensus = next((r for r in expert_records if r.get("ticker") == ticker), None)

    consensus_weight = 0.0
    target_mean = current_price
    target_high = current_price * 1.20
    target_low = current_price * 0.85

    if stock_consensus and stock_consensus.get("price_targets"):
        pt = stock_consensus["price_targets"]
        recos = stock_consensus.get("recommendation_counts", {})
        target_mean = float(pt.get("mean") or current_price)
        target_high = float(pt.get("high") or (current_price * 1.25))
        target_low = float(pt.get("low") or (current_price * 0.85))

        total_ratings = sum(recos.values()) if recos else 0
        buys = (recos.get("strongBuy", 0) + recos.get("buy", 0)) if recos else 0
        conviction = (buys / total_ratings) if total_ratings > 0 else 0.0
        dispersion = ((target_high - target_low) / target_mean) if target_mean > 0 else 0.0
        consensus_weight = consensus_blend_weight(conviction, total_ratings, dispersion)

    forecasts = {}
    for horizon_name, horizon_days in HORIZONS.items():
        ensemble, feature_cols, _ = load_or_train_model(ticker, horizon_days, df, use_cache=use_cache)

        cq = None
        if consensus_weight > 0:
            cq = consensus_quantiles(current_price, target_mean, target_high, target_low, horizon_days)

        returns = predict_horizon_returns(ensemble, feature_cols, latest_row, cq, consensus_weight, vol_mult)

        forecasts[horizon_name] = {}
        for q in QUANTILES:
            r = returns[q]
            forecasts[horizon_name][f"p{int(q * 100)}"] = {
                "expected_return_pct": round(float(np.exp(r) - 1.0) * 100, 2),
                "projected_price": round(float(current_price * np.exp(r)), 2),
            }

    forecasts = enforce_cone_across_horizons(forecasts, current_price)

    def get_latest_val(col, default=0.0):
        if col in latest_row.columns:
            val = latest_row[col].iloc[0]
            return float(val) if pd.notna(val) else default
        return default

    regime_code = int(get_latest_val("regime_code", 1))
    regime_name = REGIME_NAMES.get(regime_code, "SIDEWAYS_CONSOLIDATION")

    drivers_dict = {
        "regime": regime_name.replace("_", " "),
        "chaikin_money_flow_20d": round(get_latest_val("chaikin_money_flow_20d", 0.0), 3),
        "vol_z_score_50d": round(get_latest_val("vol_z_score_50d", 0.0), 2),
        "trend_sharpe_63d": round(get_latest_val("trend_sharpe_63d", 0.0), 2),
        "rsi_14": round(get_latest_val("rsi_14", 50.0), 1),
        "close_to_sma_200": f"{round(get_latest_val('close_to_sma_200', 0.0) * 100, 1)}%",
        "vix_level": round(get_latest_val("vix_level", 15.0), 1),
        "crude_ret_63d": f"{round(get_latest_val('crude_ret_63d', 0.0) * 100, 1)}%",
        "news_sentiment_score": round(get_latest_val("news_sentiment_score", 0.0), 3),
        "consensus_upside_pct": f"{round(get_latest_val('consensus_upside_pct', 0.0) * 100, 1)}%",
        "analyst_bullish_ratio": f"{round(get_latest_val('analyst_bullish_ratio', 0.5) * 100, 1)}%",
        "fund_pe_ratio": round(get_latest_val("fund_pe_ratio", 20.0), 1),
        "fund_peg_ratio": round(get_latest_val("fund_peg_ratio", 1.5), 2),
        "fund_rev_growth_yoy": round(get_latest_val("fund_rev_growth_yoy", 0.10), 3),
        "fund_operating_margin": f"{round(get_latest_val('fund_operating_margin', 0.15) * 100, 1)}%",
        "residual_alpha_21d": f"{round(get_latest_val('residual_alpha_21d', 0.0) * 100, 1)}%",
        "consensus_weight": round(consensus_weight, 3),
    }

    key_drivers = [
        {"feature": "regime", "label": "Macro Regime", "value": drivers_dict["regime"]},
        {"feature": "pe_ratio", "label": "P/E Valuation Multiple", "value": f"{drivers_dict['fund_pe_ratio']}x"},
        {"feature": "peg_ratio", "label": "PEG Ratio (Growth/Valuation)", "value": drivers_dict["fund_peg_ratio"]},
        {"feature": "rev_growth", "label": "Quarterly Revenue Growth", "value": f"{round(float(drivers_dict['fund_rev_growth_yoy']) * 100, 1)}% YoY"},
        {"feature": "op_margin", "label": "Operating Profit Margin", "value": drivers_dict["fund_operating_margin"]},
        {"feature": "cmf_20d", "label": "Chaikin Money Flow (CMF)", "value": drivers_dict["chaikin_money_flow_20d"]},
        {"feature": "vol_z", "label": "Volatility Z-Score", "value": f"{drivers_dict['vol_z_score_50d']} sigma"},
        {"feature": "trend_sharpe", "label": "63-Day Trend Sharpe", "value": drivers_dict["trend_sharpe_63d"]},
        {"feature": "rsi_14", "label": "RSI (14-day momentum)", "value": drivers_dict["rsi_14"]},
        {"feature": "close_to_sma_200", "label": "200-DMA Distance", "value": drivers_dict["close_to_sma_200"]},
        {"feature": "vix_level", "label": "India VIX Volatility", "value": drivers_dict["vix_level"]},
        {"feature": "crude_ret_63d", "label": "Brent Crude 3M Trend", "value": drivers_dict["crude_ret_63d"]},
        {"feature": "residual_alpha", "label": "Idiosyncratic Alpha (1M)", "value": drivers_dict["residual_alpha_21d"]},
        {"feature": "analyst_upside", "label": "Consensus Target Upside", "value": drivers_dict["consensus_upside_pct"]},
        {"feature": "analyst_conviction", "label": "Analyst Bullish Conviction", "value": drivers_dict["analyst_bullish_ratio"]},
    ]

    from src.models.narrative_engine import generate_investment_thesis
    from src.data_ingestion.stock_directory import STOCK_DIRECTORY
    from config.settings import PILOT_UNIVERSE

    stock_meta = next((s for s in STOCK_DIRECTORY if s["symbol"].upper() == ticker.upper()), None)
    stock_name = stock_meta["name"] if stock_meta else PILOT_UNIVERSE.get(ticker, {}).get("name", ticker)

    thesis = generate_investment_thesis(
        ticker=ticker,
        stock_name=stock_name,
        current_price=current_price,
        horizons=forecasts,
        drivers_dict=drivers_dict,
        regime_name=regime_name,
    )

    return {
        "ticker": ticker,
        "as_of_date": latest_date,
        "current_price": round(current_price, 2),
        "regime": regime_name,
        "horizons": forecasts,
        "key_drivers": key_drivers,
        "investment_thesis": thesis,
    }
