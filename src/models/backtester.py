"""
Walk-Forward Backtesting Engine

Evaluates out-of-sample directional accuracy and probabilistic-cone calibration.

Two properties make the result trustworthy:
1. PURGE / EMBARGO. For a horizon of h days, training stops h rows before the
   evaluation anchor, so no training label is computed from a price that occurs
   after (or on) the evaluation date.
2. It evaluates the same statistical core as production, including the regime
   volatility multiplier. The analyst-consensus overlay is intentionally NOT
   applied here because no point-in-time consensus history exists; the scorecard
   therefore describes the model-only forecast, which is stated in the output.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import FEATURES_DATA_DIR, PREDICTIONS_DIR, PILOT_UNIVERSE, MIN_TRAIN_ROWS
from src.utils.data_loader import feature_paths, sanitize_ticker
from src.models.quantile_forecaster import (
    HORIZONS,
    QUANTILES,
    load_or_train_model,
    predict_horizon_returns,
)

DEFAULT_N_WINDOWS = 8
DEFAULT_STEP_DAYS = 63


def _evaluation_indices(n_rows: int, n_test_windows: int, test_step_days: int) -> List[int]:
    """Choose out-of-sample anchors, leaving room for the longest horizon after each."""
    horizon_max = max(HORIZONS.values())
    max_eval_idx = n_rows - 1 - horizon_max
    start = max_eval_idx - (n_test_windows - 1) * test_step_days
    indices = list(range(start, max_eval_idx + 1, test_step_days))[:n_test_windows]
    return [i for i in indices if i >= MIN_TRAIN_ROWS + horizon_max]


def run_walk_forward_backtest(
    ticker: str,
    n_test_windows: int = DEFAULT_N_WINDOWS,
    test_step_days: int = DEFAULT_STEP_DAYS,
) -> Dict[str, Any]:
    """Purged walk-forward backtest with per-horizon accuracy and calibration metrics."""
    feature_file, _ = feature_paths(ticker)
    if not feature_file.exists():
        print(f"Error: Feature file not found for {ticker}", flush=True)
        return {"error": f"No feature file for {ticker}"}

    df = pd.read_parquet(feature_file)
    n_rows = len(df)
    if n_rows < MIN_TRAIN_ROWS + max(HORIZONS.values()) + 20:
        return {"error": f"Insufficient history for {ticker}"}

    eval_indices = _evaluation_indices(n_rows, n_test_windows, test_step_days)
    if not eval_indices:
        return {"error": f"No valid evaluation windows for {ticker}"}

    results_by_horizon = {h: [] for h in HORIZONS}
    total_steps = len(eval_indices) * len(HORIZONS)
    step_num = 0

    print(f"\nEvaluating {ticker} across {len(eval_indices)} purged out-of-sample windows...", flush=True)
    start_time = time.time()

    for w_i, eval_idx in enumerate(eval_indices, 1):
        eval_date = df.index[eval_idx]
        current_price = float(df["close"].iloc[eval_idx])
        vol_mult = float(df.iloc[eval_idx].get("volatility_multiplier", 1.0))
        X_eval = df.iloc[[eval_idx]]

        print(f"\n[Window {w_i}/{len(eval_indices)}] Anchor: {eval_date.strftime('%Y-%m-%d')} (Rs. {current_price:,.2f}, vol x{vol_mult})", flush=True)

        for horizon_name, horizon_days in HORIZONS.items():
            step_num += 1
            actual_future_idx = eval_idx + horizon_days
            if actual_future_idx >= n_rows:
                continue

            # PURGE: stop training h rows before the anchor.
            train_end = eval_idx + 1 - horizon_days
            if train_end < MIN_TRAIN_ROWS:
                continue
            df_train = df.iloc[:train_end]

            ensemble, feature_cols, _ = load_or_train_model(ticker, horizon_days, df_train, use_cache=False)
            returns = predict_horizon_returns(ensemble, feature_cols, X_eval, None, 0.0, vol_mult)
            pred_pcts = {q: float(np.exp(returns[q]) - 1.0) for q in QUANTILES}
            med_pred = pred_pcts[0.50]

            actual_future_price = float(df["close"].iloc[actual_future_idx])
            actual_return = (actual_future_price - current_price) / current_price

            results_by_horizon[horizon_name].append({
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "current_price": round(current_price, 2),
                "actual_price": round(actual_future_price, 2),
                "pred_median_return_pct": round(med_pred * 100, 2),
                "actual_return_pct": round(actual_return * 100, 2),
                "directional_correct": bool((med_pred > 0) == (actual_return > 0)),
                "inside_50_cone": bool(pred_pcts[0.25] <= actual_return <= pred_pcts[0.75]),
                "inside_80_cone": bool(pred_pcts[0.10] <= actual_return <= pred_pcts[0.90]),
                "abs_error_pct": round(abs(med_pred - actual_return) * 100, 2),
            })

            status = "[HIT]" if results_by_horizon[horizon_name][-1]["directional_correct"] else "[MISS]"
            print(f"  --> {horizon_name}: Pred {med_pred*100:+.1f}% | Actual {actual_return*100:+.1f}% {status}", flush=True)

    summary = {}
    for h, records in results_by_horizon.items():
        if not records:
            continue
        total = len(records)
        summary[h] = {
            "n_samples": total,
            "directional_accuracy_pct": round(sum(r["directional_correct"] for r in records) / total * 100, 1),
            "cone_50_coverage_pct": round(sum(r["inside_50_cone"] for r in records) / total * 100, 1),
            "cone_80_coverage_pct": round(sum(r["inside_80_cone"] for r in records) / total * 100, 1),
            "mean_absolute_error_pct": round(sum(r["abs_error_pct"] for r in records) / total, 2),
            "test_windows": records,
        }

    elapsed = time.time() - start_time
    print(f"\n{ticker} backtest complete in {elapsed:.1f}s.", flush=True)

    return {
        "ticker": ticker,
        "config": {
            "n_windows": len(eval_indices),
            "step_days": test_step_days,
            "purged": True,
            "consensus_overlay_applied": False,
        },
        "summary": summary,
    }


def run_targeted_backtest(ticker: str = "RELIANCE.NS") -> Dict[str, Any]:
    """Run a single-stock backtest and merge it into the scorecard."""
    scorecard_file = PREDICTIONS_DIR / "backtest_scorecard.json"
    existing = {}
    if scorecard_file.exists():
        try:
            with open(scorecard_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    res = run_walk_forward_backtest(ticker)
    existing[ticker] = res

    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nScorecard updated at {scorecard_file.name}", flush=True)
    return res


def run_full_backtest(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the backtest across the whole universe (or a supplied list)."""
    tickers = tickers or list(PILOT_UNIVERSE.keys())
    scorecard_file = PREDICTIONS_DIR / "backtest_scorecard.json"
    existing = {}
    if scorecard_file.exists():
        try:
            with open(scorecard_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*60}\n[{i}/{len(tickers)}] Backtesting {ticker}\n{'='*60}", flush=True)
        try:
            existing[ticker] = run_walk_forward_backtest(ticker)
        except Exception as e:
            print(f"  Backtest failed for {ticker}: {e}", flush=True)

    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nFull scorecard written to {scorecard_file.name}", flush=True)
    return existing


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    run_targeted_backtest(ticker_arg)
