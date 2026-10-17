"""
Walk-Forward Backtesting Engine (Fast & Transparent)
Evaluates out-of-sample directional accuracy and cone calibration with real-time progress logging.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import FEATURES_DATA_DIR, PREDICTIONS_DIR, PILOT_UNIVERSE
from src.models.quantile_forecaster import (
    prepare_training_data,
    train_stacked_horizon_ensemble,
    enforce_monotonic_quantiles,
    QUANTILES,
    HORIZONS
)


def run_walk_forward_backtest(
    ticker: str,
    n_test_windows: int = 3,
    test_step_days: int = 126
) -> Dict[str, Any]:
    """
    Performs walk-forward out-of-sample backtest across historical market windows.
    Prints real-time progress with flush=True so you always see exactly what is happening.
    """
    sanitized_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    feature_file = FEATURES_DATA_DIR / f"{sanitized_name}_features.parquet"

    if not feature_file.exists():
        print(f"Error: Feature file not found for {ticker}", flush=True)
        return {"error": f"No feature file for {ticker}"}

    df = pd.read_parquet(feature_file)
    n_rows = len(df)

    if n_rows < 500:
        return {"error": f"Insufficient history for {ticker}"}

    results_by_horizon = {h: [] for h in HORIZONS.keys()}

    # Evaluate on clean non-overlapping historical anchors
    max_eval_idx = n_rows - 127
    start_eval_idx = max(350, max_eval_idx - (n_test_windows * test_step_days))
    eval_indices = list(range(start_eval_idx, max_eval_idx, test_step_days))[:n_test_windows]

    total_steps = len(eval_indices) * len(HORIZONS)
    step_num = 0

    print(f"\nEvaluating {ticker} across {len(eval_indices)} out-of-sample historical windows...", flush=True)
    start_time = time.time()

    for w_i, eval_idx in enumerate(eval_indices, 1):
        eval_date = df.index[eval_idx]
        current_price = float(df["close"].iloc[eval_idx])
        df_train_slice = df.iloc[:eval_idx + 1]

        print(f"\n[Window {w_i}/{len(eval_indices)}] Historical Anchor: {eval_date.strftime('%Y-%m-%d')} (Base Price: Rs. {current_price:,.2f})", flush=True)

        for horizon_name, horizon_days in HORIZONS.items():
            step_num += 1
            t_sub = time.time()
            actual_future_idx = eval_idx + horizon_days
            if actual_future_idx >= n_rows:
                continue

            actual_future_price = float(df["close"].iloc[actual_future_idx])
            actual_return = (actual_future_price - current_price) / current_price

            X_train, y_train, weights, feature_cols = prepare_training_data(df_train_slice, horizon_days)
            ensemble = train_stacked_horizon_ensemble(X_train, y_train, weights)

            X_eval = df.iloc[[eval_idx]][feature_cols].ffill().bfill().fillna(0)
            X_eval_scaled = ensemble["scaler"].transform(X_eval)

            tree_preds = {q: float(ensemble["trees"][q].predict(X_eval)[0]) for q in QUANTILES}
            linear_pred = float(ensemble["linear"].predict(X_eval_scaled)[0])

            blended_preds = {}
            for q in QUANTILES:
                drift = linear_pred - tree_preds[0.50]
                blended_preds[q] = tree_preds[q] + (0.20 * drift)

            monotonic_preds = enforce_monotonic_quantiles(blended_preds)
            pred_pcts = {q: float(np.exp(monotonic_preds[q]) - 1.0) for q in QUANTILES}
            med_pred_pct = pred_pcts[0.50]

            dir_correct = ((med_pred_pct > 0) == (actual_return > 0))
            in_50_cone = (pred_pcts[0.25] <= actual_return <= pred_pcts[0.75])
            in_80_cone = (pred_pcts[0.10] <= actual_return <= pred_pcts[0.90])
            abs_err = abs(med_pred_pct - actual_return)

            results_by_horizon[horizon_name].append({
                "eval_date": eval_date.strftime("%Y-%m-%d"),
                "current_price": round(current_price, 2),
                "actual_price": round(actual_future_price, 2),
                "pred_median_return_pct": round(med_pred_pct * 100, 2),
                "actual_return_pct": round(actual_return * 100, 2),
                "directional_correct": bool(dir_correct),
                "inside_50_cone": bool(in_50_cone),
                "inside_80_cone": bool(in_80_cone),
                "abs_error_pct": round(abs_err * 100, 2)
            })

            elapsed_step = time.time() - t_sub
            status_str = "[HIT]" if dir_correct else "[MISS]"
            print(f"  --> Horizon {horizon_name}: Pred {med_pred_pct*100:+.1f}% | Actual {actual_return*100:+.1f}% {status_str} (took {elapsed_step:.1f}s)", flush=True)

    # Summarize
    summary = {}
    for h, records in results_by_horizon.items():
        if not records:
            continue
        total = len(records)
        dir_hit_rate = sum(1 for r in records if r["directional_correct"]) / total
        cone_50_cov = sum(1 for r in records if r["inside_50_cone"]) / total
        cone_80_cov = sum(1 for r in records if r["inside_80_cone"]) / total
        mean_abs_err = sum(r["abs_error_pct"] for r in records) / total

        summary[h] = {
            "total_windows_tested": total,
            "directional_accuracy_pct": round(dir_hit_rate * 100, 1),
            "cone_50_coverage_pct": round(cone_50_cov * 100, 1),
            "cone_80_coverage_pct": round(cone_80_cov * 100, 1),
            "mean_absolute_error_pct": round(mean_abs_err, 2),
            "test_windows": records
        }

    total_elapsed = time.time() - start_time
    print(f"\n{ticker} Backtest Complete in {total_elapsed:.1f} seconds (~{total_elapsed/60:.1f} min).", flush=True)
    return {"ticker": ticker, "summary": summary}


def run_targeted_backtest(ticker: str = "RELIANCE.NS"):
    """Targeted run for a specific stock."""
    scorecard_file = PREDICTIONS_DIR / "backtest_scorecard.json"
    existing_data = {}
    if scorecard_file.exists():
        try:
            with open(scorecard_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}

    res = run_walk_forward_backtest(ticker, n_test_windows=3, test_step_days=126)
    existing_data[ticker] = res

    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)

    print(f"\nScorecard updated at {scorecard_file.name}", flush=True)
    return res


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    run_targeted_backtest(ticker_arg)
