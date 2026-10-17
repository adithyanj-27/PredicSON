"""
Cross-Sectional Market Opportunity Screener & Signal Engine
Scans the entire 25-stock universe to generate:
1. Ranked Leaderboards (Highest 3M Upside, Best Risk-Reward, Valuation Value)
2. Automated Quantitative Signals (STRONG BUY, ACCUMULATE, HOLD, AVOID)
3. Risk-Reward Ratios (Projected Median Return / Downside p10 Loss)
4. Fast JSON Caching for Instant UI Loading
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, PREDICTIONS_DIR
from src.models.quantile_forecaster import generate_stock_forecast

SCREENER_CACHE_FILE = PREDICTIONS_DIR / "market_screener.json"


def generate_market_screener(use_cache: bool = False) -> pd.DataFrame:
    """
    Evaluates all stocks in the universe and returns a structured comparison matrix.
    Saves output to disk for sub-second UI access.
    """
    if use_cache and SCREENER_CACHE_FILE.exists():
        try:
            return pd.read_json(SCREENER_CACHE_FILE)
        except Exception:
            pass

    records = []
    print(f"Scanning market opportunities across {len(PILOT_UNIVERSE)} stocks...", flush=True)

    for i, (ticker, info) in enumerate(PILOT_UNIVERSE.items(), 1):
        try:
            print(f"  [{i}/{len(PILOT_UNIVERSE)}] Screening {ticker}...", flush=True)
            fc = generate_stock_forecast(ticker)
            curr_p = fc["current_price"]
            h3 = fc["horizons"]["3M"]

            med_ret_3m = h3["p50"]["expected_return_pct"]
            p10_ret_3m = h3["p10"]["expected_return_pct"]
            p90_ret_3m = h3["p90"]["expected_return_pct"]

            # Downside Risk magnitude
            downside_risk = abs(min(0.0, p10_ret_3m)) + 1e-4
            risk_reward = med_ret_3m / downside_risk if med_ret_3m > 0 else (med_ret_3m / downside_risk)

            # Extract key drivers
            drivers = {d["feature"]: d["value"] for d in fc.get("key_drivers", [])}
            cmf = float(drivers.get("cmf_20d", 0.0))
            peg = float(drivers.get("peg_ratio", 1.5))
            pe = str(drivers.get("pe_ratio", "20x"))
            rsi = float(drivers.get("rsi_14", 50.0))
            rev_growth = str(drivers.get("rev_growth", "10%"))
            analyst_conviction = str(drivers.get("analyst_conviction", "50%"))

            # Quantitative Signal Determination
            if med_ret_3m > 7.0 and cmf > 0.01 and peg < 2.0:
                signal = "STRONG BUY"
                badge = "🟢"
            elif med_ret_3m > 1.5 and cmf >= -0.06:
                signal = "ACCUMULATE"
                badge = "🔵"
            elif med_ret_3m < -6.0 or (cmf < -0.10 and med_ret_3m < 0):
                signal = "AVOID / TRIM"
                badge = "🔴"
            else:
                signal = "NEUTRAL / HOLD"
                badge = "⚪"

            records.append({
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"].split("/")[0].strip(),
                "current_price": curr_p,
                "projected_3m_price": h3["p50"]["projected_price"],
                "expected_return_3m": med_ret_3m,
                "p10_downside_3m": p10_ret_3m,
                "p90_upside_3m": p90_ret_3m,
                "risk_reward_ratio": round(risk_reward, 2),
                "signal": f"{badge} {signal}",
                "cmf_flow": cmf,
                "peg_ratio": peg,
                "pe_ratio": pe,
                "rsi_14": rsi,
                "analyst_conviction": analyst_conviction
            })
        except Exception as e:
            print(f"Error screening {ticker}: {e}", flush=True)
            continue

    df_screener = pd.DataFrame(records)
    if not df_screener.empty:
        df_screener.sort_values(by="expected_return_3m", ascending=False, inplace=True)
        df_screener.reset_index(drop=True, inplace=True)
        # Cache to disk
        df_screener.to_json(SCREENER_CACHE_FILE, orient="records", indent=2)
        print(f"Market Screener saved -> {SCREENER_CACHE_FILE.name}", flush=True)

    return df_screener


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    df = generate_market_screener(use_cache=True)
    print("\nTop 5 Bullish Outperformers:")
    print(df.head(5)[["ticker", "name", "expected_return_3m", "signal", "risk_reward_ratio"]].to_string())
