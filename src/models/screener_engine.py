"""
Cross-Sectional Market Opportunity Screener & Signal Engine
Scans the universe to produce:
1. Ranked leaderboards (expected 3M return, risk-reward, valuation)
2. Quantitative signals (STRONG BUY, ACCUMULATE, HOLD, AVOID)
3. Risk-reward ratios (projected median return / downside p10 loss)
4. JSON caching for fast UI loading
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, PREDICTIONS_DIR
from src.models.quantile_forecaster import generate_stock_forecast

SCREENER_CACHE_FILE = PREDICTIONS_DIR / "market_screener.json"


def generate_market_screener(use_cache: bool = False) -> pd.DataFrame:
    """Evaluate every stock in the universe and return a structured comparison matrix."""
    if use_cache and SCREENER_CACHE_FILE.exists():
        try:
            return pd.read_json(SCREENER_CACHE_FILE, orient="records")
        except (ValueError, OSError) as e:
            print(f"  Screener cache unreadable ({e}); regenerating.")

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

            # Downside risk magnitude (always positive: epsilon guards a zero floor)
            downside_risk = abs(min(0.0, p10_ret_3m)) + 1e-4
            risk_reward = med_ret_3m / downside_risk

            drivers = {d["feature"]: d["value"] for d in fc.get("key_drivers", [])}
            cmf = float(drivers.get("cmf_20d", 0.0))
            peg = float(drivers.get("peg_ratio", 1.5))
            pe = str(drivers.get("pe_ratio", "20x"))
            rsi = float(drivers.get("rsi_14", 50.0))
            analyst_conviction = str(drivers.get("analyst_conviction", "50%"))
            conviction_pct = float(analyst_conviction.replace("%", "").strip()) if "%" in analyst_conviction else 50.0

            if med_ret_3m > 6.0 and (cmf > 0.0 or conviction_pct >= 85.0) and peg < 2.0:
                signal, badge = "STRONG BUY", "G"
            elif med_ret_3m > 1.8 and (cmf >= -0.09 or conviction_pct >= 70.0):
                signal, badge = "ACCUMULATE", "B"
            elif med_ret_3m < -4.0 or (cmf < -0.12 and med_ret_3m < 0):
                signal, badge = "AVOID / TRIM", "R"
            else:
                signal, badge = "NEUTRAL / HOLD", "N"

            records.append({
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"].split("/")[0].strip(),
                "current_price": curr_p,
                "projected_3m_price": h3["p50"]["projected_price"],
                "expected_return_3m": med_ret_3m,
                "p10_downside_3m": p10_ret_3m,
                "p90_upside_3m": h3["p90"]["expected_return_pct"],
                "risk_reward_ratio": round(risk_reward, 2),
                "signal": signal,
                "signal_code": badge,
                "cmf_flow": cmf,
                "peg_ratio": peg,
                "pe_ratio": pe,
                "rsi_14": rsi,
                "analyst_conviction": analyst_conviction,
            })
        except Exception as e:
            print(f"  Error screening {ticker}: {e}", flush=True)
            continue

    df_screener = pd.DataFrame(records)
    if not df_screener.empty:
        df_screener.sort_values(by="expected_return_3m", ascending=False, inplace=True)
        df_screener.reset_index(drop=True, inplace=True)
        df_screener.to_json(SCREENER_CACHE_FILE, orient="records", indent=2)
        print(f"Market Screener saved -> {SCREENER_CACHE_FILE.name}", flush=True)

    return df_screener


if __name__ == "__main__":
    df = generate_market_screener(use_cache=False)
    if not df.empty:
        print("\nTop 5 Bullish Outperformers:")
        print(df.head(5)[["ticker", "name", "expected_return_3m", "signal", "risk_reward_ratio"]].to_string())
