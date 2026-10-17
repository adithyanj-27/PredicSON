"""
AI Investment Thesis & Institutional Narrative Engine
Translates quantitative predictions, fundamental health, macro regime, and alpha
metrics into an executive research brief (stance, catalysts, headwinds).
"""

from typing import Any, Dict


def _to_pct(value: Any, default: float = 0.0) -> float:
    """Parse a value that may be a number or a string like '+12.5%' into a float percent."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("%", "").replace("+", "").strip())
    except (ValueError, TypeError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def generate_investment_thesis(
    ticker: str,
    stock_name: str,
    current_price: float,
    horizons: Dict[str, Any],
    drivers_dict: Dict[str, Any],
    regime_name: str,
) -> Dict[str, Any]:
    """Synthesize the quantitative signals into an investment brief."""
    p50_3m_ret = horizons.get("3M", {}).get("p50", {}).get("expected_return_pct", 0.0)
    p90_3m_price = horizons.get("3M", {}).get("p90", {}).get("projected_price", current_price)
    p10_3m_price = horizons.get("3M", {}).get("p10", {}).get("projected_price", current_price)

    rsi = _to_float(drivers_dict.get("rsi_14", 50.0), 50.0)
    dma_dist = _to_pct(drivers_dict.get("close_to_sma_200", "0%"))
    cmf = _to_float(drivers_dict.get("chaikin_money_flow_20d", 0.0))
    pe = _to_float(drivers_dict.get("fund_pe_ratio", 20.0), 20.0)
    peg = _to_float(drivers_dict.get("fund_peg_ratio", 1.5), 1.5)
    rev_growth = _to_float(drivers_dict.get("fund_rev_growth_yoy", 0.10), 0.10) * 100
    analyst_upside = _to_pct(drivers_dict.get("consensus_upside_pct", "0%"))
    analyst_conviction = _to_pct(drivers_dict.get("analyst_bullish_ratio", "50%"), 50.0)
    crude_trend = _to_pct(drivers_dict.get("crude_ret_63d", "0%"))

    # Executive stance
    if p50_3m_ret > 5.0 and cmf > 0.05:
        stance, stance_color = "BULLISH ACCUMULATION", "green"
    elif p50_3m_ret < -5.0 or cmf < -0.10:
        stance, stance_color = "DEFENSIVE CAUTION", "red"
    else:
        stance, stance_color = "RANGE-BOUND CONSOLIDATION", "orange"

    bull_points = []
    if rev_growth > 12.0:
        bull_points.append(f"Strong top-line revenue momentum expanding at {rev_growth:.1f}% YoY.")
    if 0 < peg < 1.2:
        bull_points.append(f"Attractive valuation relative to growth with PEG ratio at {peg:.2f} (under 1.2 indicates growth at a reasonable price).")
    if cmf > 0.02:
        bull_points.append(f"Chaikin Money Flow (+{cmf:.2f}) confirms sustained institutional volume accumulation.")
    if analyst_upside > 15.0:
        bull_points.append(f"Institutional brokerages forecast {analyst_upside:.1f}% consensus upside with {analyst_conviction:.0f}% Buy ratings.")
    if rsi < 45.0:
        bull_points.append(f"Momentum oscillator (RSI {rsi:.1f}) is approaching oversold territory.")
    if not bull_points:
        bull_points.append("Steady defensive balance sheet with predictable market-share stability.")

    risk_points = []
    if dma_dist < 0:
        risk_points.append(f"Trading {dma_dist:.1f}% below its 200-Day Moving Average, acting as multi-week technical overhead resistance.")
    if crude_trend > 10.0:
        risk_points.append(f"Macro commodity pressure: Brent crude is up {crude_trend:.1f}% over the last quarter, elevating input-cost inflation.")
    if pe > 40.0:
        risk_points.append(f"Elevated price multiple at {pe:.1f}x P/E leaves little margin of error for quarterly earnings misses.")
    if rsi > 65.0:
        risk_points.append(f"Short-term momentum is overheated (RSI {rsi:.1f}), pointing to near-term profit booking.")
    if not risk_points:
        risk_points.append("Vulnerable to broader emerging-market capital outflows if US 10-Year Treasury yields spike.")

    executive_summary = (
        f"{stock_name} ({ticker}) is positioned in {stance} within the current "
        f"{regime_name.replace('_', ' ')} macro environment. Over the next 3 months, "
        f"the probabilistic engine projects a primary range of "
        f"Rs {p10_3m_price:,.0f} to Rs {p90_3m_price:,.0f} (median trajectory: {p50_3m_ret:+.1f}%). "
        f"Key structural support is bolstered by {bull_points[0].lower()}, "
        f"while primary downside vigilance is anchored by {risk_points[0].lower()}."
    )

    return {
        "stance": stance,
        "stance_color": stance_color,
        "executive_summary": executive_summary,
        "bull_catalysts": bull_points[:3],
        "risk_headwinds": risk_points[:3],
    }
