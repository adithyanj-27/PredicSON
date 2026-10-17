"""
AI Investment Thesis & Institutional Narrative Engine (Zero-Cost / Open-Source)
Translates quantitative predictions, fundamental health, macro regime, and alpha metrics
into an institutional executive research brief with Bull Theses, Bear Headwinds, and Strategic Stance.
"""

from typing import Dict, Any, List


def generate_investment_thesis(
    ticker: str,
    stock_name: str,
    current_price: float,
    horizons: Dict[str, Any],
    drivers_dict: Dict[str, Any],
    regime_name: str
) -> Dict[str, Any]:
    """
    Synthesizes all quantitative signals into an institutional-grade investment brief.
    """
    p50_3m_ret = horizons.get("3M", {}).get("p50", {}).get("expected_return_pct", 0.0)
    p90_3m_price = horizons.get("3M", {}).get("p90", {}).get("projected_price", current_price)
    p10_3m_price = horizons.get("3M", {}).get("p10", {}).get("projected_price", current_price)

    rsi = float(drivers_dict.get("rsi_14", 50.0))
    dma_dist = str(drivers_dict.get("close_to_sma_200", "0%"))
    cmf = float(drivers_dict.get("chaikin_money_flow_20d", 0.0))
    pe = float(drivers_dict.get("fund_pe_ratio", 20.0))
    peg = float(drivers_dict.get("fund_peg_ratio", 1.5))
    rev_growth = float(drivers_dict.get("fund_rev_growth_yoy", 0.10)) * 100
    analyst_upside = str(drivers_dict.get("consensus_upside_pct", "0%"))
    analyst_conviction = str(drivers_dict.get("analyst_bullish_ratio", "50%"))
    crude_trend = str(drivers_dict.get("crude_ret_63d", "0%"))

    # Determine Executive Stance
    if p50_3m_ret > 5.0 and cmf > 0.05:
        stance = "BULLISH ACCUMULATION"
        stance_color = "green"
    elif p50_3m_ret < -5.0 or cmf < -0.10:
        stance = "DEFENSIVE CAUTION"
        stance_color = "red"
    else:
        stance = "RANGE-BOUND CONSOLIDATION"
        stance_color = "orange"

    # Compile Bull Drivers
    bull_points = []
    if rev_growth > 12.0:
        bull_points.append(f"Strong top-line revenue momentum expanding at {rev_growth:.1f}% YoY.")
    if peg < 1.2:
        bull_points.append(f"Attractive valuation relative to growth with PEG ratio at {peg:.2f} (under 1.2 indicates growth at a reasonable price).")
    if cmf > 0.02:
        bull_points.append(f"Chaikin Money Flow (+{cmf:.2f}) confirms sustained institutional volume accumulation.")
    if "3" in analyst_upside or "2" in analyst_upside:
        bull_points.append(f"Institutional brokerages forecast {analyst_upside} consensus upside with {analyst_conviction} Buy ratings.")
    if rsi < 45.0:
        bull_points.append(f"Momentum oscillator (RSI {rsi:.1f}) indicates stock is approaching oversold territory ready for mean-reversion.")
    if not bull_points:
        bull_points.append("Steady defensive balance sheet with predictable market share stability.")

    # Compile Risk Headwinds
    risk_points = []
    if "-" in dma_dist:
        risk_points.append(f"Trading {dma_dist} below its 200-Day Moving Average, acting as multi-week technical overhead resistance.")
    if "+" in crude_trend and ("1" in crude_trend or "2" in crude_trend):
        risk_points.append(f"Macro commodity pressure: Brent crude is up {crude_trend} over the last quarter, elevating input cost inflation.")
    if pe > 40.0:
        risk_points.append(f"Elevated price multiple at {pe:.1f}× P/E leaves little margin of error for quarterly earnings misses.")
    if rsi > 65.0:
        risk_points.append(f"Short-term momentum is overheated (RSI {rsi:.1f}) pointing to near-term profit booking.")
    if not risk_points:
        risk_points.append("Vulnerable to broader emerging-market capital outflows if US 10-Year Treasury yields spike.")

    # Executive Summary Paragraph
    executive_summary = (
        f"{stock_name} ({ticker}) is positioned in **{stance}** within the current "
        f"**{regime_name.replace('_', ' ')}** macro environment. Over the next 3 months, "
        f"the probabilistic engine projects a primary consolidation-to-accumulation range of "
        f"**₹{p10_3m_price:,.0f} to ₹{p90_3m_price:,.0f}** (median trajectory: {p50_3m_ret:+.1f}%). "
        f"Key structural support is bolstered by {bull_points[0].lower()} "
        f"while primary downside vigilance is anchored by {risk_points[0].lower()}"
    )

    return {
        "stance": stance,
        "stance_color": stance_color,
        "executive_summary": executive_summary,
        "bull_catalysts": bull_points[:3],
        "risk_headwinds": risk_points[:3]
    }
