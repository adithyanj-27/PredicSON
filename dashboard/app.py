"""
PredicSON — Institutional AI Stock Quant Terminal (NSE/BSE)
Zerodha Kite / TradingView 2-Column Split Terminal Architecture
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PILOT_UNIVERSE, EXPERTS_DIR, NEWS_DIR, PREDICTIONS_DIR
from src.models.fan_chart_generator import generate_fan_chart_data
from src.models.screener_engine import generate_market_screener
from src.models.portfolio_allocator import optimize_portfolio
from src.utils.data_loader import load_daily_news, load_expert_consensus

# Page Config
st.set_page_config(
    page_title="PredicSON — Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Screener Data for Watchlist & Tab 2
screener_df = generate_market_screener(use_cache=True)
ticker_lookup = {row["ticker"]: row for _, row in screener_df.iterrows()} if not screener_df.empty else {}

# Premium Zerodha Kite / TradingView Dark CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Theme & Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #07090E !important;
        color: #E2E8F0 !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(14, 28, 54, 0.45) 0%, #07090E 65%) !important;
    }

    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Top Branding Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 18, 28, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 14px;
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.3);
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .brand-title span { color: #00F2FE; }
    .status-badge {
        font-size: 0.70rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 3px 8px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10B981;
    }
    .ticker-bar {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .ticker-pill {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        display: flex;
        gap: 5px;
    }
    .ticker-pill b { color: #94A3B8; }
    .pos { color: #10B981; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .neu { color: #38BDF8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* Custom Glass Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(14, 18, 28, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 6px !important;
        margin-bottom: 14px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 6px 16px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.16) 0%, rgba(79, 172, 254, 0.16) 100%) !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
    }

    /* Kite-Style Watchlist Container & Items */
    .watchlist-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px 10px 0 0;
        font-size: 0.82rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Override Streamlit Radio as Clickable Watchlist Cards */
    div[data-testid="stRadio"] > div {
        background: rgba(14, 18, 28, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: none;
        border-radius: 0 0 10px 10px;
        max-height: 720px;
        overflow-y: auto;
        padding: 6px;
        gap: 4px;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(18, 24, 38, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 9px 12px;
        margin: 0 !important;
        transition: all 0.15s ease;
        cursor: pointer;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(0, 242, 254, 0.25);
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.14) 0%, rgba(79, 172, 254, 0.08) 100%) !important;
        border: 1px solid rgba(0, 242, 254, 0.45) !important;
        box-shadow: 0 2px 10px rgba(0, 242, 254, 0.12) !important;
    }

    /* Terminal Main Canvas Header */
    .terminal-header {
        background: rgba(14, 19, 31, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .asset-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .asset-price-box {
        text-align: right;
    }
    .hero-price {
        font-size: 1.85rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }

    /* Ribbon Horizon Pills */
    .horizon-ribbon {
        display: flex;
        gap: 10px;
        margin-bottom: 12px;
    }
    .horizon-pill {
        flex: 1;
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .horizon-pill-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
    }
    .horizon-pill-val {
        font-size: 1.05rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .pill-red {
        background: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Cards */
    .fintech-card {
        background: rgba(14, 19, 31, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }
    .catalyst-item {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.84rem;
        color: #E2E8F0;
        display: flex;
        gap: 8px;
    }
    .headwind-item {
        background: rgba(244, 63, 94, 0.06);
        border: 1px solid rgba(244, 63, 94, 0.18);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.84rem;
        color: #E2E8F0;
        display: flex;
        gap: 8px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.84rem;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-row-lbl { color: #94A3B8; font-weight: 500; }
    .metric-row-val { color: #FFFFFF; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def build_tradingview_fan_chart(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Builds an edge-to-edge TradingView dark theme Plotly fan chart."""
    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["close"],
        mode="lines",
        name="Historical Price",
        line=dict(color="#38BDF8", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Historical Price: ₹%{y:,.2f}<extra></extra>"
    ))

    last_hist_date = history_df.index[-1]
    last_hist_price = float(history_df["close"].iloc[-1])
    
    fc_dates = [last_hist_date] + list(forecast_df.index)
    p10_vals = [last_hist_price] + list(forecast_df["p10"])
    p25_vals = [last_hist_price] + list(forecast_df["p25"])
    p50_vals = [last_hist_price] + list(forecast_df["p50"])
    p75_vals = [last_hist_price] + list(forecast_df["p75"])
    p90_vals = [last_hist_price] + list(forecast_df["p90"])

    # Outer 80% Cone (p10 to p90)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p90_vals, mode="lines", name="p90 (Bull Outlier)",
        line=dict(color="rgba(0, 242, 254, 0.25)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p10_vals, mode="lines", name="80% Probability Cone (p10 - p90)",
        fill="tonexty", fillcolor="rgba(0, 242, 254, 0.08)",
        line=dict(color="rgba(0, 242, 254, 0.25)", width=1),
        hovertemplate="<b>80% Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Inner 50% Core Cone (p25 to p75)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p75_vals, mode="lines", name="p75 (Upper Core)",
        line=dict(color="rgba(0, 242, 254, 0.5)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p25_vals, mode="lines", name="50% Core Probability Cone (p25 - p75)",
        fill="tonexty", fillcolor="rgba(0, 242, 254, 0.22)",
        line=dict(color="rgba(0, 242, 254, 0.5)", width=1),
        hovertemplate="<b>50% Core Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Median Path (p50)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p50_vals, mode="lines+markers", name="Expected Median (p50)",
        line=dict(color="#F59E0B", width=2.5, dash="dash"),
        marker=dict(size=4, color="#F59E0B"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Median Target: ₹%{y:,.2f}<extra></extra>"
    ))

    # Today vertical delimiter
    fig.add_vline(
        x=last_hist_date, line_width=1.5, line_dash="dash", line_color="#64748B",
        annotation_text="Today", annotation_position="top left",
        annotation_font=dict(color="#94A3B8", size=10)
    )

    fig.update_layout(
        paper_bgcolor="#0A0D15",
        plot_bgcolor="#0A0D15",
        font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=10, b=20),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#CBD5E1")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            zeroline=False,
            color="#64748B"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            zeroline=False,
            color="#64748B",
            tickprefix="₹"
        )
    )
    return fig


def main():
    # Top Branding Bar
    st.markdown("""
    <div class="top-navbar">
        <div class="brand-section">
            <div class="brand-logo">📈</div>
            <div class="brand-title">PREDIC<span>SON</span></div>
            <div class="status-badge"><span class="pulse-dot"></span> LIVE QUANT TERMINAL</div>
        </div>
        <div class="ticker-bar">
            <div class="ticker-pill"><b>NIFTY 50</b> <span class="pos">25,127 (+0.4%)</span></div>
            <div class="ticker-pill"><b>INDIA VIX</b> <span class="neu">13.20 (-1.8%)</span></div>
            <div class="ticker-pill"><b>BRENT CRUDE</b> <span class="neu">$74.20 (-0.6%)</span></div>
            <div class="ticker-pill"><b>USD/INR</b> <span class="neu">84.05</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_terminal, tab_screener, tab_portfolio = st.tabs([
        "⚡ Terminal (Watchlist & Fan Chart)",
        "🚀 Market Opportunity Screener",
        "💼 Modern Portfolio Allocator"
    ])

    # =========================================================================
    # TAB 1: KITE-STYLE 2-COLUMN SPLIT TERMINAL
    # =========================================================================
    with tab_terminal:
        col_watchlist, col_chart = st.columns([1.1, 3.4], gap="medium")

        # -------------------------------------------------------------
        # LEFT COLUMN: WATCHLIST (Zerodha Kite Style)
        # -------------------------------------------------------------
        with col_watchlist:
            search_query = st.text_input("🔍 Search Asset...", placeholder="Filter by name...", label_visibility="collapsed")
            
            # Prepare ticker list
            all_tickers = list(screener_df["ticker"]) if not screener_df.empty else list(PILOT_UNIVERSE.keys())
            
            if search_query:
                filtered_tickers = [t for t in all_tickers if search_query.lower() in t.lower() or search_query.lower() in PILOT_UNIVERSE.get(t, {}).get("name", "").lower()]
            else:
                filtered_tickers = all_tickers

            if not filtered_tickers:
                filtered_tickers = all_tickers

            st.markdown(f"<div class='watchlist-header'><span>Universe Watchlist</span><span>({len(filtered_tickers)} Assets)</span></div>", unsafe_allow_html=True)

            # Format radio card labels
            def render_watchlist_label(t):
                row = ticker_lookup.get(t, {})
                price = row.get("current_price", PILOT_UNIVERSE.get(t, {}).get("current_price", 1000.0))
                ret = row.get("expected_return_3m", 5.0)
                badge = "🟢" if ret > 6 else ("🔵" if ret > 2 else ("🔴" if ret < -4 else "⚪"))
                sign = "+" if ret > 0 else ""
                short_name = t.replace(".NS", "")
                return f"{short_name}  •  ₹{price:,.1f}  [{sign}{ret:.1f}% {badge}]"

            selected_ticker = st.radio(
                "Watchlist",
                options=filtered_tickers,
                format_func=render_watchlist_label,
                label_visibility="collapsed"
            )

        # -------------------------------------------------------------
        # RIGHT COLUMN: MAIN TRADING & FORECASTING CANVAS
        # -------------------------------------------------------------
        with col_chart:
            stock_info = PILOT_UNIVERSE.get(selected_ticker, {})
            name = stock_info.get("name", selected_ticker)
            sector = stock_info.get("sector", "Equities").split("/")[0].strip()

            # Generate forecast data
            with st.spinner(f"Computing real-time fan chart for {selected_ticker}..."):
                try:
                    history_df, forecast_df, meta = generate_fan_chart_data(selected_ticker)
                except Exception as e:
                    st.error(f"Error computing forecast: {e}")
                    return

            current_price = meta["current_price"]
            horizons = meta["horizons"]
            regime = meta.get("regime", "SIDEWAYS_CONSOLIDATION").replace("_", " ")

            h1 = horizons["1M"]["p50"]
            h3 = horizons["3M"]["p50"]
            h6 = horizons["6M"]["p50"]

            h1_ret = h1["expected_return_pct"]
            h3_ret = h3["expected_return_pct"]
            h6_ret = h6["expected_return_pct"]

            # Terminal Header Bar
            h3_pill_class = "pill-green" if h3_ret >= 0 else "pill-red"
            h3_sign = "+" if h3_ret > 0 else ""

            st.markdown(f"""
            <div class="terminal-header">
                <div>
                    <div class="asset-title">
                        {name}
                        <span class="ticker-pill"><b>NSE:</b> {selected_ticker}</span>
                        <span class="ticker-pill"><b>Sector:</b> {sector}</span>
                        <span class="ticker-pill"><b>Regime:</b> {regime}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#64748B; margin-top:4px;">
                        Black-Litterman Catalyst Blending Active • Monotonic Non-Crossing Fan Cones
                    </div>
                </div>
                <div class="asset-price-box">
                    <div class="hero-price">₹{current_price:,.2f}</div>
                    <div style="font-size:0.88rem; font-weight:700;">
                        3M Target: <span style="color:#00F2FE;">₹{h3['projected_price']:,.2f}</span>
                        <span class="{h3_pill_class}">{h3_sign}{h3_ret:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Horizontal Horizon Ribbon
            p1_class = "pill-green" if h1_ret >= 0 else "pill-red"
            p3_class = "pill-green" if h3_ret >= 0 else "pill-red"
            p6_class = "pill-green" if h6_ret >= 0 else "pill-red"

            st.markdown(f"""
            <div class="horizon-ribbon">
                <div class="horizon-pill">
                    <div>
                        <div class="horizon-pill-lbl">1-Month Target (21d)</div>
                        <div class="horizon-pill-val">₹{h1['projected_price']:,.2f}</div>
                    </div>
                    <span class="{p1_class}">{h1_ret:+.1f}%</span>
                </div>
                <div class="horizon-pill" style="border-color: rgba(0,242,254,0.3); background: rgba(0,242,254,0.06);">
                    <div>
                        <div class="horizon-pill-lbl" style="color:#00F2FE;">3-Month Target (Core 63d)</div>
                        <div class="horizon-pill-val">₹{h3['projected_price']:,.2f}</div>
                    </div>
                    <span class="{p3_class}">{h3_ret:+.1f}%</span>
                </div>
                <div class="horizon-pill">
                    <div>
                        <div class="horizon-pill-lbl">6-Month Target (126d)</div>
                        <div class="horizon-pill-val">₹{h6['projected_price']:,.2f}</div>
                    </div>
                    <span class="{p6_class}">{h6_ret:+.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Centerpiece TradingView Fan Chart
            fig = build_tradingview_fan_chart(selected_ticker, history_df, forecast_df)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Side-by-Side 50/50 Intelligence Cards
            card_col1, card_col2 = st.columns([1, 1], gap="medium")

            with card_col1:
                thesis = meta.get("investment_thesis", {})
                stance = thesis.get("stance", "NEUTRAL")
                stance_color = "#10B981" if "BULL" in stance or "ACCUMULATE" in stance else ("#F43F5E" if "BEAR" in stance or "TRIM" in stance else "#F59E0B")
                summary = thesis.get("executive_summary", "")

                catalysts = "".join([f"<div class='catalyst-item'><span>🟢</span><span>{c}</span></div>" for c in thesis.get("bull_catalysts", [])])
                headwinds = "".join([f"<div class='headwind-item'><span>🔴</span><span>{h}</span></div>" for h in thesis.get("risk_headwinds", [])])

                st.markdown(f"""
                <div class="fintech-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-size:0.95rem; font-weight:700; color:#FFFFFF;">📋 AI Investment Thesis</span>
                        <span style="font-size:0.80rem; font-weight:700; color:{stance_color}; background:rgba(255,255,255,0.05); padding:3px 8px; border-radius:12px; border:1px solid {stance_color};">● {stance}</span>
                    </div>
                    <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.5; margin-bottom:12px;">{summary}</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                        <div>
                            <div style="font-size:0.75rem; font-weight:700; color:#10B981; margin-bottom:6px; text-transform:uppercase;">Bull Catalysts</div>
                            {catalysts}
                        </div>
                        <div>
                            <div style="font-size:0.75rem; font-weight:700; color:#F43F5E; margin-bottom:6px; text-transform:uppercase;">Downside Headwinds</div>
                            {headwinds}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with card_col2:
                today_str = datetime.now().strftime("%Y-%m-%d")
                expert_records = load_expert_consensus(today_str)
                matched = next((r for r in expert_records if r.get("ticker") == selected_ticker), None)

                pt = matched.get("price_targets", {}) if matched else {}
                mean_pt = pt.get("mean")
                mean_str = f"₹{mean_pt:,.0f}" if mean_pt else "N/A"
                high_str = f"₹{pt.get('high', 0):,.0f}" if pt.get('high') else "N/A"
                low_str = f"₹{pt.get('low', 0):,.0f}" if pt.get('low') else "N/A"

                drivers_dict = {d["feature"]: d["value"] for d in meta.get("key_drivers", [])}

                st.markdown(f"""
                <div class="fintech-card">
                    <div style="font-size:0.95rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;">
                        🧠 Institutional Consensus & Key Signals
                    </div>
                    <div class="metric-row">
                        <span class="metric-row-lbl">Institutional Mean Target (1Y)</span>
                        <span class="metric-row-val" style="color:#00F2FE;">{mean_str} (Range: {low_str} - {high_str})</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-row-lbl">Analyst Buy Conviction</span>
                        <span class="metric-row-val">{drivers_dict.get('analyst_conviction', '80%')}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-row-lbl">Chaikin Money Flow (CMF)</span>
                        <span class="metric-row-val">{drivers_dict.get('cmf_20d', '0.00')}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-row-lbl">P/E Multiple & PEG Ratio</span>
                        <span class="metric-row-val">{drivers_dict.get('pe_ratio', '20x')} • PEG: {drivers_dict.get('peg_ratio', '1.5')}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-row-lbl">RSI (14-Day Momentum)</span>
                        <span class="metric-row-val">{drivers_dict.get('rsi_14', '50.0')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: MARKET OPPORTUNITY SCREENER
    # =========================================================================
    with tab_screener:
        st.markdown("### 🚀 Cross-Sectional Opportunity Screener")
        st.caption("Real-time ranking of 25 titans across Expected Return, Downside Risk, Risk-Reward, and Money Flow.")

        if not screener_df.empty:
            strong_buys = len(screener_df[screener_df["signal"].str.contains("STRONG BUY")])
            accumulates = len(screener_df[screener_df["signal"].str.contains("ACCUMULATE")])
            avoids = len(screener_df[screener_df["signal"].str.contains("AVOID")])

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Universe</div><div class='horizon-pill-val'>{len(screener_df)} Titans</div></div></div>", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Strong Buy</div><div class='horizon-pill-val' style='color:#10B981;'>{strong_buys}</div></div><span class='pill-green'>Top Alpha</span></div>", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Accumulate</div><div class='horizon-pill-val' style='color:#00F2FE;'>{accumulates}</div></div><span class='pill-green'>Positive</span></div>", unsafe_allow_html=True)
            with sc4:
                st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Avoid / Trim</div><div class='horizon-pill-val' style='color:#F43F5E;'>{avoids}</div></div><span class='pill-red'>Caution</span></div>", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            f1, f2 = st.columns([1, 1])
            with f1:
                sec_opts = sorted(screener_df["sector"].unique())
                sec_filter = st.multiselect("Filter Sector", options=sec_opts, default=[])
            with f2:
                sig_opts = sorted(screener_df["signal"].unique())
                sig_filter = st.multiselect("Filter Signal", options=sig_opts, default=[])

            filtered = screener_df.copy()
            if sec_filter:
                filtered = filtered[filtered["sector"].isin(sec_filter)]
            if sig_filter:
                filtered = filtered[filtered["signal"].isin(sig_filter)]

            display_cols = [
                "ticker", "name", "sector", "current_price",
                "expected_return_3m", "p10_downside_3m", "risk_reward_ratio",
                "signal", "cmf_flow", "peg_ratio", "pe_ratio"
            ]
            renamed = filtered[display_cols].copy()
            renamed.columns = [
                "Ticker", "Company", "Sector", "Price (₹)",
                "3M Return %", "Downside p10 %", "Risk-Reward",
                "Signal", "Money Flow", "PEG", "P/E"
            ]

            st.dataframe(renamed, hide_index=True, use_container_width=True)

            # Chart
            fig_bar = px.bar(
                screener_df.sort_values(by="expected_return_3m", ascending=True),
                x="expected_return_3m",
                y="ticker",
                orientation="h",
                color="expected_return_3m",
                color_continuous_scale=["#F43F5E", "#38BDF8", "#10B981"],
                labels={"expected_return_3m": "Expected Return (%)", "ticker": "Asset"},
                template="plotly_dark",
                height=520
            )
            fig_bar.update_layout(
                paper_bgcolor="#0A0D15",
                plot_bgcolor="#0A0D15",
                font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # TAB 3: MODERN PORTFOLIO ALLOCATOR
    # =========================================================================
    with tab_portfolio:
        st.markdown("### 💼 Modern Portfolio Theory (MPT) Capital Allocator")
        st.caption("Solves for the Maximum Sharpe Ratio Portfolio using model expected returns and 252-day empirical covariance.")

        p1, p2 = st.columns([1, 2])
        with p1:
            total_invest = st.number_input("Total Investment Capital (₹ INR)", min_value=10000.0, max_value=50000000.0, value=100000.0, step=10000.0)
            max_weight = st.slider("Maximum Cap per Asset (%)", min_value=10, max_value=40, value=25) / 100.0
            recalc = st.button("⚡ Recalculate Markowitz Allocation", use_container_width=True)

        with p2:
            st.markdown("""
            <div class="fintech-card">
                <div style="font-weight:700; color:#00F2FE; margin-bottom:4px;">Markowitz Formulation</div>
                <div style="font-size:0.85rem; color:#94A3B8; line-height:1.5;">
                    • Maximizes risk-adjusted excess return: <span style="font-family:'JetBrains Mono'; color:#FFFFFF;">max (w<sup>T</sup>r - r<sub>f</sub>) / √(w<sup>T</sup>Σw)</span><br>
                    • Allocates across non-correlated sectors to minimize portfolio drawdown.<br>
                    • Enforces single-asset ceiling to avoid idiosyncratic shock exposure.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if not screener_df.empty:
            port_result = optimize_portfolio(screener_df, total_capital=total_invest, max_stock_weight=max_weight)

            if port_result and "allocations" in port_result:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Capital</div><div class='horizon-pill-val'>₹{port_result['total_capital']:,.0f}</div></div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Expected 3M Return</div><div class='horizon-pill-val' style='color:#10B981;'>{port_result['expected_3m_portfolio_return_pct']:+.1f}%</div></div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Volatility</div><div class='horizon-pill-val' style='color:#F59E0B;'>{port_result['portfolio_volatility_pct']:.1f}%</div></div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='horizon-pill'><div><div class='horizon-pill-lbl'>Sharpe Ratio</div><div class='horizon-pill-val' style='color:#00F2FE;'>{port_result['portfolio_sharpe']:.2f}</div></div></div>", unsafe_allow_html=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                c_alloc1, c_alloc2 = st.columns([1.2, 0.8])
                alloc_df = pd.DataFrame(port_result["allocations"])
                
                with c_alloc1:
                    st.markdown("#### 🎯 Optimal Asset Deployment Plan")
                    disp_alloc = alloc_df[["ticker", "name", "sector", "weight_pct", "capital_inr", "expected_3m_return"]].copy()
                    disp_alloc.columns = ["Ticker", "Company", "Sector", "Weight %", "Capital (₹)", "Expected 3M Return %"]
                    st.dataframe(disp_alloc, hide_index=True, use_container_width=True)

                with c_alloc2:
                    st.markdown("#### 🍩 Capital Allocation Breakdown")
                    fig_donut = px.pie(
                        alloc_df,
                        values="weight_pct",
                        names="name",
                        hole=0.55,
                        template="plotly_dark",
                        color_discrete_sequence=["#00F2FE", "#38BDF8", "#10B981", "#818CF8", "#F59E0B"]
                    )
                    fig_donut.update_layout(
                        paper_bgcolor="#0A0D15",
                        plot_bgcolor="#0A0D15",
                        margin=dict(l=20, r=20, t=20, b=20),
                        font=dict(family="Plus Jakarta Sans", color="#94A3B8")
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)


if __name__ == "__main__":
    main()
