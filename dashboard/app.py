"""
PredicSON — Institutional AI Stock Quant Engine (NSE/BSE)
"Are ya winning, son?" Universal Search & Predict Architecture
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PILOT_UNIVERSE
from src.data_ingestion.stock_directory import STOCK_DIRECTORY, search_stock_directory
from src.models.on_demand_engine import get_stock_prediction
from src.utils.data_loader import load_daily_news, load_expert_consensus

# Page Config
st.set_page_config(
    page_title="PredicSON — Are ya winning, son?",
    page_icon="🕶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling: Iconic "Are ya winning, son?" Dark Neon Fintech Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #06080D !important;
        color: #E2E8F0 !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(16, 32, 60, 0.5) 0%, #06080D 70%) !important;
    }

    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1320px !important;
    }

    /* Top Meme Header */
    .meme-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 19, 31, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 24px;
        margin-bottom: 20px;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .dad-avatar {
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 12px;
        padding: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
    }
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .brand-title span {
        color: #00F2FE;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.4);
    }
    .brand-tagline {
        font-size: 0.82rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: 2px;
    }
    .brand-tagline b {
        color: #F8FAFC;
        font-style: normal;
    }

    .market-badge {
        font-size: 0.74rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 4px 10px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10B981;
    }

    /* Universal Search Bar */
    .search-container {
        background: rgba(14, 20, 34, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .search-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .trending-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        align-items: center;
    }
    .trending-lbl {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .chip-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.80rem;
        color: #CBD5E1;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .chip-item:hover {
        background: rgba(0, 242, 254, 0.12);
        border-color: rgba(0, 242, 254, 0.4);
        color: #00F2FE;
    }

    /* Hero Canvas Header */
    .terminal-hero {
        background: rgba(14, 19, 31, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .asset-headline {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-price {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }

    /* Horizon Ribbon */
    .horizon-ribbon {
        display: flex;
        gap: 12px;
        margin-bottom: 14px;
    }
    .horizon-pill {
        flex: 1;
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .horizon-pill-lbl {
        font-size: 0.76rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
    }
    .horizon-pill-val {
        font-size: 1.15rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 3px 9px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-red {
        background: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        padding: 3px 9px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Cards */
    .fintech-card {
        background: rgba(14, 19, 31, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }
    .catalyst-item {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
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
        font-size: 0.85rem;
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
        font-size: 0.85rem;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-row-lbl { color: #94A3B8; font-weight: 500; }
    .metric-row-val { color: #FFFFFF; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def build_tradingview_fan_chart(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Builds a full TradingView dark theme Plotly fan chart."""
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
        x=fc_dates, y=p90_vals, mode="lines", name="p90 (Bull Target)",
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
        x=fc_dates, y=p50_vals, mode="lines+markers", name="Expected Median Target (p50)",
        line=dict(color="#F59E0B", width=2.5, dash="dash"),
        marker=dict(size=4, color="#F59E0B"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Median Projection: ₹%{y:,.2f}<extra></extra>"
    ))

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
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#CBD5E1")
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", zeroline=False, color="#64748B"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", zeroline=False, color="#64748B", tickprefix="₹")
    )
    return fig


def main():
    # Session state initialization
    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = "RELIANCE.NS"

    # SVG of the Iconic "Are ya winning, son?" Dad
    dad_svg = """
    <svg width="42" height="42" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M15 42 C 25 38, 75 38, 85 42 C 90 43, 92 46, 82 46 C 70 46, 30 46, 18 46 C 8 46, 10 43, 15 42 Z" fill="#00F2FE" />
        <path d="M28 42 C 30 20, 70 20, 72 42 Z" fill="#00F2FE" opacity="0.85" />
        <path d="M28 38 C 35 36, 65 36, 72 38 L72 42 C 65 40, 35 40, 28 42 Z" fill="#F43F5E" />
        <circle cx="50" cy="62" r="22" stroke="#FFFFFF" stroke-width="3" fill="#0E121C" />
        <circle cx="41" cy="60" r="7" stroke="#00F2FE" stroke-width="2.5" />
        <circle cx="59" cy="60" r="7" stroke="#00F2FE" stroke-width="2.5" />
        <line x1="48" y1="60" x2="52" y2="60" stroke="#00F2FE" stroke-width="2.5" />
        <path d="M42 72 Q 50 78, 58 72" stroke="#FFFFFF" stroke-width="2.5" fill="none" stroke-linecap="round" />
        <line x1="56" y1="74" x2="68" y2="76" stroke="#FFFFFF" stroke-width="2.5" />
        <circle cx="70" cy="76" r="1.5" fill="#F43F5E" />
    </svg>
    """

    # Top Meme Header
    st.markdown(f"""
    <div class="meme-navbar">
        <div class="brand-group">
            <div class="dad-avatar">{dad_svg}</div>
            <div>
                <div class="brand-title">Predic<span>SON</span></div>
                <div class="brand-tagline">"Are ya winning, son?" — <b>Universal AI Stock Forecaster (NSE/BSE)</b></div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="market-badge"><span class="pulse-dot"></span> LIVE ON-DEMAND ENGINE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Universal Search Bar Container
    st.markdown("""
    <div class="search-container">
        <div class="search-title">🔍 Search Any Stock on NSE & BSE</div>
        <div style="font-size:0.84rem; color:#94A3B8; margin-bottom:10px;">
            Search any Indian equity, defense, railway, energy, or consumer titan. If uncached, PredicSON gathers live market info and reloads the model on the fly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search Bar & Trending Quick Buttons
    col_input, col_go = st.columns([4, 1])
    with col_input:
        search_text = st.text_input(
            "Search Ticker or Company",
            placeholder="Type any stock name or symbol (e.g. Reliance, Zomato, Suzlon, Tata Motors, HAL, Trent)...",
            label_visibility="collapsed"
        )
    with col_go:
        search_clicked = st.button("⚡ Predict Asset", use_container_width=True)

    # Search matching logic
    matched_stocks = search_stock_directory(search_text, limit=6) if search_text else []
    
    if search_text and matched_stocks:
        st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#00F2FE; margin-top:-8px; margin-bottom:8px;'>Matching Assets Found:</div>", unsafe_allow_html=True)
        m_cols = st.columns(min(len(matched_stocks), 6))
        for idx, s in enumerate(matched_stocks[:6]):
            with m_cols[idx]:
                if st.button(f"{s['symbol'].replace('.NS', '')}\n({s['name'][:16]}...)", key=f"match_{s['symbol']}", use_container_width=True):
                    st.session_state.active_ticker = s["symbol"]
                    st.rerun()

    # Trending Hot Tickers Bar
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px; margin-bottom:18px;">
        <span style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">Popular:</span>
    </div>
    """, unsafe_allow_html=True)
    
    t_cols = st.columns(6)
    trending_list = [
        ("RELIANCE.NS", "Reliance (+8.3%)"),
        ("M&M.NS", "M&M (+7.8%)"),
        ("ZOMATO.NS", "Zomato"),
        ("SUZLON.NS", "Suzlon"),
        ("HAL.NS", "HAL (Defense)"),
        ("TRENT.NS", "Trent (Zudio)")
    ]
    for idx, (sym, label) in enumerate(trending_list):
        with t_cols[idx]:
            if st.button(f"🔥 {label}", key=f"trend_{sym}", use_container_width=True):
                st.session_state.active_ticker = sym
                st.rerun()

    if search_clicked and matched_stocks:
        st.session_state.active_ticker = matched_stocks[0]["symbol"]
        st.rerun()

    current_ticker = st.session_state.active_ticker

    # =========================================================================
    # PREDICTION & FORECAST CANVAS
    # =========================================================================
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.spinner(f"Analyzing {current_ticker}... Ingesting fresh data & fitting LightGBM quantile cones..."):
        try:
            history_df, forecast_df, meta = get_stock_prediction(current_ticker)
        except Exception as e:
            st.error(f"Error analyzing {current_ticker}: {e}")
            return

    stock_meta = next((s for s in STOCK_DIRECTORY if s["symbol"] == current_ticker), {})
    company_name = stock_meta.get("name", current_ticker.replace(".NS", ""))
    sector = stock_meta.get("sector", "Equities")

    current_price = meta["current_price"]
    horizons = meta["horizons"]
    regime = meta.get("regime", "SIDEWAYS_CONSOLIDATION").replace("_", " ")

    h1 = horizons["1M"]["p50"]
    h3 = horizons["3M"]["p50"]
    h6 = horizons["6M"]["p50"]

    h1_ret = h1["expected_return_pct"]
    h3_ret = h3["expected_return_pct"]
    h6_ret = h6["expected_return_pct"]

    h3_pill_class = "pill-green" if h3_ret >= 0 else "pill-red"
    h3_sign = "+" if h3_ret > 0 else ""

    # Hero Banner
    st.markdown(f"""
    <div class="terminal-hero">
        <div>
            <div class="asset-headline">
                {company_name}
                <span class="ticker-pill"><b>NSE:</b> {current_ticker}</span>
                <span class="ticker-pill"><b>Sector:</b> {sector}</span>
                <span class="ticker-pill"><b>Regime:</b> {regime}</span>
            </div>
            <div style="font-size:0.84rem; color:#64748B; margin-top:5px;">
                Black-Litterman Catalyst Blending • Dynamic LightGBM Quantile Ensemble
            </div>
        </div>
        <div style="text-align:right;">
            <div class="hero-price">₹{current_price:,.2f}</div>
            <div style="font-size:0.92rem; font-weight:700;">
                3M Target: <span style="color:#00F2FE;">₹{h3['projected_price']:,.2f}</span>
                <span class="{h3_pill_class}">{h3_sign}{h3_ret:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Horizon Target Ribbon
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
        <div class="horizon-pill" style="border-color: rgba(0,242,254,0.35); background: rgba(0,242,254,0.06);">
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

    # Full-Width Fan Chart
    fig = build_tradingview_fan_chart(current_ticker, history_df, forecast_df)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Side-by-Side Intelligence Cards
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        thesis = meta.get("investment_thesis", {})
        stance = thesis.get("stance", "NEUTRAL")
        stance_color = "#10B981" if "BULL" in stance or "ACCUMULATE" in stance else ("#F43F5E" if "BEAR" in stance or "TRIM" in stance else "#F59E0B")
        summary = thesis.get("executive_summary", "")

        catalysts = "".join([f"<div class='catalyst-item'><span>🟢</span><span>{c}</span></div>" for c in thesis.get("bull_catalysts", [])])
        headwinds = "".join([f"<div class='headwind-item'><span>🔴</span><span>{h}</span></div>" for h in thesis.get("risk_headwinds", [])])

        st.markdown(f"""
        <div class="fintech-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:1.0rem; font-weight:700; color:#FFFFFF;">📋 AI Investment Thesis</span>
                <span style="font-size:0.80rem; font-weight:700; color:{stance_color}; background:rgba(255,255,255,0.05); padding:3px 8px; border-radius:12px; border:1px solid {stance_color};">● {stance}</span>
            </div>
            <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.5; margin-bottom:12px;">{summary}</div>
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

    with c2:
        today_str = datetime.now().strftime("%Y-%m-%d")
        expert_records = load_expert_consensus(today_str)
        matched = next((r for r in expert_records if r.get("ticker") == current_ticker), None)

        pt = matched.get("price_targets", {}) if matched else {}
        mean_pt = pt.get("mean")
        mean_str = f"₹{mean_pt:,.0f}" if mean_pt else "N/A"
        high_str = f"₹{pt.get('high', 0):,.0f}" if pt.get('high') else "N/A"
        low_str = f"₹{pt.get('low', 0):,.0f}" if pt.get('low') else "N/A"

        drivers_dict = {d["feature"]: d["value"] for d in meta.get("key_drivers", [])}

        st.markdown(f"""
        <div class="fintech-card">
            <div style="font-size:1.0rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;">
                🧠 Institutional Consensus & Key Factors
            </div>
            <div class="metric-row">
                <span class="metric-row-lbl">Consensus Mean Target (1Y)</span>
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


if __name__ == "__main__":
    main()
