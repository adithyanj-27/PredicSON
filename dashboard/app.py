"""
PredicSON — Institutional AI Stock Quant Forecaster (NSE/BSE)
Clean Google/Perplexity-style Search Architecture & Stoxify Brokerage Aesthetic.
"""

import html
import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.stock_directory import STOCK_DIRECTORY, search_stock_directory
from src.models.on_demand_engine import get_stock_prediction
from src.utils.data_loader import load_expert_consensus

# Page Configuration
st.set_page_config(
    page_title="PredicSON — Institutional AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stoxify-inspired Dark Brokerage Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #080D14 !important;
        color: #F8FAFC !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(14, 165, 233, 0.12) 0%, #080D14 65%) !important;
    }

    #MainMenu, footer, header, .stDeployButton {
        display: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
    }

    /* Primary Gradient Brand Text */
    .brand-gradient {
        background: linear-gradient(135deg, #0EA5E9 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Google/Perplexity Centered Homepage */
    .home-hero {
        text-align: center;
        padding-top: 6vh;
        padding-bottom: 24px;
    }
    .home-logo-wrap {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 18px;
        filter: drop-shadow(0 0 24px rgba(14, 165, 233, 0.35));
    }
    .home-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: #F8FAFC;
        margin-bottom: 6px;
        line-height: 1.1;
    }
    .home-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 28px;
    }

    /* Persistent Top Navbar */
    .terminal-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 18px;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
    }
    .nav-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #F8FAFC;
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

    /* Stock Terminal Hero Canvas */
    .terminal-hero {
        background: rgba(15, 23, 42, 0.75);
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
    .ticker-pill {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #CBD5E1;
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
        background: rgba(15, 23, 42, 0.6);
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

    /* Research Cards */
    .fintech-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .catalyst-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 0.84rem;
        color: #E2E8F0;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .headwind-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 0.84rem;
        color: #CBD5E1;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.84rem;
    }
    .metric-row-lbl { color: #94A3B8; font-weight: 500; }
    .metric-row-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F8FAFC; }

    /* Match Dropdown Cards */
    .match-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.15s ease;
    }
    .match-card:hover {
        border-color: #0EA5E9;
        background: rgba(14, 165, 233, 0.12);
    }
</style>
""", unsafe_allow_html=True)


def get_stoxify_icon_svg(size: int = 64) -> str:
    """Stoxify-inspired vector brand icon representing multi-quantile forecast cone and upward trajectory."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="stoxGrad" x1="15%" y1="85%" x2="85%" y2="15%">
          <stop offset="0%" stop-color="#0284C7"/>
          <stop offset="50%" stop-color="#0EA5E9"/>
          <stop offset="100%" stop-color="#10B981"/>
        </linearGradient>
        <linearGradient id="coneGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="rgba(14, 165, 233, 0.35)"/>
          <stop offset="100%" stop-color="rgba(16, 185, 129, 0.05)"/>
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill="#0F172A" stroke="rgba(14, 165, 233, 0.45)" stroke-width="1.5"/>
      <path d="M14 44 L48 18 L48 38 Z" fill="url(#coneGrad)" />
      <path d="M14 44 L25 33 L31 38 L48 20" stroke="url(#stoxGrad)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="48" cy="20" r="3.5" fill="#10B981"/>
    </svg>
    """


def build_tradingview_fan_chart(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """TradingView-grade dark fan chart with historical series and probabilistic quantiles."""
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
        line=dict(color="rgba(14, 165, 233, 0.25)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p10_vals, mode="lines", name="80% Probability Cone (p10 - p90)",
        fill="tonexty", fillcolor="rgba(14, 165, 233, 0.08)",
        line=dict(color="rgba(14, 165, 233, 0.25)", width=1),
        hovertemplate="<b>80% Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Inner 50% Core Cone (p25 to p75)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p75_vals, mode="lines", name="p75 (Upper Core)",
        line=dict(color="rgba(14, 165, 233, 0.5)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p25_vals, mode="lines", name="50% Core Probability Cone (p25 - p75)",
        fill="tonexty", fillcolor="rgba(14, 165, 233, 0.22)",
        line=dict(color="rgba(14, 165, 233, 0.5)", width=1),
        hovertemplate="<b>50% Core Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Median Path (p50)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p50_vals, mode="lines+markers", name="Expected Median Target (p50)",
        line=dict(color="#10B981", width=2.5, dash="dash"),
        marker=dict(size=4, color="#10B981"),
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
        font=dict(family="Sora", color="#94A3B8"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=10, b=20),
        height=500,
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


def render_homepage():
    """Clean Google/Perplexity-style Centered Search Landing Page."""
    # Centered Brand Hero
    icon_svg = get_stoxify_icon_svg(size=72)
    st.markdown(f"""
    <div class="home-hero">
        <div class="home-logo-wrap">{icon_svg}</div>
        <div class="home-title">Predic<span class="brand-gradient">SON</span></div>
        <div class="home-subtitle">Institutional AI Stock Quant & Forecast Engine • Indian Equities (NSE & BSE)</div>
    </div>
    """, unsafe_allow_html=True)

    # Centered Search Bar Container
    c_left, c_mid, c_right = st.columns([1, 4.5, 1])
    with c_mid:
        with st.form("search_form", clear_on_submit=False, border=False):
            col_search_input, col_search_btn = st.columns([4.2, 1])
            with col_search_input:
                search_query = st.text_input(
                    "Search Input",
                    placeholder="Search any stock, ticker, brand, or sector (e.g. Federal Bank, Tata Motors, Jio, Zomato)...",
                    label_visibility="collapsed"
                )
            with col_search_btn:
                search_submitted = st.form_submit_button("⚡ Forecast", use_container_width=True)

        matched_stocks = search_stock_directory(search_query, limit=5) if search_query else []

        # If user pressed Enter or clicked the Forecast button:
        if search_submitted and search_query:
            if matched_stocks:
                st.session_state.active_ticker = matched_stocks[0]["symbol"]
                st.rerun()
            else:
                st.error(f"No Indian equity found matching '{search_query}'. Please check the symbol or name.")

        # If user is searching, render immediate match cards
        if search_query and matched_stocks:
            st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#0EA5E9; margin-top:8px; margin-bottom:8px;'>Matching Stocks on NSE / BSE (Press Enter or Click to Forecast):</div>", unsafe_allow_html=True)
            for idx, s in enumerate(matched_stocks):
                clean_sym = s['symbol'].replace('.NS', '').replace('.BO', '')
                sector_tag = s.get('sector', 'Equities')
                c_info, c_action = st.columns([4.2, 1])
                with c_info:
                    st.markdown(f"""
                    <div style="padding: 10px 14px; background: rgba(15,23,42,0.85); border:1px solid rgba(255,255,255,0.08); border-radius:8px; margin-bottom:6px;">
                        <span style="font-weight:700; font-size:1.0rem; color:#F8FAFC;">{html.escape(clean_sym)}</span>
                        <span style="color:#94A3B8; font-size:0.86rem; margin-left:10px;">{html.escape(str(s['name']))}</span>
                        <span style="font-size:0.75rem; color:#0EA5E9; margin-left:10px; background:rgba(14,165,233,0.12); padding:2px 8px; border-radius:4px; font-weight:600;">{html.escape(str(sector_tag))}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c_action:
                    if st.button(f"View {clean_sym}", key=f"home_pick_{s['symbol']}_{idx}", use_container_width=True):
                        st.session_state.active_ticker = s["symbol"]
                        st.rerun()

        # Popular / Trending Assets Row
        st.markdown("""
        <div style="text-align:center; margin-top:22px; margin-bottom:12px;">
            <span style="font-size:0.78rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;">Trending Assets:</span>
        </div>
        """, unsafe_allow_html=True)

        t_cols = st.columns(4)
        trending_list = [
            ("RELIANCE.NS", "Reliance (Jio)"),
            ("ZOMATO.NS", "Zomato (Blinkit)"),
            ("TRENT.NS", "Trent (Zudio)"),
            ("TATAMOTORS.NS", "Tata Motors"),
            ("SUZLON.NS", "Suzlon (Green)"),
            ("HAL.NS", "HAL (Defense)"),
            ("IRFC.NS", "IRFC (Railways)"),
            ("HDFCBANK.NS", "HDFC Bank")
        ]
        for idx, (sym, label) in enumerate(trending_list):
            col_target = t_cols[idx % 4]
            with col_target:
                if st.button(label, key=f"home_trend_{sym}", use_container_width=True):
                    st.session_state.active_ticker = sym
                    st.rerun()

        # Bottom Feature Badges
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:20px; margin-top:40px; font-size:0.80rem; color:#64748B;">
            <span>⚡ <b>Sub-Second Caching</b></span>
            <span>🎯 <b>LightGBM Quantile Ensemble</b></span>
            <span>🇮🇳 <b>2,590+ Listed Indian Equities</b></span>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_canvas(current_ticker: str):
    """Stock Prediction Dashboard when a stock is actively searched."""
    # Top Persistent Navigation Bar
    mini_icon = get_stoxify_icon_svg(size=36)
    
    col_nav_left, col_nav_search, col_nav_back, col_nav_reload = st.columns([2.5, 3.5, 1.3, 1.3])
    
    with col_nav_left:
        st.markdown(f"""
        <div class="nav-brand">
            {mini_icon}
            <div class="nav-brand-title">Predic<span class="brand-gradient">SON</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_nav_search:
        with st.form("nav_search_form", border=False):
            c_nav_inp, c_nav_btn = st.columns([3.5, 1])
            with c_nav_inp:
                quick_search = st.text_input(
                    "Quick Search",
                    placeholder="Search another stock (e.g. Federal Bank, TCS)...",
                    label_visibility="collapsed"
                )
            with c_nav_btn:
                nav_submitted = st.form_submit_button("Go", use_container_width=True)

            if nav_submitted and quick_search:
                quick_matches = search_stock_directory(quick_search, limit=1)
                if quick_matches:
                    st.session_state.active_ticker = quick_matches[0]["symbol"]
                    st.rerun()
                else:
                    st.error(f"No stock found matching '{quick_search}'")

    with col_nav_back:
        if st.button("← Back to Search", use_container_width=True):
            st.session_state.active_ticker = None
            st.rerun()

    with col_nav_reload:
        force_reload = st.button("🔄 Force Reload", use_container_width=True, help="Fetch fresh live data and retrain model")

    # Load / Compute Prediction Data
    with st.spinner(f"Gathering latest market information & fitting model for {current_ticker}..."):
        try:
            history_df, forecast_df, meta = get_stock_prediction(current_ticker, force_refresh=force_reload)
            if force_reload:
                st.toast(f"✅ Real-time data & model retrained for {current_ticker}!", icon="⚡")
        except Exception as e:
            st.error(f"Error analyzing {current_ticker}: {e}")
            if st.button("← Back to Search"):
                st.session_state.active_ticker = None
                st.rerun()
            return

    stock_meta = next((s for s in STOCK_DIRECTORY if s["symbol"] == current_ticker), {})
    company_name = stock_meta.get("name", current_ticker.replace(".NS", ""))
    sector = stock_meta.get("sector", "Equities")
    exchange_label = "BSE" if current_ticker.endswith(".BO") else "NSE"

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
                {html.escape(str(company_name))}
                <span class="ticker-pill"><b>{exchange_label}:</b> {html.escape(str(current_ticker))}</span>
                <span class="ticker-pill"><b>Sector:</b> {html.escape(str(sector))}</span>
                <span class="ticker-pill"><b>Regime:</b> {html.escape(str(regime))}</span>
            </div>
            <div style="font-size:0.84rem; color:#64748B; margin-top:5px;">
                Capped Consensus Overlay • Dynamic LightGBM Quantile Ensemble
            </div>
        </div>
        <div style="text-align:right;">
            <div class="hero-price">₹{current_price:,.2f}</div>
            <div style="font-size:0.92rem; font-weight:700;">
                3M Target: <span style="color:#0EA5E9;">₹{h3['projected_price']:,.2f}</span>
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
        <div class="horizon-pill" style="border-color: rgba(14,165,233,0.35); background: rgba(14,165,233,0.06);">
            <div>
                <div class="horizon-pill-lbl" style="color:#0EA5E9;">3-Month Target (Core 63d)</div>
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

    # Side-by-Side Research Cards
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
                <span class="metric-row-val" style="color:#0EA5E9;">{mean_str} (Range: {low_str} - {high_str})</span>
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


def main():
    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = None

    if st.session_state.active_ticker is None:
        render_homepage()
    else:
        render_prediction_canvas(st.session_state.active_ticker)


if __name__ == "__main__":
    main()
