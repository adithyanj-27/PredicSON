"""
PredicSON — Institutional AI Stock Quant Terminal (NSE/BSE)
High-End Dark Fintech UI (TradingView / Bloomberg / Linear aesthetic)
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
    page_title="PredicSON Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Fintech CSS Design System
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
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Top Branding Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 18, 28, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 14px rgba(0, 242, 254, 0.35);
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-title span {
        color: #00F2FE;
    }
    .status-badge {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
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
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    .ticker-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        overflow-x: auto;
    }
    .ticker-pill {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.82rem;
        display: flex;
        gap: 6px;
        white-space: nowrap;
    }
    .ticker-pill b {
        color: #94A3B8;
    }
    .ticker-pos { color: #10B981; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .ticker-neu { color: #38BDF8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

    /* Custom Glass Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(14, 18, 28, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        gap: 8px !important;
        margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 9px 22px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.16) 0%, rgba(79, 172, 254, 0.16) 100%) !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        box-shadow: 0 4px 14px rgba(0, 242, 254, 0.18) !important;
    }

    /* Fintech Glass Card Container */
    .fintech-card {
        background: rgba(14, 19, 31, 0.65);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
        transition: border 0.2s ease, box-shadow 0.2s ease;
    }
    .fintech-card:hover {
        border-color: rgba(0, 242, 254, 0.25);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    /* Metric Stat Tiles */
    .stat-tile {
        background: rgba(18, 24, 38, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 105px;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 3px 9px;
        width: fit-content;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-red {
        background: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .pill-cyan {
        background: rgba(0, 242, 254, 0.15);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Executive Thesis Box */
    .thesis-box {
        background: linear-gradient(145deg, rgba(14, 22, 38, 0.8) 0%, rgba(9, 13, 22, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
    }
    .thesis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .thesis-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .thesis-summary {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #CBD5E1;
        margin-bottom: 18px;
    }
    .catalyst-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #E2E8F0;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }
    .headwind-card {
        background: rgba(244, 63, 94, 0.05);
        border: 1px solid rgba(244, 63, 94, 0.18);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #E2E8F0;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }

    /* Factor Grid */
    .factor-tile {
        background: rgba(18, 24, 38, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .factor-name {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .factor-value {
        font-size: 0.88rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #FFFFFF;
    }

    /* News Box */
    .news-tile {
        background: rgba(18, 24, 38, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: background 0.15s ease;
    }
    .news-tile:hover {
        background: rgba(255, 255, 255, 0.04);
    }
    .news-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 4px;
    }
    .news-meta {
        font-size: 0.75rem;
        color: #64748B;
        display: flex;
        gap: 10px;
    }

    /* Streamlit Selectbox override */
    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


def build_fintech_fan_chart(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Builds a TradingView-caliber Plotly fan chart with custom dark neon styling."""
    fig = go.Figure()

    # Historical price curve
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
        line=dict(color="rgba(0, 242, 254, 0.3)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p10_vals, mode="lines", name="80% Probability Cone (p10 - p90)",
        fill="tonexty", fillcolor="rgba(0, 242, 254, 0.08)",
        line=dict(color="rgba(0, 242, 254, 0.3)", width=1),
        hovertemplate="<b>80% Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Inner 50% Core Probability Cone (p25 to p75)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p75_vals, mode="lines", name="p75 (Upper Core)",
        line=dict(color="rgba(0, 242, 254, 0.6)", width=1), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p25_vals, mode="lines", name="50% Core Probability Cone (p25 - p75)",
        fill="tonexty", fillcolor="rgba(0, 242, 254, 0.22)",
        line=dict(color="rgba(0, 242, 254, 0.6)", width=1),
        hovertemplate="<b>50% Core Range</b>: ₹%{y:,.2f}<extra></extra>"
    ))

    # Expected Median Path (p50)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p50_vals, mode="lines+markers", name="Expected Median Path (p50)",
        line=dict(color="#F59E0B", width=2.5, dash="dash"),
        marker=dict(size=4, color="#F59E0B"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Median Projection: ₹%{y:,.2f}<extra></extra>"
    ))

    # Vertical demarcation for Today
    fig.add_vline(
        x=last_hist_date, line_width=1.5, line_dash="dash", line_color="#64748B",
        annotation_text="Today (Forecast Pivot)", annotation_position="top left",
        annotation_font=dict(color="#94A3B8", size=11)
    )

    fig.update_layout(
        paper_bgcolor="#0A0D15",
        plot_bgcolor="#0A0D15",
        font=dict(family="Plus Jakarta Sans", color="#94A3B8"),
        hovermode="x unified",
        margin=dict(l=30, r=30, t=30, b=30),
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#CBD5E1")
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
    # Top Branding Header
    st.markdown("""
    <div class="top-navbar">
        <div class="brand-section">
            <div class="brand-logo">📈</div>
            <div class="brand-title">PREDIC<span>SON</span></div>
            <div class="status-badge"><span class="pulse-dot"></span> LIVE QUANT TERMINAL</div>
        </div>
        <div class="ticker-bar">
            <div class="ticker-pill"><b>NIFTY 50</b> <span class="ticker-pos">25,127 (+0.4%)</span></div>
            <div class="ticker-pill"><b>INDIA VIX</b> <span class="ticker-neu">13.20 (-1.8%)</span></div>
            <div class="ticker-pill"><b>BRENT CRUDE</b> <span class="ticker-neu">$74.20 (-0.6%)</span></div>
            <div class="ticker-pill"><b>USD/INR</b> <span class="ticker-neu">84.05</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Master Tabs
    tab_deep_dive, tab_screener, tab_portfolio = st.tabs([
        "⚡ Single-Stock Deep Dive",
        "🚀 Market Opportunity Screener",
        "💼 Modern Portfolio Allocator"
    ])

    # =========================================================================
    # TAB 1: SINGLE-STOCK DEEP DIVE
    # =========================================================================
    with tab_deep_dive:
        # Stock Selector Bar
        col_select, col_info = st.columns([1.5, 2.5])
        with col_select:
            ticker_keys = list(PILOT_UNIVERSE.keys())
            selected_ticker = st.selectbox(
                "Choose Asset",
                options=ticker_keys,
                index=0,
                format_func=lambda x: f"{x.replace('.NS', '')}  —  {PILOT_UNIVERSE[x]['name']}",
                label_visibility="collapsed"
            )
        with col_info:
            info = PILOT_UNIVERSE[selected_ticker]
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; height:38px;">
                <span style="font-weight:700; color:#F1F5F9; font-size:1.05rem;">{info['name']}</span>
                <span class="ticker-pill"><b>Sector:</b> {info['sector'].split('/')[0].strip()}</span>
                <span class="ticker-pill"><b>NSE:</b> {selected_ticker}</span>
                <span class="ticker-pill"><b>BSE:</b> {info['bse_code']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Generate Forecast Data
        with st.spinner("Computing 72-factor quantile trajectory..."):
            try:
                history_df, forecast_df, meta = generate_fan_chart_data(selected_ticker)
            except Exception as e:
                st.error(f"Error computing forecast: {e}")
                return

        current_price = meta["current_price"]
        horizons = meta["horizons"]
        regime = meta.get("regime", "SIDEWAYS_CONSOLIDATION").replace("_", " ")

        # 4 Hero Metric Cards
        p50_1m = horizons["1M"]["p50"]["projected_price"]
        ret_1m = horizons["1M"]["p50"]["expected_return_pct"]
        p50_3m = horizons["3M"]["p50"]["projected_price"]
        ret_3m = horizons["3M"]["p50"]["expected_return_pct"]
        p50_6m = horizons["6M"]["p50"]["projected_price"]
        ret_6m = horizons["6M"]["p50"]["expected_return_pct"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">Current Market Price</div>
                <div class="stat-val">₹{current_price:,.2f}</div>
                <div class="stat-pill pill-cyan">Live EOD Close</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            pill_class = "pill-green" if ret_1m >= 0 else "pill-red"
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">1-Month Target (21d)</div>
                <div class="stat-val">₹{p50_1m:,.2f}</div>
                <div class="stat-pill {pill_class}">{ret_1m:+.1f}% Expected</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            pill_class = "pill-green" if ret_3m >= 0 else "pill-red"
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">3-Month Target (63d)</div>
                <div class="stat-val">₹{p50_3m:,.2f}</div>
                <div class="stat-pill {pill_class}">{ret_3m:+.1f}% Core Horizon</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            pill_class = "pill-green" if ret_6m >= 0 else "pill-red"
            st.markdown(f"""
            <div class="stat-tile">
                <div class="stat-label">6-Month Target (126d)</div>
                <div class="stat-val">₹{p50_6m:,.2f}</div>
                <div class="stat-pill {pill_class}">{ret_6m:+.1f}% Macro Horizon</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Main TradingView Fan Chart
        fig = build_fintech_fan_chart(selected_ticker, history_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Executive AI Investment Thesis
        thesis = meta.get("investment_thesis", {})
        if thesis:
            stance = thesis.get("stance", "NEUTRAL")
            stance_color = "#10B981" if "BULL" in stance or "ACCUMULATE" in stance else ("#F43F5E" if "BEAR" in stance or "TRIM" in stance else "#F59E0B")
            summary = thesis.get("executive_summary", "")

            catalysts_html = "".join([f"<div class='catalyst-card'><span>🟢</span><span>{b}</span></div>" for b in thesis.get("bull_catalysts", [])])
            headwinds_html = "".join([f"<div class='headwind-card'><span>🔴</span><span>{h}</span></div>" for h in thesis.get("risk_headwinds", [])])

            st.markdown(f"""
            <div class="thesis-box">
                <div class="thesis-header">
                    <div class="thesis-title">📋 Institutional Investment Thesis</div>
                    <div class="stat-pill" style="background:rgba(255,255,255,0.06); border-color:{stance_color}; color:{stance_color}; font-size:0.85rem;">
                        ● {stance}
                    </div>
                </div>
                <div class="thesis-summary">{summary}</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                    <div>
                        <div style="font-size:0.82rem; font-weight:700; color:#10B981; margin-bottom:8px; text-transform:uppercase;">Key Bull Catalysts</div>
                        {catalysts_html}
                    </div>
                    <div>
                        <div style="font-size:0.82rem; font-weight:700; color:#F43F5E; margin-bottom:8px; text-transform:uppercase;">Downside Risk Headwinds</div>
                        {headwinds_html}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Bottom Grid: 72 Factors & Analyst Intelligence
        col_factors, col_intel = st.columns([1.1, 0.9])
        
        with col_factors:
            st.markdown("#### 🔍 Key Quantitative Factors (72-Feature Stack)")
            drivers = meta.get("key_drivers", [])
            if drivers:
                for d in drivers[:10]:
                    st.markdown(f"""
                    <div class="factor-tile">
                        <span class="factor-name">{d['label']}</span>
                        <span class="factor-value">{d['value']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col_intel:
            st.markdown("#### 🧠 Institutional Analyst Consensus")
            today_str = datetime.now().strftime("%Y-%m-%d")
            expert_records = load_expert_consensus(today_str)
            matched = next((r for r in expert_records if r.get("ticker") == selected_ticker), None)

            if matched and matched.get("price_targets"):
                pt = matched["price_targets"]
                mean_pt = pt.get("mean")
                mean_str = f"₹{mean_pt:,.0f}" if mean_pt else "N/A"
                high_str = f"₹{pt.get('high', 0):,.0f}" if pt.get('high') else "N/A"
                low_str = f"₹{pt.get('low', 0):,.0f}" if pt.get('low') else "N/A"

                st.markdown(f"""
                <div class="stat-tile" style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span class="stat-label">Consensus Mean Target</span>
                        <span style="font-family:'JetBrains Mono'; font-weight:700; color:#00F2FE;">{mean_str}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8;">Analyst Range: Low {low_str}  •  High {high_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📰 Verified Material News")
            news_items = load_daily_news(today_str)
            tagged_news = [a for a in news_items if selected_ticker in a.get("matched_tickers", [])]
            if tagged_news:
                for n in tagged_news[:4]:
                    st.markdown(f"""
                    <div class="news-tile">
                        <div class="news-title">{n['title']}</div>
                        <div class="news-meta">
                            <span>{n.get('source', 'Financial Wire')}</span>
                            <span>•</span>
                            <span>{n.get('published_raw', 'Today')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption(f"No breaking news alerts tagged for {selected_ticker} today.")

    # =========================================================================
    # TAB 2: MARKET OPPORTUNITY SCREENER
    # =========================================================================
    with tab_screener:
        st.markdown("### 🚀 Cross-Sectional Opportunity Screener")
        st.caption("Real-time ranking of 25 titans across Expected Return, Downside Risk, Risk-Reward, and Money Flow.")

        screener_df = generate_market_screener(use_cache=True)
        if not screener_df.empty:
            # Metric Counter Bar
            strong_buys = len(screener_df[screener_df["signal"].str.contains("STRONG BUY")])
            accumulates = len(screener_df[screener_df["signal"].str.contains("ACCUMULATE")])
            avoids = len(screener_df[screener_df["signal"].str.contains("AVOID")])

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"<div class='stat-tile'><div class='stat-label'>Universe Size</div><div class='stat-val'>{len(screener_df)}</div><div class='stat-pill pill-cyan'>Nifty Titans</div></div>", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"<div class='stat-tile'><div class='stat-label'>Strong Buys</div><div class='stat-val'>{strong_buys}</div><div class='stat-pill pill-green'>Top Momentum</div></div>", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"<div class='stat-tile'><div class='stat-label'>Accumulate</div><div class='stat-val'>{accumulates}</div><div class='stat-pill pill-cyan'>Positive Trajectory</div></div>", unsafe_allow_html=True)
            with sc4:
                st.markdown(f"<div class='stat-tile'><div class='stat-label'>Avoid / Trim</div><div class='stat-val'>{avoids}</div><div class='stat-pill pill-red'>Negative Asymmetry</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Filter Controls
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

            # Clean Display Columns
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

            st.dataframe(
                renamed,
                hide_index=True,
                use_container_width=True
            )

            # Return Comparison Chart
            st.markdown("#### 📊 3-Month Expected Return Ranking")
            fig_bar = px.bar(
                screener_df.sort_values(by="expected_return_3m", ascending=True),
                x="expected_return_3m",
                y="ticker",
                orientation="h",
                color="expected_return_3m",
                color_continuous_scale=["#F43F5E", "#38BDF8", "#10B981"],
                labels={"expected_return_3m": "Expected Return (%)", "ticker": "Asset"},
                template="plotly_dark",
                height=560
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
            <div class="fintech-card" style="padding:16px;">
                <div style="font-weight:700; color:#00F2FE; margin-bottom:6px;">Markowitz Max-Sharpe Formulation</div>
                <div style="font-size:0.88rem; color:#94A3B8; line-height:1.6;">
                    • <b>Objective</b>: Maximizes risk-adjusted excess return: <span style="font-family:'JetBrains Mono'; color:#FFFFFF;">max (w<sup>T</sup>r - r<sub>f</sub>) / √(w<sup>T</sup>Σw)</span><br>
                    • <b>Covariance Alignment</b>: Allocates across non-correlated sectors to minimize portfolio drawdown.<br>
                    • <b>Concentration Guard</b>: Enforces single-asset ceiling to avoid idiosyncratic shock exposure.
                </div>
            </div>
            """, unsafe_allow_html=True)

        screener_for_port = generate_market_screener(use_cache=True)
        if not screener_for_port.empty:
            port_result = optimize_portfolio(screener_for_port, total_capital=total_invest, max_stock_weight=max_weight)

            if port_result and "allocations" in port_result:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"<div class='stat-tile'><div class='stat-label'>Capital Allocated</div><div class='stat-val'>₹{port_result['total_capital']:,.0f}</div><div class='stat-pill pill-cyan'>100% Deployed</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='stat-tile'><div class='stat-label'>Expected 3M Return</div><div class='stat-val'>{port_result['expected_3m_portfolio_return_pct']:+.1f}%</div><div class='stat-pill pill-green'>Forward Target</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='stat-tile'><div class='stat-label'>Annualized Volatility</div><div class='stat-val'>{port_result['portfolio_volatility_pct']:.1f}%</div><div class='stat-pill pill-amber'>Empirical Risk</div></div>", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"<div class='stat-tile'><div class='stat-label'>Portfolio Sharpe Ratio</div><div class='stat-val'>{port_result['portfolio_sharpe']:.2f}</div><div class='stat-pill pill-green'>Risk-Adjusted</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

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
