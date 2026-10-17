"""
PredicSON Institutional Quant Terminal
Features:
- Tab 1: Single-Stock Deep Dive (Continuous Fan Chart, 72 Signals, AI Investment Thesis)
- Tab 2: Market Opportunity Screener (Cross-sectional leaderboard, signals, risk-reward ratios)
- Tab 3: Modern Portfolio Theory Capital Allocator (Optimal Markowitz Sharpe ratio weighting)
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


st.set_page_config(
    page_title="PredicSON — Institutional AI Stock Forecaster (NSE/BSE)",
    page_icon="📈",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def build_fan_chart_figure(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Build interactive Plotly fan chart with expanding 50% and 80% confidence bands."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["close"],
        mode="lines",
        name="Historical Price",
        line=dict(color="#00D4FF", width=2.5)
    ))

    last_hist_date = history_df.index[-1]
    last_hist_price = float(history_df["close"].iloc[-1])
    
    fc_dates = [last_hist_date] + list(forecast_df.index)
    p10_vals = [last_hist_price] + list(forecast_df["p10"])
    p25_vals = [last_hist_price] + list(forecast_df["p25"])
    p50_vals = [last_hist_price] + list(forecast_df["p50"])
    p75_vals = [last_hist_price] + list(forecast_df["p75"])
    p90_vals = [last_hist_price] + list(forecast_df["p90"])

    # Outer 80% cone
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p90_vals, mode="lines", name="80% Upper (p90)",
        line=dict(color="rgba(0, 212, 255, 0.2)", width=1), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p10_vals, mode="lines", name="80% Prediction Cone",
        fill="tonexty", fillcolor="rgba(0, 212, 255, 0.12)",
        line=dict(color="rgba(0, 212, 255, 0.2)", width=1)
    ))

    # Inner 50% cone
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p75_vals, mode="lines", name="50% Upper (p75)",
        line=dict(color="rgba(0, 212, 255, 0.4)", width=1), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p25_vals, mode="lines", name="50% Core Probability Cone",
        fill="tonexty", fillcolor="rgba(0, 212, 255, 0.25)",
        line=dict(color="rgba(0, 212, 255, 0.4)", width=1)
    ))

    # Median Path
    fig.add_trace(go.Scatter(
        x=fc_dates, y=p50_vals, mode="lines+markers", name="Median Forecast (p50)",
        line=dict(color="#FFB800", width=2.5, dash="dash"), marker=dict(size=4)
    ))

    fig.add_vline(
        x=last_hist_date, line_width=1.5, line_dash="dot", line_color="#888888",
        annotation_text="Today", annotation_position="top left"
    )

    fig.update_layout(
        title=f"<b>{ticker}</b>: 3 to 6-Month Probabilistic Price Cone (Fan Chart)",
        xaxis_title="Timeline", yaxis_title="Price (INR ₹)",
        template="plotly_dark", hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550
    )
    return fig


def main():
    st.title("🏛️ PredicSON — Institutional AI Stock Forecaster (NSE/BSE)")
    st.caption("Cross-Sectional Market Screener • 72-Factor Quantile Fan Charts • Modern Portfolio Theory Allocation")

    # Master Navigation Tabs
    tab_deep_dive, tab_screener, tab_portfolio = st.tabs([
        "📈 Single-Stock Deep Dive",
        "🚀 Market Opportunity Screener",
        "💼 AI Portfolio Allocator"
    ])

    # ==========================================
    # TAB 1: SINGLE-STOCK DEEP DIVE
    # ==========================================
    with tab_deep_dive:
        ticker_options = list(PILOT_UNIVERSE.keys())
        selected_ticker = st.sidebar.selectbox(
            "Select Company to Analyze",
            options=ticker_options,
            format_func=lambda x: f"{x.replace('.NS', '')} — {PILOT_UNIVERSE[x]['name']} ({PILOT_UNIVERSE[x]['sector'].split('/')[0].strip()})"
        )
        stock_info = PILOT_UNIVERSE[selected_ticker]
        st.sidebar.markdown(f"**Sector:** {stock_info['sector']}")
        st.sidebar.markdown(f"**Exchange:** NSE (`{selected_ticker}`) / BSE (`{stock_info['bse_code']}`)")
        st.sidebar.markdown("---")
        st.sidebar.info("""
        - **Model:** Stacked Quantile Ensemble
        - **Signals:** 72 Factors (Fundamentals, CMF, Alpha, Macro)
        - **Horizon:** 1M, 3M, 6M Cones
        - **Spend:** ₹0 (Open-Source / Local CPU)
        """)

        with st.spinner(f"Computing 72-factor forecast for {selected_ticker}..."):
            try:
                history_df, forecast_df, meta = generate_fan_chart_data(selected_ticker)
            except Exception as e:
                st.error(f"Error computing forecast: {e}")
                return

        current_price = meta["current_price"]
        horizons = meta["horizons"]

        regime_display = meta.get("regime", "SIDEWAYS CONSOLIDATION").replace("_", " ")
        st.markdown(f"**Current Macro Regime:** :violet[**{regime_display}**] • As of {meta['as_of_date']}")

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Market Price", f"₹{current_price:,.2f}", delta="Live EOD")
        with col2:
            p50_1m = horizons["1M"]["p50"]["projected_price"]
            ret_1m = horizons["1M"]["p50"]["expected_return_pct"]
            st.metric("1-Month Median Target", f"₹{p50_1m:,.2f}", delta=f"{ret_1m:+.1f}% (Cone: ₹{horizons['1M']['p10']['projected_price']:,.0f} - ₹{horizons['1M']['p90']['projected_price']:,.0f})")
        with col3:
            p50_3m = horizons["3M"]["p50"]["projected_price"]
            ret_3m = horizons["3M"]["p50"]["expected_return_pct"]
            st.metric("3-Month Median Target", f"₹{p50_3m:,.2f}", delta=f"{ret_3m:+.1f}% (Cone: ₹{horizons['3M']['p10']['projected_price']:,.0f} - ₹{horizons['3M']['p90']['projected_price']:,.0f})")
        with col4:
            p50_6m = horizons["6M"]["p50"]["projected_price"]
            ret_6m = horizons["6M"]["p50"]["expected_return_pct"]
            st.metric("6-Month Median Target", f"₹{p50_6m:,.2f}", delta=f"{ret_6m:+.1f}% (Cone: ₹{horizons['6M']['p10']['projected_price']:,.0f} - ₹{horizons['6M']['p90']['projected_price']:,.0f})")

        # Executive Investment Thesis
        thesis = meta.get("investment_thesis", {})
        if thesis:
            st.markdown("---")
            stance_str = thesis.get("stance", "NEUTRAL")
            stance_color = thesis.get("stance_color", "orange")
            st.markdown(f"### 📋 Institutional Investment Thesis: :{stance_color}[**{stance_str}**]")
            st.info(thesis.get("executive_summary", ""))

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("#### 🟢 Key Bull Catalysts")
                for b in thesis.get("bull_catalysts", []):
                    st.markdown(f"* {b}")
            with t_col2:
                st.markdown("#### 🔴 Key Downside Headwinds")
                for r in thesis.get("risk_headwinds", []):
                    st.markdown(f"* {r}")

        # Main Fan Chart
        st.markdown("---")
        fig = build_fan_chart_figure(selected_ticker, history_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("🔍 Model Drivers & Institutional Signals (72 Factors)")
            drivers = meta.get("key_drivers", [])
            driver_df = pd.DataFrame(drivers)
            if not driver_df.empty:
                st.table(driver_df[["label", "value"]].rename(columns={"label": "Factor Indicator", "value": "Current Value"}))

        with col_right:
            st.subheader("🧠 Institutional Analyst Consensus")
            today_str = datetime.now().strftime("%Y-%m-%d")
            expert_records = load_expert_consensus(today_str)
            matched_consensus = next((r for r in expert_records if r.get("ticker") == selected_ticker), None)

            if matched_consensus and matched_consensus.get("price_targets"):
                pt = matched_consensus["price_targets"]
                st.markdown(f"**Mean Analyst Target:** ₹{pt.get('mean', 'N/A')}")
                st.markdown(f"**Analyst Range:** Low: ₹{pt.get('low', 'N/A')}  |  High: ₹{pt.get('high', 'N/A')}")
                recos = matched_consensus.get("recommendation_counts", {})
                if recos:
                    st.markdown("**Analyst Ratings Breakdown:**")
                    st.dataframe(pd.DataFrame([recos]), hide_index=True)

            st.subheader("📰 Recent Material News (Noise Gate Verified)")
            news_articles = load_daily_news(today_str)
            ticker_news = [a for a in news_articles if selected_ticker in a.get("matched_tickers", [])]
            if ticker_news:
                for article in ticker_news[:5]:
                    with st.expander(f"📌 {article['title']}"):
                        st.write(article.get("summary", "No summary provided."))
                        st.caption(f"Source: {article.get('source')} • {article.get('published_raw')}")
                        if article.get("link"):
                            st.markdown(f"[Read full story]({article['link']})")
            else:
                st.caption(f"No specific breaking news tagged for {selected_ticker} today.")

    # ==========================================
    # TAB 2: MARKET OPPORTUNITY SCREENER
    # ==========================================
    with tab_screener:
        st.subheader("🚀 Cross-Sectional Market Opportunity Screener (Top 25 Titans)")
        st.caption("Live cross-sectional ranking across Expected Return, Risk-Reward, PEG Valuation, and Institutional Money Flow.")

        with st.spinner("Loading master market screener matrix..."):
            screener_df = generate_market_screener(use_cache=True)

        if not screener_df.empty:
            # Quick Stats
            strong_buys = len(screener_df[screener_df["signal"].str.contains("STRONG BUY")])
            accumulates = len(screener_df[screener_df["signal"].str.contains("ACCUMULATE")])
            avoids = len(screener_df[screener_df["signal"].str.contains("AVOID")])

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Titans Screened", len(screener_df))
            sc2.metric("🟢 Strong Buy Signals", strong_buys)
            sc3.metric("🔵 Accumulate Signals", accumulates)
            sc4.metric("🔴 Avoid / Trim Signals", avoids)

            st.markdown("---")
            # Interactive Filters
            f_col1, f_col2 = st.columns([1, 1])
            with f_col1:
                sector_filter = st.multiselect("Filter by Sector", options=sorted(screener_df["sector"].unique()), default=[])
            with f_col2:
                signal_filter = st.multiselect("Filter by Signal", options=sorted(screener_df["signal"].unique()), default=[])

            filtered_df = screener_df.copy()
            if sector_filter:
                filtered_df = filtered_df[filtered_df["sector"].isin(sector_filter)]
            if signal_filter:
                filtered_df = filtered_df[filtered_df["signal"].isin(signal_filter)]

            # Display Table
            display_cols = [
                "ticker", "name", "sector", "current_price",
                "expected_return_3m", "p10_downside_3m", "risk_reward_ratio",
                "signal", "cmf_flow", "peg_ratio", "pe_ratio"
            ]
            renamed_df = filtered_df[display_cols].copy()
            renamed_df.columns = [
                "Ticker", "Company Name", "Sector", "Price (₹)",
                "3M Expected Return %", "3M Downside p10 %", "Risk-Reward Ratio",
                "Signal", "Money Flow (CMF)", "PEG Ratio", "P/E Multiple"
            ]

            st.dataframe(
                renamed_df.style.background_gradient(subset=["3M Expected Return %"], cmap="RdYlGn", vmin=-10, vmax=25),
                hide_index=True,
                use_container_width=True
            )

            # Bar Chart Comparison
            st.markdown("#### 📊 Expected 3-Month Return Comparison")
            fig_bar = px.bar(
                screener_df.sort_values(by="expected_return_3m", ascending=True),
                x="expected_return_3m",
                y="ticker",
                orientation="h",
                color="expected_return_3m",
                color_continuous_scale="RdYlGn",
                labels={"expected_return_3m": "Expected 3-Month Return (%)", "ticker": "Company"},
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # TAB 3: MODERN PORTFOLIO THEORY ALLOCATOR
    # ==========================================
    with tab_portfolio:
        st.subheader("💼 Modern Portfolio Theory (MPT) & Capital Allocator")
        st.caption("Solves for the Maximum Sharpe Ratio Portfolio using forward model forecasts and historical covariance.")

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            total_invest = st.number_input("Enter Total Investment Capital (₹ INR)", min_value=10000.0, max_value=10000000.0, value=100000.0, step=10000.0)
            max_weight = st.slider("Maximum Allocation per Stock (%)", min_value=10, max_value=40, value=25) / 100.0
            optimize_btn = st.button("⚡ Calculate Mathematically Optimal Allocation", use_container_width=True)

        with p_col2:
            st.info("""
            **How the Allocator Works:**
            * Optimizes weights $w$ to maximize the forward Sharpe Ratio: $\\max \\frac{w^T r - r_f}{\\sqrt{w^T \\Sigma w}}$.
            * Diversifies across uncorrelated sectors (e.g. balances Auto + Pharma + Tech + Energy).
            * Protects capital by capping individual single-stock risk exposure.
            """)

        screener_for_port = generate_market_screener(use_cache=True)
        if not screener_for_port.empty:
            port_result = optimize_portfolio(screener_for_port, total_capital=total_invest, max_stock_weight=max_weight)
            
            if port_result and "allocations" in port_result:
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Capital Allocated", f"₹{port_result['total_capital']:,.0f}")
                m2.metric("Expected 3M Return", f"{port_result['expected_3m_portfolio_return_pct']:+.1f}%")
                m3.metric("Annual Volatility", f"{port_result['portfolio_volatility_pct']:.1f}%")
                m4.metric("Portfolio Sharpe Ratio", f"{port_result['portfolio_sharpe']:.2f}")

                alloc_df = pd.DataFrame(port_result["allocations"])
                
                c_alloc1, c_alloc2 = st.columns([1, 1])
                with c_alloc1:
                    st.markdown("#### 🎯 Capital Allocation Plan")
                    disp_alloc = alloc_df[["ticker", "name", "sector", "weight_pct", "capital_inr", "expected_3m_return"]].copy()
                    disp_alloc.columns = ["Ticker", "Company", "Sector", "Weight %", "Capital (₹)", "Expected 3M Return %"]
                    st.dataframe(disp_alloc, hide_index=True, use_container_width=True)

                with c_alloc2:
                    st.markdown("#### 🍩 Portfolio Weight Breakdown")
                    fig_pie = px.pie(
                        alloc_df,
                        values="weight_pct",
                        names="name",
                        hole=0.4,
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == "__main__":
    main()
