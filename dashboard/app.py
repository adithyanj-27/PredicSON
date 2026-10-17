"""
PredicSON Interactive Dashboard
Displays multi-horizon probabilistic fan charts, macro drivers, news feeds,
and expert consensus for Indian equities on NSE/BSE.
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

from config.settings import PILOT_UNIVERSE, EXPERTS_DIR, NEWS_DIR
from src.models.fan_chart_generator import generate_fan_chart_data
from src.utils.data_loader import load_daily_news, load_expert_consensus


st.set_page_config(
    page_title="PredicSON — AI Indian Stock Market Forecaster",
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
    .stSelectbox label {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def build_fan_chart_figure(ticker: str, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Build interactive Plotly fan chart with expanding 50% and 80% confidence bands."""
    fig = go.Figure()

    # 1. Historical Prices (Solid Dark Gray / Cyan Line)
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["close"],
        mode="lines",
        name="Historical Price",
        line=dict(color="#00D4FF", width=2.5)
    ))

    # Connect history to forecast seamlessly
    last_hist_date = history_df.index[-1]
    last_hist_price = float(history_df["close"].iloc[-1])
    
    # Prepend last historical point to forecast curves
    fc_dates = [last_hist_date] + list(forecast_df.index)
    p10_vals = [last_hist_price] + list(forecast_df["p10"])
    p25_vals = [last_hist_price] + list(forecast_df["p25"])
    p50_vals = [last_hist_price] + list(forecast_df["p50"])
    p75_vals = [last_hist_price] + list(forecast_df["p75"])
    p90_vals = [last_hist_price] + list(forecast_df["p90"])

    # 2. Outer Band: p90 (Top of 80% cone)
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=p90_vals,
        mode="lines",
        name="80% Upper Band (p90)",
        line=dict(color="rgba(0, 212, 255, 0.2)", width=1),
        showlegend=False
    ))

    # Outer Band: p10 (Bottom of 80% cone with fill up to p90)
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=p10_vals,
        mode="lines",
        name="80% Prediction Cone",
        fill="tonexty",
        fillcolor="rgba(0, 212, 255, 0.12)",
        line=dict(color="rgba(0, 212, 255, 0.2)", width=1)
    ))

    # 3. Inner Band: p75 (Top of 50% cone)
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=p75_vals,
        mode="lines",
        name="50% Upper Band (p75)",
        line=dict(color="rgba(0, 212, 255, 0.4)", width=1),
        showlegend=False
    ))

    # Inner Band: p25 (Bottom of 50% cone with fill up to p75)
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=p25_vals,
        mode="lines",
        name="50% Core Probability Cone",
        fill="tonexty",
        fillcolor="rgba(0, 212, 255, 0.25)",
        line=dict(color="rgba(0, 212, 255, 0.4)", width=1)
    ))

    # 4. Median Path (p50 Expected Trajectory)
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=p50_vals,
        mode="lines+markers",
        name="Median Forecast (p50)",
        line=dict(color="#FFB800", width=2.5, dash="dash"),
        marker=dict(size=4)
    ))

    # Vertical line indicating forecast boundary
    fig.add_vline(
        x=last_hist_date,
        line_width=1.5,
        line_dash="dot",
        line_color="#888888",
        annotation_text="Today",
        annotation_position="top left"
    )

    fig.update_layout(
        title=f"<b>{ticker}</b>: 3 to 6-Month Probabilistic Price Cone (Fan Chart)",
        xaxis_title="Timeline",
        yaxis_title="Price (INR ₹)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=550
    )
    return fig


def main():
    st.title("📈 PredicSON — AI Multi-Horizon Stock Forecaster")
    st.caption("Zero-Cost Quantitative AI Engine for NSE / BSE Indian Equities • Continuous Probabilistic Trajectories")

    # Sidebar: Stock Selector
    ticker_options = list(PILOT_UNIVERSE.keys())
    selected_ticker = st.sidebar.selectbox(
        "Select Stock to Analyze",
        options=ticker_options,
        format_func=lambda x: f"{x.replace('.NS', '')} — {PILOT_UNIVERSE[x]['name']}"
    )
    stock_info = PILOT_UNIVERSE[selected_ticker]
    st.sidebar.markdown(f"**Sector:** {stock_info['sector']}")
    st.sidebar.markdown(f"**Exchange:** NSE (`{selected_ticker}`) / BSE (`{stock_info['bse_code']}`)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Architecture")
    st.sidebar.info("""
    - **Ensemble:** Quantile Gradient Boosters
    - **Quantiles:** 10%, 25%, 50% (Median), 75%, 90%
    - **Horizons:** 1M (21d), 3M (63d), 6M (126d)
    - **Macro Alignment:** Brent Crude, USD/INR, VIX, US 10Y, Nifty 50
    - **Spend:** ₹0 (All free / open data)
    """)

    # Generate Fan Chart Forecast
    with st.spinner(f"Generating probabilistic forecast for {selected_ticker}..."):
        try:
            history_df, forecast_df, meta = generate_fan_chart_data(selected_ticker)
        except Exception as e:
            st.error(f"Error computing forecast: {e}")
            return

    current_price = meta["current_price"]
    horizons = meta["horizons"]

    # Top Level Metrics Bar
    regime_display = meta.get("regime", "SIDEWAYS CONSOLIDATION").replace("_", " ")
    st.markdown(f"**Current Macro Regime:** :violet[**{regime_display}**] • As of {meta['as_of_date']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Current Market Price",
            value=f"₹{current_price:,.2f}",
            delta="Live EOD"
        )

    with col2:
        p50_1m = horizons["1M"]["p50"]["projected_price"]
        ret_1m = horizons["1M"]["p50"]["expected_return_pct"]
        st.metric(
            label="1-Month Median Target",
            value=f"₹{p50_1m:,.2f}",
            delta=f"{ret_1m:+.1f}% (Cone: ₹{horizons['1M']['p10']['projected_price']:,.0f} - ₹{horizons['1M']['p90']['projected_price']:,.0f})"
        )

    with col3:
        p50_3m = horizons["3M"]["p50"]["projected_price"]
        ret_3m = horizons["3M"]["p50"]["expected_return_pct"]
        st.metric(
            label="3-Month Median Target",
            value=f"₹{p50_3m:,.2f}",
            delta=f"{ret_3m:+.1f}% (Cone: ₹{horizons['3M']['p10']['projected_price']:,.0f} - ₹{horizons['3M']['p90']['projected_price']:,.0f})"
        )

    with col4:
        p50_6m = horizons["6M"]["p50"]["projected_price"]
        ret_6m = horizons["6M"]["p50"]["expected_return_pct"]
        st.metric(
            label="6-Month Median Target",
            value=f"₹{p50_6m:,.2f}",
            delta=f"{ret_6m:+.1f}% (Cone: ₹{horizons['6M']['p10']['projected_price']:,.0f} - ₹{horizons['6M']['p90']['projected_price']:,.0f})"
        )

    # Main Chart
    st.markdown("---")
    fig = build_fan_chart_figure(selected_ticker, history_df, forecast_df)
    st.plotly_chart(fig, use_container_width=True)

    # Bottom Sections: Two Columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🔍 Model Drivers & Institutional Signals")
        drivers = meta.get("key_drivers", [])
        driver_df = pd.DataFrame(drivers)
        if not driver_df.empty:
            st.table(driver_df[["label", "value"]].rename(columns={"label": "Factor Indicator", "value": "Current Value"}))

        # Quantile Range Breakdown Table
        st.subheader("📊 Quantile Price Horizon Table")
        table_rows = []
        for h in ["1M", "3M", "6M"]:
            row = {
                "Horizon": h,
                "p10 (Bearish Limit)": f"₹{horizons[h]['p10']['projected_price']:,.2f}",
                "p25 (Low Prob)": f"₹{horizons[h]['p25']['projected_price']:,.2f}",
                "p50 (Median)": f"₹{horizons[h]['p50']['projected_price']:,.2f}",
                "p75 (High Prob)": f"₹{horizons[h]['p75']['projected_price']:,.2f}",
                "p90 (Bullish Limit)": f"₹{horizons[h]['p90']['projected_price']:,.2f}",
            }
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("🧠 Institutional Analyst & Brokerage Consensus")
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
                reco_df = pd.DataFrame([recos])
                st.dataframe(reco_df, hide_index=True)

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
            st.caption(f"No specific breaking news tagged for {selected_ticker} in today's RSS stream.")

    # Empirical Backtest Scorecard Expander
    st.markdown("---")
    with st.expander("🎯 Empirical Backtest & Out-of-Sample Accuracy Scorecard", expanded=True):
        scorecard_path = PREDICTIONS_DIR / "backtest_scorecard.json"
        if scorecard_path.exists():
            with open(scorecard_path, "r", encoding="utf-8") as f:
                scorecard_data = json.load(f)
            
            stock_bt = scorecard_data.get(selected_ticker, {}).get("summary", {})
            if stock_bt:
                bc1, bc2, bc3, bc4 = st.columns(4)
                h3_res = stock_bt.get("3M", {})
                bc1.metric("3M Directional Hit Rate", f"{h3_res.get('directional_accuracy_pct', 0)}%")
                bc2.metric("80% Cone Calibration", f"{h3_res.get('cone_80_coverage_pct', 0)}%")
                bc3.metric("50% Cone Calibration", f"{h3_res.get('cone_50_coverage_pct', 0)}%")
                bc4.metric("Mean Abs Error (3M)", f"{h3_res.get('mean_absolute_error_pct', 0)}%")

                # Historical test windows table
                windows = h3_res.get("test_windows", [])
                if windows:
                    st.markdown("**Historical Out-of-Sample Test Windows (3-Month Horizon):**")
                    w_df = pd.DataFrame(windows)[["eval_date", "current_price", "actual_price", "pred_median_return_pct", "actual_return_pct", "directional_correct", "inside_80_cone"]]
                    w_df.columns = ["Evaluation Date", "Base Price", "Actual Price", "Pred Median Return %", "Actual Return %", "Direction Correct?", "Inside 80% Cone?"]
                    st.dataframe(w_df, hide_index=True, use_container_width=True)
            else:
                st.info(f"Backtest evaluation available. Run `main.py backtest` to refresh.")
        else:
            st.info("Run `python main.py backtest` to generate out-of-sample accuracy validation metrics.")


if __name__ == "__main__":
    main()
