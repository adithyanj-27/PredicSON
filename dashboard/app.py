"""
PredicSON — Institutional Consensus & Expert Intelligence Terminal (NSE/BSE)
Aggregates authentic sell-side analyst price targets, individual institutional predictions,
company fundamentals, and Dalal Street research calls across 10,680+ Indian equities.
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
from src.data_ingestion.institutional_engine import (
    fetch_institutional_consensus,
    build_institutional_corridor_figure,
    get_universe_institutional_rankings,
    INSTITUTION_METADATA
)

# Page Configuration
st.set_page_config(
    page_title="PredicSON — Institutional Consensus Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stoxify-grade Dark Brokerage Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #06090E !important;
    color: #F8FAFC !important;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, rgba(14, 165, 233, 0.14) 0%, rgba(6, 9, 14, 0.95) 55%, #06090E 100%) !important;
}

#MainMenu, footer, header, .stDeployButton {
    display: none !important;
}

.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1340px !important;
}

/* Glassmorphic card styling */
.fintech-card {
    background: rgba(13, 19, 31, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5) !important;
}

/* Primary Gradient Brand Text */
.brand-gradient {
    background: linear-gradient(135deg, #0EA5E9 0%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Homepage Hero */
.home-hero {
    text-align: center;
    padding-top: 2.5vh;
    padding-bottom: 14px;
}
.home-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #F8FAFC;
    margin-bottom: 4px;
    line-height: 1.1;
}
.home-subtitle {
    font-size: 1.05rem;
    font-weight: 500;
    color: #94A3B8;
    margin-bottom: 14px;
}
.home-badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.home-badge {
    font-size: 0.74rem;
    font-weight: 700;
    background: rgba(14, 165, 233, 0.10);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 20px;
    padding: 4px 14px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Prediction Canvas Top Bar & Hero */
.terminal-hero {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 29, 0.90) 100%);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 14px;
    padding: 18px 24px;
    margin-top: 8px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.6);
}
.asset-headline {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.badge-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.ticker-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #CBD5E1;
}
.hero-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
    text-align: right;
}

/* 3-Pill Horizon Target Ribbon */
.horizon-ribbon {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 16px;
}
.horizon-pill {
    background: rgba(13, 19, 31, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.horizon-pill-lbl {
    font-size: 0.74rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.horizon-pill-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 800;
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
.pill-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
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

/* Broker Avatar Badge */
.broker-avatar {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
    border: 1px solid rgba(14, 165, 233, 0.4);
    color: #38BDF8;
}

/* Metric Row */
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.85rem;
}
.metric-row-lbl { color: #94A3B8; font-weight: 500; }
.metric-row-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #F8FAFC; }

/* Global Button styling */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    color: #F8FAFC !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #0EA5E9 !important;
    color: #38BDF8 !important;
    background: rgba(14, 165, 233, 0.12) !important;
    box-shadow: 0 0 14px rgba(14, 165, 233, 0.25) !important;
}

/* Input Fields */
.stTextInput > div > div > input {
    background: rgba(13, 19, 31, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.3) !important;
}

/* Tab Headers */
button[data-baseweb="tab"] {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #94A3B8 !important;
    padding: 10px 20px !important;
    background: transparent !important;
    border: none !important;
}
button[data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0EA5E9 !important;
    border-bottom: 2px solid #0EA5E9 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #0EA5E9 !important;
}
div[data-baseweb="tab-border"] {
    background-color: rgba(255, 255, 255, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)


def get_broker_avatar_initials(inst_name: str) -> str:
    """Return 2-letter uppercase initials for brokerage avatar badges."""
    clean = inst_name.replace("Securities", "").replace("Institutional", "").replace("Equities", "").replace("Direct", "").strip()
    words = clean.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    elif len(words) == 1 and len(words[0]) >= 2:
        return words[0][:2].upper()
    return "IB"


def render_all_stocks_tab():
    """Browse, search, and paginate through all 10,680+ Indian equities on NSE and BSE."""
    st.markdown(
        "<div style='margin-top:12px; margin-bottom:14px;'>"
        "<div style='font-size:1.15rem; font-weight:800; color:#FFFFFF;'>National & Bombay Stock Exchange Directory</div>"
        "<div style='font-size:0.82rem; color:#94A3B8;'>Filter through all 10,680+ listed equities across NSE and BSE. Click <b>⚡ Inspect</b> on any company to pull real-time sell-side consensus, analyst targets, and institutional research calls on demand.</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Filter & Search Controls
    col_sec, col_exch_flt, col_search, col_per = st.columns([1.5, 1.2, 2.8, 1.0])
    
    with col_sec:
        all_sectors = sorted(list({s.get("sector", "Equities") for s in STOCK_DIRECTORY if s.get("sector")}))
        sector_options = ["All Sectors"] + all_sectors
        selected_sector = st.selectbox("Sector", options=sector_options, index=0)

    with col_exch_flt:
        selected_exch = st.selectbox("Exchange", ["All Exchanges", "NSE Only", "BSE Only"], index=0)

    with col_search:
        filter_kw = st.text_input("Filter Directory", placeholder="Search symbol or company (e.g. Suzlon, MRF, Gensol, Tata)...").strip().lower()

    with col_per:
        per_page = st.selectbox("Per Page", [25, 50, 100], index=1)

    # Filter Logic
    matched_stocks = []
    if filter_kw:
        search_candidates = search_stock_directory(filter_kw, limit=500)
        for s in search_candidates:
            sec = s.get("sector", "Equities")
            if selected_sector != "All Sectors" and sec != selected_sector:
                continue
            if selected_exch == "NSE Only" and not s["symbol"].endswith(".NS"):
                continue
            if selected_exch == "BSE Only" and not s["symbol"].endswith(".BO"):
                continue
            matched_stocks.append(s)
    else:
        for s in STOCK_DIRECTORY:
            sec = s.get("sector", "Equities")
            if selected_sector != "All Sectors" and sec != selected_sector:
                continue
            if selected_exch == "NSE Only" and not s["symbol"].endswith(".NS"):
                continue
            if selected_exch == "BSE Only" and not s["symbol"].endswith(".BO"):
                continue
            matched_stocks.append(s)

    total_matched = len(matched_stocks)
    total_pages = max(1, (total_matched + per_page - 1) // per_page)

    if "all_stocks_page" not in st.session_state:
        st.session_state.all_stocks_page = 1
    if st.session_state.all_stocks_page > total_pages:
        st.session_state.all_stocks_page = total_pages
    if st.session_state.all_stocks_page < 1:
        st.session_state.all_stocks_page = 1

    # Pagination Bar
    col_p1, col_p2, col_p3 = st.columns([1.2, 3.6, 1.2])
    with col_p1:
        if st.button("← Previous", disabled=(st.session_state.all_stocks_page <= 1), use_container_width=True):
            st.session_state.all_stocks_page -= 1
            st.rerun()
    with col_p2:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px; font-size:0.86rem; color:#94A3B8;'>"
            f"Showing Page <b style='color:#38BDF8;'>{st.session_state.all_stocks_page}</b> of <b>{total_pages}</b> "
            f"({total_matched:,} Equities Matched)"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_p3:
        if st.button("Next →", disabled=(st.session_state.all_stocks_page >= total_pages), use_container_width=True):
            st.session_state.all_stocks_page += 1
            st.rerun()

    # Table Header
    st.markdown(
        "<div style='display:flex; justify-content:space-between; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:6px; margin-top:10px; margin-bottom:6px; font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;'>"
        "<span style='width:18%;'>Ticker / Symbol</span>"
        "<span style='width:42%;'>Company Name</span>"
        "<span style='width:20%;'>Sector</span>"
        "<span style='width:8%; text-align:center;'>Exchange</span>"
        "<span style='width:12%; text-align:right;'>Action</span>"
        "</div>",
        unsafe_allow_html=True
    )

    start_idx = (st.session_state.all_stocks_page - 1) * per_page
    end_idx = min(start_idx + per_page, total_matched)
    page_items = matched_stocks[start_idx:end_idx]

    for idx, item in enumerate(page_items):
        clean_sym = item["symbol"].replace(".NS", "").replace(".BO", "")
        exch = "BSE" if item["symbol"].endswith(".BO") else "NSE"
        c_sym, c_name, c_sec, c_exch, c_action = st.columns([1.8, 4.2, 2.0, 0.8, 1.2])

        with c_sym:
            st.markdown(f"<div style='padding-top:8px; font-family:JetBrains Mono, monospace; font-weight:800; font-size:0.92rem; color:#38BDF8;'>{html.escape(clean_sym)}</div>", unsafe_allow_html=True)
        with c_name:
            st.markdown(f"<div style='padding-top:8px; font-size:0.86rem; color:#F8FAFC; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{html.escape(item['name'])}</div>", unsafe_allow_html=True)
        with c_sec:
            st.markdown(f"<div style='padding-top:8px; font-size:0.76rem; color:#94A3B8;'>{html.escape(item.get('sector', 'Equities'))}</div>", unsafe_allow_html=True)
        with c_exch:
            st.markdown(f"<div style='padding-top:8px; text-align:center;'><span class='ticker-pill' style='font-size:0.68rem; padding:2px 6px;'>{exch}</span></div>", unsafe_allow_html=True)
        with c_action:
            if st.button("Inspect", key=f"dir_btn_{item['symbol']}_{start_idx + idx}", use_container_width=True):
                st.session_state.active_ticker = item["symbol"]
                st.rerun()


def render_desks_directory_tab():
    """Directory of 50+ Tracked Sell-Side Institutional Research Desks & Investment Banks."""
    st.markdown(
        "<div style='margin-top:12px; margin-bottom:14px;'>"
        "<div style='font-size:1.15rem; font-weight:800; color:#FFFFFF;'>Monitored Institutional Desks & Research Houses</div>"
        "<div style='font-size:0.82rem; color:#94A3B8;'>Sell-side price targets, ratings revisions, and research calls are parsed from verified reports published by these institutional desks.</div>"
        "</div>",
        unsafe_allow_html=True
    )

    items = list(INSTITUTION_METADATA.items())
    cols = st.columns(3)
    for idx, (desk_name, meta) in enumerate(items):
        with cols[idx % 3]:
            tier_color = "#38BDF8" if "Global" in meta["type"] else "#10B981"
            avatar = get_broker_avatar_initials(desk_name)
            st.markdown(
                f"<div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px 14px; margin-bottom:10px; display:flex; gap:12px; align-items:center;'>"
                f"<div class='broker-avatar'>{avatar}</div>"
                f"<div style='flex:1;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;'>"
                f"<span style='font-weight:800; font-size:0.92rem; color:#F8FAFC;'>{html.escape(desk_name)}</span>"
                f"<span style='font-size:0.68rem; font-weight:700; color:{tier_color}; background:rgba(255,255,255,0.04); padding:2px 6px; border-radius:4px;'>{html.escape(meta['type'])}</span>"
                f"</div>"
                f"<div style='font-size:0.72rem; color:#64748B;'>📍 {html.escape(meta['hq'])}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )


def render_homepage():
    """Clean, Google/TradingView-grade Centered Search Landing Page & Institutional Screener."""
    # Clean Minimalist Hero Banner
    st.markdown(
        "<div style='text-align:center; padding-top:4vh; padding-bottom:20px;'>"
        "<div style='font-size:3.5rem; font-weight:800; letter-spacing:-1.8px; color:#F8FAFC; line-height:1.0; margin-bottom:6px;'>"
        "Predic<span class='brand-gradient'>SON</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # 1. Universal Search (Covers all 10,680+ stocks + live exchange fallback)
    c_left, c_mid, c_right = st.columns([0.8, 4.4, 0.8])
    with c_mid:
        with st.form("universal_search_form", clear_on_submit=False, border=False):
            col_inp, col_btn = st.columns([4.2, 1.2])
            with col_inp:
                search_query = st.text_input(
                    "Universal Search",
                    placeholder="Search company, brand, or symbol (e.g. Reliance, Zudio, Blinkit, Tata Motors, HDFC Bank)...",
                    label_visibility="collapsed"
                ).strip()
            with col_btn:
                search_submitted = st.form_submit_button("Inspect ↵", use_container_width=True, type="primary")

        matched_stocks = search_stock_directory(search_query, limit=6) if search_query else []

        # If user pressed Enter or clicked Inspect:
        if search_submitted and search_query:
            if matched_stocks:
                st.session_state.active_ticker = matched_stocks[0]["symbol"]
                st.rerun()
            else:
                st.error(f"No Indian equity found matching '{search_query}'. Please check the symbol or name.")

        if search_query and matched_stocks:
            st.markdown(
                f"<div style='font-size:0.80rem; font-weight:700; color:#38BDF8; margin-top:8px; margin-bottom:6px;'>"
                f"Matching Equities on NSE & BSE (Press Enter for #1, or select below):"
                f"</div>",
                unsafe_allow_html=True
            )
            for idx, m in enumerate(matched_stocks):
                c_sym, c_action = st.columns([4.2, 1.2])
                clean_sym = m["symbol"].replace(".NS", "").replace(".BO", "")
                is_nse = m["symbol"].endswith(".NS")
                exch_badge = "NSE" if is_nse else "BSE"
                exch_style = "background:rgba(16,185,129,0.15); color:#10B981; border:1px solid rgba(16,185,129,0.3);" if is_nse else "background:rgba(14,165,233,0.15); color:#38BDF8; border:1px solid rgba(14,165,233,0.3);"
                with c_sym:
                    st.markdown(
                        f"<div style='padding:9px 14px; background:rgba(15,23,42,0.85); border:1px solid rgba(255,255,255,0.08); border-radius:8px; margin-bottom:4px; font-size:0.90rem; display:flex; justify-content:space-between; align-items:center;'>"
                        f"<div><b style='color:#38BDF8; font-family:JetBrains Mono, monospace; font-size:0.95rem;'>{html.escape(clean_sym)}</b> "
                        f"<span style='color:#F8FAFC; margin-left:10px; font-weight:500;'>{html.escape(m['name'])}</span></div>"
                        f"<div><span style='font-size:0.70rem; font-weight:700; border-radius:4px; padding:2px 7px; margin-right:8px; {exch_style}'>{exch_badge}</span>"
                        f"<span style='font-size:0.75rem; color:#94A3B8;'>[{html.escape(m.get('sector', 'Equities'))}]</span></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with c_action:
                    if st.button("Inspect →", key=f"srch_res_{m['symbol']}_{idx}", use_container_width=True):
                        st.session_state.active_ticker = m["symbol"]
                        st.rerun()

    # Quick Jump Trending Equities Pills
    st.markdown("<div style='text-align:center; margin-top:16px; margin-bottom:12px;'><span style='font-size:0.74rem; font-weight:700; color:#64748B; text-transform:uppercase;'>Trending Heavyweights:</span></div>", unsafe_allow_html=True)
    t_cols = st.columns(8)
    trending_list = [
        ("RELIANCE.NS", "Reliance"),
        ("HDFCBANK.NS", "HDFC Bank"),
        ("TCS.NS", "TCS"),
        ("INFY.NS", "Infosys"),
        ("SBIN.NS", "SBI"),
        ("TRENT.NS", "Trent"),
        ("HAL.NS", "HAL"),
        ("SUZLON.NS", "Suzlon")
    ]
    for idx, (sym, label) in enumerate(trending_list):
        with t_cols[idx]:
            if st.button(label, key=f"home_trend_{sym}", use_container_width=True):
                st.session_state.active_ticker = sym
                st.rerun()

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # 3 Primary Workflow Tabs
    tab_screener, tab_all_stocks, tab_desks = st.tabs([
        "Institutional Screener",
        "All Equities (NSE & BSE)",
        "Institutional Desks"
    ])

    with tab_screener:
        # Top Institutional Conviction Cards (Highest Consensus Upside)
        rankings = get_universe_institutional_rankings()
        if rankings:
            st.markdown(
                "<div style='margin-top:12px; margin-bottom:12px;'>"
                "<div style='font-size:1.15rem; font-weight:800; color:#FFFFFF;'>Top Consensus Upside</div>"
                "<div style='font-size:0.82rem; color:#94A3B8;'>Equities where covering investment banks and institutional desks project the greatest 12-month upside.</div>"
                "</div>",
                unsafe_allow_html=True
            )

            top_4 = rankings[:4]
            cols = st.columns(4)
            for i, stock in enumerate(top_4):
                with cols[i]:
                    st.markdown(
                        f"<div style='background:rgba(15,23,42,0.75); border:1px solid rgba(14,165,233,0.3); border-radius:12px; padding:14px; margin-bottom:10px;'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;'>"
                        f"<div>"
                        f"<div style='font-size:1.05rem; font-weight:800; color:#FFFFFF;'>{html.escape(stock['ticker'].replace('.NS', ''))}</div>"
                        f"<div style='font-size:0.76rem; color:#94A3B8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:140px;'>{html.escape(stock['name'])}</div>"
                        f"</div>"
                        f"<span class='pill-green'>+{stock['mean_upside_pct']:.1f}%</span>"
                        f"</div>"
                        f"<div style='margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.06);'>"
                        f"<div style='display:flex; justify-content:space-between; font-size:0.80rem; margin-bottom:3px;'>"
                        f"<span style='color:#64748B;'>CMP</span>"
                        f"<span style='font-family:JetBrains Mono, monospace; font-weight:700; color:#CBD5E1;'>₹{stock['current_price']:,.1f}</span>"
                        f"</div>"
                        f"<div style='display:flex; justify-content:space-between; font-size:0.80rem; margin-bottom:4px;'>"
                        f"<span style='color:#64748B;'>Target (1Y)</span>"
                        f"<span style='font-family:JetBrains Mono, monospace; font-weight:700; color:#10B981;'>₹{stock['mean_target']:,.1f}</span>"
                        f"</div>"
                        f"<div style='display:flex; justify-content:space-between; font-size:0.74rem; color:#38BDF8; font-weight:600;'>"
                        f"<span>🏛️ {stock['analysts']} Desks</span>"
                        f"<span style='color:#94A3B8;'>{stock.get('market_cap_str', '')}</span>"
                        f"</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    if st.button(f"Analyze {stock['ticker'].replace('.NS', '')}", key=f"top_btn_{stock['ticker']}", use_container_width=True):
                        st.session_state.active_ticker = stock["ticker"]
                        st.rerun()

            # Institutional Universe Screener Table
            st.markdown(
                "<div style='margin-top:24px; margin-bottom:12px;'>"
                "<div style='font-size:1.15rem; font-weight:800; color:#FFFFFF;'>Universe Rankings</div>"
                "<div style='font-size:0.82rem; color:#94A3B8;'>Consensus targets, valuation multiples, and institutional stance across heavyweights.</div>"
                "</div>",
                unsafe_allow_html=True
            )

            for idx, row in enumerate(rankings):
                c_rank, c_name, c_cmp, c_target, c_upside, c_reco, c_mcap, c_btn = st.columns([0.5, 2.5, 1.1, 1.1, 1.1, 1.5, 1.2, 1.1])
                with c_rank:
                    st.markdown(f"<div style='padding-top:8px; font-weight:700; color:#64748B; font-size:0.88rem;'>#{idx+1}</div>", unsafe_allow_html=True)
                with c_name:
                    clean_sym = row['ticker'].replace('.NS', '')
                    st.markdown(
                        f"<div style='padding-top:2px;'>"
                        f"<span style='font-weight:800; color:#F8FAFC; font-size:0.92rem;'>{html.escape(clean_sym)}</span>"
                        f"<span style='font-size:0.76rem; color:#94A3B8; margin-left:6px;'>{html.escape(row['name'])}</span>"
                        f"<div style='font-size:0.70rem; color:#64748B;'>{html.escape(row['sector'])}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with c_cmp:
                    st.markdown(f"<div style='padding-top:8px; font-family:JetBrains Mono, monospace; font-size:0.88rem; color:#CBD5E1;'>₹{row['current_price']:,.1f}</div>", unsafe_allow_html=True)
                with c_target:
                    st.markdown(f"<div style='padding-top:8px; font-family:JetBrains Mono, monospace; font-weight:700; font-size:0.88rem; color:#38BDF8;'>₹{row['mean_target']:,.1f}</div>", unsafe_allow_html=True)
                with c_upside:
                    u_color = "pill-green" if row['mean_upside_pct'] >= 0 else "pill-red"
                    st.markdown(f"<div style='padding-top:6px;'><span class='{u_color}'>+{row['mean_upside_pct']:.1f}%</span></div>", unsafe_allow_html=True)
                with c_reco:
                    row_color = row.get("color", "#10B981")
                    row_stance = row.get("stance", "BUY")
                    st.markdown(
                        f"<div style='padding-top:4px;'>"
                        f"<span style='font-size:0.72rem; font-weight:700; color:{row_color}; background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:10px; border:1px solid {row_color};'>● {row_stance}</span>"
                        f"<div style='font-size:0.68rem; color:#64748B; margin-top:2px;'>{row['analysts']} Desks ({row['buy_ratio']:.0f}% Buy)</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with c_mcap:
                    st.markdown(f"<div style='padding-top:8px; font-size:0.78rem; color:#94A3B8;'>{row.get('market_cap_str', 'N/A')}</div>", unsafe_allow_html=True)
                with c_btn:
                    if st.button("Inspect", key=f"tbl_btn_{row['ticker']}", use_container_width=True):
                        st.session_state.active_ticker = row["ticker"]
                        st.rerun()

    with tab_all_stocks:
        render_all_stocks_tab()

    with tab_desks:
        render_desks_directory_tab()


def render_prediction_canvas(current_ticker: str):
    """Institutional Consensus & Fundamental Intelligence Canvas."""
    # Top Persistent Navigation Bar
    col_nav_brand, col_nav_switch, col_nav_back, col_nav_reload = st.columns([2.0, 4.4, 1.3, 1.3])

    with col_nav_brand:
        st.markdown(
            "<div style='display:flex; align-items:center; gap:8px; padding-top:6px;'>"
            ""
            "<span style='font-size:1.25rem; font-weight:800; color:#F8FAFC;'>Predic<span style='background:linear-gradient(135deg, #0EA5E9 0%, #10B981 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>SON</span></span>"
            "</div>",
            unsafe_allow_html=True
        )

    with col_nav_switch:
        with st.form("nav_switch_form", border=False):
            col_sw_inp, col_sw_btn = st.columns([3.8, 1.2])
            with col_sw_inp:
                sw_text = st.text_input(
                    "Switch Stock",
                    placeholder=f"Active: {current_ticker.replace('.NS', '').replace('.BO', '')} — Type stock & press Enter...",
                    label_visibility="collapsed"
                ).strip()
            with col_sw_btn:
                sw_btn = st.form_submit_button("Go ↵", use_container_width=True)
        if sw_btn and sw_text:
            sw_matches = search_stock_directory(sw_text, limit=1)
            if sw_matches and sw_matches[0]["symbol"] != current_ticker:
                st.session_state.active_ticker = sw_matches[0]["symbol"]
                st.rerun()

    with col_nav_back:
        if st.button("← Terminal", use_container_width=True):
            st.session_state.active_ticker = None
            st.rerun()

    with col_nav_reload:
        force_reload = st.button("🔄 Live Sync", use_container_width=True, help="Force fresh live data fetch from financial desks")

    # Fetch institutional and fundamental data
    with st.spinner(f"Fetching real-time institutional targets & individual broker calls for {current_ticker}..."):
        try:
            consensus = fetch_institutional_consensus(current_ticker, force_refresh=force_reload)
            if force_reload:
                st.toast(f"✅ Real-time institutional consensus synced for {current_ticker}!", icon="🏛️")
        except Exception as e:
            st.error(f"Error fetching data for {current_ticker}: {e}")
            if st.button("← Back to Terminal"):
                st.session_state.active_ticker = None
                st.rerun()
            return

    stock_meta = next((s for s in STOCK_DIRECTORY if s["symbol"] == current_ticker), {})
    company_name = stock_meta.get("name", current_ticker.replace(".NS", ""))
    sector = consensus.get("fundamentals", {}).get("sector") or stock_meta.get("sector", "Equities")
    exchange_label = "BSE" if current_ticker.endswith(".BO") else "NSE"

    curr_p = consensus["current_price"]
    pt = consensus.get("price_targets", {})
    has_coverage = consensus.get("has_analyst_coverage", False)
    funds = consensus.get("fundamentals", {})

    mean_p = pt.get("mean")
    high_p = pt.get("high")
    low_p = pt.get("low")

    mean_upside = consensus["mean_upside_pct"]
    bull_upside = consensus["bull_upside_pct"]
    stance = consensus["consensus_stance"]
    stance_color = consensus["stance_color"]
    analyst_count = consensus["total_analysts"]

    upside_sign = "+" if mean_upside > 0 else ""
    upside_pill_class = "pill-green" if mean_upside >= 0 else "pill-red"

    # Hero Banner (Responsive 2-Row Design)
    coverage_badge = (
        f"<span class='ticker-pill' style='border-color:rgba(14,165,233,0.3); color:#38BDF8;'>🏛️ {analyst_count} Covering Desks</span>"
        if has_coverage
        else "<span class='ticker-pill' style='color:#94A3B8;'>⚠️ Coverage Limited</span>"
    )

    st.markdown(
        f"<div class='terminal-hero'>"
        f"<div>"
        f"<div class='asset-headline'>"
        f"{html.escape(str(company_name))}"
        f"<span class='ticker-pill'><b>{exchange_label}:</b> {html.escape(str(current_ticker))}</span>"
        f"<span class='ticker-pill'><b>Sector:</b> {html.escape(str(sector))}</span>"
        f"</div>"
        f"<div class='badge-row'>"
        f"<span class='ticker-pill' style='border-color:{stance_color}; color:{stance_color}; background:rgba(255,255,255,0.04);'>● {stance}</span>"
        f"{coverage_badge}"
        f"<span class='ticker-pill'><b>MCap:</b> {funds.get('market_cap_str', 'N/A')}</span>"
        f"<span class='ticker-pill'><b>P/E:</b> {funds.get('pe_ratio') if funds.get('pe_ratio') else 'N/A'}x</span>"
        f"</div>"
        f"</div>"
        f"<div>"
        f"<div class='hero-price'>₹{curr_p:,.2f}</div>"
        f"<div style='font-size:0.90rem; font-weight:700; text-align:right; margin-top:4px;'>"
        + (f"Consensus Target (1Y): <span style='color:#10B981;'>₹{mean_p:,.2f}</span> <span class='{upside_pill_class}'>{upside_sign}{mean_upside:.1f}%</span>" if has_coverage and mean_p else "<span style='color:#94A3B8;'>No sell-side targets</span>") +
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 3-Pill Institutional Target Ribbon
    if has_coverage and mean_p and high_p and low_p:
        b_class = "pill-green" if bull_upside >= 0 else "pill-red"
        m_class = "pill-green" if mean_upside >= 0 else "pill-red"

        st.markdown(
            f"<div class='horizon-ribbon'>"
            f"<div class='horizon-pill'>"
            f"<div>"
            f"<div class='horizon-pill-lbl' style='color:#F59E0B;'>Bear Case Floor (Min Target)</div>"
            f"<div class='horizon-pill-val'>₹{low_p:,.2f}</div>"
            f"</div>"
            f"<span class='pill-amber'>Floor Target</span>"
            f"</div>"
            f"<div class='horizon-pill' style='border-color:rgba(14,165,233,0.45); background:rgba(14,165,233,0.08); box-shadow:0 0 16px rgba(14,165,233,0.15);'>"
            f"<div>"
            f"<div class='horizon-pill-lbl' style='color:#0EA5E9;'>Consensus Expected Target (Mean)</div>"
            f"<div class='horizon-pill-val' style='color:#10B981;'>₹{mean_p:,.2f}</div>"
            f"</div>"
            f"<span class='{m_class}'>{mean_upside:+.1f}% Upside</span>"
            f"</div>"
            f"<div class='horizon-pill'>"
            f"<div>"
            f"<div class='horizon-pill-lbl' style='color:#10B981;'>Bull Case Ceiling (Max Target)</div>"
            f"<div class='horizon-pill-val'>₹{high_p:,.2f}</div>"
            f"</div>"
            f"<span class='{b_class}'>{bull_upside:+.1f}% Bull Target</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        h52 = funds.get("fifty_two_week_high")
        l52 = funds.get("fifty_two_week_low")
        h52_str = f"₹{h52:,.2f}" if h52 else "N/A"
        l52_str = f"₹{l52:,.2f}" if l52 else "N/A"
        st.markdown(
            f"<div class='horizon-ribbon'>"
            f"<div class='horizon-pill'>"
            f"<div><div class='horizon-pill-lbl' style='color:#F43F5E;'>52-Week Low</div><div class='horizon-pill-val'>{l52_str}</div></div>"
            f"<span class='pill-red'>52W Floor</span>"
            f"</div>"
            f"<div class='horizon-pill' style='border-color:rgba(245,158,11,0.3); background:rgba(245,158,11,0.06);'>"
            f"<div><div class='horizon-pill-lbl' style='color:#F59E0B;'>Coverage Status</div><div class='horizon-pill-val' style='font-size:0.95rem; color:#F59E0B;'>Limited Sell-Side Coverage</div></div>"
            f"<span class='pill-amber'>Uncovered</span>"
            f"</div>"
            f"<div class='horizon-pill'>"
            f"<div><div class='horizon-pill-lbl' style='color:#10B981;'>52-Week High</div><div class='horizon-pill-val'>{h52_str}</div></div>"
            f"<span class='pill-green'>52W Ceiling</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 4. PROMINENT SECTION: Individual Institutional Predictions & Research Desks
    ind_preds = consensus.get("individual_institution_predictions", [])
    if ind_preds:
        st.markdown(
            "<div style='margin-top:20px; margin-bottom:12px;'>"
            "<div style='font-size:1.25rem; font-weight:800; color:#FFFFFF;'>Individual Institutional Predictions & Broker Calls</div>"
            "<div style='font-size:0.82rem; color:#94A3B8;'>Verified individual price targets, ratings, and research calls published by specific investment banks and brokerages.</div>"
            "</div>",
            unsafe_allow_html=True
        )

        num_cards = min(len(ind_preds), 4)
        pred_cols = st.columns(num_cards)
        for i in range(num_cards):
            p = ind_preds[i]
            avatar = get_broker_avatar_initials(p["institution"])
            with pred_cols[i]:
                tp_str = f"₹{p['target_price']:,.1f}" if p.get('target_price') else "Target In Report"
                up_val = p.get('implied_upside')
                if up_val is not None:
                    up_str = f"+{up_val:.1f}%" if up_val >= 0 else f"{up_val:.1f}%"
                    up_badge = f"<span class='pill-green'>{up_str}</span>" if up_val >= 0 else f"<span class='pill-red'>{up_str}</span>"
                else:
                    up_badge = ""

                st.markdown(
                    f"<div style='background:rgba(15,23,42,0.85); border:1px solid rgba(14,165,233,0.3); border-radius:12px; padding:14px; margin-bottom:12px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
                    f"<div style='display:flex; align-items:center; gap:8px;'>"
                    f"<div class='broker-avatar'>{avatar}</div>"
                    f"<div>"
                    f"<div style='font-weight:800; font-size:0.95rem; color:#F8FAFC;'>{html.escape(p['institution'])}</div>"
                    f"<div style='font-size:0.68rem; color:#64748B;'>{html.escape(p['institution_type'])}</div>"
                    f"</div>"
                    f"</div>"
                    f"<span style='font-size:0.72rem; font-weight:700; color:{p['stance_color']}; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px;'>{p['stance']}</span>"
                    f"</div>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline; margin-top:8px; margin-bottom:6px;'>"
                    f"<span style='font-family:JetBrains Mono, monospace; font-weight:800; font-size:1.20rem; color:#38BDF8;'>{tp_str}</span>"
                    f"{up_badge}"
                    f"</div>"
                    f"<div style='font-size:0.72rem; color:#94A3B8; margin-bottom:8px; line-height:1.3; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;'>"
                    f"{html.escape(p['headline'])}"
                    f"</div>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center; font-size:0.70rem; color:#64748B; padding-top:6px; border-top:1px solid rgba(255,255,255,0.05);'>"
                    f"<span>📅 {html.escape(p['published_date'])}</span>"
                    f"<a href='{p['link']}' target='_blank' style='color:#0EA5E9; text-decoration:none; font-weight:600;'>Read Report ↗</a>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        if len(ind_preds) > 4:
            with st.expander(f"📋 View All {len(ind_preds)} Covering Institutional Reports"):
                for p in ind_preds[4:]:
                    avatar = get_broker_avatar_initials(p["institution"])
                    tp_str = f"₹{p['target_price']:,.1f}" if p.get('target_price') else "Target In Report"
                    up_val = p.get('implied_upside')
                    up_str = f"+{up_val:.1f}%" if up_val is not None and up_val >= 0 else (f"{up_val:.1f}%" if up_val is not None else "")
                    c_inst, c_stance, c_tp, c_source, c_link = st.columns([2, 1.2, 1.2, 2.5, 1])
                    with c_inst:
                        st.markdown(f"<b>{html.escape(p['institution'])}</b> <span style='font-size:0.70rem; color:#64748B;'>({p['institution_type']})</span>", unsafe_allow_html=True)
                    with c_stance:
                        st.markdown(f"<span style='color:{p['stance_color']}; font-weight:700; font-size:0.75rem;'>● {p['stance']}</span>", unsafe_allow_html=True)
                    with c_tp:
                        st.markdown(f"<span style='font-family:JetBrains Mono, monospace; color:#38BDF8; font-weight:700;'>{tp_str}</span> <span style='font-size:0.72rem; color:#10B981;'>{up_str}</span>", unsafe_allow_html=True)
                    with c_source:
                        st.markdown(f"<span style='font-size:0.74rem; color:#94A3B8;'>{html.escape(p['source'])} ({p['published_date']})</span>", unsafe_allow_html=True)
                    with c_link:
                        st.markdown(f"<a href='{p['link']}' target='_blank' style='color:#0EA5E9; text-decoration:none; font-size:0.75rem;'>Report ↗</a>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:rgba(15,23,42,0.7); border:1px solid rgba(14,165,233,0.25); border-radius:10px; padding:14px 18px; margin-top:14px; margin-bottom:14px;'>"
            f"<div style='font-weight:700; color:#38BDF8; font-size:0.92rem; margin-bottom:3px;'>🏛️ Aggregated Sell-Side Consensus</div>"
            f"<div style='font-size:0.82rem; color:#CBD5E1; line-height:1.4;'>"
            f"This company has sell-side analyst tracking with an aggregated expected target of <b>₹{mean_p:,.2f}</b> (High: ₹{high_p:,.2f}, Low: ₹{low_p:,.2f}). Specific named broker research notes are parsed from live financial media wires on demand. Click <b>'🔄 Live Sync'</b> above to query fresh updates."
            f"</div>"
            f"</div>" if (has_coverage and mean_p) else
            f"<div style='background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px 16px; margin-top:14px; margin-bottom:14px; font-size:0.82rem; color:#94A3B8;'>"
            f"ℹ️ No sell-side institutional targets or broker research reports are indexed for this symbol yet. Audited fundamental ratios and technical trading range are displayed below."
            f"</div>",
            unsafe_allow_html=True
        )

    # Full-Width Institutional Corridor Chart
    fig = build_institutional_corridor_figure(current_ticker, consensus)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Deep Dive Columns
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        recos = consensus.get("recommendation_breakdown", {})
        sb = recos.get("strongBuy", 0)
        b = recos.get("buy", 0)
        h = recos.get("hold", 0)
        s = recos.get("sell", 0)
        ss = recos.get("strongSell", 0)
        tot = sb + b + h + s + ss

        buy_pct = round(((sb + b) / tot * 100), 1) if tot > 0 else 0
        hold_pct = round((h / tot * 100), 1) if tot > 0 else 0
        sell_pct = round(((s + ss) / tot * 100), 1) if tot > 0 else 0

        trend_rows = ""
        for tr in consensus.get("recommendation_trend", []):
            p_label = "Current Month" if tr['period'] == "0m" else (f"{tr['period'].replace('-', '')} Months Ago")
            trend_rows += (
                f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.80rem;'>"
                f"<span style='color:#94A3B8; font-weight:600;'>{p_label}</span>"
                f"<span style='font-family:JetBrains Mono, monospace; color:#10B981;'>{tr['strongBuy'] + tr['buy']} Buy</span>"
                f"<span style='font-family:JetBrains Mono, monospace; color:#F59E0B;'>{tr['hold']} Hold</span>"
                f"<span style='font-family:JetBrains Mono, monospace; color:#F43F5E;'>{tr['sell'] + tr['strongSell']} Sell</span>"
                f"</div>"
            )

        st.markdown(
            f"<div class='fintech-card'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>"
            f"<span style='font-size:1.0rem; font-weight:700; color:#FFFFFF;'>Institutional Rating Distribution</span>"
            f"<span style='font-size:0.78rem; font-weight:700; color:{stance_color}; background:rgba(255,255,255,0.05); padding:3px 8px; border-radius:12px; border:1px solid {stance_color};'>● {stance}</span>"
            f"</div>"
            + (
                f"<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:12px;'>"
                f"<b>{buy_pct}%</b> of institutional analysts recommend buying ({sb + b} of {tot} opinions)."
                f"</div>"
                f"<div style='display:flex; height:10px; border-radius:6px; overflow:hidden; margin-bottom:12px; background:rgba(255,255,255,0.05);'>"
                f"<div style='width:{buy_pct}%; background:#10B981;' title='Buy: {buy_pct}%'></div>"
                f"<div style='width:{hold_pct}%; background:#F59E0B;' title='Hold: {hold_pct}%'></div>"
                f"<div style='width:{sell_pct}%; background:#F43F5E;' title='Sell: {sell_pct}%'></div>"
                f"</div>"
                f"<div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:14px;'>"
                f"<span style='color:#10B981; font-weight:700;'>🟢 Buy ({sb + b})</span>"
                f"<span style='color:#F59E0B; font-weight:700;'>🟡 Hold ({h})</span>"
                f"<span style='color:#F43F5E; font-weight:700;'>🔴 Sell ({s + ss})</span>"
                f"</div>"
                f"<div style='font-size:0.76rem; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:6px;'>3-Month Revision Trend:</div>"
                f"{trend_rows}"
                f"<div style='margin-top:12px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between; font-size:0.80rem;'>"
                f"<span style='color:#64748B;'>Analyst Uncertainty Spread (Dispersion):</span>"
                f"<span style='font-family:JetBrains Mono, monospace; font-weight:700; color:#38BDF8;'>{consensus['dispersion_pct']:.1f}%</span>"
                f"</div>"
                if tot > 0
                else "<div style='color:#94A3B8; font-size:0.85rem; padding:16px 0;'>No institutional ratings published for this symbol.</div>"
            ) +
            f"</div>",
            unsafe_allow_html=True
        )

    with c2:
        pe_str = f"{funds.get('pe_ratio')}x" if funds.get('pe_ratio') else "N/A"
        fwd_pe_str = f"{funds.get('forward_pe')}x" if funds.get('forward_pe') else "N/A"
        pb_str = f"{funds.get('pb_ratio')}x" if funds.get('pb_ratio') else "N/A"
        div_str = f"{funds.get('dividend_yield')}%" if funds.get('dividend_yield') else "0.0%"
        h52_str = f"₹{funds.get('fifty_two_week_high'):,.1f}" if funds.get('fifty_two_week_high') else "N/A"
        l52_str = f"₹{funds.get('fifty_two_week_low'):,.1f}" if funds.get('fifty_two_week_low') else "N/A"
        pct_52w = funds.get("fifty_two_week_pct", 50.0)

        st.markdown(
            f"<div class='fintech-card'>"
            f"<div style='font-size:1.0rem; font-weight:700; color:#FFFFFF; margin-bottom:4px;'>Fundamental & Valuation Intelligence</div>"
            f"<div style='font-size:0.80rem; color:#94A3B8; margin-bottom:12px;'>Audited financial metrics and balance sheet multiples.</div>"
            f"<div class='metric-row'>"
            f"<span class='metric-row-lbl'>Market Capitalization</span>"
            f"<span class='metric-row-val' style='color:#38BDF8;'>{funds.get('market_cap_str', 'N/A')}</span>"
            f"</div>"
            f"<div class='metric-row'>"
            f"<span class='metric-row-lbl'>P/E Multiple (Trailing / Forward)</span>"
            f"<span class='metric-row-val'>{pe_str} • Fwd: {fwd_pe_str}</span>"
            f"</div>"
            f"<div class='metric-row'>"
            f"<span class='metric-row-lbl'>Price-to-Book (P/B) Ratio</span>"
            f"<span class='metric-row-val'>{pb_str}</span>"
            f"</div>"
            f"<div class='metric-row'>"
            f"<span class='metric-row-lbl'>Dividend Yield</span>"
            f"<span class='metric-row-val'>{div_str}</span>"
            f"</div>"
            f"<div class='metric-row'>"
            f"<span class='metric-row-lbl'>52-Week Range</span>"
            f"<span class='metric-row-val'>{l52_str} — {h52_str}</span>"
            f"</div>"
            f"<div style='margin-top:10px; margin-bottom:6px;'>"
            f"<div style='display:flex; justify-content:space-between; font-size:0.72rem; color:#64748B; margin-bottom:4px;'>"
            f"<span>52W Low</span><span>Current Position: {pct_52w:.0f}%</span><span>52W High</span>"
            f"</div>"
            f"<div style='height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;'>"
            f"<div style='width:{pct_52w}%; height:100%; background:linear-gradient(90deg, #0EA5E9, #10B981);'></div>"
            f"</div>"
            f"</div>"
            f"<div class='metric-row' style='border-bottom:none; padding-bottom:0;'>"
            f"<span class='metric-row-lbl'>Industry / Segment</span>"
            f"<span class='metric-row-val' style='font-size:0.80rem;'>{funds.get('industry', 'Equities')}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Full-Width Expert Media Wire Section
    news_items = consensus.get("expert_news", [])
    if news_items:
        news_cards = ""
        for n in news_items:
            news_cards += (
                f"<div style='padding:10px 14px; background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); border-radius:8px; margin-bottom:8px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px;'>"
                f"<a href='{n['link']}' target='_blank' style='font-size:0.88rem; font-weight:600; color:#F8FAFC; text-decoration:none;'>"
                f"{html.escape(n['title'])} ↗"
                f"</a>"
                f"<span style='font-size:0.72rem; color:#0EA5E9; background:rgba(14,165,233,0.1); padding:2px 6px; border-radius:4px; margin-left:10px; white-space:nowrap;'>{html.escape(n['source'])}</span>"
                f"</div>"
                f"<div style='font-size:0.72rem; color:#64748B;'>{html.escape(n['published'])}</div>"
                f"</div>"
            )

        st.markdown(
            f"<div class='fintech-card' style='margin-top:10px;'>"
            f"<div style='font-size:1.0rem; font-weight:700; color:#FFFFFF; margin-bottom:4px;'>Research Reports & Wire</div>"
            f"<div style='font-size:0.80rem; color:#94A3B8; margin-bottom:12px;'>Real-time research commentary, earnings previews, and broker target updates from Moneycontrol, Economic Times, and Livemint.</div>"
            f"{news_cards}"
            f"</div>",
            unsafe_allow_html=True
        )


def main():
    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = None

    if st.session_state.active_ticker is None:
        render_homepage()
    else:
        render_prediction_canvas(st.session_state.active_ticker)


if __name__ == "__main__":
    main()
