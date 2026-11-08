"""
PredicSON unit tests for Institutional Consensus Terminal.
Run with:  pytest -q
"""

import pytest
from config.settings import PILOT_UNIVERSE, NEWS_RSS_FEEDS
from src.data_ingestion.news_collector import identify_tickers
from src.data_ingestion.stock_directory import search_stock_directory
from src.data_ingestion.institutional_engine import (
    INSTITUTION_METADATA,
    format_inr_crore,
    fetch_institutional_consensus,
    build_institutional_corridor_figure,
    get_universe_institutional_rankings
)
from src.utils.data_loader import sanitize_ticker


# --- 1. Entity Tagging (News Wire Ticker Extraction) ---

def test_entity_tagging_avoids_false_positives():
    assert identify_tickers("Kitchen appliances maker reports profit") == []
    assert identify_tickers("Titanium prices surge on strong demand") == []
    assert identify_tickers("April auto sales rise sharply") == []


def test_entity_tagging_true_positives():
    assert "ITC.NS" in identify_tickers("ITC Ltd reports strong quarterly profit")
    assert "HDFCBANK.NS" in identify_tickers("HDFC Bank posts record profit")
    assert "HDFCBANK.NS" not in identify_tickers("HDFC Life Insurance posts profit")


# --- 2. Stock Directory Universal Search ---

def test_search_directory_known_and_custom():
    res = search_stock_directory("reliance", limit=3)
    assert any(r["symbol"] == "RELIANCE.NS" for r in res)

    custom = search_stock_directory("ZZZZNONEXISTENT", limit=3)
    assert custom and custom[0]["symbol"].endswith(".NS")
    assert " " not in custom[0]["symbol"]


def test_search_directory_multi_token_and_brand_aliases():
    # Multi-token company name matching
    fed = search_stock_directory("federal bank", limit=3)
    assert any(r["symbol"] == "FEDERALBNK.NS" for r in fed)

    sbi = search_stock_directory("state bank of india", limit=3)
    assert any(r["symbol"] == "SBIN.NS" for r in sbi)

    pnb = search_stock_directory("punjab national bank", limit=3)
    assert any(r["symbol"] == "PNB.NS" for r in pnb)

    tm = search_stock_directory("tata motors", limit=3)
    assert any(r["symbol"] == "TATAMOTORS.NS" for r in tm)

    # Brand and alias matching
    zudio = search_stock_directory("zudio", limit=3)
    assert any(r["symbol"] == "TRENT.NS" for r in zudio)

    blinkit = search_stock_directory("blinkit", limit=3)
    assert any(r["symbol"] == "ZOMATO.NS" for r in blinkit)


def test_search_directory_fuzzy_typo_tolerance():
    # Common user typos should gracefully resolve to the target stock
    assert search_stock_directory("relaince", limit=1)[0]["symbol"] == "RELIANCE.NS"
    assert search_stock_directory("tatamotor", limit=1)[0]["symbol"] == "TATAMOTORS.NS"
    assert search_stock_directory("infoss", limit=1)[0]["symbol"] == "INFY.NS"
    assert search_stock_directory("hdfcbnk", limit=1)[0]["symbol"] == "HDFCBANK.NS"


def test_search_directory_prioritizes_nse():
    # Default search on Dalal Street should return the NSE (.NS) listing before BSE (.BO)
    infy_res = search_stock_directory("infosys", limit=1)
    assert infy_res[0]["symbol"] == "INFY.NS"

    ril_res = search_stock_directory("reliance", limit=1)
    assert ril_res[0]["symbol"] == "RELIANCE.NS"


def test_custom_fallback_never_creates_spaces_in_ticker():
    res = search_stock_directory("completely unknown stock name xyz 123", limit=3)
    for r in res:
        assert " " not in r["symbol"]


# --- 3. Utilities & Formatting ---

def test_sanitize_ticker():
    assert sanitize_ticker("RELIANCE.NS") == "RELIANCE_NS"
    assert sanitize_ticker("^NSEI") == "NSEI"
    assert sanitize_ticker("INR=X") == "INR_X"


def test_format_inr_crore():
    assert format_inr_crore(None) == "N/A"
    assert format_inr_crore(0) == "N/A"
    assert "Lakh Cr" in format_inr_crore(1.5e12)
    assert "Cr" in format_inr_crore(500e7)


# --- 4. Configuration & Metadata ---

def test_pilot_universe_config():
    assert len(PILOT_UNIVERSE) >= 25
    assert "RELIANCE.NS" in PILOT_UNIVERSE
    assert "TCS.NS" in PILOT_UNIVERSE


def test_institutional_metadata():
    assert "Goldman Sachs" in INSTITUTION_METADATA
    assert "Motilal Oswal" in INSTITUTION_METADATA
    assert "ICICI Direct" in INSTITUTION_METADATA
    assert "Jefferies" in INSTITUTION_METADATA


# --- 5. Institutional Consensus Engine ---

def test_institutional_engine():
    data = fetch_institutional_consensus("RELIANCE.NS")
    assert data["ticker"] == "RELIANCE.NS"
    assert "price_targets" in data
    assert data["current_price"] > 0
    assert "mean" in data["price_targets"]
    assert "consensus_stance" in data
    assert data["total_analysts"] >= 1

    fig = build_institutional_corridor_figure("RELIANCE.NS", data)
    assert len(fig.data) >= 3

    rankings = get_universe_institutional_rankings()
    assert isinstance(rankings, list)
    assert len(rankings) > 0
    assert "mean_upside_pct" in rankings[0]
