"""
PredicSON unit tests (no network, no large data required).
Run with:  pytest -q
"""

import pandas as pd

from config.settings import FEATURE_SCHEMA_VERSION
from src.data_ingestion.news_collector import identify_tickers
from src.data_ingestion.stock_directory import search_stock_directory
from src.models.quantile_forecaster import (
    QUANTILES,
    consensus_blend_weight,
    drop_constant_columns,
    enforce_monotonic_quantiles,
)
from src.models.fan_chart_generator import HORIZON_ORDER, _monotone_cone
from src.models.narrative_engine import _to_pct
import src.utils.data_loader as dl


# --- entity tagging (regression: substring false positives) ---

def test_entity_tagging_avoids_false_positives():
    assert identify_tickers("Kitchen appliances maker reports profit") == []
    assert identify_tickers("Titanium prices surge on strong demand") == []
    assert identify_tickers("April auto sales rise sharply") == []


def test_entity_tagging_true_positives():
    assert "ITC.NS" in identify_tickers("ITC Ltd reports strong quarterly profit")
    assert "HDFCBANK.NS" in identify_tickers("HDFC Bank posts record profit")
    assert "HDFCBANK.NS" not in identify_tickers("HDFC Life Insurance posts profit")


# --- feature hygiene ---

def test_drop_constant_columns():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "c": [1, 1, 2]})
    X2, keep = drop_constant_columns(X)
    assert keep == ["a", "c"]
    assert "b" not in X2.columns


def test_enforce_monotonic_quantiles():
    out = enforce_monotonic_quantiles({0.10: 0.05, 0.25: 0.01, 0.50: 0.02, 0.75: 0.03, 0.90: 0.04})
    values = [out[q] for q in sorted(QUANTILES)]
    assert values == sorted(values)


# --- consensus overlay is gated and capped ---

def test_consensus_weight_gating_and_cap():
    assert consensus_blend_weight(0.9, 1, 0.2) == 0.0            # too few analysts
    w = consensus_blend_weight(1.0, 20, 0.0)
    assert 0.0 < w <= 0.35
    # more disagreement -> strictly lower weight
    assert consensus_blend_weight(1.0, 20, 1.0) < consensus_blend_weight(1.0, 20, 0.0)


# --- fan-chart cone widens with horizon ---

def _horizon(p10, p25, p50, p75, p90):
    return {"p10": {"projected_price": p10}, "p25": {"projected_price": p25},
            "p50": {"projected_price": p50}, "p75": {"projected_price": p75},
            "p90": {"projected_price": p90}}


def test_cone_widens_with_horizon():
    current = 100.0
    horizons = {
        "1M": _horizon(98, 99, 101, 103, 105),
        "3M": _horizon(99, 100, 102, 104, 106),   # p10 rises -> must be corrected
        "6M": _horizon(95, 97, 104, 108, 112),
    }
    cone = _monotone_cone(current, horizons)

    p10 = [cone[h][0] for h in HORIZON_ORDER]
    p90 = [cone[h][4] for h in HORIZON_ORDER]
    assert all(p10[i] >= p10[i + 1] for i in range(len(p10) - 1))   # non-increasing
    assert all(p90[i] <= p90[i + 1] for i in range(len(p90) - 1))   # non-decreasing
    for h in HORIZON_ORDER:
        assert list(cone[h]) == sorted(cone[h])                     # internal ordering


def test_published_cone_monotone_across_horizons():
    from src.models.quantile_forecaster import enforce_cone_across_horizons

    def h(p10, p25, p50, p75, p90):
        return {f"p{q}": {"projected_price": v, "expected_return_pct": 0.0}
                for q, v in zip([10, 25, 50, 75, 90], [p10, p25, p50, p75, p90])}

    fc = {"1M": h(98, 99, 101, 103, 105),
          "3M": h(99, 100, 102, 104, 106),   # p10 rises -> corrected
          "6M": h(95, 97, 104, 108, 112)}
    out = enforce_cone_across_horizons(fc, 100.0)

    p10 = [out[k]["p10"]["projected_price"] for k in ["1M", "3M", "6M"]]
    p90 = [out[k]["p90"]["projected_price"] for k in ["1M", "3M", "6M"]]
    assert p10 == sorted(p10, reverse=True)   # non-increasing
    assert p90 == sorted(p90)                 # non-decreasing
    for k in ["1M", "3M", "6M"]:
        vals = [out[k][f"p{q}"]["projected_price"] for q in [10, 25, 50, 75, 90]]
        assert vals == sorted(vals)


# --- narrative parsing (regression: substring sniffing) ---

def test_to_pct_parsing():
    assert _to_pct("+12.5%") == 12.5
    assert _to_pct("-3%") == -3.0
    assert _to_pct(0.25) == 0.25
    assert _to_pct(None, 7.0) == 7.0
    assert _to_pct("n/a", 5.0) == 5.0


# --- directory search ---

def test_search_directory_known_and_custom():
    res = search_stock_directory("reliance", limit=3)
    assert any(r["symbol"] == "RELIANCE.NS" for r in res)

    custom = search_stock_directory("ZZZZNONEXISTENT", limit=3)
    assert custom and custom[0]["symbol"].endswith(".NS")


# --- feature-matrix versioning ---

def test_feature_meta_versioning(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "FEATURES_DATA_DIR", tmp_path)
    assert dl.feature_is_current("TEST.NS") is False

    dl.write_feature_meta("TEST.NS", n_rows=10, n_cols=5)
    assert dl.feature_is_current("TEST.NS") is False  # meta exists but parquet missing

    pd.DataFrame({"x": [1, 2]}).to_parquet(dl.feature_paths("TEST.NS")[0])
    assert dl.feature_is_current("TEST.NS") is True
    assert FEATURE_SCHEMA_VERSION >= 1
