"""
Data Loader Utility
Provides clean, typed accessors to load saved raw and feature datasets from Parquet and JSON.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OHLCV_DIR, MACRO_DIR, NEWS_DIR, EXPERTS_DIR, FEATURES_DATA_DIR, FEATURE_SCHEMA_VERSION


def sanitize_ticker(ticker: str) -> str:
    """Return the filesystem-safe stem used for a ticker's data files."""
    return ticker.replace("^", "").replace("=", "_").replace(".", "_")


def load_stock_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load historical OHLCV parquet for a ticker.

    If the file carries a separate `adj_close` column (older ingestion runs used
    auto_adjust=False), promote it to `close` so all downstream return/target
    maths operate on split- and dividend-adjusted prices. Splits alone are
    already handled by yfinance; this closes the dividend gap too.
    """
    target = OHLCV_DIR / f"{sanitize_ticker(ticker)}.parquet"
    if not target.exists():
        return None
    df = pd.read_parquet(target)
    if "adj_close" in df.columns and "close" in df.columns:
        df = df.assign(close=df["adj_close"])
    return df


def load_macro_indicators() -> Optional[pd.DataFrame]:
    """Load consolidated macro indicators dataset."""
    target = MACRO_DIR / "macro_indicators.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


def load_daily_news(date_str: str) -> List[Dict]:
    """Load news articles collected on a specific date (YYYY-MM-DD), with fallback to latest."""
    target = NEWS_DIR / f"news_{date_str}.jsonl"
    if not target.exists():
        news_files = sorted(NEWS_DIR.glob("news_*.jsonl"))
        if not news_files:
            return []
        target = news_files[-1]
    articles = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return articles


def load_expert_consensus(date_str: str) -> List[Dict]:
    """Load expert consensus snapshot for a given date, with fallback to latest."""
    target = EXPERTS_DIR / f"expert_consensus_{date_str}.json"
    if not target.exists():
        expert_files = sorted(EXPERTS_DIR.glob("expert_consensus_*.json"))
        if not expert_files:
            return []
        target = expert_files[-1]
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature-matrix versioning
# ---------------------------------------------------------------------------

def feature_paths(ticker: str):
    """Return (parquet_path, meta_path) for a ticker's feature matrix."""
    stem = f"{sanitize_ticker(ticker)}_features"
    return FEATURES_DATA_DIR / f"{stem}.parquet", FEATURES_DATA_DIR / f"{stem}.meta.json"


def write_feature_meta(ticker: str, n_rows: int, n_cols: int) -> None:
    """Record the schema version a feature matrix was built with."""
    _, meta_path = feature_paths(ticker)
    meta = {
        "ticker": ticker,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def feature_is_current(ticker: str) -> bool:
    """True only if a feature matrix exists and was built with the current schema version."""
    parquet_path, meta_path = feature_paths(ticker)
    if not parquet_path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("schema_version", -1)) == FEATURE_SCHEMA_VERSION
    except (json.JSONDecodeError, OSError, ValueError):
        return False
