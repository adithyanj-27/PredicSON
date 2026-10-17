"""
Data Loader Utility
Provides clean, typed accessors to load saved raw and feature datasets from Parquet and JSON.
"""

from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import json

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import OHLCV_DIR, MACRO_DIR, NEWS_DIR, EXPERTS_DIR


def load_stock_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Load historical OHLCV parquet file for a given ticker."""
    sanitized_name = ticker.replace("^", "").replace("=", "_").replace(".", "_")
    target = OHLCV_DIR / f"{sanitized_name}.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


def load_macro_indicators() -> Optional[pd.DataFrame]:
    """Load consolidated macro indicators dataset."""
    target = MACRO_DIR / "macro_indicators.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)


def load_daily_news(date_str: str) -> List[Dict]:
    """Load news articles collected on a specific date (YYYY-MM-DD)."""
    target = NEWS_DIR / f"news_{date_str}.jsonl"
    if not target.exists():
        return []
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
    """Load expert consensus snapshot for a given date."""
    target = EXPERTS_DIR / f"expert_consensus_{date_str}.json"
    if not target.exists():
        return []
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)
