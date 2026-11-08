"""
Data Loader Utility
Provides clean, typed accessors to load saved news records and sanitize ticker strings.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import NEWS_DIR


def sanitize_ticker(ticker: str) -> str:
    """Return the filesystem-safe stem used for a ticker's cache files."""
    return ticker.replace("^", "").replace("=", "_").replace(".", "_")


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
