"""
Financial News Ingestion Engine (Free Tier)
Aggregates live financial news from RSS feeds across MoneyControl, Economic Times, LiveMint, etc.
Performs basic entity matching against the Indian Stock Universe and saves structured news records.
"""

import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import feedparser
from bs4 import BeautifulSoup

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    NEWS_RSS_FEEDS,
    PILOT_UNIVERSE,
    NEWS_DIR
)

# Company keyword matching rules for entity identification
COMPANY_KEYWORD_MAP = {
    "RELIANCE.NS": ["reliance", "ril", "mukesh ambani", "jio", "reliance retail"],
    "TCS.NS": ["tcs", "tata consultancy", "tata consultancy services"],
    "HDFCBANK.NS": ["hdfc bank", "hdfc", "sashidhar jagdishan"],
    "INFY.NS": ["infosys", "salil parekh", "nandan nilekani"],
    "ICICIBANK.NS": ["icici bank", "icici", "sandeep bakhshi"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel", "sunil mittal"],
    "SBIN.NS": ["sbi", "state bank of india", "dinesh khara"],
    "ITC.NS": ["itc", "itc ltd", "sanjiv puri"],
    "LT.NS": ["l&t", "larsen & toubro", "larsen and toubro", "subrahmanyan"],
    "HINDUNILVR.NS": ["hindustan unilever", "hul", "unilever", "rohit jawa"]
}


def clean_html(raw_html: str) -> str:
    """Strip HTML markup from summaries."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ").strip()


def identify_tickers(text: str) -> List[str]:
    """Identify which stocks in our universe are mentioned in the text."""
    lower_text = text.lower()
    matched = []
    for ticker, keywords in COMPANY_KEYWORD_MAP.items():
        if any(kw in lower_text for kw in keywords):
            matched.append(ticker)
    return matched


def generate_article_id(title: str, link: str) -> str:
    """Create a unique deterministic ID for deduplication."""
    return hashlib.md5(f"{title}_{link}".encode("utf-8")).hexdigest()


def fetch_rss_feed(source_name: str, feed_url: str) -> List[Dict[str, Any]]:
    """Parse a single RSS feed and return normalized articles."""
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary = clean_html(entry.get("summary", entry.get("description", "")))
            published = entry.get("published", entry.get("updated", datetime.utcnow().isoformat()))

            full_text = f"{title}. {summary}"
            tickers = identify_tickers(full_text)

            article_record = {
                "id": generate_article_id(title, link),
                "source": source_name,
                "title": title,
                "summary": summary,
                "link": link,
                "published_raw": published,
                "fetched_at": datetime.utcnow().isoformat(),
                "matched_tickers": tickers
            }
            articles.append(article_record)
    except Exception as e:
        print(f"  Error fetching feed {source_name}: {e}")
    return articles


def run_news_ingestion() -> Path:
    """Fetch all configured RSS feeds, deduplicate, and append to daily news log."""
    print("=" * 60)
    print("PredicSON: Starting Zero-Cost Financial News Ingestion")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_articles = []
    for source_name, feed_url in NEWS_RSS_FEEDS.items():
        print(f"Fetching {source_name}...")
        feed_articles = fetch_rss_feed(source_name, feed_url)
        print(f"  Retrieved {len(feed_articles)} articles.")
        all_articles.extend(feed_articles)

    # Load existing IDs to prevent duplicates
    date_str = datetime.now().strftime("%Y-%m-%d")
    daily_file = NEWS_DIR / f"news_{date_str}.jsonl"

    existing_ids = set()
    if daily_file.exists():
        with open(daily_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue

    # Filter out duplicates
    new_articles = [a for a in all_articles if a["id"] not in existing_ids]

    # Append new articles
    with open(daily_file, "a", encoding="utf-8") as f:
        for article in new_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    # Metrics
    stocks_mentioned = sum(1 for a in new_articles if len(a["matched_tickers"]) > 0)
    print(f"\nSaved {len(new_articles)} new articles to {daily_file.name} ({stocks_mentioned} tagged to pilot stocks).")
    return daily_file


if __name__ == "__main__":
    run_news_ingestion()
