"""
Financial Sentiment & Noise Filtration Engine (Zero-Cost / Open-Source)
Implements:
1. Strict Noise Gate: filters out promotional clickbait, penny stock ads, and repetitive fluff.
2. Financial Lexicon Analyzer: specialized financial dictionary (based on Loughran-McDonald principles)
   measuring positive sentiment, negative headwind, litigious risk, and uncertainty.
3. Exponential Temporal Decay: models the fading market impact of historical events over time.
"""

import re
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import NEWS_DIR, FEATURES_DATA_DIR, PILOT_UNIVERSE


# --- 1. THE NOISE GATE RULES (WASTE DATA ELIMINATION) ---

# Clickbait & spam patterns to completely reject
CLICKBAIT_PATTERNS = [
    r"multibagger",
    r"penny stock",
    r"can turn \d+ into",
    r"horoscope",
    r"astrology",
    r"get rich",
    r"guaranteed return",
    r"secret stock",
    r"ready to explode",
    r"double your money",
    r"stocks to watch today: check list",
    r"top \d+ stocks to buy"
]

# Material business keywords required to consider an article high-signal
MATERIAL_EVENT_KEYWORDS = [
    "profit", "revenue", "ebitda", "pat", "margin", "guidance",
    "dividend", "bonus", "buyback", "split", "acquisition", "merger",
    "order win", "contract", "deal", "expansion", "capex", "commissioning",
    "sebi", "rbi", "penalty", "investigation", "resignation", "appointed",
    "q1", "q2", "q3", "q4", "quarterly results", "downgrade", "upgrade",
    "target price", "tariff", "export", "import", "sanction"
]


def is_high_signal_article(title: str, summary: str) -> bool:
    """
    Strict filter to reject low-substance news, clickbait, and promotional spam.
    Returns True only if the article contains genuine material business facts.
    """
    combined = f"{title} {summary}".lower()

    # Reject if too short (e.g., automated 1-sentence placeholder)
    if len(combined.strip()) < 50:
        return False

    # Reject if matching any clickbait patterns
    for pat in CLICKBAIT_PATTERNS:
        if re.search(pat, combined):
            return False

    # Must contain at least one material event keyword
    has_material_fact = any(kw in combined for kw in MATERIAL_EVENT_KEYWORDS)
    return has_material_fact


# --- 2. SPECIALIZED FINANCIAL LEXICON (LOUGHRAN-MCDONALD TAXONOMY) ---
# Standard general-English sentiment tools fail in finance (e.g., words like "liability" or "share" 
# have unique domain meanings). Here we implement domain-calibrated finance terms.

FINANCIAL_POSITIVE_WORDS = {
    "beat", "surge", "jumped", "growth", "record", "exceeded", "outperformed",
    "expansion", "strengthened", "highest", "bullish", "profitability",
    "dividend", "upgrade", "approved", "momentum", "gain", "rebounds",
    "accelerates", "robust", "resilient", "traction", "breakthrough"
}

FINANCIAL_NEGATIVE_WORDS = {
    "missed", "slumped", "plunged", "decline", "fell", "loss", "losses",
    "underperformed", "contraction", "weakened", "lowest", "bearish",
    "downgrade", "penalty", "probe", "investigation", "fraud", "default",
    "debt", "curb", "ban", "slashed", "headwind", "pressured", "resigned"
}

FINANCIAL_UNCERTAINTY_WORDS = {
    "cautious", "uncertain", "volatile", "unpredictable", "risk", "risks",
    "contingent", "fluctuate", "unclear", "doubt", "pending", "exposure"
}


def score_text_sentiment(text: str) -> Dict[str, float]:
    """
    Calculates positive, negative, net polarity (-1.0 to +1.0), and uncertainty scores
    using financial domain word frequencies.
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return {"net_sentiment": 0.0, "pos_score": 0.0, "neg_score": 0.0, "uncertainty": 0.0}

    pos_count = sum(1 for w in words if w in FINANCIAL_POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in FINANCIAL_NEGATIVE_WORDS)
    unc_count = sum(1 for w in words if w in FINANCIAL_UNCERTAINTY_WORDS)
    total_relevant = pos_count + neg_count

    if total_relevant == 0:
        net_score = 0.0
    else:
        # Standard bounded ratio: (Pos - Neg) / (Pos + Neg)
        net_score = (pos_count - neg_count) / float(total_relevant)

    total_words = len(words)
    return {
        "net_sentiment": round(float(net_score), 4),
        "pos_density": round(pos_count / total_words, 4),
        "neg_density": round(neg_count / total_words, 4),
        "uncertainty_density": round(unc_count / total_words, 4)
    }


# --- 3. TEMPORAL DECAY & AGGREGATION ENGINE ---

def process_daily_news_sentiment(start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Iterates through all collected news records, filters out waste/noise,
    scores financial sentiment, and calculates decaying sentiment series per stock.
    """
    news_files = sorted(NEWS_DIR.glob("news_*.jsonl"))
    if not news_files:
        print("No raw news files available.")
        return pd.DataFrame()

    cleaned_records = []
    dropped_count = 0

    for nf in news_files:
        with open(nf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    article = json.loads(line)
                    title = article.get("title", "")
                    summary = article.get("summary", "")

                    # Apply Noise Gate
                    if not is_high_signal_article(title, summary):
                        dropped_count += 1
                        continue

                    # Score sentiment
                    scores = score_text_sentiment(f"{title} {summary}")
                    
                    # Parse timestamp
                    pub_str = article.get("published_raw", article.get("fetched_at", ""))
                    # Extract YYYY-MM-DD
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", pub_str)
                    pub_date = date_match.group(1) if date_match else datetime.utcnow().strftime("%Y-%m-%d")

                    for ticker in article.get("matched_tickers", []):
                        if ticker in PILOT_UNIVERSE:
                            cleaned_records.append({
                                "date": pd.to_datetime(pub_date),
                                "ticker": ticker,
                                "net_sentiment": scores["net_sentiment"],
                                "uncertainty": scores["uncertainty_density"],
                                "title": title
                            })
                except Exception:
                    continue

    print(f"Noise Gate Filtered: Kept {len(cleaned_records)} material records (Discarded {dropped_count} noise/clickbait articles).")

    if not cleaned_records:
        return pd.DataFrame()

    df_news = pd.DataFrame(cleaned_records)
    return df_news


def compute_stock_decayed_sentiment_series(
    df_news: pd.DataFrame,
    ticker: str,
    calendar_index: pd.DatetimeIndex,
    half_life_days: float = 7.0
) -> pd.DataFrame:
    """
    Computes an exponentially decaying daily sentiment index for a specific stock:
    S_t = sum_{i} (sentiment_i * exp(-lambda * (t - t_i)))
    Ensures that older events fade smoothly without sudden cliff drops.
    """
    sentiment_series = pd.DataFrame(index=calendar_index)
    sentiment_series["news_sentiment_score"] = 0.0
    sentiment_series["news_uncertainty_score"] = 0.0
    sentiment_series["news_volume_count"] = 0

    if df_news.empty:
        return sentiment_series

    stock_news = df_news[df_news["ticker"] == ticker].copy()
    if stock_news.empty:
        return sentiment_series

    # Daily aggregation of incoming signals
    daily_signals = stock_news.groupby("date").agg({
        "net_sentiment": "mean",
        "uncertainty": "mean",
        "title": "count"
    }).rename(columns={"title": "count"})

    # Align with stock's trading days
    decay_rate = math.log(2) / half_life_days
    current_decayed_sent = 0.0
    current_decayed_unc = 0.0

    scores = []
    uncs = []
    counts = []

    last_date = None
    for d in calendar_index:
        if last_date is not None:
            delta_days = (d - last_date).days
            current_decayed_sent *= math.exp(-decay_rate * delta_days)
            current_decayed_unc *= math.exp(-decay_rate * delta_days)

        if d in daily_signals.index:
            row = daily_signals.loc[d]
            # Blend in today's fresh news
            current_decayed_sent += float(row["net_sentiment"])
            current_decayed_unc += float(row["uncertainty"])
            counts.append(int(row["count"]))
        else:
            counts.append(0)

        # Normalize score bounds smoothly using tanh to prevent runaway accumulation
        scores.append(math.tanh(current_decayed_sent))
        uncs.append(math.tanh(current_decayed_unc))
        last_date = d

    sentiment_series["news_sentiment_score"] = np.round(scores, 4)
    sentiment_series["news_uncertainty_score"] = np.round(uncs, 4)
    sentiment_series["news_volume_count"] = counts

    return sentiment_series
