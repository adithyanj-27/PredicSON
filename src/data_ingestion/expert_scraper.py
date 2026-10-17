"""
Expert Predictions & Consensus Ingestion Engine (Free Tier)
Captures financial expert research, target prices, and brokerage recommendations:
1. Structured analyst consensus (Mean/Median/High/Low targets & Buy/Sell ratings) via yfinance.
2. Brokerage upgrades/downgrades history.
3. Expert call extraction from financial news.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, EXPERTS_DIR, NEWS_DIR


def fetch_yfinance_expert_consensus(ticker: str) -> Dict[str, Any]:
    """Fetch analyst price targets, recommendations, and recent broker actions."""
    print(f"Fetching expert consensus for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        
        # 1. Price Targets (Current, High, Low, Mean, Median)
        price_targets = {}
        try:
            pt = t.analyst_price_targets
            if pt and isinstance(pt, dict):
                price_targets = {
                    "current": pt.get("current"),
                    "mean": pt.get("mean"),
                    "median": pt.get("median"),
                    "high": pt.get("high"),
                    "low": pt.get("low"),
                }
        except Exception as e:
            print(f"  Note: Analyst price targets not available for {ticker}: {e}")

        # 2. Recommendations summary (Strong Buy, Buy, Hold, Sell)
        reco_summary = {}
        try:
            recos = t.recommendations_summary
            if recos is not None and not recos.empty:
                # Get the most recent period row
                latest_reco = recos.iloc[0].to_dict()
                reco_summary = {k: int(v) if pd.notna(v) else 0 for k, v in latest_reco.items() if k != "period"}
        except Exception as e:
            try:
                # Fallback to recommendations dataframe
                recos_df = t.recommendations
                if recos_df is not None and not recos_df.empty:
                    reco_summary = recos_df.iloc[-1].to_dict()
            except Exception:
                pass

        # 3. Recent Upgrades & Downgrades from Brokerages
        recent_actions = []
        try:
            upgrades = t.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                # Take the latest 5 actions
                latest_upgrades = upgrades.head(5).reset_index()
                for _, row in latest_upgrades.iterrows():
                    action_date = str(row.get("index", row.get("Date", "")))
                    recent_actions.append({
                        "date": action_date,
                        "firm": str(row.get("Firm", "")),
                        "to_grade": str(row.get("ToGrade", "")),
                        "from_grade": str(row.get("FromGrade", "")),
                        "action": str(row.get("Action", ""))
                    })
        except Exception as e:
            pass

        return {
            "ticker": ticker,
            "fetched_at": datetime.utcnow().isoformat(),
            "price_targets": price_targets,
            "recommendation_counts": reco_summary,
            "recent_broker_actions": recent_actions
        }
    except Exception as e:
        print(f"  Error fetching consensus for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def extract_expert_calls_from_news() -> List[Dict[str, Any]]:
    """Scan collected news files for expert calls, brokerage recommendations, and target prices."""
    extracted_calls = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    news_file = NEWS_DIR / f"news_{today_str}.jsonl"

    if not news_file.exists():
        return extracted_calls

    expert_keywords = ["buy", "sell", "target", "brokerage", "jefferies", "nomura", "morgan stanley",
                       "motilal", "icici direct", "upgrade", "downgrade", "overweight", "recommends"]

    with open(news_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                article = json.loads(line)
                title = article.get("title", "").lower()
                summary = article.get("summary", "").lower()
                matched_stocks = article.get("matched_tickers", [])

                if matched_stocks and any(kw in title or kw in summary for kw in expert_keywords):
                    extracted_calls.append({
                        "id": article["id"],
                        "matched_tickers": matched_stocks,
                        "title": article["title"],
                        "source": article["source"],
                        "published": article.get("published_raw"),
                        "link": article.get("link")
                    })
            except Exception:
                continue

    return extracted_calls


def run_expert_ingestion() -> Path:
    """Fetch expert forecasts and consensus across the pilot universe."""
    print("=" * 60)
    print("PredicSON: Starting Expert Predictions & Consensus Ingestion")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    consensus_records = []
    for ticker in PILOT_UNIVERSE.keys():
        record = fetch_yfinance_expert_consensus(ticker)
        consensus_records.append(record)

    # Save to JSON
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_file = EXPERTS_DIR / f"expert_consensus_{date_str}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(consensus_records, f, indent=2)

    print(f"\nSaved expert consensus for {len(consensus_records)} stocks -> {out_file.name}")

    # Also check news for expert commentary
    news_calls = extract_expert_calls_from_news()
    if news_calls:
        calls_file = EXPERTS_DIR / f"expert_news_calls_{date_str}.json"
        with open(calls_file, "w", encoding="utf-8") as f:
            json.dump(news_calls, f, indent=2)
        print(f"Extracted {len(news_calls)} expert calls from daily news -> {calls_file.name}")

    return out_file


if __name__ == "__main__":
    run_expert_ingestion()
