"""
PredicSON Master Execution Script
Unified entrypoint for institutional consensus, Dalal Street wire ingestion, screener, and dashboard.
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path

# Set UTF-8 encoding for standard output if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_ingestion.institutional_engine import (
    fetch_institutional_consensus,
    get_universe_institutional_rankings
)
from src.data_ingestion.news_collector import run_news_ingestion


def handle_consensus(ticker: str, force_refresh: bool = False):
    """Fetch and display institutional consensus for a specific stock."""
    c = fetch_institutional_consensus(ticker, force_refresh=force_refresh)
    print(f"\n{'='*70}")
    print(f"🏛️  INSTITUTIONAL CONSENSUS: {c['ticker']} ({c.get('fundamentals', {}).get('company_name', '')})")
    print(f"Current Price: ₹{c['current_price']:,.2f}")
    mean_target = c['price_targets'].get('mean')
    if mean_target:
        print(f"Consensus Target (1Y Mean): ₹{mean_target:,.2f} ({c['mean_upside_pct']:+.1f}%)")
        print(f"Corridor: Bear ₹{c['price_targets'].get('low', 0):,.2f} <---> Bull ₹{c['price_targets'].get('high', 0):,.2f}")
    else:
        print("Consensus Target: Limited sell-side coverage")
    print(f"Consensus Stance: {c['consensus_stance']} ({c['total_analysts']} Covering Desks)")

    broker_calls = c.get("individual_institution_predictions", [])
    if broker_calls:
        print(f"\nRecent Individual Institutional Research Calls ({len(broker_calls)}):")
        for b in broker_calls[:5]:
            tp_str = f"₹{b['target_price']:,.1f}" if b.get('target_price') else "N/A"
            upside_str = f"({b['implied_upside']:+.1f}%)" if b.get('implied_upside') is not None else ""
            print(f"  • {b['institution']:<24} {b['stance']:<16} Target: {tp_str:<10} {upside_str}")
    print(f"{'='*70}\n")


def handle_rankings():
    """Display universe rankings sorted by sell-side institutional upside."""
    rankings = get_universe_institutional_rankings()
    print(f"\n{'='*80}")
    print(f"{'RANK':<5} {'TICKER':<14} {'NAME':<24} {'CMP':<10} {'TARGET':<10} {'UPSIDE':<10} {'STANCE'}")
    print(f"{'='*80}")
    for idx, r in enumerate(rankings):
        print(f"#{idx+1:<4} {r['ticker']:<14} {r['name'][:22]:<24} ₹{r['current_price']:<8.1f} ₹{r['mean_target']:<8.1f} {r['mean_upside_pct']:+5.1f}%    {r['stance']}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PredicSON — Institutional Consensus & Expert Intelligence Terminal (NSE/BSE)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # consensus
    consensus_parser = subparsers.add_parser("consensus", help="Fetch real-time institutional consensus for a ticker")
    consensus_parser.add_argument("--ticker", default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS, TCS.NS)")
    consensus_parser.add_argument("--refresh", action="store_true", help="Force refresh from live sources")

    # consensus-all
    subparsers.add_parser("consensus-all", help="Generate institutional rankings across the universe")

    # screener
    subparsers.add_parser("screener", help="Refresh institutional screener rankings")

    # news
    subparsers.add_parser("news", help="Ingest latest Dalal Street news feeds & brokerage reports")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch the interactive Streamlit terminal")

    args = parser.parse_args()

    if args.command == "consensus":
        handle_consensus(args.ticker, force_refresh=args.refresh)
    elif args.command in ("consensus-all", "screener"):
        handle_rankings()
    elif args.command == "news":
        print("Ingesting live Dalal Street financial news feeds...")
        run_news_ingestion()
        print("News wire updated.")
    elif args.command == "dashboard":
        app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    else:
        # Default action: launch dashboard or show help
        parser.print_help()


if __name__ == "__main__":
    main()
