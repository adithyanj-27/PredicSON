"""
PredicSON Master Execution Script
Unified entrypoint for ingestion, feature generation, modelling, and the dashboard.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_ingestion.market_data import run_full_market_ingestion
from src.data_ingestion.news_collector import run_news_ingestion
from src.data_ingestion.expert_scraper import run_expert_ingestion


def handle_ingestion(start_date: str = "2018-01-01"):
    """Execute the complete zero-cost data ingestion suite."""
    print("=" * 70)
    print(">>> STEP 1: DATA INGESTION SUITE (ALL FREE SOURCES) <<<")
    print("=" * 70)

    print("\n--- Subtask 1: Equities & Macro Indicators ---")
    run_full_market_ingestion(start_date=start_date)

    print("\n--- Subtask 2: Indian Financial News Feeds ---")
    run_news_ingestion()

    print("\n--- Subtask 3: Financial Experts & Brokerage Consensus ---")
    run_expert_ingestion()

    print("\n--- Subtask 4: Corporate Balance Sheets & Valuations ---")
    from src.data_ingestion.fundamentals_scraper import run_fundamentals_ingestion
    run_fundamentals_ingestion()

    print("\n" + "=" * 70)
    print(">>> INGESTION COMPLETE. RAW DATA STORED IN Parquet/JSON <<<")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PredicSON AI Stock Prediction Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw market, macro, news & expert data")
    ingest_parser.add_argument("--start", default="2018-01-01", help="Start date for historical price data (YYYY-MM-DD)")

    subparsers.add_parser("features", help="Compute the feature matrices")

    predict_parser = subparsers.add_parser("predict", help="Generate a multi-horizon quantile forecast")
    predict_parser.add_argument("--ticker", default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")

    backtest_parser = subparsers.add_parser("backtest", help="Run a purged walk-forward backtest for one ticker")
    backtest_parser.add_argument("--ticker", default="RELIANCE.NS", help="Ticker symbol to backtest")

    subparsers.add_parser("backtest-all", help="Run the purged backtest across the whole universe")

    subparsers.add_parser("screener", help="Build the cross-sectional market screener")

    subparsers.add_parser("dashboard", help="Launch the interactive Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "ingest" or args.command is None:
        start_date = getattr(args, "start", "2018-01-01") if args.command else "2018-01-01"
        handle_ingestion(start_date=start_date)
    elif args.command == "features":
        from src.feature_engineering.build_dataset import build_all_features
        build_all_features()
    elif args.command == "predict":
        import json
        from src.models.quantile_forecaster import generate_stock_forecast
        print(json.dumps(generate_stock_forecast(getattr(args, "ticker", "RELIANCE.NS")), indent=2))
    elif args.command == "backtest":
        from src.models.backtester import run_targeted_backtest
        run_targeted_backtest(getattr(args, "ticker", "RELIANCE.NS"))
    elif args.command == "backtest-all":
        from src.models.backtester import run_full_backtest
        run_full_backtest()
    elif args.command == "screener":
        from src.models.screener_engine import generate_market_screener
        generate_market_screener(use_cache=False)
    elif args.command == "dashboard":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
