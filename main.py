"""
PredicSON Master Execution Script
Unified entrypoint for running ingestion, feature generation, modeling, and dashboard.
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
    """Execute complete Layer 1 zero-cost data ingestion."""
    print("=" * 70)
    print(">>> STEP 1: RUNNING DATA INGESTION SUITE (ALL FREE SOURCES) <<<")
    print("=" * 70)
    
    # 1. Market & Macro Data
    print("\n--- Subtask 1: Equities & Macro Indicators ---")
    run_full_market_ingestion(start_date=start_date)

    # 2. Live News Feeds
    print("\n--- Subtask 2: Indian Financial News Feeds ---")
    run_news_ingestion()

    # 3. Expert Research & Consensus
    print("\n--- Subtask 3: Financial Experts & Brokerage Consensus ---")
    run_expert_ingestion()

    # 4. Corporate Fundamentals & Balance Sheet Health
    print("\n--- Subtask 4: Corporate Balance Sheets & Valuations ---")
    from src.data_ingestion.fundamentals_scraper import run_fundamentals_ingestion
    run_fundamentals_ingestion()

    print("\n" + "=" * 70)
    print(">>> INGESTION COMPLETE. ALL RAW DATASTORED IN Parquet/JSON <<<")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PredicSON AI Stock Prediction Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest raw market, macro, news & expert data")
    ingest_parser.add_argument("--start", default="2018-01-01", help="Start date for historical price data (YYYY-MM-DD)")

    # Feature engineering command
    subparsers.add_parser("features", help="Compute technical, macro, and alignment features")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate multi-horizon quantile forecast")
    predict_parser.add_argument("--ticker", default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward out-of-sample backtesting scorecard")
    backtest_parser.add_argument("--ticker", default="RELIANCE.NS", help="Ticker symbol to backtest")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch interactive Streamlit dashboard")

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
        ticker = getattr(args, "ticker", "RELIANCE.NS")
        result = generate_stock_forecast(ticker)
        print(json.dumps(result, indent=2))
    elif args.command == "backtest":
        from src.models.backtester import run_targeted_backtest
        ticker = getattr(args, "ticker", "RELIANCE.NS")
        run_targeted_backtest(ticker)
    elif args.command == "dashboard":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
