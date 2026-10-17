# PredicSON — AI Multi-Horizon Stock Prediction Platform (NSE / BSE)

> **Zero-Cost Quantitative AI Engine for Indian Equities**  
> Predicts probabilistic price cones (1-Month, 3-Month, and 6-Month Fan Charts) using open-source market data, live financial news sentiment, macroeconomic drivers, and institutional analyst consensus.

---

## 🌟 Key Capabilities

1. **Probabilistic Fan Charts (Not Single Guess Lines)**:
   - Models the future as an expanding cone of probability: $p_{10}$ (bearish boundary), $p_{25}$, $p_{50}$ (most likely median trajectory), $p_{75}$, and $p_{90}$ (bullish boundary).
   - Horizontally projects 1-Month (21 trading days), 3-Month (63 trading days), and 6-Month (126 trading days) forward horizons.

2. **Multi-Signal Input Engine (44+ Engineered Features)**:
   - **Market Momentum & Technicals**: Multi-period log returns, distance to 20/50/200 DMA, Wilder's RSI (14), MACD & Signal histogram, realized volatility, Bollinger bandwidth, ATR, and 52-week high/low extremes.
   - **Macroeconomic Drivers**: Brent Crude oil shocks (3M trend), USD/INR currency pressure, India VIX percentile rank, and US 10-Year Treasury Yield shifts.
   - **Benchmark Alignment**: Rolling 60-day stock beta vs Nifty 50 and excess relative returns.

3. **Financial Expert & Brokerage Consensus**:
   - Institutional analyst targets (Mean, Median, High, Low) and recommendation breakdowns (Strong Buy / Buy / Hold / Sell).
   - Brokerage upgrade/downgrade history tracking.

4. **Live Financial News Stream**:
   - Real-time ingestion from Indian financial RSS feeds (MoneyControl, Economic Times, LiveMint).
   - Entity recognition tagging mentions of Indian companies and market events.

5. **100% Free & Open-Source (Total Spend: ₹0)**:
   - All data sourced via `yfinance`, official RSS feeds, and open economic series.
   - High-performance local Parquet storage (no expensive database subscriptions needed).
   - Fast local CPU training (no GPU or cloud costs needed).

---

## 📁 System Architecture & Directory Structure

```
PredicSON/
├── config/
│   └── settings.py             # Stock universe, macro tickers, RSS feeds & paths
├── data/
│   ├── raw/                    # Clean zero-cost raw storage
│   │   ├── ohlcv/              # 8+ years of daily equity data (.parquet)
│   │   ├── macro/              # Consolidated macro indicators (.parquet)
│   │   ├── news/               # Live tagged news articles (.jsonl)
│   │   └── experts/            # Analyst price targets & recommendations (.json)
│   ├── features/               # Aligned multi-factor feature matrices (.parquet)
│   └── predictions/            # Quantile model predictions
├── src/
│   ├── data_ingestion/
│   │   ├── market_data.py      # Historical equities & macro indicators
│   │   ├── news_collector.py   # Indian financial news feeds & company tagger
│   │   └── expert_scraper.py   # Analyst targets, ratings & brokerage actions
│   ├── feature_engineering/
│   │   ├── technical_features.py # Multi-horizon momentum & volatility
│   │   ├── macro_features.py   # Crude, Currency, VIX & Nifty Beta
│   │   └── build_dataset.py    # Merges all factors into unified datasets
│   ├── models/
│   │   ├── quantile_forecaster.py # Multi-quantile Gradient Boosted Forecaster
│   │   └── fan_chart_generator.py # Smooth continuous trajectory interpolation
│   └── utils/
│       └── data_loader.py      # Clean dataset accessors
├── dashboard/
│   └── app.py                  # Streamlit interactive Fan Chart Dashboard
├── requirements.txt            # Zero-cost dependencies
└── main.py                     # Master CLI runner
```

---

## 🚀 Quickstart Guide

### 1. Ingest Data (Equities, Macro, News, Experts)
Pulls 8+ years of historical data for the pilot universe, consolidated macro drivers, live financial news, and expert consensus:
```bash
.\venv\Scripts\python.exe main.py ingest
```

### 2. Compile Feature Matrices
Calculates all 44+ technical, momentum, volatility, and macro alignment signals:
```bash
.\venv\Scripts\python.exe main.py features
```

### 3. Generate a Forecast (CLI)
Computes the probabilistic price cone and key drivers for any stock (e.g. Reliance):
```bash
.\venv\Scripts\python.exe main.py predict --ticker RELIANCE.NS
```

### 4. Launch Interactive Web Dashboard
Spins up the Streamlit interface with interactive Plotly Fan Charts:
```bash
.\venv\Scripts\python.exe main.py dashboard
```
Open your browser at `http://localhost:8501`.
