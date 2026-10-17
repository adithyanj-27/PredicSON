# PredicSON — Institutional AI Stock Prediction & Quant Allocation Terminal (NSE / BSE)

> **Zero-Cost Quantitative AI Engine & Portfolio Allocator for Indian Equities**  
> Predicts probabilistic price cones (1-Month, 3-Month, and 6-Month Fan Charts) across 25 Nifty titans using 72 engineered factors, macroeconomic regimes, Loughran-McDonald news sentiment, corporate fundamentals, and institutional analyst consensus. Includes an automated opportunity screener and Modern Portfolio Theory (MPT) capital allocator.

---

## 🌟 Key Capabilities

### 1. Probabilistic Fan Charts (Not Single Guess Lines)
- Models the future as an expanding continuous cone of probability: $p_{10}$ (bearish floor), $p_{25}$, $p_{50}$ (most likely median trajectory), $p_{75}$, and $p_{90}$ (bullish breakout ceiling).
- Strict non-crossing monotonic guarantees ($p_{10} \le p_{25} \le p_{50} \le p_{75} \le p_{90}$) to eliminate noise distortion.
- Horizontally projects 1-Month (21 trading days), 3-Month (63 trading days), and 6-Month (126 trading days) forward horizons.

### 2. Multi-Signal Quantitative Architecture (72 Engineered Factors)
- **Technical & Price Action**: Multi-horizon returns, 20/50/200 DMA distance ratios, Wilder's EWM RSI (14), MACD & signal divergence, Parkinson extreme-value volatility, ATR, Volatility Z-scores ($Z = \frac{\text{close} - \text{SMA}_{50}}{\text{ATR}}$), Chaikin Money Flow (CMF), Rolling 63-day Sharpe ratio.
- **Corporate Balance Sheet Fundamentals**: P/E, PEG ratio, P/B, Operating Margins, YoY Revenue & Profit growth, Promoter & Institutional (FII/DII) ownership.
- **Financial News Sentiment (Strict Noise Gate + Decay)**: Regex-based noise filter blocking clickbait and penny-stock promotional spam, Loughran-McDonald financial dictionary scoring, and exponential half-life decay ($\lambda = 7$ days).
- **Macroeconomic Drivers & Regime Classification**: Brent Crude oil shocks (3M trend), USD/INR currency pressure, India VIX percentile rank, US 10-Year Treasury Yield shifts, and rolling beta vs Nifty 50.
- **Idiosyncratic Residual Alpha**: Residual excess performance ($R_{\text{stock}} - \beta R_{\text{nifty}}$), momentum acceleration, and volume expansion.

### 3. Cross-Sectional Market Screener & Signal Engine
- Automatically ranks all 25 titans across Expected 3M Return, Downside Risk ($p_{10}$), Risk-Reward Ratio, Valuation (PEG, P/E), and Money Flow (CMF).
- Issues quantitative conviction badges:
  - 🟢 **STRONG BUY**: High expected return + positive institutional money flow + PEG < 2.0.
  - 🔵 **ACCUMULATE**: Positive expected trajectory + stable capital flow.
  - ⚪ **NEUTRAL / HOLD**: Range-bound consolidation.
  - 🔴 **AVOID / TRIM**: Deteriorating momentum or extreme downside asymmetry.

### 4. Modern Portfolio Theory (MPT) Capital Allocator
- Solves for the **Maximum Sharpe Ratio Portfolio**: $\max \frac{w^T r - r_f}{\sqrt{w^T \Sigma w}}$ using forward model expected returns and empirical covariance.
- Inputs custom investment capital (in ₹ INR) and applies user-defined single-stock concentration caps (e.g. max 25% per company).
- Visualizes optimal rupee allocations and sector breakdown donuts.

### 5. High-Performance & 100% Free (Total Spend: ₹0)
- **Zero API Costs**: 100% open-source market data, live financial RSS feeds, and local lexicon NLP.
- **Blazing Fast LightGBM Quantile Boosting**: Sub-3-second multi-horizon inference per stock on CPU.
- **No Paid Infrastructure**: Runs locally or deploys for free on Streamlit Community Cloud.

---

## 🏛️ Tracked Universe (25 Heavyweights across 8 Sectors)

| Sector | Tickers |
| :--- | :--- |
| **Energy & Conglomerate** | Reliance Industries (`RELIANCE.NS`), ONGC (`ONGC.NS`), NTPC (`NTPC.NS`), Power Grid (`POWERGRID.NS`) |
| **Banking & Financials** | HDFC Bank (`HDFCBANK.NS`), ICICI Bank (`ICICIBANK.NS`), SBI (`SBIN.NS`), Kotak Mahindra (`KOTAKBANK.NS`), Axis Bank (`AXISBANK.NS`), Bajaj Finance (`BAJFINANCE.NS`) |
| **Information Technology** | TCS (`TCS.NS`), Infosys (`INFY.NS`) |
| **Automotive** | Maruti Suzuki (`MARUTI.NS`), Mahindra & Mahindra (`M&M.NS`) |
| **Metals & Materials** | Tata Steel (`TATASTEEL.NS`), Hindalco (`HINDALCO.NS`), UltraTech Cement (`ULTRACEMCO.NS`) |
| **Pharma & Healthcare** | Sun Pharma (`SUNPHARMA.NS`), Cipla (`CIPLA.NS`) |
| **FMCG & Consumer** | ITC (`ITC.NS`), Hindustan Unilever (`HINDUNILVR.NS`), Titan (`TITAN.NS`), Asian Paints (`ASIANPAINT.NS`) |
| **Telecom & Infrastructure** | Bharti Airtel (`BHARTIARTL.NS`), Larsen & Toubro (`LT.NS`) |

---

## 📁 System Architecture

```
PredicSON/
├── config/
│   └── settings.py             # 25-stock universe, macro tickers, RSS feeds & paths
├── data/
│   ├── raw/                    # Zero-cost raw storage
│   │   ├── ohlcv/              # 8+ years of daily equity data (.parquet)
│   │   ├── macro/              # Consolidated macro indicators (.parquet)
│   │   ├── news/               # Live tagged news articles (.jsonl)
│   │   ├── experts/            # Analyst price targets & recommendations (.json)
│   │   └── fundamentals/       # Balance sheet metrics (.json)
│   ├── features/               # 72-factor feature matrices (.parquet)
│   └── predictions/            # Cached screener & quantile predictions (.json)
├── src/
│   ├── data_ingestion/
│   │   ├── market_data.py      # Equities & macro indicators
│   │   ├── news_collector.py   # Indian financial news feeds & company tagger
│   │   ├── expert_scraper.py   # Analyst targets & ratings
│   │   └── fundamentals_scraper.py # Balance sheet metrics & ownership
│   ├── feature_engineering/
│   │   ├── technical_features.py # 72 quantitative momentum, volatility & flow factors
│   │   ├── macro_features.py   # Crude, Currency, VIX, US 10Y & Nifty Beta
│   │   ├── sentiment_engine.py # Noise Gate + Loughran-McDonald + decay
│   │   ├── fundamental_features.py # P/E, PEG, profit margin standardization
│   │   ├── consensus_engine.py # Analyst dispersion & conviction ratio
│   │   ├── alpha_engine.py     # Idiosyncratic residual alpha
│   │   └── build_dataset.py    # Merges all factors into unified matrices
│   ├── models/
│   │   ├── quantile_forecaster.py # Stacked LightGBM + Drift Quantile Forecaster
│   │   ├── regime_classifier.py   # Macro regime detection (Bull/Bear/Vol/Consolidation)
│   │   ├── narrative_engine.py    # AI Investment Thesis (Stance, Catalysts, Headwinds)
│   │   ├── screener_engine.py     # Cross-sectional opportunity ranking & signals
│   │   ├── portfolio_allocator.py # Modern Portfolio Theory Sharpe optimizer
│   │   ├── backtester.py          # Out-of-sample walk-forward validation
│   │   └── fan_chart_generator.py # Smooth continuous trajectory interpolation
│   └── utils/
│       └── data_loader.py      # Dataset accessors
├── dashboard/
│   └── app.py                  # Streamlit 3-Tab Quant Terminal
├── requirements.txt            # Zero-cost dependencies
└── main.py                     # Master CLI runner
```

---

## 🚀 Quickstart Guide

### 1. Ingest Data (Equities, Macro, News, Fundamentals, Experts)
Pulls 8+ years of historical data for the 25-stock universe, consolidated macro drivers, live financial news, corporate balance sheet ratios, and analyst consensus:
```bash
.\venv\Scripts\python.exe main.py ingest
```

### 2. Compile Feature Matrices
Calculates all 72 technical, momentum, volatility, fundamental, sentiment, and macro signals:
```bash
.\venv\Scripts\python.exe main.py features
```

### 3. Run the Cross-Sectional Screener
Evaluates and ranks all 25 stocks in under 60 seconds:
```bash
.\venv\Scripts\python.exe src/models/screener_engine.py
```

### 4. Launch Interactive Web Terminal
Spins up the Streamlit interface with 3 full tabs (Single-Stock Deep Dive, Market Screener, MPT Allocator):
```bash
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.
