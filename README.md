# PredicSON — Probabilistic Stock Forecaster & Quant Allocator (NSE / BSE)

> **Zero-cost quantitative forecasting for Indian equities.**
> Produces probabilistic price cones (1M / 3M / 6M) for 25 Nifty heavyweights, plus a
> cross-sectional screener and a maximum-Sharpe capital allocator. Everything runs on
> free data sources and local, open-source models.

---

## What it actually does

- **Probabilistic price cones**, not point guesses. For each horizon it reports
  p10 / p25 / p50 / p75 / p90 with a strict non-crossing guarantee
  (`p10 <= p25 <= p50 <= p75 <= p90`), and the cone is forced to widen with horizon.
- **Multi-signal model.** Technical/momentum/volatility features, macro drivers
  (Brent, USD/INR, India VIX, US 10Y, Nifty beta), a noise-gated financial-news
  sentiment series with exponential decay, and a macro regime classifier.
- **Purged walk-forward backtest** that reports directional accuracy, cone coverage,
  and MAE out-of-sample.
- **Cross-sectional screener** ranking the universe by expected 3M return,
  risk-reward, valuation, and money flow.
- **Max-Sharpe allocator** (Modern Portfolio Theory) with per-stock caps.

### Honest scope and limitations

- The **analyst-consensus overlay is a live-only adjustment**, capped at 0.35 and
  disabled when fewer than three analysts cover a name. It is *not* applied in the
  backtest, because no point-in-time consensus history exists — so the scorecard
  describes the model-only forecast.
- **Fundamental and consensus features are a single "as-of" snapshot** broadcast flat
  across history. They carry no historical variation, so they are **dropped before
  model fitting** (a per-stock tree model cannot split on a constant) and used only
  for the narrative and the live overlay. The model itself is driven by the
  time-varying technical, macro, alpha, regime, and sentiment features.
- **News sentiment is thin in the bundled snapshot** (a single day's articles), so it
  contributes little for most names. Ingest more history to give it weight.
- This is a research/education tool. It is **not investment advice**.

---

## Tracked universe

25 NSE large-caps across energy, banking, IT, autos, metals, pharma, FMCG, and
telecom/infra — see `config/settings.py` (`PILOT_UNIVERSE`). On-demand search can
also fetch any NSE/BSE symbol from the bundled directory (~2,600 names).

---

## Architecture

```
PredicSON/
├── config/settings.py              # universe, macro tickers, RSS feeds, paths, schema version
├── data/
│   ├── raw/                        # ohlcv, macro, news, experts, fundamentals
│   ├── features/                   # per-stock feature matrices (+ .meta.json version stamps)
│   ├── predictions/                # screener & backtest scorecard caches
│   └── model_cache/                # persisted trained models (git-ignored)
├── src/
│   ├── data_ingestion/             # market_data, news_collector, expert_scraper,
│   │                               # fundamentals_scraper, stock_directory
│   ├── feature_engineering/        # technical, macro, alpha, sentiment, consensus,
│   │                               # fundamental, build_dataset
│   ├── models/                     # quantile_forecaster, backtester, regime_classifier,
│   │                               # narrative_engine, screener_engine, portfolio_allocator,
│   │                               # fan_chart_generator, on_demand_engine
│   └── utils/data_loader.py        # typed loaders + feature-version checks
├── dashboard/app.py                # Streamlit terminal (search homepage + forecast canvas)
├── tests/test_predicson.py         # unit tests
├── main.py                         # CLI entrypoint
├── requirements.txt / pyproject.toml
└── LICENSE (MIT)
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Ingest data (equities, macro, news, fundamentals, expert consensus)
python main.py ingest

# 3. Build the feature matrices (writes data/features/*.parquet + .meta.json)
python main.py features

# 4. Forecast a single stock
python main.py predict --ticker RELIANCE.NS

# 5. Purged walk-forward backtest (one ticker, or the whole universe)
python main.py backtest --ticker RELIANCE.NS
python main.py backtest-all

# 6. Build the screener, or launch the dashboard
python main.py screener
python main.py dashboard      # or: streamlit run dashboard/app.py
```

Trained models are persisted under `data/model_cache/` and reused until the
underlying data changes, so repeat forecasts are fast; the first forecast per stock
fits the models and takes noticeably longer.

---

## Methodology notes

- **No look-ahead.** Labels use `close.shift(-h)`; the backtest purges the last `h`
  training rows before each evaluation anchor so no training label is derived from a
  post-anchor price.
- **Adjusted prices.** Prices are split/dividend-adjusted (`auto_adjust=True`, and the
  loader promotes a legacy `adj_close` column when present) before any return maths.
- **Feature versioning.** Each feature matrix records the schema version it was built
  with; a mismatch forces a rebuild, so stale artifacts can't silently feed the model.
- **Reproducibility.** Fixed random seed (`RANDOM_SEED`) throughout.

---

## Testing

```bash
pip install pytest
pytest -q
```

CI runs the suite on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

---

## License

MIT — see `LICENSE`.
