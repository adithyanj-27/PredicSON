# PredicSON — Institutional Consensus & Expert Intelligence Terminal (NSE / BSE)

> **Wisdom of Institutional Crowds for Indian Equities.**
> Aggregates sell-side research, institutional price targets, and analyst ratings from 800+ covering brokerage desks
> across Dalal Street and premier global investment banks. Zero brittle retail models — pure institutional consensus.

---

## What it actually does

- **Institutional Consensus & Price Targets**: For any equity across NSE/BSE, retrieves
  Mean, Median, High (Bull Case), and Low (Bear Case) price targets published by sell-side research desks.
- **Institutional Corridor Visualizer**: TradingView-grade interactive chart charting 1Y historical
  price extending forward into the 12-month sell-side target corridor.
- **Rating Breakdown & 3-Month Momentum**: Full distribution of Strong Buy, Buy, Hold, and
  Sell recommendations, along with quarterly stance revision trends.
- **Brokerage Actions & Research Estimates**: Firm-by-firm breakdown of price targets and recommendations
  from global and domestic institutions (Jefferies, Morgan Stanley, Goldman Sachs, Motilal Oswal, ICICI Direct, Kotak, CLSA, Nomura).
- **Dalal Street Wire**: Real-time aggregation of brokerage reports, Q2 preview commentary,
  and target upgrades from Moneycontrol, Economic Times, and Livemint.
- **Universe Screener**: Ranks 25 large caps and ~2,600 Indian equities by highest institutional upside potential.

### Honest scope and limitations

- Consensus data reflects sell-side analyst price targets and recommendations published via formal research reports.
- Small-cap or micro-cap equities with limited institutional coverage (< 3 covering desks) display transparent coverage status.
- This is a research and market intelligence tool. It is **not investment advice**.

---

## Tracked universe

25 NSE large-caps across energy, banking, IT, autos, metals, pharma, FMCG, and
telecom/infra — see `config/settings.py` (`PILOT_UNIVERSE`). On-demand search can
also fetch any NSE/BSE symbol from the bundled directory (~2,600 names).

---

## Architecture

```
PredicSON/
├── config/settings.py              # universe, RSS feeds, cache paths
├── data/
│   ├── raw/news/                   # live financial news wire snapshots
│   ├── predictions/institutional/  # same-day sell-side consensus caches
│   └── indian_stocks_master.json   # 10,680+ Indian equities master directory
├── src/
│   ├── data_ingestion/
│   │   ├── institutional_engine.py # ⭐ Sell-side targets, broker calls, corridor chart
│   │   ├── stock_directory.py      # ⭐ Multi-token search & company directory
│   │   └── news_collector.py       # Live RSS feed ingestion from financial press
│   └── utils/data_loader.py        # Ticker sanitization & news loader
├── dashboard/app.py                # Streamlit institutional terminal UI
├── tests/test_predicson.py         # Unit tests (entity tagging, search, consensus)
├── main.py                         # CLI entrypoint
├── requirements.txt / pyproject.toml
└── LICENSE (MIT)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check institutional consensus for a specific stock
python main.py consensus --ticker RELIANCE.NS

# 3. View institutional universe rankings (highest upside)
python main.py consensus-all

# 4. Launch the Streamlit Institutional Terminal
python main.py dashboard      # or: streamlit run dashboard/app.py
```

---

## Testing

```bash
pytest -q
```

---

## License

MIT — see `LICENSE`.
