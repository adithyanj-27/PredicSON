"""
Institutional & Expert Consensus Engine (Zero-Cost Sell-Side Intelligence)
Aggregates authentic financial analyst targets, institutional recommendations,
company fundamentals, and Dalal Street research calls:
1. Real individual institutional predictions (Nomura, Goldman Sachs, Motilal Oswal,
   ICICI Securities, Jefferies, Sharekhan, Anand Rathi, Emkay, Religare, etc.)
   extracted in real-time with exact targets, ratings, dates, and direct report links.
2. Sell-side price targets (Mean, Median, High Bull, Low Bear) from covering institutions via yfinance.
3. Full recommendation distribution & 3-month stance revisions (Strong Buy, Buy, Hold, Sell, Strong Sell).
4. Comprehensive fundamental & valuation metrics (Market Cap, P/E, P/B, Dividend Yield, 52-Week Range).
5. Real expert news wire & brokerage reports parsed from financial media (Moneycontrol, ET, Livemint).
"""

import sys
import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PILOT_UNIVERSE, PREDICTIONS_DIR, NEWS_DIR
from src.utils.data_loader import sanitize_ticker, load_daily_news
from src.data_ingestion.news_collector import run_news_ingestion

INSTITUTIONAL_CACHE_DIR = PREDICTIONS_DIR / "institutional"
INSTITUTIONAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 40+ Recognized Institutional Research Desks & Brokerages covering Indian Equities
INSTITUTION_METADATA = {
    'Goldman Sachs': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'Morgan Stanley': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'Jefferies': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'Nomura': {'type': 'Global Investment Bank', 'hq': 'Tokyo / Mumbai'},
    'Citi': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'Citigroup': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'CLSA': {'type': 'Asian Institutional Brokerage', 'hq': 'Hong Kong / Mumbai'},
    'Bernstein': {'type': 'Global Research Desk', 'hq': 'New York / Mumbai'},
    'Macquarie': {'type': 'Global Investment Bank', 'hq': 'Sydney / Mumbai'},
    'HSBC': {'type': 'Global Investment Bank', 'hq': 'London / Mumbai'},
    'Motilal Oswal': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'ICICI Securities': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'ICICI Direct': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Kotak Institutional Equities': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Kotak Equities': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Kotak Securities': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Kotak': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Prabhudas Lilladher': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Sharekhan': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Nuvama': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Emkay': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Anand Rathi': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Axis Direct': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Axis Capital': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'HDFC Securities': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Religare': {'type': 'Domestic Institutional Desk', 'hq': 'New Delhi'},
    'Geojit': {'type': 'Domestic Research House', 'hq': 'Kochi'},
    'JM Financial': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Antique': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'InCred': {'type': 'Domestic Equities Desk', 'hq': 'Mumbai'},
    'Yes Securities': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'Centrum': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'SBI Securities': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'IDBI Capital': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Choice Broking': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'KRChoksey': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Equirus': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Elara Capital': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'UBS': {'type': 'Global Investment Bank', 'hq': 'Zurich / Mumbai'},
    'JPMorgan': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'JP Morgan': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'BofA': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'Bank of America': {'type': 'Global Investment Bank', 'hq': 'New York / Mumbai'},
    'BNP Paribas': {'type': 'Global Investment Bank', 'hq': 'Paris / Mumbai'},
    'Nirmal Bang': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'Ventura': {'type': 'Domestic Research House', 'hq': 'Mumbai'},
    'SMIFS': {'type': 'Domestic Institutional Desk', 'hq': 'Kolkata'},
    'Systematix': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'PhillipCapital': {'type': 'Global Brokerage Desk', 'hq': 'Singapore / Mumbai'},
    'IIFL Securities': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
    'IIFL': {'type': 'Domestic Institutional Desk', 'hq': 'Mumbai'},
}


def format_inr_crore(val: Optional[float]) -> str:
    """Format market cap and large currency figures in Indian Lakh Cr / Cr notation."""
    if val is None or pd.isna(val) or val <= 0:
        return "N/A"
    if val >= 1e12:
        return f"₹{val / 1e12:.2f} Lakh Cr"
    if val >= 1e7:
        return f"₹{val / 1e7:,.0f} Cr"
    return f"₹{val:,.2f}"


def fetch_individual_institution_predictions(company_name: str, ticker: str, current_price: float = 0.0) -> List[Dict[str, Any]]:
    """
    Search and extract authentic, real-time research predictions, target prices,
    and ratings published by specific institutions (e.g. Goldman Sachs, Nomura,
    Motilal Oswal, ICICI Securities, Jefferies, Sharekhan, Emkay) from live financial news wire.
    """
    clean_sym = ticker.replace(".NS", "").replace(".BO", "")
    clean_name = re.sub(r'\b(ltd|limited|corp|corporation|industries|india)\b', '', company_name, flags=re.IGNORECASE).strip()
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    if not clean_name:
        clean_name = company_name
    query_parts = [clean_name]
    if clean_sym.lower() not in clean_name.lower():
        query_parts.append(clean_sym)
    
    query_str = f"({' OR '.join(query_parts)}) (target OR buy OR sell OR brokerage OR recommends OR initiate OR upgrade)"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_str)}&hl=en-IN&gl=IN&ceid=IN:en"
    
    predictions = []
    seen_brokers = set()
    
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:35]:
            title = entry.get("title", "")
            title_lower = title.lower()
            
            # Match covering institution
            matched_inst = None
            for inst_name in INSTITUTION_METADATA.keys():
                pattern = r"\b" + re.escape(inst_name.lower()) + r"\b"
                if re.search(pattern, title_lower):
                    matched_inst = inst_name
                    break
                    
            if not matched_inst:
                continue
                
            # Classify rating / stance
            if re.search(r"\b(sell|reduce|underweight|underperform|downgrade|cuts target)\b", title_lower):
                stance = "Sell / Reduce"
                stance_color = "#F43F5E"
            elif re.search(r"\b(accumulate|add|overweight|outperform|raises target|bullish)\b", title_lower):
                stance = "Accumulate"
                stance_color = "#0EA5E9"
            elif re.search(r"\b(buy|strong buy|initiates buy|maintains buy|buy call|recommends buy|top buy)\b", title_lower):
                stance = "Buy"
                stance_color = "#10B981"
            elif re.search(r"\b(hold|neutral|market perform)\b", title_lower):
                stance = "Hold"
                stance_color = "#F59E0B"
            else:
                stance = "Research Call"
                stance_color = "#38BDF8"
                
            # Extract absolute target price or percentage upside
            tp = None
            m_price = re.search(r'(?:target|tp|target price|target of|target at|target to)\s*(?:of|at|is|with|to)?\s*(?:rs\.?|inr|₹)?\s*([0-9,]+(?:\.[0-9]+)?)', title, re.IGNORECASE)
            if not m_price:
                m_price = re.search(r'(?:rs\.?|inr|₹)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:target|tp)', title, re.IGNORECASE)
            m_pct = re.search(r'(?:gain|rally|upside|jump|surge)\s*(?:of|up to)?\s*([0-9]+(?:\.[0-9]+)?)\s*%', title, re.IGNORECASE)
            if not m_pct:
                m_pct = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:upside|gain|rally)', title, re.IGNORECASE)
            
            if m_price:
                try:
                    val = float(m_price.group(1).replace(',', ''))
                    if 5 < val < 500000:
                        # Sanity check: target should be within 35% to 300% of CMP to prevent penny-stock headline artifacts
                        if current_price <= 0 or (current_price * 0.35 <= val <= current_price * 3.0):
                            tp = val
                except Exception:
                    pass
            elif m_pct and current_price > 0:
                try:
                    pct = float(m_pct.group(1))
                    if 0 < pct < 150:
                        tp = round(current_price * (1 + pct / 100), 1)
                except Exception:
                    pass
                    
            upside = round(((tp - current_price) / current_price * 100), 1) if (tp and current_price > 0) else None
            
            # Format report date
            raw_date = entry.get("published", "")
            pub_date = raw_date[:16] if raw_date else "Recent Report"
            
            inst_info = INSTITUTION_METADATA.get(matched_inst, {'type': 'Institutional Desk', 'hq': 'India'})
            
            new_item = {
                "institution": matched_inst,
                "institution_type": inst_info["type"],
                "hq": inst_info["hq"],
                "stance": stance,
                "stance_color": stance_color,
                "target_price": tp,
                "implied_upside": upside,
                "headline": title,
                "published_date": pub_date,
                "source": entry.get("source", {}).get("title", "Financial Press"),
                "link": entry.get("link", "#")
            }

            # Deduplicate by institution: keep the one with target price or most recent
            existing_idx = next((i for i, x in enumerate(predictions) if x["institution"] == matched_inst), None)
            if existing_idx is not None:
                if tp and not predictions[existing_idx].get("target_price"):
                    predictions[existing_idx] = new_item
                continue

            predictions.append(new_item)
    except Exception as e:
        print(f"  Note: error extracting individual institution predictions for {company_name}: {e}")
        
    return predictions


def fetch_institutional_consensus(ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch comprehensive institutional consensus, price targets, recommendations,
    fundamental valuation ratios, 1Y historical prices, and REAL individual institution
    predictions for any NSE/BSE equity.
    """
    stem = sanitize_ticker(ticker)
    cache_file = INSTITUTIONAL_CACHE_DIR / f"{stem}.json"
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Check same-day cache unless forced
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("as_of_date") == today_str and "price_targets" in data and "individual_institution_predictions" in data:
                return data
        except Exception:
            pass

    print(f"[Institutional Engine] Fetching live institutional data for {ticker}...", flush=True)

    t = yf.Ticker(ticker)

    # 2. Historical Prices (1 Year daily closes)
    hist_records = []
    current_price = 0.0
    try:
        hist_df = t.history(period="1y", interval="1d", auto_adjust=True)
        if hist_df is not None and not hist_df.empty:
            if isinstance(hist_df.columns, pd.MultiIndex):
                hist_df.columns = [str(c[0]).lower() for c in hist_df.columns]
            else:
                hist_df.columns = [str(c).lower() for c in hist_df.columns]
            hist_df = hist_df[["close"]].dropna()
            if not hist_df.empty:
                current_price = round(float(hist_df["close"].iloc[-1]), 2)
                hist_records = [
                    {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
                    for d, c in zip(hist_df.index, hist_df["close"])
                ]
    except Exception as e:
        print(f"  Note: history fetch issue for {ticker}: {e}")

    # Fallback to fast_info if price not captured
    if current_price <= 0:
        try:
            current_price = round(float(getattr(t.fast_info, "last_price", 0)), 2)
        except Exception:
            pass

    # 3. Fast Info & Full Info for Fundamentals
    fi = getattr(t, "fast_info", None)
    raw_info = {}
    try:
        raw_info = t.info
    except Exception:
        raw_info = {}

    mcap = getattr(fi, "market_cap", None) or (raw_info.get("marketCap") if isinstance(raw_info, dict) else None)
    year_high = getattr(fi, "year_high", None) or (raw_info.get("fiftyTwoWeekHigh") if isinstance(raw_info, dict) else None)
    year_low = getattr(fi, "year_low", None) or (raw_info.get("fiftyTwoWeekLow") if isinstance(raw_info, dict) else None)

    pe_val = raw_info.get("trailingPE") if isinstance(raw_info, dict) else None
    fwd_pe_val = raw_info.get("forwardPE") if isinstance(raw_info, dict) else None
    pb_val = raw_info.get("priceToBook") if isinstance(raw_info, dict) else None
    div_yield_val = raw_info.get("dividendYield") if isinstance(raw_info, dict) else None
    sector_val = raw_info.get("sector") if isinstance(raw_info, dict) else None
    industry_val = raw_info.get("industry") if isinstance(raw_info, dict) else None
    summary_val = raw_info.get("longBusinessSummary", "") if isinstance(raw_info, dict) else ""
    company_name_val = raw_info.get("longName") or raw_info.get("shortName") or ticker.replace(".NS", "").replace(".BO", "")

    # Calculate 52-week range percentage position
    range_pct = 50.0
    if year_high and year_low and year_high > year_low and current_price > 0:
        range_pct = max(0.0, min(100.0, round(((current_price - year_low) / (year_high - year_low)) * 100, 1)))

    fundamentals = {
        "company_name": company_name_val,
        "market_cap": float(mcap) if mcap else 0.0,
        "market_cap_str": format_inr_crore(mcap),
        "pe_ratio": round(float(pe_val), 1) if pe_val and pd.notna(pe_val) else None,
        "forward_pe": round(float(fwd_pe_val), 1) if fwd_pe_val and pd.notna(fwd_pe_val) else None,
        "pb_ratio": round(float(pb_val), 2) if pb_val and pd.notna(pb_val) else None,
        "dividend_yield": round(float(div_yield_val) * 100, 2) if div_yield_val and div_yield_val < 1.0 else (round(float(div_yield_val), 2) if div_yield_val else None),
        "fifty_two_week_high": round(float(year_high), 2) if year_high else None,
        "fifty_two_week_low": round(float(year_low), 2) if year_low else None,
        "fifty_two_week_pct": range_pct,
        "sector": sector_val or "Equities",
        "industry": industry_val or "General",
        "business_summary": (summary_val[:280] + "...") if summary_val else ""
    }

    # 4. Analyst Price Targets (Consensus)
    price_targets = {}
    has_analyst_coverage = False
    try:
        pt = t.analyst_price_targets
        if pt and isinstance(pt, dict) and pt.get("mean") is not None and float(pt.get("mean")) > 0:
            has_analyst_coverage = True
            mean_p = float(pt.get("mean"))
            med_p = float(pt.get("median") or mean_p)
            high_p = float(pt.get("high") or mean_p)
            low_p = float(pt.get("low") or mean_p)
            curr_p = float(pt.get("current") or current_price)
            if current_price <= 0 and curr_p > 0:
                current_price = curr_p

            price_targets = {
                "current": round(curr_p, 2),
                "mean": round(mean_p, 2),
                "median": round(med_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
            }
    except Exception as e:
        print(f"  Note: price target extraction error for {ticker}: {e}")

    # 5. Recommendation Summary & 3-Month Momentum
    reco_trend = []
    latest_recos = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    total_ratings = 0

    try:
        recs_df = t.recommendations_summary
        if recs_df is not None and not recs_df.empty:
            for _, row in recs_df.iterrows():
                period = str(row.get("period", ""))
                sb = int(row.get("strongBuy", 0) or 0)
                b = int(row.get("buy", 0) or 0)
                h = int(row.get("hold", 0) or 0)
                s = int(row.get("sell", 0) or 0)
                ss = int(row.get("strongSell", 0) or 0)
                tot = sb + b + h + s + ss
                reco_trend.append({
                    "period": period,
                    "strongBuy": sb,
                    "buy": b,
                    "hold": h,
                    "sell": s,
                    "strongSell": ss,
                    "total": tot
                })
            if reco_trend:
                latest_recos = {
                    "strongBuy": reco_trend[0]["strongBuy"],
                    "buy": reco_trend[0]["buy"],
                    "hold": reco_trend[0]["hold"],
                    "sell": reco_trend[0]["sell"],
                    "strongSell": reco_trend[0]["strongSell"],
                }
                total_ratings = reco_trend[0]["total"]
                if total_ratings > 0:
                    has_analyst_coverage = True
    except Exception as e:
        print(f"  Note: recommendation summary error for {ticker}: {e}")

    # 6. Consensus Calculations
    if has_analyst_coverage and price_targets:
        mean_val = price_targets.get("mean", current_price)
        high_val = price_targets.get("high", current_price)
        low_val = price_targets.get("low", current_price)
        med_val = price_targets.get("median", current_price)

        mean_upside_pct = round(((mean_val - current_price) / current_price * 100), 2) if current_price > 0 else 0.0
        bull_upside_pct = round(((high_val - current_price) / current_price * 100), 2) if current_price > 0 else 0.0
        bear_downside_pct = round(((low_val - current_price) / current_price * 100), 2) if current_price > 0 else 0.0
        dispersion_pct = round(((high_val - low_val) / current_price * 100), 2) if current_price > 0 else 0.0

        buy_count = latest_recos.get("strongBuy", 0) + latest_recos.get("buy", 0)
        buy_ratio = round((buy_count / total_ratings * 100), 1) if total_ratings > 0 else 50.0

        if buy_ratio >= 75:
            consensus_stance = "STRONG BUY"
            stance_color = "#10B981"
        elif buy_ratio >= 50:
            consensus_stance = "ACCUMULATE / BUY"
            stance_color = "#0EA5E9"
        elif buy_ratio >= 35:
            consensus_stance = "HOLD / NEUTRAL"
            stance_color = "#F59E0B"
        else:
            consensus_stance = "REDUCE / SELL"
            stance_color = "#F43F5E"
    else:
        has_analyst_coverage = False
        price_targets = {"current": current_price, "mean": None, "median": None, "high": None, "low": None}
        mean_upside_pct = 0.0
        bull_upside_pct = 0.0
        bear_downside_pct = 0.0
        dispersion_pct = 0.0
        buy_ratio = 0.0
        consensus_stance = "UNCOVERED / NEUTRAL"
        stance_color = "#64748B"

    # 7. Real Individual Institution Predictions Extraction
    individual_preds = fetch_individual_institution_predictions(company_name_val, ticker, current_price)

    # 8. Real Dalal Street Media News Wire
    today_news_file = NEWS_DIR / f"news_{today_str}.jsonl"
    if force_refresh or not today_news_file.exists():
        try:
            run_news_ingestion()
        except Exception as e:
            print(f"  Note: news ingestion error: {e}")

    relevant_calls = []
    all_news = load_daily_news(today_str)
    ticker_clean = ticker.replace(".NS", "").replace(".BO", "")
    for art in all_news:
        matched = art.get("matched_tickers", [])
        title = art.get("title", "")
        summary = art.get("summary", "")
        text = (title + " " + summary).lower()
        if ticker in matched or ticker_clean.lower() in text or company_name_val.lower() in text:
            relevant_calls.append({
                "title": title,
                "source": art.get("source", "Financial News").replace("_", " ").title(),
                "published": art.get("published_raw", ""),
                "link": art.get("link", "#")
            })
        if len(relevant_calls) >= 6:
            break

    result = {
        "ticker": ticker,
        "as_of_date": today_str,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,
        "has_analyst_coverage": has_analyst_coverage,
        "total_analysts": total_ratings,
        "price_targets": price_targets,
        "mean_upside_pct": mean_upside_pct,
        "bull_upside_pct": bull_upside_pct,
        "bear_downside_pct": bear_downside_pct,
        "dispersion_pct": dispersion_pct,
        "recommendation_breakdown": latest_recos,
        "recommendation_trend": reco_trend,
        "buy_ratio_pct": buy_ratio,
        "consensus_stance": consensus_stance,
        "stance_color": stance_color,
        "fundamentals": fundamentals,
        "individual_institution_predictions": individual_preds,
        "expert_news": relevant_calls,
        "history": hist_records,
    }

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"  Note: error writing cache: {e}")

    return result


def build_institutional_corridor_figure(ticker: str, consensus: Dict[str, Any]) -> go.Figure:
    """
    Builds a TradingView-grade institutional corridor chart with generous right margin
    to prevent any text truncation, plotting historical price extending forward
    into the 12-month sell-side target corridor.
    """
    fig = go.Figure()

    hist_records = consensus.get("history", [])
    if not hist_records:
        return fig

    hist_df = pd.DataFrame(hist_records)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df.set_index("date", inplace=True)

    # 1. Historical Price Line
    fig.add_trace(go.Scatter(
        x=hist_df.index,
        y=hist_df["close"],
        mode="lines",
        name="Historical Price",
        line=dict(color="#38BDF8", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Market Price: ₹%{y:,.2f}<extra></extra>"
    ))

    last_date = hist_df.index[-1]
    curr_p = consensus["current_price"]
    pt = consensus.get("price_targets", {})
    has_coverage = consensus.get("has_analyst_coverage", False)

    if has_coverage and pt.get("mean"):
        mean_p = pt.get("mean", curr_p)
        high_p = pt.get("high", curr_p)
        low_p = pt.get("low", curr_p)
        med_p = pt.get("median", curr_p)

        # 12-Month forward projection dates
        fwd_12m_date = last_date + timedelta(days=365)
        fwd_dates = [last_date, fwd_12m_date]

        # Bull Target Line (Upper Corridor limit)
        fig.add_trace(go.Scatter(
            x=fwd_dates,
            y=[curr_p, high_p],
            mode="lines+text",
            name=f"Bull Target: ₹{high_p:,.0f}",
            line=dict(color="#10B981", width=1.5, dash="dash"),
            text=["", f" Bull ₹{high_p:,.0f}"],
            textposition="middle right",
            cliponaxis=False,
            textfont=dict(color="#10B981", size=11, family="Sora"),
            hovertemplate="<b>Bull Target (12M)</b>: ₹%{y:,.2f}<extra></extra>"
        ))

        # Bear Target Line (Lower Corridor limit)
        fig.add_trace(go.Scatter(
            x=fwd_dates,
            y=[curr_p, low_p],
            mode="lines+text",
            name=f"Bear Floor: ₹{low_p:,.0f}",
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.08)",
            line=dict(color="#F43F5E", width=1.5, dash="dash"),
            text=["", f" Floor ₹{low_p:,.0f}"],
            textposition="middle right",
            cliponaxis=False,
            textfont=dict(color="#F43F5E", size=11, family="Sora"),
            hovertemplate="<b>Bear Floor (12M)</b>: ₹%{y:,.2f}<extra></extra>"
        ))

        # Median Target Line
        fig.add_trace(go.Scatter(
            x=fwd_dates,
            y=[curr_p, med_p],
            mode="lines",
            name=f"Median: ₹{med_p:,.0f}",
            line=dict(color="#0EA5E9", width=2, dash="dot"),
            hovertemplate="<b>Median Target (12M)</b>: ₹%{y:,.2f}<extra></extra>"
        ))

        # Consensus Mean Target Line (Primary Focus)
        fig.add_trace(go.Scatter(
            x=fwd_dates,
            y=[curr_p, mean_p],
            mode="lines+markers+text",
            name=f"Consensus Target: ₹{mean_p:,.0f}",
            line=dict(color="#10B981", width=3),
            marker=dict(size=[0, 8], color="#10B981"),
            text=["", f" Consensus ₹{mean_p:,.0f}"],
            textposition="middle right",
            cliponaxis=False,
            textfont=dict(color="#F8FAFC", size=12, family="Sora"),
            hovertemplate="<b>Consensus Mean (12M)</b>: ₹%{y:,.2f}<extra></extra>"
        ))

        # Add individual institution dots if available
        ind_preds = consensus.get("individual_institution_predictions", [])
        for p in ind_preds[:5]:
            if p.get("target_price"):
                inst_tp = p["target_price"]
                inst_name = p["institution"]
                fig.add_trace(go.Scatter(
                    x=[fwd_12m_date],
                    y=[inst_tp],
                    mode="markers+text",
                    name=f"{inst_name}: ₹{inst_tp:,.0f}",
                    marker=dict(size=9, color="#38BDF8", symbol="diamond"),
                    text=[f"  {inst_name}"],
                    textposition="middle right",
                    cliponaxis=False,
                    textfont=dict(color="#38BDF8", size=10, family="Sora"),
                    hovertemplate=f"<b>{inst_name}</b> Target: ₹{inst_tp:,.2f}<extra></extra>",
                    showlegend=False
                ))
    else:
        # For uncovered stocks: show 52-Week High & Low trading boundaries
        fund = consensus.get("fundamentals", {})
        h52 = fund.get("fifty_two_week_high")
        l52 = fund.get("fifty_two_week_low")
        if h52 and l52:
            fig.add_hline(
                y=h52, line_dash="dash", line_color="#10B981",
                annotation_text=f" 52W High: ₹{h52:,.1f}", annotation_position="top right",
                annotation_font=dict(color="#10B981", size=11)
            )
            fig.add_hline(
                y=l52, line_dash="dash", line_color="#F43F5E",
                annotation_text=f" 52W Low: ₹{l52:,.1f}", annotation_position="bottom right",
                annotation_font=dict(color="#F43F5E", size=11)
            )

    # Today boundary marker
    fig.add_vline(
        x=last_date, line_width=1.5, line_dash="dash", line_color="#64748B",
        annotation_text="Today (CMP)", annotation_position="top left",
        annotation_font=dict(color="#94A3B8", size=10, family="Sora")
    )

    fig.update_layout(
        paper_bgcolor="#0A0D15",
        plot_bgcolor="#0A0D15",
        font=dict(family="Sora", color="#94A3B8"),
        hovermode="x unified",
        margin=dict(l=25, r=250, t=25, b=25),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#CBD5E1")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            zeroline=False,
            color="#64748B",
            range=[hist_df.index[0], fwd_12m_date + timedelta(days=70)] if (has_coverage and pt.get("mean")) else None
        ),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)", zeroline=False, color="#64748B", tickprefix="₹")
    )

    return fig


def get_universe_institutional_rankings() -> List[Dict[str, Any]]:
    """
    Return list of Indian large-caps ranked by Institutional Upside %.
    """
    screener_file = PREDICTIONS_DIR / "institutional_screener.json"
    today_str = datetime.now().strftime("%Y-%m-%d")

    if screener_file.exists():
        try:
            with open(screener_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("as_of_date") == today_str and "rankings" in cached:
                rankings = cached["rankings"]
                # Check if there are newly cached stocks from on-demand inspection
                existing_syms = {r["ticker"] for r in rankings}
                added_new = False
                for f in INSTITUTIONAL_CACHE_DIR.glob("*.json"):
                    try:
                        with open(f, "r", encoding="utf-8") as jf:
                            c_data = json.load(jf)
                        act_ticker = c_data.get("ticker")
                        if act_ticker and act_ticker not in existing_syms and c_data.get("current_price", 0) > 0:
                            existing_syms.add(act_ticker)
                            added_new = True
                            pt_data = c_data.get("price_targets", {})
                            fund_data = c_data.get("fundamentals", {})
                            rankings.append({
                                "ticker": act_ticker,
                                "name": fund_data.get("company_name", act_ticker.replace(".NS", "")),
                                "sector": fund_data.get("sector", "Equities"),
                                "current_price": c_data["current_price"],
                                "mean_target": pt_data.get("mean", c_data["current_price"]),
                                "high_target": pt_data.get("high", c_data["current_price"]),
                                "low_target": pt_data.get("low", c_data["current_price"]),
                                "mean_upside_pct": c_data.get("mean_upside_pct", 0.0),
                                "bull_upside_pct": c_data.get("bull_upside_pct", 0.0),
                                "bear_downside_pct": c_data.get("bear_downside_pct", 0.0),
                                "analysts": c_data.get("total_analysts", 0),
                                "buy_ratio": c_data.get("buy_ratio_pct", 0.0),
                                "stance": c_data.get("consensus_stance", "UNCOVERED"),
                                "color": c_data.get("stance_color", "#64748B"),
                                "market_cap_str": fund_data.get("market_cap_str", "N/A"),
                                "pe_ratio": fund_data.get("pe_ratio")
                            })
                    except Exception:
                        pass
                if added_new:
                    rankings.sort(key=lambda x: x["mean_upside_pct"], reverse=True)
                    try:
                        with open(screener_file, "w", encoding="utf-8") as f_out:
                            json.dump({"as_of_date": today_str, "rankings": rankings}, f_out, indent=2)
                    except Exception:
                        pass
                return rankings
        except Exception:
            pass

    print("[Institutional Screener] Rebuilding universe institutional rankings...", flush=True)
    rankings = []
    for ticker, meta in PILOT_UNIVERSE.items():
        try:
            c = fetch_institutional_consensus(ticker)
            if c and c.get("current_price", 0) > 0:
                rankings.append({
                    "ticker": ticker,
                    "name": meta["name"],
                    "sector": meta["sector"],
                    "current_price": c["current_price"],
                    "mean_target": c["price_targets"].get("mean", c["current_price"]),
                    "high_target": c["price_targets"].get("high", c["current_price"]),
                    "low_target": c["price_targets"].get("low", c["current_price"]),
                    "mean_upside_pct": c["mean_upside_pct"],
                    "bull_upside_pct": c["bull_upside_pct"],
                    "bear_downside_pct": c["bear_downside_pct"],
                    "analysts": c["total_analysts"],
                    "buy_ratio": c["buy_ratio_pct"],
                    "stance": c["consensus_stance"],
                    "color": c["stance_color"],
                    "market_cap_str": c["fundamentals"]["market_cap_str"],
                    "pe_ratio": c["fundamentals"]["pe_ratio"]
                })
        except Exception as e:
            print(f"  Error processing {ticker} for screener: {e}")

    # Also include any on-demand inspected equities in local cache
    cached_tickers = {r["ticker"] for r in rankings}
    for f in INSTITUTIONAL_CACHE_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as jf:
                c_data = json.load(jf)
            act_ticker = c_data.get("ticker")
            if act_ticker and act_ticker not in cached_tickers and c_data.get("current_price", 0) > 0:
                cached_tickers.add(act_ticker)
                pt_data = c_data.get("price_targets", {})
                fund_data = c_data.get("fundamentals", {})
                rankings.append({
                    "ticker": act_ticker,
                    "name": fund_data.get("company_name", act_ticker.replace(".NS", "")),
                    "sector": fund_data.get("sector", "Equities"),
                    "current_price": c_data["current_price"],
                    "mean_target": pt_data.get("mean", c_data["current_price"]),
                    "high_target": pt_data.get("high", c_data["current_price"]),
                    "low_target": pt_data.get("low", c_data["current_price"]),
                    "mean_upside_pct": c_data.get("mean_upside_pct", 0.0),
                    "bull_upside_pct": c_data.get("bull_upside_pct", 0.0),
                    "bear_downside_pct": c_data.get("bear_downside_pct", 0.0),
                    "analysts": c_data.get("total_analysts", 0),
                    "buy_ratio": c_data.get("buy_ratio_pct", 0.0),
                    "stance": c_data.get("consensus_stance", "UNCOVERED"),
                    "color": c_data.get("stance_color", "#64748B"),
                    "market_cap_str": fund_data.get("market_cap_str", "N/A"),
                    "pe_ratio": fund_data.get("pe_ratio")
                })
        except Exception:
            pass

    rankings.sort(key=lambda x: x["mean_upside_pct"], reverse=True)

    try:
        with open(screener_file, "w", encoding="utf-8") as f:
            json.dump({"as_of_date": today_str, "rankings": rankings}, f, indent=2)
    except Exception as e:
        print(f"  Error writing screener cache: {e}")

    return rankings
