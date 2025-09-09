import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from openai import OpenAI
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar 24h Scanner", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_24h(ticker):
    """Scan last 24h price action (premarket + intraday + postmarket)."""
    data = yf.download(ticker, period="2d", interval="5m", prepost=True, progress=False)
    if data.empty:
        return None
    
    last_price = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    pct_change = (last_price - prev_close) / prev_close * 100
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return pct_change, rel_vol

@st.cache_data(show_spinner=False, ttl=300)
def get_news_headline(ticker: str):
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={today}&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, list) and js:
            for item in sorted(js, key=lambda x: x.get("datetime", 0), reverse=True):
                headline = item.get("headline") or item.get("title")
                if headline:
                    return headline
        return "No major news"
    except Exception:
        return "News fetch error"

def ai_playbook(ticker, gap, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY in Secrets."

    # Safe fallback values
    try:
        gap = float(gap) if gap is not None else 0.0
    except Exception:
        gap = 0.0

    try:
        relvol = float(relvol) if relvol is not None else 0.0
    except Exception:
        relvol = 0.0

    prompt = f"""
    Ticker: {ticker}
    24h Change: {gap:.2f}%
    RelVol: {relvol:.2f}x
    Catalyst: {catalyst}

    Generate a 3-sentence trading playbook:
    1) Bias (long/short).
    2) Expected duration (scalp vs swing).
    3) Risks (fade, IV crush, market pullback).
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.strip()}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"

# =========================
# EARNINGS CALENDAR
# =========================
@st.cache_data(show_spinner=True, ttl=300)
def get_earnings_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={today}&token={NEWS_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json().get("earningsCalendar", [])
        return [item["symbol"] for item in js if "symbol" in item]
    except Exception:
        return []

# =========================
# TOP MOVERS FEED
# =========================
@st.cache_data(show_spinner=True, ttl=300)
def get_top_movers(limit=15):
    movers = []
    try:
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        symbols = r.json()[:200]  # sample subset
        tickers = [s["symbol"] for s in symbols]

        for t in tickers:
            try:
                q = requests.get(f"https://finnhub.io/api/v1/quote?symbol={t}&token={NEWS_API_KEY}").json()
                pct = ((q["c"] - q["pc"]) / q["pc"]) * 100 if q.get("pc") else 0
                vol = q.get("v", 0)
                if abs(pct) >= 5 and vol > 500000:  # filter movers
                    movers.append((t, pct, vol))
            except Exception:
                continue
    except Exception:
        return ["AAPL","NVDA","TSLA","AMD","SPY"]

    movers = sorted(movers, key=lambda x: abs(x[1]), reverse=True)[:limit]
    return [m[0] for m in movers]

def scan_list(tickers, tag="24h"):
    rows = []
    for t in tickers:
        scan = scan_24h(t)
        if not scan:
            continue
        gap, relvol = scan
        catalyst = get_news_headline(t)
        playbook = ai_playbook(t, gap, relvol, catalyst)
        rows.append([t, round(gap, 2), round(relvol, 2), catalyst, playbook])
    return pd.DataFrame(rows, columns=["Ticker", "Change %", "RelVol", "Catalyst", "AI Playbook"])

# =========================
# STREAMLIT UI
# =========================
st.title("🔥 AI Radar 24h Scanner")
st.caption("Scans Premarket, Intraday, Postmarket, and Earnings Movers (24/7)")

tabs = st.tabs(["📊 Premarket", "💥 Intraday", "🌙 Postmarket"])

# Always grab top movers + earnings
tickers = list(set(get_top_movers(limit=15) + get_earnings_today()))

with tabs[0]:
    st.subheader("Premarket Movers (4am–9:30am)")
    df = scan_list(tickers, tag="premarket")
    st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.subheader("Intraday Explosives (9:30am–4pm)")
    df = scan_list(tickers, tag="intraday")
    st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.subheader("Postmarket Movers + Earnings")
    df = scan_list(tickers, tag="postmarket")
    st.dataframe(df, use_container_width=True)

