import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from openai import OpenAI
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Radar Pro",
    layout="wide",
    page_icon="🔥"
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")  # Finnhub
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# SAFE DATAFRAME HELPER
# =========================
def safe_dataframe(df):
    if df is None or df.empty:
        return st.write("No data available.")
    try:
        return st.dataframe(df.style.format({
            "Change %": "{:+.2f}%",
            "RelVol": "{:.2f}x"
        }), use_container_width=True)
    except:
        return st.dataframe(df, use_container_width=True)

# =========================
# HELPERS
# =========================
@st.cache_data(ttl=300)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False, prepost=True)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_24h(ticker):
    data = yf.download(ticker, period="2d", interval="5m", prepost=True, progress=False)
    if data.empty:
        return None
    last_price = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    pct_change = (last_price - prev_close) / prev_close * 100
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return pct_change, rel_vol, last_price

# =========================
# NEWS SOURCES
# =========================
def get_finnhub_news(ticker: str):
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={today}&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if isinstance(r, list) and r:
            return r[0].get("headline") or r[0].get("title")
        return "No major Finnhub news"
    except:
        return "Finnhub error"

def get_polygon_news(ticker: str):
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if "results" in r and r["results"]:
            latest = r["results"][0]
            return f"{latest.get('publisher', {}).get('name','News')}: {latest.get('title')}"
        return "No major Polygon/Benzinga news"
    except:
        return "Polygon news error"

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, price, change, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "⚠️ Add OPENAI_API_KEY in Secrets."
    prompt = f"""
    Ticker: {ticker}
    Price: {price:.2f}
    24h Change: {change:.2f}%
    RelVol: {relvol:.2f}x
    Catalyst: {catalyst}

    Generate a trading playbook:
    1) Bias (Bullish/Bearish/Neutral).
    2) Entry & exit strategy for scalp (1-5m), day trade (15-30m), swing (4H-Daily).
    3) Risks to watch (IV crush, macro events, pullback).
    Keep concise and actionable.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"

# =========================
# SCANNERS
# =========================
def scan_list(tickers, use_polygon=True, use_finnhub=True):
    rows = []
    for t in tickers:
        scan = scan_24h(t)
        if not scan:
            continue
        change, relvol, price = scan
        catalyst = ""
        if use_polygon:
            catalyst = get_polygon_news(t)
        elif use_finnhub:
            catalyst = get_finnhub_news(t)
        playbook = ai_playbook(t, price, change, relvol, catalyst)
        rows.append([t, f"${price:.2f}", round(change,2), round(relvol,2), catalyst, playbook])
    return pd.DataFrame(rows, columns=["Ticker","Price","Change %","RelVol","Catalyst","AI Playbook"])

# =========================
# STREAMLIT UI
# =========================
st.title("🔥 AI Radar Pro — Live Market Scanner")

# Sidebar settings
st.sidebar.header("⚙️ News Sources")
use_polygon = st.sidebar.checkbox("Use Polygon (Benzinga)", value=True)
use_finnhub = st.sidebar.checkbox("Use Finnhub", value=True)

# Search ticker
search_ticker = st.text_input("🔍 Search a ticker (e.g. TSLA, NVDA, SPY)")
if search_ticker:
    df = scan_list([search_ticker.upper()], use_polygon, use_finnhub)
    safe_dataframe(df)

# Tabs
tabs = st.tabs(["📊 Premarket","💥 Intraday","🌙 Postmarket","📰 Catalysts"])
watchlist = ["AAPL","NVDA","TSLA","MSFT","AMZN","META","AMD","GOOG"]

with tabs[0]:
    st.subheader("Premarket Movers")
    df = scan_list(watchlist, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[1]:
    st.subheader("Intraday Movers")
    df = scan_list(watchlist, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[2]:
    st.subheader("Postmarket Movers")
    df = scan_list(watchlist, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[3]:
    st.subheader("📰 Catalyst Feed")
    df = pd.DataFrame([[t, get_polygon_news(t), get_finnhub_news(t)] for t in watchlist],
                      columns=["Ticker","Polygon/Benzinga","Finnhub"])
    safe_dataframe(df)
