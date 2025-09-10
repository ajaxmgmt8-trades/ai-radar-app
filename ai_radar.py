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
    """Render styled DataFrame with fallback if matplotlib is missing."""
    try:
        return st.dataframe(
            df.style.format({
                "Change %": "{:+.2f}%",
                "RelVol": "{:.2f}x"
            }).background_gradient(subset=["Change %"], cmap="RdYlGn"),
            use_container_width=True
        )
    except ImportError:
        return st.dataframe(
            df.style.format({
                "Change %": "{:+.2f}%",
                "RelVol": "{:.2f}x"
            }),
            use_container_width=True
        )

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
    data = yf.download(ticker, period="2d", interval="5m", prepost=True, progress=False)
    if data.empty:
        return None
    last_price = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    pct_change = (last_price - prev_close) / prev_close * 100
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return pct_change, rel_vol

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
            headline = r[0].get("headline") or r[0].get("title")
            return headline
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
    except Exception as e:
        return f"Polygon news error: {e}"

def get_stocktwits_messages(ticker, limit=3):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = requests.get(url, timeout=10).json()
        msgs = []
        for m in r.get("messages", [])[:limit]:
            user = m.get("user", {}).get("username", "user")
            body = m.get("body", "")
            msgs.append(f"@{user}: {body}")
        return msgs if msgs else ["No chatter"]
    except:
        return ["StockTwits error"]

def get_stocktwits_trending(limit=5):
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        r = requests.get(url, timeout=10).json()
        syms = [s["symbol"] for s in r.get("symbols", [])[:limit]]
        return syms
    except:
        return []

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, change, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY in Secrets."

    # ✅ Safe numeric conversion
    try:
        change = float(change) if change is not None else 0.0
    except:
        change = 0.0
    try:
        relvol = float(relvol) if relvol is not None else 0.0
    except:
        relvol = 0.0

    prompt = f"""
    Ticker: {ticker}
    24h Change: {change:.2f}%
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
# SCANNERS
# =========================
def get_top_movers(limit=10):
    return ["AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG"]

def scan_list(tickers, use_polygon, use_finnhub):
    rows = []
    for t in tickers:
        scan = scan_24h(t)
        if not scan:
            continue
        change, relvol = scan
        catalyst = get_polygon_news(t) if use_polygon else get_finnhub_news(t)
        try:
            playbook = ai_playbook(t, change, relvol, catalyst)
        except:
            playbook = "AI Playbook error"
        rows.append([t, round(change,2), round(relvol,2), catalyst, playbook])
    return pd.DataFrame(rows, columns=["Ticker","Change %","RelVol","Catalyst","AI Playbook"])

def scan_catalysts(tickers, use_polygon=True, use_finnhub=True, use_stocktwits=True):
    rows = []
    for t in tickers:
        polygon_headline = get_polygon_news(t) if use_polygon else ""
        finnhub_headline = get_finnhub_news(t) if use_finnhub else ""
        stocktwits_headline = get_stocktwits_messages(t,1)[0] if use_stocktwits else ""
        rows.append([t, polygon_headline, finnhub_headline, stocktwits_headline])
    return pd.DataFrame(rows, columns=["Ticker","Polygon/Benzinga","Finnhub","StockTwits"])

# =========================
# STREAMLIT UI
# =========================
st.markdown(
    "<h1 style='text-align: center; color: orange;'>🔥 AI Radar Pro — Market Scanner</h1>",
    unsafe_allow_html=True
)
st.caption("Premarket, Intraday, Postmarket, StockTwits, and AI Playbooks")

# Sidebar
st.sidebar.header("⚙️ News Settings")
use_polygon = st.sidebar.checkbox("Use Polygon (Benzinga)", value=True)
use_finnhub = st.sidebar.checkbox("Use Finnhub", value=True)
use_stocktwits = st.sidebar.checkbox("Use StockTwits chatter", value=True)

# Search box
search_ticker = st.text_input("🔍 Search a ticker (e.g. TSLA, NVDA, SPY)")
if search_ticker:
    st.subheader(f"Search Results for {search_ticker.upper()}")
    df = scan_list([search_ticker.upper()], use_polygon, use_finnhub)
    safe_dataframe(df)

# Tabs
tabs = st.tabs(["📊 Premarket","💥 Intraday","🌙 Postmarket","💬 StockTwits Feed","📰 Catalysts"])
tickers = get_top_movers(limit=8)

with tabs[0]:
    st.subheader("Premarket Movers")
    df = scan_list(tickers, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[1]:
    st.subheader("Intraday Movers")
    df = scan_list(tickers, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[2]:
    st.subheader("Postmarket Movers")
    df = scan_list(tickers, use_polygon, use_finnhub)
    safe_dataframe(df)

with tabs[3]:
    st.subheader("💬 StockTwits Feed (Watchlist + Trending)")
    for t in tickers:
        st.markdown(f"### {t}")
        msgs = get_stocktwits_messages(t, limit=3)
        for m in msgs:
            st.write(f"👉 {m}")

    st.markdown("---")
    st.markdown("### 📈 Market Trending")
    trending = get_stocktwits_trending(limit=5)
    st.write(", ".join(trending))

with tabs[4]:
    st.subheader("📰 Catalyst Feed (Polygon + Finnhub + StockTwits)")
    refresh_rate = st.slider("Refresh Catalysts every X minutes", 1, 10, 3)
    st_autorefresh = st.experimental_autorefresh(interval=refresh_rate * 60 * 1000, key="catalyst_refresh")
    df = scan_catalysts(tickers, use_polygon=True, use_finnhub=use_finnhub, use_stocktwits=use_stocktwits)
    st.dataframe(df, use_container_width=True)
