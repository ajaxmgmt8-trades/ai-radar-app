import streamlit as st
import pandas as pd
import requests
import datetime
import yfinance as yf
import time
from typing import Dict, List
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
CORE_TICKERS = ["AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","GOOG","NFLX","CELH","PDD"]

# API Keys
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------------- HELPERS ----------------
@st.cache_data(ttl=15)
def get_live_quote(ticker: str) -> Dict:
    """Fetch live price + session breakdown via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d", interval="1m", prepost=True)
        if hist.empty: return {"error": "No data"}

        last = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[0]
        change_pct = (last - prev_close) / prev_close * 100 if prev_close else 0
        volume = hist["Volume"].iloc[-1]

        # crude rel vol (today vs 20-day avg daily vol)
        hist20 = stock.history(period="1mo", interval="1d")
        avg_vol = hist20["Volume"].mean() if not hist20.empty else 1
        relvol = volume / avg_vol if avg_vol else 1

        return {
            "last": float(last),
            "prev_close": float(prev_close),
            "change_pct": float(change_pct),
            "volume": int(volume),
            "relvol": float(relvol),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=600)
def get_news(symbol: str = None) -> List[Dict]:
    """Fetch last 24h news from Finnhub + Polygon."""
    news_items = []
    try:
        if FINNHUB_KEY:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()-datetime.timedelta(days=1)}&to={datetime.date.today()}&token={FINNHUB_KEY}" if symbol else f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
            r = requests.get(url, timeout=10).json()
            for n in r[:5]:
                news_items.append({
                    "title": n.get("headline"),
                    "summary": n.get("summary",""),
                    "source": "Finnhub",
                    "url": n.get("url","")
                })
    except: pass
    try:
        if POLYGON_KEY:
            url = f"https://api.polygon.io/v2/reference/news?limit=10&apiKey={POLYGON_KEY}"
            r = requests.get(url, timeout=10).json()
            for n in r.get("results", []):
                news_items.append({
                    "title": n.get("title"),
                    "summary": n.get("description",""),
                    "source": "Polygon",
                    "url": n.get("article_url","")
                })
    except: pass
    return news_items[:10]

def ai_playbook(ticker: str, change: float, catalyst: str = "") -> str:
    """AI playbook using OpenAI."""
    if not openai_client:
        return f"OpenAI API not configured. Change: {change:+.2f}%. Catalyst: {catalyst}"
    prompt = f"""
    Analyze {ticker}:
    Price change: {change:+.2f}%
    Catalyst: {catalyst if catalyst else "None"}
    Provide:
    1. Sentiment with confidence
    2. Scalp setup (1-5m)
    3. Day trade setup (15-30m)
    4. Swing setup (4H-D)
    Keep concise, bullet points under 200 words.
    """
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# ---------------- UI ----------------
st.title("🔥 AI Radar Pro — Live Trading Assistant")
tabs = st.tabs(["📊 Live Quotes","🔥 Catalyst Scanner","🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    for ticker in CORE_TICKERS:
        quote = get_live_quote(ticker)
        if quote["error"]: 
            st.warning(f"{ticker}: {quote['error']}")
            continue

        col1, col2, col3, col4 = st.columns([2,2,2,2])
        col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_pct']:+.2f}%")
        col2.write(f"**RelVol:** {quote['relvol']:.2f}x")
        col3.write(f"**Volume:** {quote['volume']:,}")
        col4.caption(f"Prev Close: ${quote['prev_close']:.2f}")

        with st.expander(f"🔎 Expand {ticker}"):
            # Catalyst headlines
            news = get_news(ticker)
            if news:
                st.write("### 📰 Catalysts (last 24h)")
                for n in news:
                    st.write(f"- [{n['title']}]({n['url']}) ({n['source']})")
            else:
                st.info("No recent news.")
            # AI Playbook
            st.markdown("### 🎯 AI Playbook")
            st.markdown(ai_playbook(ticker, quote['change_pct'], news[0]['title'] if news else ""))

# TAB 2: Catalyst Scanner
with tabs[1]:
    st.subheader("🔥 Market-Wide Catalyst Scanner")
    news_items = get_news()
    if news_items:
        for n in news_items:
            with st.expander(f"{n['title']} ({n['source']})"):
                st.write(n['summary'])
                if st.button(f"🤖 Generate Playbook from {n['title']}", key=f"pb_{n['title']}"):
                    st.markdown(ai_playbook("Market", 0, n['title']))
    else:
        st.info("No fresh news found.")

# TAB 3: AI Playbooks
with tabs[2]:
    st.subheader("🤖 Auto + Custom AI Playbooks")

    # Auto-generate for top movers
    st.markdown("### 🔥 Auto-Generated for Movers")
    for ticker in CORE_TICKERS[:5]:
        q = get_live_quote(ticker)
        if not q["error"] and abs(q["change_pct"]) > 1:
            st.write(f"**{ticker}** {q['change_pct']:+.2f}%")
            st.markdown(ai_playbook(ticker, q["change_pct"]))

    # Custom
    st.markdown("### 🎯 Custom Ticker Playbook")
    t = st.text_input("Enter ticker")
    if st.button("Generate Playbook") and t:
        q = get_live_quote(t)
        st.markdown(ai_playbook(t, q['change_pct'] if not q["error"] else 0))
