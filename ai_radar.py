import streamlit as st
import pandas as pd
import requests
import datetime
import json
import yfinance as yf
from typing import Dict, List
import numpy as np
import time

# Configure page
st.set_page_config(page_title="AI Radar Pro", layout="wide")

# Core tickers for selection
CORE_TICKERS = [
    "AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG",
    "NFLX","SPX","APP","NDX","SMCI","QUBT","IONQ","QBTS","SOFI","IBM",
    "COST","MSTR","COIN","OSCR","LYFT","JOBY","ACHR","LLY","UNH","OPEN",
    "UPST","NOW","ISRG","RR","FIG","HOOD","IBIT","WULF","WOLF","OKLO",
    "APLD","HUT","SNPS","SE","ETHU","TSM","AVGO","BITF","HIMS","BULL",
    "SPOT","LULU","CRCL","SOUN","QMMM","BMNR","SBET","GEMI","CRWV","KLAR",
    "BABA","INTC","CMG","UAMY","IREN","BBAI","BRKB","TEM","GLD","IWM","LMND",
    "CELH","PDD"
]

# Initialize session state
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"Default": ["AAPL", "NVDA", "TSLA", "SPY", "AMD", "MSFT"]}
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30

# API Keys
try:
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if OPENAI_KEY:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    else:
        openai_client = None
except:
    FINNHUB_KEY = ""
    POLYGON_KEY = ""
    openai_client = None

# ==================
# Data Functions
# ==================
@st.cache_data(ttl=10)
def get_live_quote(ticker: str) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_2d = stock.history(period="2d", interval="1m")
        hist_1d = stock.history(period="1d", interval="1m")
        if hist_1d.empty:
            hist_1d = stock.history(period="1d")
        if hist_2d.empty:
            hist_2d = stock.history(period="2d")
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        regular_market_open = info.get('regularMarketOpen', 0)
        previous_close = info.get('previousClose', hist_2d['Close'].iloc[-2] if len(hist_2d) >= 2 else 0)
        premarket_change = 0
        intraday_change = 0
        postmarket_change = 0
        if regular_market_open and previous_close:
            premarket_change = ((regular_market_open - previous_close) / previous_close) * 100
            if current_price and regular_market_open:
                intraday_change = ((current_price - regular_market_open) / regular_market_open) * 100
        current_hour = datetime.datetime.now().hour
        if current_hour >= 16 or current_hour < 4:
            regular_close = info.get('regularMarketPrice', current_price)
            if current_price != regular_close and regular_close:
                postmarket_change = ((current_price - regular_close) / regular_close) * 100
        total_change = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
        return {
            "last": float(current_price),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "change": float(info.get('regularMarketChange', current_price - previous_close if previous_close else 0)),
            "change_percent": float(total_change),
            "premarket_change": float(premarket_change),
            "intraday_change": float(intraday_change),
            "postmarket_change": float(postmarket_change),
            "previous_close": float(previous_close),
            "market_open": float(regular_market_open) if regular_market_open else 0,
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"),
            "error": None
        }
    except Exception as e:
        return {"last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
                "change": 0.0, "change_percent": 0.0,
                "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
                "previous_close": 0.0, "market_open": 0.0,
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"),
                "error": str(e)}

@st.cache_data(ttl=600)
def get_finnhub_news(symbol: str = None) -> List[Dict]:
    if not FINNHUB_KEY: return []
    try:
        if symbol:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()}&to={datetime.date.today()}&token={FINNHUB_KEY}"
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()[:10]
    except: pass
    return []

@st.cache_data(ttl=600)
def get_polygon_news() -> List[Dict]:
    if not POLYGON_KEY: return []
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/reference/news?published_utc.gte={today}&limit=20&apikey={POLYGON_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
    except: pass
    return []

def get_all_news() -> List[Dict]:
    all_news = []
    finnhub_news = get_finnhub_news()
    for item in finnhub_news:
        all_news.append({"title": item.get("headline", ""), "summary": item.get("summary", ""),
                         "source": "Finnhub", "url": item.get("url", ""),
                         "datetime": item.get("datetime", 0), "related": item.get("related", "")})
    polygon_news = get_polygon_news()
    for item in polygon_news:
        all_news.append({"title": item.get("title", ""), "summary": item.get("description", ""),
                         "source": "Polygon", "url": item.get("article_url", ""),
                         "datetime": item.get("published_utc", ""), "related": ",".join(item.get("tickers", []))})
    try: all_news.sort(key=lambda x: x["datetime"], reverse=True)
    except: pass
    return all_news[:15]

def ai_playbook(ticker: str, change: float, catalyst: str = "") -> str:
    if not openai_client:
        return f"**{ticker} Analysis** (OpenAI not set)\nChange: {change:+.2f}%\nCatalyst: {catalyst}"
    try:
        prompt = f"""Analyze {ticker} with {change:+.2f}% change today.
Catalyst: {catalyst or "Market movement"}.
Give sentiment, scalp/day/swing setups, and key levels under 250 words."""
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=400)
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    if not openai_client: return "OpenAI not set."
    try:
        news_context = "\n".join([f"- {n['title']}" for n in news_items[:5]])
        movers_context = "\n".join([f"- {m['ticker']}: {m['change_pct']:+.2f}%" for m in movers[:5]])
        prompt = f"""Analyze market with:
News:\n{news_context}\nMovers:\n{movers_context}
Give sentiment, themes, sectors, and opportunities under 200 words."""
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300)
        return resp.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# NEW: Auto-generate plays
def ai_auto_generate_plays(limit: int = 5):
    plays = []
    tickers = st.session_state.watchlists.get(st.session_state.active_watchlist, CORE_TICKERS[:20])
    for ticker in tickers:
        quote = get_live_quote(ticker)
        if quote["error"]: continue
        if abs(quote["change_percent"]) < 1.5: continue
        news = get_finnhub_news(ticker)
        catalyst = news[0].get("headline", "") if news else "No major catalyst"
        analysis = ai_playbook(ticker, quote["change_percent"], catalyst)
        plays.append({
            "ticker": ticker, "current_price": quote["last"],
            "change_percent": quote["change_percent"],
            "volume": quote["volume"], "catalyst": catalyst,
            "play_analysis": analysis,
            "session_data": {"premarket": quote["premarket_change"],
                             "intraday": quote["intraday_change"],
                             "afterhours": quote["postmarket_change"]}
        })
    plays.sort(key=lambda x: abs(x["change_percent"]), reverse=True)
    return plays[:limit]

# ==================
# Main App
# ==================
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Auto-refresh controls
col1, col2, col3, col4 = st.columns([2,1,1,2])
with col1: st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
with col2: st.session_state.refresh_interval = st.selectbox("Interval", [10,30,60], index=1)
with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear(); st.rerun()
with col4:
    now = datetime.datetime.now()
    st.write(f"**{'🟢 Open' if 9<=now.hour<16 else '🔴 Closed'}** | {now.strftime('%I:%M:%S %p ET')}")

tabs = st.tabs(["📊 Live Quotes","📋 Watchlist","🔥 Catalyst Scanner","📈 Market Analysis","🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    for t in tickers:
        q = get_live_quote(t)
        if q["error"]: continue
        with st.expander(f"{t} — ${q['last']:.2f} ({q['change_percent']:+.2f}%)"):
            st.write(f"Bid/Ask: ${q['bid']:.2f} / ${q['ask']:.2f}")
            st.write(f"Volume: {q['volume']:,}")
            st.write(f"Premarket {q['premarket_change']:+.2f}% | Day {q['intraday_change']:+.2f}% | AH {q['postmarket_change']:+.2f}%")
            if abs(q['change_percent']) >= 2:
                st.markdown("### 🎯 AI Details")
                st.markdown(ai_playbook(t, q['change_percent']))

# TAB 2: Watchlist Manager
with tabs[1]:
    st.subheader("📋 Manage Watchlist")
    # (Claude's manager code omitted here for brevity — keep as is in your file)

# TAB 3: Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Real-Time Catalyst Scanner")
    all_news = get_all_news()
    for n in all_news[:10]:
        with st.expander(f"{n['title']}"):
            st.write(n['summary'])
            st.write(f"Source: {n['source']}")
            if n['url']: st.markdown(f"[Read]({n['url']})")

# TAB 4: Market Analysis
with tabs[3]:
    st.subheader("📈 AI Market Analysis")
    movers = []
    for t in CORE_TICKERS[:10]:
        q = get_live_quote(t)
        if not q["error"]:
            movers.append({"ticker":t,"change_pct":q["change_percent"],"price":q["last"]})
    st.markdown(ai_market_analysis(get_all_news(), movers))

# TAB 5: AI Playbooks
with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    if st.button("🚀 Generate Auto Plays"):
        plays = ai_auto_generate_plays()
        for p in plays:
            with st.expander(f"{p['ticker']} — ${p['current_price']:.2f} ({p['change_percent']:+.2f}%)"):
                st.write(f"Catalyst: {p['catalyst']}")
                st.write(p['play_analysis'])

