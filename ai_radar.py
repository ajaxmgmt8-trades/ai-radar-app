import streamlit as st
import pandas as pd
import requests
import datetime
import json
import yfinance as yf
from typing import Dict, List
import time
import openai

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar Pro", layout="wide")

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

# =========================
# API KEYS
# =========================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

if OPENAI_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_KEY)
else:
    openai_client = None

# =========================
# HELPERS
# =========================
@st.cache_data(ttl=10)
def get_live_quote(ticker: str) -> Dict:
    """Live quote & session data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d", interval="1m", prepost=True)
        if hist.empty:
            return {"error": "No data"}

        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[0])
        change_pct = (last - prev) / prev * 100 if prev else 0

        return {
            "last": last,
            "volume": int(hist["Volume"].iloc[-1]),
            "change_percent": change_pct,
            "previous_close": prev,
            "last_updated": datetime.datetime.now().strftime("%I:%M:%S %p"),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=600)
def get_polygon_news(symbol: str = None) -> List[Dict]:
    if not POLYGON_KEY:
        return []
    try:
        url = f"https://api.polygon.io/v2/reference/news?limit=10&apiKey={POLYGON_KEY}"
        if symbol:
            url += f"&ticker={symbol}"
        r = requests.get(url, timeout=10).json()
        return r.get("results", [])
    except:
        return []

@st.cache_data(ttl=600)
def get_finnhub_news(symbol: str = None) -> List[Dict]:
    if not FINNHUB_KEY:
        return []
    try:
        if symbol:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()-datetime.timedelta(days=1)}&to={datetime.date.today()}&token={FINNHUB_KEY}"
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10).json()
        return r if isinstance(r, list) else []
    except:
        return []

def analyze_sentiment(title, summary=""):
    text = (title + " " + summary).lower()
    if any(w in text for w in ["soars","surge","beats","record","rocket"]):
        return "🚀 Explosive"
    elif any(w in text for w in ["rise","gain","partnership","approval"]):
        return "📈 Bullish"
    elif any(w in text for w in ["fall","weak","delay","warning"]):
        return "📉 Bearish"
    return "⚪ Neutral"

def ai_playbook(ticker, change, catalyst=""):
    if not openai_client:
        return f"⚠️ OpenAI key missing. Change: {change:+.2f}%. Catalyst: {catalyst}"
    try:
        prompt = f"""
        Analyze {ticker}:
        Change: {change:+.2f}%
        Catalyst: {catalyst or "Market Movement"}
        Give:
        1. Sentiment
        2. Scalp setup
        3. Day trade setup
        4. Swing setup
        """
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# =========================
# APP LAYOUT
# =========================
st.title("🔥 AI Radar Pro — Live Trading Assistant")

tabs = st.tabs(["📊 Live Quotes","📋 Watchlist","🔥 Catalyst Scanner","🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")

    search = st.text_input("🔍 Search ticker", "").upper()
    if search:
        q = get_live_quote(search)
        if not q["error"]:
            st.metric(search, f"${q['last']:.2f}", f"{q['change_percent']:+.2f}%")
            with st.expander(f"More on {search}"):
                news = get_polygon_news(search)[:3] + get_finnhub_news(search)[:3]
                for n in news:
                    st.write(f"- {n.get('title', '')}")
                st.markdown("### 🤖 AI Playbook")
                st.write(ai_playbook(search, q["change_percent"], news[0].get("title") if news else ""))
        else:
            st.error(q["error"])

    st.markdown("### Your Watchlist")
    for t in CORE_TICKERS[:10]:  # quick demo slice
        q = get_live_quote(t)
        if q["error"]: continue
        col1, col2 = st.columns([2,1])
        col1.metric(t, f"${q['last']:.2f}", f"{q['change_percent']:+.2f}%")
        with col2.expander("📂 Details"):
            news = get_polygon_news(t)[:2] + get_finnhub_news(t)[:2]
            for n in news:
                st.write(f"{analyze_sentiment(n.get('title',''))} {n.get('title','')}")
            st.write(ai_playbook(t, q["change_percent"], news[0].get("title") if news else ""))

# TAB 2: Watchlist
with tabs[1]:
    st.subheader("📋 Manage Watchlist")
    st.info("Coming soon: Persistent custom watchlists")

# TAB 3: Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 24h Catalyst Scanner")
    news = get_polygon_news() + get_finnhub_news()
    for n in news[:15]:
        sent = analyze_sentiment(n.get("title",""), n.get("summary",""))
        with st.expander(f"{sent} {n.get('title','')[:80]}"):
            st.write(n.get("summary",""))
            related = n.get("tickers") or n.get("related")
            if related: st.write(f"Tickers: {related}")
            if st.button(f"AI Playbook {related}", key=n.get("title","")):
                st.write(ai_playbook(related,0,n.get("title","")))

# TAB 4: AI Playbooks
with tabs[3]:
    st.subheader("🤖 AI Playbook Generator")
    custom = st.text_input("Enter ticker for playbook", "").upper()
    if st.button("Generate"):
        q = get_live_quote(custom)
        if not q["error"]:
            news = get_polygon_news(custom)
            st.write(ai_playbook(custom, q["change_percent"], news[0].get("title") if news else ""))

    st.markdown("### Auto Plays (Top Movers)")
    movers = []
    for t in CORE_TICKERS[:15]:
        q = get_live_quote(t)
        if not q["error"] and abs(q["change_percent"])>2:
            movers.append((t,q))
    for t,q in movers:
        with st.expander(f"{t} {q['change_percent']:+.2f}%"):
            news = get_polygon_news(t)
            st.write(ai_playbook(t, q["change_percent"], news[0].get("title") if news else ""))
