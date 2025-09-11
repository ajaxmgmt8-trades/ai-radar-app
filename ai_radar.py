import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro — Live Trading Assistant", layout="wide")

REFRESH_INTERVAL = 15 * 1000  # 15 sec refresh
st_autorefresh = st.autorefresh(interval=REFRESH_INTERVAL, key="refresh")

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

# ---------------- API KEYS ----------------
try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
    FINNHUB_KEY = st.secrets["FINNHUB_API_KEY"]
except KeyError:
    OPENAI_KEY, POLYGON_KEY, FINNHUB_KEY = None, None, None

openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------------- HELPERS ----------------
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty:
        return None
    return hist["Volume"].mean()

def scan_ticker(ticker):
    """Return last price, % change, relative volume"""
    data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
    if data.empty:
        return None
    last = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[0])
    change = (last - prev) / prev * 100 if prev > 0 else 0
    relvol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return last, change, relvol

def ai_playbook(ticker, change, relvol, catalyst=""):
    if not openai_client:
        return "⚠️ No API key"
    prompt = f"""
    Analyze {ticker}:
    - Change: {change:+.2f}%
    - RelVol: {relvol:.2f}x
    - Catalyst: {catalyst if catalyst else "None"}
    Provide:
    1. Sentiment (bullish/bearish/neutral)
    2. Scalp setup (1-5m)
    3. Day trade setup (15-30m)
    4. Swing setup (4H-Daily)
    5. Risks
    """
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"

def scan_list(tickers):
    rows = []
    for t in tickers:
        try:
            scan = scan_ticker(t)
            if not scan:
                continue
            last, change, relvol = scan
            playbook = ai_playbook(t, change, relvol)
            rows.append({
                "Ticker": t,
                "Price": f"${last:.2f}",
                "Change %": f"{change:+.2f}%",
                "RelVol": f"{relvol:.2f}x",
                "AI Playbook": playbook
            })
        except:
            continue
    return pd.DataFrame(rows)

# ---------------- NEWS SOURCES ----------------
def get_polygon_news(limit=10):
    if not POLYGON_KEY:
        return []
    url = f"https://api.polygon.io/v2/reference/news?limit={limit}&apiKey={POLYGON_KEY}"
    try:
        return requests.get(url, timeout=10).json().get("results", [])
    except:
        return []

def get_finnhub_news():
    if not FINNHUB_KEY:
        return []
    today = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        return r[:10] if isinstance(r, list) else []
    except:
        return []

# ---------------- TABS ----------------
tabs = st.tabs(["🔥 Premarket", "📈 Intraday", "🌙 Postmarket", "📰 Catalysts", "📌 Watchlist", "🤖 AI Playbooks"])

# Session Tabs
session_map = {
    "🔥 Premarket": CORE_TICKERS[:20],
    "📈 Intraday": CORE_TICKERS[20:40],
    "🌙 Postmarket": CORE_TICKERS[40:60]
}

for i, (label, tickers) in enumerate(session_map.items()):
    with tabs[i]:
        st.subheader(f"{label} Movers")
        df = scan_list(tickers)
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=500)

# Catalysts Tab
with tabs[3]:
    st.subheader("📰 Catalyst Scanner")
    sources = st.multiselect("Select News Sources", ["Polygon","Finnhub"], default=["Polygon","Finnhub"])
    news_items = []
    if "Polygon" in sources:
        news_items.extend(get_polygon_news())
    if "Finnhub" in sources:
        news_items.extend(get_finnhub_news())
    if news_items:
        for n in news_items:
            title = n.get("title") or n.get("headline")
            desc = n.get("description") or n.get("summary","")
            st.markdown(f"**{title}**")
            st.caption(desc)
            if st.button(f"AI Playbook for {title[:20]}", key=title):
                st.markdown(ai_playbook("Market", 0, 0, title))
            st.divider()

# Watchlist Tab
with tabs[4]:
    st.subheader("📌 Your Watchlist")
    wl = st.text_area("Enter tickers (comma separated)", "MSFT,AMZN,NVDA").upper().split(",")
    wl = [t.strip() for t in wl if t.strip()]
    df = scan_list(wl)
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=500)

# AI Playbooks Tab
with tabs[5]:
    st.subheader("🤖 AI Playbooks")
    ticker = st.text_input("Enter ticker", "AAPL").upper().strip()
    if ticker:
        scan = scan_ticker(ticker)
        if scan:
            last, change, relvol = scan
            pb = ai_playbook(ticker, change, relvol)
            st.markdown(f"### {ticker}")
            st.write(f"Price: ${last:.2f}, Change: {change:+.2f}%, RelVol: {relvol:.2f}x")
            st.markdown(pb)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🔥 Live data via yfinance | News via Polygon/Finnhub | AI via OpenAI | Built with Streamlit")
