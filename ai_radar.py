import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from openai import OpenAI
import requests

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="live_refresh")

# API Keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Core tickers
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
# HELPERS
# =========================
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_ticker(ticker):
    data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
    if data.empty:
        return None
    last = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[0])
    change = (last - prev) / prev * 100 if prev > 0 else 0
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return last, change, rel_vol

def ai_playbook(ticker, change, relvol, catalyst=""):
    if not OPENAI_API_KEY:
        return "⚠️ Missing OpenAI API Key"
    prompt = f"""
    Ticker: {ticker}
    Change: {change:+.2f}%
    Relative Volume: {relvol:.2f}x
    Catalyst: {catalyst if catalyst else "None"}

    Generate a short trading playbook:
    1. Sentiment & confidence
    2. Scalp setup (1-5m)
    3. Day trade setup (15-30m)
    4. Swing setup (4H-Daily)
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# =========================
# UI: Tabs
# =========================
st.title("🔥 AI Radar Pro — Live Trading Assistant")

tabs = st.tabs([
    "📊 Premarket", "💥 Intraday", "🌙 Postmarket",
    "🔥 Catalyst Scanner", "📌 Watchlist", "🤖 AI Playbooks"
])

# Shared playbook panel state
if "playbook_output" not in st.session_state:
    st.session_state.playbook_output = ""

# Function to render session movers
def render_session(tab, session_name, tickers):
    with tab:
        st.subheader(f"{session_name} Movers")

        rows = []
        for t in tickers:
            result = scan_ticker(t)
            if not result: 
                continue
            last, change, relvol = result
            rows.append({
                "Ticker": t,
                "Price": f"${last:.2f}",
                "Change %": f"{change:+.2f}%",
                "RelVol": f"{relvol:.2f}x"
            })
        df = pd.DataFrame(rows)

        if df.empty:
            st.info("No data available.")
        else:
            st.dataframe(df, use_container_width=True, height=500)

            # Playbook buttons
            for t in df["Ticker"]:
                if st.button(f"📊 Playbook {t}", key=f"{session_name}_{t}"):
                    last, change, relvol = scan_ticker(t)
                    st.session_state.playbook_output = ai_playbook(t, change, relvol)

        # Show playbook output
        if st.session_state.playbook_output:
            st.markdown("### 🎯 AI Playbook Result")
            st.markdown(st.session_state.playbook_output)

# TAB 1: Premarket
render_session(tabs[0], "Premarket", CORE_TICKERS[:20])

# TAB 2: Intraday
render_session(tabs[1], "Intraday", CORE_TICKERS[20:40])

# TAB 3: Postmarket
render_session(tabs[2], "Postmarket", CORE_TICKERS[40:60])

# TAB 4: Catalyst Scanner
with tabs[3]:
    st.subheader("🔥 Catalyst Scanner (Polygon/Finnhub demo)")
    st.info("🚧 Hook live APIs here for market-wide catalysts")

# TAB 5: Watchlist
with tabs[4]:
    st.subheader("📌 Watchlist Movers")
    rows = []
    for t in CORE_TICKERS:
        result = scan_ticker(t)
        if not result:
            continue
        last, change, relvol = result
        rows.append({
            "Ticker": t,
            "Price": f"${last:.2f}",
            "Change %": f"{change:+.2f}%",
            "RelVol": f"{relvol:.2f}x"
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=500)

    for t in df["Ticker"]:
        if st.button(f"📊 Playbook {t}", key=f"watchlist_{t}"):
            last, change, relvol = scan_ticker(t)
            st.session_state.playbook_output = ai_playbook(t, change, relvol)

    if st.session_state.playbook_output:
        st.markdown("### 🎯 AI Playbook Result")
        st.markdown(st.session_state.playbook_output)

# TAB 6: AI Playbooks
with tabs[5]:
    st.subheader("🤖 Generate Custom AI Playbook")
    ticker_in = st.text_input("Enter ticker (e.g. TSLA)").upper().strip()
    catalyst = st.text_input("Catalyst (optional)")
    if st.button("Generate Playbook", key="custom_playbook"):
        if ticker_in:
            result = scan_ticker(ticker_in)
            if result:
                last, change, relvol = result
                st.session_state.playbook_output = ai_playbook(ticker_in, change, relvol, catalyst)
    if st.session_state.playbook_output:
        st.markdown("### 🎯 AI Playbook Result")
        st.markdown(st.session_state.playbook_output)
