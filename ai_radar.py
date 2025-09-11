import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
REFRESH_INTERVAL = 5000  # ms (5s live refresh)

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
    openai_client = OpenAI(api_key=OPENAI_KEY)
except Exception:
    openai_client = None

if "playbook_output" not in st.session_state:
    st.session_state.playbook_output = ""

# ---------------- HELPERS ----------------
@st.cache_data(ttl=300, show_spinner=False)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_ticker(ticker):
    try:
        data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
        if data.empty:
            return None
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[0])
        change = (last - prev) / prev * 100 if prev > 0 else 0
        rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
        return float(last), float(change or 0), float(rel_vol or 1.0)
    except Exception:
        return None

def ai_playbook(ticker: str, change: float, relvol: float, catalyst: str = ""):
    if not openai_client:
        return "⚠️ OpenAI key missing."
    prompt = f"""
    You are an expert trader. Analyze {ticker}:
    - Change: {change:+.2f}%
    - Relative Volume: {relvol:.2f}x
    - Catalyst: {catalyst if catalyst else "None"}

    Provide:
    1. Sentiment + confidence
    2. Scalp setup (1-5m)
    3. Day trade setup (15-30m)
    4. Swing setup (4H-Daily)
    """
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

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
                    result = scan_ticker(t)
                    if result:
                        last, change, relvol = result
                        st.session_state.playbook_output = ai_playbook(t, change, relvol)

        # Show playbook output panel
        if st.session_state.playbook_output:
            st.markdown("### 🎯 AI Playbook Result")
            st.markdown(st.session_state.playbook_output)

# ---------------- MAIN ----------------
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# 🔄 Auto refresh every REFRESH_INTERVAL ms
st_autorefresh = st.experimental_autorefresh(interval=REFRESH_INTERVAL, key="refresh")

tabs = st.tabs(["📊 Premarket", "💥 Intraday", "🌙 Postmarket", "📌 Watchlist", "🤖 AI Playbooks"])

# Session Tabs
render_session(tabs[0], "Premarket", CORE_TICKERS[:20])
render_session(tabs[1], "Intraday", CORE_TICKERS[20:40])
render_session(tabs[2], "Postmarket", CORE_TICKERS[40:60])

# Watchlist Tab
with tabs[3]:
    st.subheader("📌 Your Watchlist Movers")
    render_session(st, "Watchlist", CORE_TICKERS)

# AI Playbooks Tab
with tabs[4]:
    st.subheader("🤖 Generate Custom AI Playbook")
    search_ticker = st.text_input("Enter ticker symbol").upper().strip()
    catalyst = st.text_input("Catalyst (optional)")
    if st.button("Generate Playbook"):
        result = scan_ticker(search_ticker)
        if result:
            last, change, relvol = result
            st.session_state.playbook_output = ai_playbook(search_ticker, change, relvol, catalyst)
    if st.session_state.playbook_output:
        st.markdown("### 🎯 AI Playbook Result")
        st.markdown(st.session_state.playbook_output)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🔥 Powered by Yahoo Finance & OpenAI")
