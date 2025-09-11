import streamlit as st
import pandas as pd
import yfinance as yf
from openai import OpenAI
import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
st_autorefresh = st.experimental_autorefresh(interval=5000, key="live_refresh")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

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

if "watchlist" not in st.session_state:
    st.session_state.watchlist = CORE_TICKERS[:15]  # default smaller set for speed

# ---------------- HELPERS ----------------
def get_quote_yf(ticker):
    try:
        data = yf.Ticker(ticker).history(period="2d", interval="1m", prepost=True)
        if data.empty: return None, None, None
        last = data["Close"].iloc[-1]
        prev = data["Close"].iloc[0]
        change = ((last - prev) / prev) * 100
        return last, prev, change
    except Exception:
        return None, None, None

def get_rel_volume(ticker):
    try:
        hist = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if hist.empty: return None
        avg_vol = hist["Volume"].mean()
        today_vol = hist["Volume"].iloc[-1]
        return today_vol / avg_vol if avg_vol > 0 else None
    except Exception:
        return None

def ai_playbook(ticker, change):
    if not OPENAI_API_KEY:
        return "Demo Playbook (add API key for live AI)"
    try:
        prompt = f"""
        Analyze {ticker} with {change:+.2f}% change.
        Provide 3 short bullet points:
        1. Sentiment (bullish/bearish/neutral)
        2. Best trade setup (scalp/day/swing)
        3. Key risk to watch
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except:
        return "AI Error"

def build_table(tickers):
    rows = []
    for t in tickers:
        price, prev, change = get_quote_yf(t)
        if price is None or prev is None: continue
        relvol = get_rel_volume(t)
        playbook = ai_playbook(t, change)
        rows.append({
            "Ticker": t,
            "Price": f"${price:.2f}",
            "Change %": f"{change:+.2f}%",
            "RelVol": f"{relvol:.2f}x" if relvol else "—",
            "AI Playbook": playbook
        })
    return pd.DataFrame(rows)

# ---------------- LAYOUT ----------------
st.title("🔥 AI Radar Pro — Live Trading Assistant")

tabs = st.tabs(["📊 Sessions", "📌 Watchlist", "📰 Catalysts", "🤖 AI Playbooks"])

# TAB 1: Sessions (Premarket, Intraday, Postmarket in columns)
with tabs[0]:
    st.subheader("📊 Market Sessions (Live)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔥 Premarket Movers")
        df = build_table(st.session_state.watchlist[:20])
        st.dataframe(df, use_container_width=True, height=500)

    with col2:
        st.markdown("### 💥 Intraday Movers")
        df = build_table(st.session_state.watchlist[20:40])
        st.dataframe(df, use_container_width=True, height=500)

    with col3:
        st.markdown("### 🌙 Postmarket Movers")
        df = build_table(st.session_state.watchlist[40:60])
        st.dataframe(df, use_container_width=True, height=500)

# TAB 2: Watchlist Management
with tabs[1]:
    st.subheader("📌 Manage Watchlist")
    new_ticker = st.text_input("Add Ticker").upper()
    if st.button("➕ Add"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            st.success(f"Added {new_ticker}")
    st.write("### Current Watchlist")
    for t in st.session_state.watchlist:
        col1, col2 = st.columns([4,1])
        col1.write(t)
        if col2.button("🗑️", key=f"rm_{t}"):
            st.session_state.watchlist.remove(t)
            st.rerun()

# TAB 3: Catalysts (placeholder)
with tabs[2]:
    st.subheader("📰 Catalyst Scanner")
    st.info("Coming soon: Polygon, Finnhub, StockTwits integration here")

# TAB 4: AI Playbooks Search
with tabs[3]:
    st.subheader("🤖 Search AI Playbook")
    search = st.text_input("Enter Ticker", "").upper()
    if st.button("Generate Playbook") and search:
        price, prev, change = get_quote_yf(search)
        if price:
            pb = ai_playbook(search, change)
            st.markdown(f"### {search} Playbook")
            st.markdown(pb)
        else:
            st.warning("No data for this ticker")
