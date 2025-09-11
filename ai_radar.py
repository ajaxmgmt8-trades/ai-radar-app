import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
from zoneinfo import ZoneInfo
from openai import OpenAI
import yfinance as yf   # ✅ using yfinance instead of Polygon

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")

TZ_CT = ZoneInfo("America/Chicago")

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
WATCHLIST_FILE = "watchlists.json"

# ---------------- API KEYS ----------------
try:
    FINNHUB_KEY = st.secrets["FINNHUB_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError as e:
    st.error(f"Missing API key: {e}")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_KEY)

# ---------------- WATCHLIST ----------------
def load_watchlists():
    try:
        with open(WATCHLIST_FILE, "r") as f:
            watchlists = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        watchlists = {"Default": CORE_TICKERS.copy()}
    if "Default" not in watchlists:
        watchlists["Default"] = CORE_TICKERS.copy()
    return watchlists

def save_watchlists(watchlists):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlists, f, indent=2)

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"

# ---------------- HELPERS ----------------
def get_quote(ticker: str):
    """Get live last price using yfinance"""
    try:
        data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
        if data.empty:
            return None
        last_price = float(data["Close"].iloc[-1])
        return {"last": last_price}
    except Exception:
        return None

def get_previous_close(ticker: str):
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data.empty:
            return None
        return float(data["Close"].iloc[-2])  # previous close
    except Exception:
        return None

def get_top_movers():
    """Scan top movers from a fixed universe using yfinance"""
    movers = []
    for t in CORE_TICKERS[:50]:  # limit for speed
        try:
            data = yf.download(t, period="2d", interval="1m", prepost=True, progress=False)
            if data.empty:
                continue
            last = data["Close"].iloc[-1]
            prev = data["Close"].iloc[0]
            change = (last - prev) / prev * 100
            movers.append({"Ticker": t, "Price": last, "Change %": change})
        except:
            continue

    if not movers:
        return pd.DataFrame(columns=["Ticker","Price","Change %"])

    df = pd.DataFrame(movers)
    df = df.sort_values("Change %", ascending=False).head(10)

    # ✅ format only for display
    df["Price"] = df["Price"].map(lambda x: f"${x:.2f}")
    df["Change %"] = df["Change %"].map(lambda x: f"{x:+.2f}%")

    return df


def ai_playbook(ticker: str, change: float, catalyst: str = ""):
    prompt = f"""
    You are an expert trader. Analyze {ticker}:
    - Change: {change:+.2f}%
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

def get_news_finnhub():
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
    r = requests.get(url, timeout=10).json()
    return r[:10] if isinstance(r, list) else []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Watchlist Manager")
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name
    tickers = st.session_state.watchlists[list_name].copy()

    # Add symbol
    new_ticker = st.text_input("Add Symbol", "").upper().strip()
    if st.button("➕ Add"):
        if new_ticker and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)
            st.rerun()

    # Remove
    for t in tickers:
        col1, col2 = st.columns([4,1])
        col1.write(t)
        if col2.button("🗑️", key=f"rm_{t}"):
            tickers.remove(t)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)
            st.rerun()

# ---------------- MAIN ----------------
st.title("🔥 AI Radar Pro — Live Trading Assistant")

tabs = st.tabs(["📊 Premarket", "📈 Intraday", "🌙 Postmarket", "🔥 Catalyst Scanner", "🤖 AI Playbooks"])

# TAB 1-3: Sessions
for i, session in enumerate(["Premarket","Intraday","Postmarket"]):
    with tabs[i]:
        st.subheader(f"{session} Movers")
        auto = st.checkbox(f"🔄 Auto-refresh {session}", key=f"auto_{session}")
        interval = st.selectbox("Refresh interval", [30,60,120], key=f"int_{session}")
        if st.button(f"Refresh {session}"):
            st.rerun()

        # Top movers
        st.markdown("### 🔥 Top 10 Market Movers")
        st.dataframe(get_top_movers(), use_container_width=True)

        # Watchlist movers
        st.markdown("### 📌 Your Watchlist Movers")
        wl_data = []
        for t in tickers:
            q = get_quote(t)
            prev = get_previous_close(t)
            if not q or not prev: continue
            change = ((q["last"]-prev)/prev)*100
            wl_data.append({
                "Ticker": t,
                "Price": f"${q['last']:.2f}",
                "Change %": f"{change:+.2f}%"
            })
        if wl_data:
            st.dataframe(pd.DataFrame(wl_data), use_container_width=True)

# TAB 4: Catalyst Scanner
with tabs[3]:
    st.subheader("🔥 Catalyst Scanner (Finnhub)")
    news = get_news_finnhub()
    for item in news:
        title = item.get("headline") or item.get("title")
        summary = item.get("summary","")
        st.markdown(f"**{title}**")
        st.caption(summary)
        if st.button(f"📊 AI Playbook {title}", key=f"ai_{title}"):
            with st.spinner("Analyzing..."):
                analysis = ai_playbook("SPY", 0, title)
                st.markdown(analysis)
        st.divider()

# TAB 5: AI Playbooks
with tabs[4]:
    st.subheader("🤖 AI Playbooks")
    sel = st.selectbox("Select Ticker", tickers+["Custom"])
    if sel=="Custom":
        sel = st.text_input("Enter ticker").upper()
    catalyst = st.text_input("Catalyst (optional)")
    if st.button("Generate AI Playbook"):
        q = get_quote(sel)
        prev = get_previous_close(sel)
        if q and prev:
            change = ((q["last"]-prev)/prev)*100
        else:
            change=0
        pb = ai_playbook(sel, change, catalyst)
        st.markdown(pb)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🔥 Powered by yfinance, Finnhub, OpenAI")
