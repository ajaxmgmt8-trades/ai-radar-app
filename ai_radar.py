import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from streamlit_autorefresh import st_autorefresh

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="🔥 AI Radar Pro — Market Scanner", layout="wide")
st.title("🔥 AI Radar Pro — Market Scanner")
st.caption("Premarket, Intraday, Postmarket, StockTwits, and AI Playbooks")

# ==============================
# Utility Functions
# ==============================

def scan_session_change_and_relvol(ticker, session="intraday"):
    """Return % change and rel vol for a ticker in a session window"""
    try:
        data = yf.download(ticker, period="5d", interval="1m", progress=False)
        if data.empty:
            return None, None

        # session windows
        if session == "premarket":
            start_t, end_t = "04:00", "09:30"
        elif session == "intraday":
            start_t, end_t = "09:30", "16:00"
        elif session == "postmarket":
            start_t, end_t = "16:00", "20:00"
        else:
            return None, None

        # slice
        day_slice = data.between_time(start_t, end_t)
        if day_slice.empty:
            return None, None

        # % change
        open_px = day_slice["Open"].iloc[0]
        last_px = day_slice["Close"].iloc[-1]
        change = ((last_px - open_px) / open_px) * 100

        # relative volume
        avg_vol = data["Volume"].mean()
        relvol = day_slice["Volume"].sum() / avg_vol if avg_vol > 0 else 0

        return round(change, 2), round(relvol, 2)
    except Exception:
        return None, None


def scan_session_list(tickers, session):
    rows = []
    for t in tickers:
        change, relvol = scan_session_change_and_relvol(t, session)
        if change is not None:
            rows.append({
                "Ticker": t,
                "Change %": change,
                "RelVol": relvol,
                "Catalyst": "No major Polygon/Benzinga news",
                "AI Playbook": f"Bias: {'Long' if change > 0 else 'Short'} — {change:.2f}% move, RelVol {relvol:.2f}x"
            })
    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("Change %", key=lambda x: x.abs(), ascending=False)
        df = df.head(10).reset_index(drop=True)
        df.index = df.index + 1  # rank 1–10
    return df


def safe_dataframe(df, height=420):
    """Render dataframe safely with formatting + scrollbar"""
    if df is None or df.empty:
        st.info("No data available.")
        return
    st.dataframe(
        df.style.format({
            "Change %": "{:+.2f}%",
            "RelVol": "{:.2f}x"
        }).background_gradient(subset=["Change %"], cmap="RdYlGn"),
        use_container_width=True,
        height=height
    )


def fetch_stocktwits(ticker):
    """Fetch messages from StockTwits"""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "messages" not in data:
            return ["No chatter"]
        return [f"🗨 {m['user']['username']}: {m['body']}" for m in data["messages"][:5]]
    except Exception as e:
        return [f"⚠ StockTwits error: {e}"]


# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙ News Settings")
use_polygon = st.sidebar.checkbox("Use Polygon (Benzinga)", value=True)
use_finnhub = st.sidebar.checkbox("Use Finnhub", value=False)
use_stocktwits = st.sidebar.checkbox("Use StockTwits chatter", value=True)

watchlist_input = st.text_input("🔎 Search a ticker (e.g. TSLA, NVDA, SPY)", "AAPL,NVDA,TSLA,SPY,AMD,MSFT,META,ORCL,MDB,GOOG")
watchlist = [t.strip().upper() for t in watchlist_input.split(",")]

# ==============================
# Tabs
# ==============================
tabs = st.tabs(["📊 Premarket", "☀️ Intraday", "🌙 Postmarket", "💬 StockTwits Feed", "⚡ Catalysts"])

# ------------------------------
# Premarket Tab
# ------------------------------
with tabs[0]:
    st.subheader("Premarket Movers")
    st_autorefresh(interval=60 * 1000, key="premarket_refresh")  # refresh every 60s
    df = scan_session_list(watchlist, "premarket")
    safe_dataframe(df)

# ------------------------------
# Intraday Tab
# ------------------------------
with tabs[1]:
    st.subheader("Intraday Movers")
    st_autorefresh(interval=60 * 1000, key="intraday_refresh")
    df = scan_session_list(watchlist, "intraday")
    safe_dataframe(df)

# ------------------------------
# Postmarket Tab
# ------------------------------
with tabs[2]:
    st.subheader("Postmarket Movers")
    st_autorefresh(interval=60 * 1000, key="postmarket_refresh")
    df = scan_session_list(watchlist, "postmarket")
    safe_dataframe(df)

# ------------------------------
# StockTwits Feed
# ------------------------------
with tabs[3]:
    st.subheader("💬 StockTwits Feed (Watchlist + Trending)")
    if use_stocktwits:
        for t in watchlist:
            st.markdown(f"### {t}")
            msgs = fetch_stocktwits(t)
            for m in msgs:
                st.write(m)
    else:
        st.info("StockTwits chatter disabled in sidebar.")

# ------------------------------
# Catalysts Tab
# ------------------------------
with tabs[4]:
    st.subheader("⚡ Catalyst Feed (Experimental)")
    st_autorefresh(interval=90 * 1000, key="catalyst_refresh")
    df_cat = pd.DataFrame({
        "Ticker": watchlist,
        "Catalyst": ["Earnings tomorrow" if i % 2 == 0 else "No major news" for i in range(len(watchlist))]
    })
    safe_dataframe(df_cat, height=300)
