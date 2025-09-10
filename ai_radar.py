import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="🔥 AI Radar Pro — Market Scanner", layout="wide")
st.title("🔥 AI Radar Pro — Market Scanner")
st.caption("Live Premarket, Intraday, Postmarket Movers with RelVol")

TZ_ET = ZoneInfo("US/Eastern")
POLYGON_KEY = st.secrets["POLYGON_API_KEY"]

# ==============================
# HELPERS
# ==============================

def get_avg_daily_vol(ticker: str, lookback: int = 20) -> float | None:
    """20d average daily volume via yfinance."""
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def get_polygon_session_change(ticker, session="intraday"):
    """Session % change and RelVol from Polygon 1m bars."""
    today = datetime.now(TZ_ET).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_KEY}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None, None
    bars = r.json().get("results", [])
    if not bars:
        return None, None

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ_ET)
    df = df.set_index("t")

    if session == "premarket":
        start, end = "04:00", "09:30"
    elif session == "intraday":
        start, end = "09:30", "16:00"
    elif session == "postmarket":
        start, end = "16:00", "20:00"
    else:
        return None, None

    sess = df.between_time(start, end)
    if sess.empty:
        return None, None

    open_px = sess["o"].iloc[0]
    last_px = sess["c"].iloc[-1]
    change = (last_px - open_px) / open_px * 100

    vol = sess["v"].sum()
    avg_vol = get_avg_daily_vol(ticker) or 1.0
    relvol = vol / avg_vol

    return round(change, 2), round(relvol, 2)

def polygon_top_universe(direction="gainers"):
    """Fetch top gainers/losers from Polygon snapshot."""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/{direction}?apiKey={POLYGON_KEY}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return []
    return [t["ticker"] for t in r.json().get("tickers", [])]

def get_top10(session="intraday"):
    """Rank top 10 movers by session-specific % change."""
    tickers = polygon_top_universe("gainers") + polygon_top_universe("losers")
    movers = []
    for t in tickers[:50]:  # limit universe for speed
        change, relvol = get_polygon_session_change(t, session)
        if change is None:
            continue
        movers.append({
            "Ticker": t,
            "Change %": change,
            "RelVol": relvol,
            "Catalyst": "No major Polygon/Benzinga news",
            "AI Playbook": f"Bias: {'Long' if change > 0 else 'Short'} — {change:.2f}%, RelVol {relvol:.2f}x"
        })
    df = pd.DataFrame(movers)
    if df.empty:
        return df
    df = df.reindex(df["Change %"].abs().sort_values(ascending=False).index)
    df = df.head(10).reset_index(drop=True)
    df.index = df.index + 1
    return df

def safe_dataframe(df: pd.DataFrame, height=480):
    if df is None or df.empty:
        st.info("No data available.")
        return
    styler = df.style.format({
        "Change %": "{:+.2f}%",
        "RelVol": "{:.2f}x"
    }).background_gradient(subset=["Change %"], cmap="RdYlGn")
    st.dataframe(styler, use_container_width=True, height=height)

# ==============================
# SEARCH BAR
# ==============================
search_ticker = st.text_input("🔍 Search a ticker (e.g., TSLA, NVDA, SPY)", "").upper()
if search_ticker:
    st.subheader(f"Search: {search_ticker}")
    change, relvol = get_polygon_session_change(search_ticker, "intraday")
    if change is not None:
        st.write(f"Change: {change:+.2f}%, RelVol: {relvol:.2f}x")
    else:
        st.write("No intraday data available")

# ==============================
# TABS
# ==============================
tabs = st.tabs(["📊 Premarket", "☀️ Intraday", "🌙 Postmarket"])

with tabs[0]:
    st.subheader("Premarket Movers (04:00–09:30 ET)")
    st_autorefresh(interval=60*1000, key="pre_refresh")
    safe_dataframe(get_top10("premarket"))

with tabs[1]:
    st.subheader("Intraday Movers (09:30–16:00 ET)")
    st_autorefresh(interval=60*1000, key="intra_refresh")
    safe_dataframe(get_top10("intraday"))

with tabs[2]:
    st.subheader("Postmarket Movers (16:00–20:00 ET)")
    st_autorefresh(interval=60*1000, key="post_refresh")
    safe_dataframe(get_top10("postmarket"))
