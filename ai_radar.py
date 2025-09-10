import streamlit as st
import pandas as pd
import requests
import datetime
import json
from polygon import RESTClient
import finnhub
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro — Market Scanner", layout="wide")

# 🔑 API KEYS
POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", None)

polygon_client = RESTClient(POLYGON_KEY)
finnhub_client = finnhub.Client(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

TZ_CT = ZoneInfo("US/Central")  # convert everything to Central time
WATCHLIST_FILE = "watchlists.json"

# ---------------- WATCHLIST ----------------
def load_watchlists():
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except:
        return {"Default": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]}

def save_watchlists(watchlists):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlists, f)

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]

# ---------------- HELPERS ----------------
def get_session_times():
    return {
        "premarket": (datetime.time(3, 0), datetime.time(8, 30)),   # CT
        "intraday": (datetime.time(8, 30), datetime.time(15, 0)),  # CT
        "postmarket": (datetime.time(15, 0), datetime.time(19, 0)) # CT
    }

def get_session_data(ticker, session, date_override=None):
    start_t, end_t = get_session_times()[session]
    target_date = date_override or datetime.date.today()

    try:
        bars = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=target_date.strftime("%Y-%m-%d"),
            to=target_date.strftime("%Y-%m-%d")
        )
    except Exception:
        return None, None

    if not bars:
        return None, None

    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ_CT)
    df.set_index("timestamp", inplace=True)

    session_df = df.between_time(start_t, end_t)
    if session_df.empty:
        return None, None

    open_price = session_df.iloc[0]["open"]
    close_price = session_df.iloc[-1]["close"]
    change_pct = ((close_price - open_price) / open_price) * 100

    relvol = session_df["volume"].sum() / (df["volume"].sum() / 3)
    return round(change_pct, 2), round(relvol, 2)

def scan_session(session, tickers):
    movers = []

    # Try today first
    for t in tickers:
        change, relvol = get_session_data(t, session)
        if change is not None:
            movers.append({"Ticker": t, "Change %": change, "RelVol": relvol})

    # If empty, look back up to 5 previous sessions
    if not movers:
        today = datetime.date.today()
        lookback = 1
        while not movers and lookback <= 5:
            prev_day = today - datetime.timedelta(days=lookback)
            for t in tickers:
                change, relvol = get_session_data(t, session, date_override=prev_day)
                if change is not None:
                    movers.append({"Ticker": t, "Change %": change, "RelVol": relvol})
            lookback += 1

    if not movers:
        return pd.DataFrame(columns=["Ticker", "Change %", "RelVol"])

    df = pd.DataFrame(movers).sort_values("Change %", ascending=False).head(10)
    df.index = range(1, len(df) + 1)
    return df

def get_quote(ticker):
    url = f"https://api.polygon.io/v2/last/nbbo/{ticker}?apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url, timeout=5)
        q = r.json().get("results", {})
        return {
            "last": q.get("p", 0),
            "bid": q.get("bP", 0),
            "ask": q.get("aP", 0)
        }
    except:
        return {"last": 0, "bid": 0, "ask": 0}

def get_catalysts(tickers):
    if not finnhub_client:
        return pd.DataFrame()
    news_items = []
    for t in tickers:
        try:
            res = finnhub_client.company_news(
                t,
                _from=str(datetime.date.today()),
                to=str(datetime.date.today())
            )
            for n in res[:2]:
                news_items.append({
                    "Ticker": t,
                    "Headline": n["headline"],
                    "Source": n["source"],
                    "Time": datetime.datetime.fromtimestamp(n["datetime"]).astimezone(TZ_CT).strftime("%H:%M")
                })
        except:
            pass
    return pd.DataFrame(news_items)

def ai_playbook(ticker, change, relvol, catalyst="None"):
    bias = "Long" if change > 0 else "Short"
    return f"📊 {ticker}: {bias} bias | Change {change:+.2f}%, RelVol {relvol:.2f}x | Catalyst: {catalyst}"

# ---------------- UI ----------------
st.title("🔥 AI Radar Pro — Market Scanner")

# --- Sidebar Watchlist ---
with st.sidebar:
    st.header("📌 Watchlist Manager")

    # Watchlist selector
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name

    tickers = st.session_state.watchlists[list_name]

    # Add new ticker
    new_ticker = st.text_input("Add Ticker (e.g. TSLA)").upper()
    if st.button("➕ Add"):
        if new_ticker and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)

    # Remove tickers
    for t in tickers:
        col1, col2 = st.columns([4,1])
        col1.write(t)
        if col2.button("❌", key=f"remove_{t}"):
            tickers.remove(t)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)

    st_autorefresh(interval=30*1000, key="watchlist_refresh")

    # Display live table
    rows = []
    for t in tickers:
        change, relvol = get_session_data(t, "intraday")  # default intraday
        q = get_quote(t)
        rows.append({
            "Ticker": t,
            "Last": q["last"],
            "Change %": change if change else 0,
            "RelVol": relvol if relvol else 0,
            "Bid": q["bid"],
            "Ask": q["ask"]
        })
    df_watch = pd.DataFrame(rows)
    st.dataframe(df_watch, use_container_width=True)

# --- Main Tabs ---
tabs = st.tabs([
    "📈 Premarket", "🌞 Intraday", "🌙 Postmarket", "🚨 Market Radar", "📰 Catalysts & News"
])

with tabs[0]:
    st.subheader("Premarket Movers (03:00–08:30 CT)")
    st_autorefresh(interval=30*1000, key="premarket_refresh")
    df = scan_session("premarket", tickers)
    st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.subheader("Intraday Movers (08:30–15:00 CT)")
    st_autorefresh(interval=30*1000, key="intraday_refresh")
    df = scan_session("intraday", tickers)
    st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.subheader("Postmarket Movers (15:00–19:00 CT)")
    st_autorefresh(interval=30*1000, key="postmarket_refresh")
    df = scan_session("postmarket", tickers)
    st.dataframe(df, use_container_width=True)

with tabs[3]:
    st.subheader("🚨 Market Radar (Unusual Volume Movers)")
    st_autorefresh(interval=60*1000, key="radar_refresh")
    # Here we could expand to fetch whole-market movers. For now, scan watchlist.
    df = scan_session("intraday", tickers)
    st.dataframe(df, use_container_width=True)

with tabs[4]:
    st.subheader("📰 Catalysts & News Feed")
    st_autorefresh(interval=60*1000, key="catalysts_refresh")
    df_cat = get_catalysts(tickers)
    if not df_cat.empty:
        st.dataframe(df_cat, use_container_width=True, height=400)
    else:
        st.info("No major catalysts today.")
