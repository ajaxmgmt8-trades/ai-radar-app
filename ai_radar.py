import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from openai import OpenAI
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar Pro", layout="wide", page_icon="🔥")

OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY     = st.secrets.get("NEWS_API_KEY", "")  # Finnhub optional
POLYGON_API_KEY  = st.secrets.get("POLYGON_API_KEY", "")  # Your Polygon/Benzinga key

client = OpenAI(api_key=OPENAI_API_KEY)
TZ_ET = ZoneInfo("US/Eastern")

# =========================
# SAFE DATAFRAME HELPER
# =========================
def safe_dataframe(df, height=500):
    """Render styled DataFrame with scrollbars and clean formatting."""
    if df is None or df.empty:
        return st.write("No data available.")

    # Force numeric cols to 2 decimals
    fmt = {}
    if "Change %" in df.columns and pd.api.types.is_numeric_dtype(df["Change %"]):
        fmt["Change %"] = lambda x: f"{x:+.2f}%"
    if "RelVol" in df.columns and pd.api.types.is_numeric_dtype(df["RelVol"]):
        fmt["RelVol"] = lambda x: f"{x:.2f}x"

    styler = (
        df.style
        .format(fmt, na_rep="—")
        .set_properties(subset=["AI Playbook"], **{"white-space": "normal"})  # wrap playbook
    )

    try:
        styler = styler.background_gradient(subset=["Change %"], cmap="RdYlGn")
    except Exception:
        pass

    return st.dataframe(styler, use_container_width=True, height=height)

# =========================
# SESSION HELPERS
# =========================
def _localize_to_et(df):
    if df.empty:
        return df
    try:
        idx = df.index.tz_convert(TZ_ET)
    except Exception:
        idx = df.index.tz_localize("UTC").tz_convert(TZ_ET)
    df = df.copy()
    df.index = idx
    return df

def _session_times(session: str):
    if session == "premarket":   return time(4, 0),  time(9, 30)
    if session == "regular":     return time(9, 30), time(16, 0)
    if session == "postmarket":  return time(16, 0), time(20, 0)
    raise ValueError("Invalid session")

@st.cache_data(show_spinner=False, ttl=300)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_session_change_and_relvol(ticker: str, session: str):
    data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
    if data.empty: return None
    data = _localize_to_et(data)

    start_t, end_t = _session_times(session)
    today = datetime.now(TZ_ET).date()
    day_slice = data.loc[str(today)]
    if day_slice.empty: return None

    session_slice = day_slice.between_time(start_t, end_t)
    if session_slice.empty: return None

    first = float(session_slice["Close"].iloc[0])
    last  = float(session_slice["Close"].iloc[-1])
    pct_change = (last - first) / first * 100

    session_vol = float(session_slice["Volume"].sum())
    daily_avg = avg_volume(ticker) or 1.0
    rel_vol = session_vol / daily_avg

    return round(pct_change, 2), round(rel_vol, 2)

# =========================
# NEWS SOURCES
# =========================
def get_polygon_news(ticker: str):
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if "results" in r and r["results"]:
            latest = r["results"][0]
            headline = latest.get("title")
            source = latest.get("publisher", {}).get("name", "News")
            ts = latest.get("published_utc", "")[:16].replace("T", " ")
            return f"{source}: {headline} ({ts} ET)"
        return "No major Polygon/Benzinga news"
    except Exception as e:
        return f"Polygon news error: {e}"

def get_stocktwits_messages(ticker, limit=3):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = requests.get(url, timeout=10).json()
        msgs = []
        for m in r.get("messages", [])[:limit]:
            user = m.get("user", {}).get("username", "user")
            body = m.get("body", "")
            msgs.append(f"@{user}: {body}")
        return msgs if msgs else ["No chatter"]
    except Exception as e:
        return [f"StockTwits error: {e}"]

def get_stocktwits_trending(limit=5):
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        r = requests.get(url, timeout=10).json()
        return [s["symbol"] for s in r.get("symbols", [])[:limit]]
    except Exception as e:
        return [f"Error: {e}"]

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, change, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY in Secrets."

    prompt = f"""
    Ticker: {ticker}
    Session Change: {change:.2f}%
    Session RelVol: {relvol:.2f}x
    Catalyst: {catalyst}

    Generate a concise 3-sentence trading playbook:
    1) Bias (long/short).
    2) Expected duration (scalp vs swing).
    3) Key risks (fade, IV crush, headlines, market beta).
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.strip()}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"

# =========================
# SCANNERS
# =========================
def get_watchlist(limit=10):
    return ["AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG"][:limit]

def scan_session_list(tickers, session):
    rows = []
    for t in tickers:
        r = scan_session_change_and_relvol(t, session)
        if not r: continue
        change, relvol = r
        catalyst = get_polygon_news(t)
        play = ai_playbook(t, change, relvol, catalyst)
        rows.append([t, change, relvol, catalyst, play])
    return pd.DataFrame(rows, columns=["Ticker","Change %","RelVol","Catalyst","AI Playbook"])

def scan_catalysts(tickers):
    rows = []
    for t in tickers:
        rows.append([t, get_polygon_news(t), " | ".join(get_stocktwits_messages(t,1))])
    return pd.DataFrame(rows, columns=["Ticker","Polygon/Benzinga","StockTwits"])

# =========================
# STREAMLIT UI
# =========================
st.markdown("<h1 style='text-align:center;color:orange'>🔥 AI Radar Pro — Market Scanner</h1>", unsafe_allow_html=True)

search_ticker = st.text_input("🔍 Search a ticker (e.g. TSLA, NVDA, SPY)")
watchlist = [search_ticker.upper()] if search_ticker else get_watchlist(8)

tabs = st.tabs(["📊 Premarket","💥 Intraday","🌙 Postmarket","💬 StockTwits Feed","📰 Catalysts"])

with tabs[0]:
    st.subheader("Premarket Movers (04:00–09:30 ET)")
    safe_dataframe(scan_session_list(watchlist, "premarket"))

with tabs[1]:
    st.subheader("Intraday Movers (09:30–16:00 ET)")
    safe_dataframe(scan_session_list(watchlist, "regular"))

with tabs[2]:
    st.subheader("Postmarket Movers (16:00–20:00 ET)")
    safe_dataframe(scan_session_list(watchlist, "postmarket"))

with tabs[3]:
    st.subheader("💬 StockTwits Feed")
    for t in watchlist:
        st.markdown(f"### {t}")
        for m in get_stocktwits_messages(t, 3):
            st.write(f"👉 {m}")
    st.markdown("---")
    st.markdown("### 📈 Market Trending")
    st.write(", ".join(get_stocktwits_trending(6)))

with tabs[4]:
    st.subheader("📰 Catalyst Feed (Polygon + StockTwits)")
    refresh_rate = st.slider("Refresh every X minutes", 1, 10, 3)
    st_autorefresh(interval=refresh_rate * 60 * 1000, key="catalyst_refresh")
    safe_dataframe(scan_catalysts(watchlist), height=420)

