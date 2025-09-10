import yfinance as yf
import pandas as pd
import streamlit as st
import requests
from openai import OpenAI
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh  # modern auto-refresh

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar Pro", layout="wide", page_icon="🔥")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY   = st.secrets.get("NEWS_API_KEY", "")     # Finnhub
POLYGON_API_KEY = st.secrets.get("POLYGON_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

TZ_ET = ZoneInfo("US/Eastern")

# =========================
# SAFE DATAFRAME HELPER
# =========================
def safe_dataframe(df, height=560):
    """
    Styled dataframe:
      - Formats Change % and RelVol safely
      - Adds gradient (if matplotlib present)
      - Wraps AI Playbook text
      - Provides a vertical scrollbar (height)
    """
    try:
        if df is None or df.empty:
            return st.write("No data available.")

        # Build a per-column format map (only for numeric cols)
        fmt = {}
        if "Change %" in df.columns and pd.api.types.is_numeric_dtype(df["Change %"]):
            fmt["Change %"] = lambda x: f"{x:+.2f}%"
        if "RelVol" in df.columns and pd.api.types.is_numeric_dtype(df["RelVol"]):
            fmt["RelVol"] = lambda x: f"{x:.2f}x"

        styler = (
            df.style
            .format(fmt, na_rep="—")
            .set_properties(subset=[c for c in df.columns if c.lower().startswith("ai playbook")],
                            **{"white-space": "normal"})  # wrap long playbooks
        )

        # Try gradient; if matplotlib unavailable, fall back
        try:
            styler = styler.background_gradient(
                subset=[c for c in ["Change %"] if c in df.columns], cmap="RdYlGn"
            )
        except ImportError:
            pass

        return st.dataframe(styler, use_container_width=True, height=height)

    except Exception as e:
        st.warning(f"⚠️ Styling disabled: {e}")
        return st.dataframe(df, use_container_width=True, height=height)

# =========================
# DATA HELPERS
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def _localize_to_et(df):
    """Ensure DatetimeIndex is in US/Eastern."""
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
    """Return (start_time, end_time) as time objects (ET)."""
    if session == "premarket":
        return time(4, 0), time(9, 30)
    if session == "regular":
        return time(9, 30), time(16, 0)
    if session == "postmarket":
        return time(16, 0), time(20, 0)
    raise ValueError("Invalid session")

def scan_session_change_and_relvol(ticker: str, session: str):
    """Compute % change and session rel vol for a single ticker & session (today ET)."""
    # 2d 1m gives enough bars to cover pre/post
    data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
    if data.empty:
        return None
    data = _localize_to_et(data)

    start_t, end_t = _session_times(session)

    # Take today's bars in ET between session times
    today = datetime.now(TZ_ET).date()
    day_slice = data.loc[str(today)]
    if day_slice.empty:
        return None

    session_slice = day_slice.between_time(start_t, end_t, include_end=False)
    if session_slice.empty:
        return None

    first = float(session_slice["Close"].iloc[0])
    last  = float(session_slice["Close"].iloc[-1])
    pct_change = (last - first) / first * 100

    # RelVol: session volume vs 20d average daily volume
    session_vol = float(session_slice["Volume"].sum())
    daily_avg = avg_volume(ticker) or 1.0
    rel_vol = session_vol / daily_avg

    return float(pct_change), float(rel_vol)

# =========================
# NEWS SOURCES
# =========================
def get_finnhub_news(ticker: str):
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={today}&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if isinstance(r, list) and r:
            return r[0].get("headline") or r[0].get("title")
        return "No major Finnhub news"
    except Exception:
        return "Finnhub error"

def get_polygon_news(ticker: str):
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if "results" in r and r["results"]:
            latest = r["results"][0]
            return f"{latest.get('publisher', {}).get('name','News')}: {latest.get('title')}"
        return "No major Polygon/Benzinga news"
    except Exception as e:
        return f"Polygon news error: {e}"

def _get_json(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (AI-Radar)",
        "Accept": "application/json"
    }
    resp = requests.get(url, timeout=10, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.json()

def get_stocktwits_messages(ticker, limit=3):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = _get_json(url)
        msgs = []
        for m in r.get("messages", [])[:limit]:
            user = m.get("user", {}).get("username", "user")
            body = m.get("body", "")
            msgs.append(f"@{user}: {body}")
        return msgs if msgs else ["No chatter"]
    except Exception as e:
        return [f"StockTwits error ({e})"]

def get_stocktwits_trending(limit=5):
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        r = _get_json(url)
        return [s["symbol"] for s in r.get("symbols", [])[:limit]]
    except Exception as e:
        return [f"StockTwits trending error ({e})"]

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, change, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY in Secrets."

    try:
        change = float(change) if change is not None else 0.0
    except Exception:
        change = 0.0
    try:
        relvol = float(relvol) if relvol is not None else 0.0
    except Exception:
        relvol = 0.0

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
    # you can replace with your own universe
    return ["AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG"][:limit]

def scan_session_list(tickers, session, use_polygon, use_finnhub):
    rows = []
    for t in tickers:
        r = scan_session_change_and_relvol(t, session)
        if not r:
            continue
        change, relvol = r
        catalyst = get_polygon_news(t) if use_polygon else get_finnhub_news(t)
        try:
            play = ai_playbook(t, change, relvol, catalyst)
        except Exception:
            play = "AI Playbook error"
        rows.append([t, float(round(change,2)), float(round(relvol,2)), catalyst, play])
    return pd.DataFrame(rows, columns=["Ticker","Change %","RelVol","Catalyst","AI Playbook"])

def scan_catalysts(tickers, use_polygon=True, use_finnhub=True, use_stocktwits=True):
    rows = []
    for t in tickers:
        poly = get_polygon_news(t) if use_polygon else ""
        finn = get_finnhub_news(t) if use_finnhub else ""
        stw  = get_stocktwits_messages(t,1)[0] if use_stocktwits else ""
        rows.append([t, poly, finn, stw])
    return pd.DataFrame(rows, columns=["Ticker","Polygon/Benzinga","Finnhub","StockTwits"])

# =========================
# STREAMLIT UI
# =========================
st.markdown("<h1 style='text-align:center;color:orange'>🔥 AI Radar Pro — Market Scanner</h1>", unsafe_allow_html=True)
st.caption("Premarket, Intraday, Postmarket (session-aware), StockTwits, and AI Playbooks")

# Sidebar options
st.sidebar.header("⚙️ News Settings")
use_polygon     = st.sidebar.checkbox("Use Polygon (Benzinga)", value=True)
use_finnhub     = st.sidebar.checkbox("Use Finnhub", value=True)
use_stocktwits  = st.sidebar.checkbox("Use StockTwits chatter", value=True)

# Search box
search_ticker = st.text_input("🔍 Search a ticker (e.g. TSLA, NVDA, SPY)")
if search_ticker:
    wl = [search_ticker.upper()]
else:
    wl = get_watchlist(limit=10)

# Tabs
tabs = st.tabs(["📊 Premarket","💥 Intraday","🌙 Postmarket","💬 StockTwits Feed","📰 Catalysts"])

with tabs[0]:
    st.subheader("Premarket Movers (04:00–09:30 ET)")
    df = scan_session_list(wl, "premarket", use_polygon, use_finnhub)
    safe_dataframe(df, height=500)

with tabs[1]:
    st.subheader("Intraday Movers (09:30–16:00 ET)")
    df = scan_session_list(wl, "regular", use_polygon, use_finnhub)
    safe_dataframe(df, height=500)

with tabs[2]:
    st.subheader("Postmarket Movers (16:00–20:00 ET)")
    df = scan_session_list(wl, "postmarket", use_polygon, use_finnhub)
    safe_dataframe(df, height=500)

with tabs[3]:
    st.subheader("💬 StockTwits Feed (Watchlist + Trending)")
    for t in wl:
        st.markdown(f"### {t}")
        msgs = get_stocktwits_messages(t, limit=3)
        for m in msgs:
            st.write(f"👉 {m}")

    st.markdown("---")
    st.markdown("### 📈 Market Trending")
    trending = get_stocktwits_trending(limit=6)
    st.write(", ".join(trending))

with tabs[4]:
    st.subheader("📰 Catalyst Feed (Auto-refresh)")
    refresh_rate = st.slider("Refresh every X minutes", 1, 10, 3)
    st_autorefresh(interval=refresh_rate * 60 * 1000, key="catalyst_refresh")
    df = scan_catalysts(wl, use_polygon=True, use_finnhub=use_finnhub, use_stocktwits=use_stocktwits)
    safe_dataframe(df, height=420)
