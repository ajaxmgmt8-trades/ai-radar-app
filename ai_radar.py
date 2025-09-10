import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, time
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="🔥 AI Radar Pro — Market Scanner", layout="wide")
st.title("🔥 AI Radar Pro — Market Scanner")
st.caption("Premarket, Intraday, Postmarket, StockTwits, and Catalysts")

TZ_ET = ZoneInfo("US/Eastern")

# ==============================
# Helpers
# ==============================

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the index is tz-aware in US/Eastern."""
    if df.empty:
        return df
    try:
        idx = df.index.tz_convert(TZ_ET)
    except Exception:
        idx = df.index.tz_localize("UTC").tz_convert(TZ_ET)
    out = df.copy()
    out.index = idx
    return out

@st.cache_data(show_spinner=False, ttl=600)
def get_avg_daily_vol(ticker: str, lookback: int = 20) -> float | None:
    """20-day average daily volume."""
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def _session_window(session: str) -> tuple[time, time]:
    if session == "premarket":
        return time(4, 0), time(9, 30)
    if session in ("intraday", "regular"):
        return time(9, 30), time(16, 0)
    if session == "postmarket":
        return time(16, 0), time(20, 0)
    raise ValueError("invalid session")

@st.cache_data(show_spinner=False, ttl=60)
def scan_session_change_and_relvol(ticker: str, session: str) -> tuple[float | None, float | None]:
    """
    % change from session open to last trade and
    RelVol = (session volume) / (20d avg daily volume).
    """
    data = yf.download(ticker, period="3d", interval="1m", prepost=True, progress=False)
    if data.empty:
        return None, None

    data = _to_et(data)
    start_t, end_t = _session_window(session)

    today = datetime.now(TZ_ET).date()
    try:
        today_slice = data.loc[str(today)]
    except KeyError:
        return None, None

    session_slice = today_slice.between_time(start_t, end_t)
    if session_slice.empty:
        return None, None

    open_px = float(session_slice["Open"].iloc[0])
    last_px = float(session_slice["Close"].iloc[-1])
    change = (last_px - open_px) / open_px * 100

    sess_vol = float(session_slice["Volume"].sum())
    avg_vol = get_avg_daily_vol(ticker) or 1.0
    relvol = sess_vol / avg_vol

    return round(change, 2), round(relvol, 2)

def scan_session_list(tickers: list[str], session: str) -> pd.DataFrame:
    rows = []
    for t in tickers:
        change, relvol = scan_session_change_and_relvol(t, session)
        if change is None:
            continue
        rows.append({
            "Ticker": t,
            "Change %": change,
            "RelVol": relvol,
            "Catalyst": "No major Polygon/Benzinga news",  # placeholder; add real catalysts if desired
            "AI Playbook": f"Bias: {'Long' if change > 0 else 'Short'} — {change:.2f}% move, RelVol {relvol:.2f}x"
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Rank by absolute move, 1–10
    df = df.reindex(df["Change %"].abs().sort_values(ascending=False).index)
    df = df.head(10).reset_index(drop=True)
    df.index = df.index + 1
    return df

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)

def safe_dataframe(df: pd.DataFrame, height: int = 460):
    """Style only if columns exist; avoid KeyErrors on non-numeric tables."""
    if df is None or df.empty:
        st.info("No data available.")
        return

    styler = df.style

    if "AI Playbook" in df.columns:
        styler = styler.set_properties(subset=["AI Playbook"], **{"white-space": "normal"})

    # Apply numeric formats only if present & numeric
    if "Change %" in df.columns and _is_numeric_series(df["Change %"]):
        styler = styler.format({"Change %": "{:+.2f}%"})
        try:
            styler = styler.background_gradient(subset=["Change %"], cmap="RdYlGn")
        except Exception:
            pass

    if "RelVol" in df.columns and _is_numeric_series(df["RelVol"]):
        styler = styler.format({"RelVol": "{:.2f}x"})

    st.dataframe(styler, use_container_width=True, height=height)

# ==============================
# StockTwits (stable + cached)
# ==============================

@st.cache_data(show_spinner=False, ttl=90)
def fetch_stocktwits(ticker: str, limit: int = 5) -> list[str]:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return [f"⚠ StockTwits HTTP {r.status_code}"]
        try:
            data = r.json()
        except Exception:
            return ["⚠ StockTwits: invalid JSON response"]
        msgs = []
        for m in data.get("messages", [])[:limit]:
            u = m.get("user", {}).get("username", "user")
            b = m.get("body", "")
            msgs.append(f"🗨 @{u}: {b}")
        return msgs if msgs else ["No chatter"]
    except Exception as e:
        return [f"⚠ StockTwits error: {e}"]

@st.cache_data(show_spinner=False, ttl=120)
def stocktwits_trending(limit: int = 6) -> list[str]:
    url = "https://api.stocktwits.com/api/2/trending/symbols.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return [f"HTTP {r.status_code}"]
        data = r.json()
        syms = [s["symbol"] for s in data.get("symbols", [])[:limit]]
        return syms if syms else ["No trending"]
    except Exception as e:
        return [f"error: {e}"]

# ==============================
# Sidebar / Inputs
# ==============================
st.sidebar.header("⚙ Settings")
watchlist_input = st.sidebar.text_input(
    "Watchlist (comma-separated)",
    "AAPL,NVDA,TSLA,SPY,AMD,MSFT,META,ORCL,MDB,GOOG"
)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

# ==============================
# Tabs
# ==============================
tabs = st.tabs(["📊 Premarket", "☀️ Intraday", "🌙 Postmarket", "💬 StockTwits Feed", "📰 Catalysts"])

# ---- Premarket ----
with tabs[0]:
    st.subheader("Premarket Movers (04:00–09:30 ET)")
    st_autorefresh(interval=60 * 1000, key="premarket_refresh")  # 60s
    df = scan_session_list(watchlist, "premarket")
    safe_dataframe(df, height=480)

# ---- Intraday ----
with tabs[1]:
    st.subheader("Intraday Movers (09:30–16:00 ET)")
    st_autorefresh(interval=60 * 1000, key="intraday_refresh")
    df = scan_session_list(watchlist, "intraday")
    safe_dataframe(df, height=480)

# ---- Postmarket ----
with tabs[2]:
    st.subheader("Postmarket Movers (16:00–20:00 ET)")
    st_autorefresh(interval=60 * 1000, key="postmarket_refresh")
    df = scan_session_list(watchlist, "postmarket")
    safe_dataframe(df, height=480)

# ---- StockTwits ----
with tabs[3]:
    st.subheader("💬 StockTwits Feed (Watchlist + Trending)")
    for t in watchlist:
        st.markdown(f"### {t}")
        for m in fetch_stocktwits(t, limit=4):
            st.write(m)
    st.markdown("---")
    st.markdown("**Trending:** " + ", ".join(stocktwits_trending(6)))

# ---- Catalysts (placeholder demo; won’t style % columns) ----
with tabs[4]:
    st.subheader("📰 Catalysts (placeholder; plug Polygon/Benzinga here)")
    st_autorefresh(interval=90 * 1000, key="catalyst_refresh")
    df_cat = pd.DataFrame(
        {"Ticker": watchlist, "Catalyst": ["No major news"] * len(watchlist)}
    )
    # safe_dataframe handles tables without Change % columns (no KeyError)
    safe_dataframe(df_cat, height=320)
