# ai_radar.py
import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
from polygon import RESTClient
from zoneinfo import ZoneInfo
from openai import OpenAI
from typing import Dict, List, Optional

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
TZ_CT = ZoneInfo("America/Chicago")
WATCHLIST_FILE = "watchlists.json"

# 🔑 API KEYS
try:
    POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    polygon_client = RESTClient(POLYGON_KEY)
    openai_client = OpenAI(api_key=OPENAI_KEY)
except Exception as e:
    st.error(f"API Key Error: {e}")
    st.stop()

# ---------------- WATCHLIST ----------------
def load_watchlists() -> Dict:
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"Default": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]}

def save_watchlists(watchlists: Dict):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlists, f, indent=2)

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]
if "show_sparklines" not in st.session_state:
    st.session_state.show_sparklines = True

# ---------------- HELPERS ----------------
@st.cache_data(ttl=30)
def get_quote(ticker: str) -> Dict:
    url = f"https://api.polygon.io/v2/last/trade/{ticker}?apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "results" not in data:
            return {"last": 0, "bid": 0, "ask": 0, "error": "No data"}
        q = data["results"]
        return {
            "last": q.get("p", 0),
            "bid": q.get("bP", 0),
            "ask": q.get("aP", 0),
            "error": None
        }
    except Exception as e:
        return {"last": 0, "bid": 0, "ask": 0, "error": str(e)}

@st.cache_data(ttl=300)
def get_previous_close(ticker: str) -> float:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        return r["results"][0]["c"] if "results" in r else 0
    except:
        return 0

@st.cache_data(ttl=300)
def get_intraday_sparkline(ticker: str) -> Optional[pd.DataFrame]:
    today = datetime.date.today().strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{today}/{today}?adjusted=true&sort=asc&apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        if "results" not in r:
            return None
        data = [
            {
                "timestamp": bar["t"],
                "close": bar["c"],
                "volume": bar["v"]
            }
            for bar in r["results"]
        ]
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ_CT)
        return df
    except:
        return None

def render_sparkline(df: pd.DataFrame, change: float) -> go.Figure:
    color = "#00ff88" if change >= 0 else "#ff4444"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["close"], mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=0,b=0), height=40, width=120,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def calculate_rel_volume(ticker: str, current_vol: float) -> float:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    try:
        bars = polygon_client.get_aggs(ticker, 1, "day", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        vols = [b.volume for b in bars]
        avg = sum(vols[-20:]) / len(vols[-20:]) if len(vols) >= 5 else 1
        return current_vol / avg if avg > 0 else 1.0
    except:
        return 1.0

def ai_playbook(ticker: str, change: float, relvol: float, catalyst: str = "") -> str:
    prompt = f"""
    Analyze {ticker}:
    - Change: {change:.2f}%
    - Relative Volume: {relvol:.2f}x
    - Catalyst: {catalyst or "None"}

    Provide:
    1. Sentiment (Bullish/Bearish/Neutral, confidence %)
    2. Scalp Setup (1-5m): entries, targets, stops
    3. Day Trade Setup (15-30m)
    4. Swing Setup (4H-Daily)
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Watchlist")
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name
    tickers = st.session_state.watchlists[list_name].copy()

    new_ticker = st.text_input("Add Symbol", placeholder="TSLA").upper().strip()
    if st.button("➕ Add"):
        if new_ticker and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)
            st.rerun()

    for t in tickers:
        if st.button(f"🗑️ Remove {t}", key=f"rm_{t}"):
            tickers.remove(t)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)
            st.rerun()

    st.session_state.show_sparklines = st.checkbox("⚡ Show Sparklines", value=st.session_state.show_sparklines)

# ---------------- MAIN ----------------
st.title("🔥 AI Radar Pro — Trading Assistant")

tabs = st.tabs(["📊 Market Movers", "📰 Catalysts & News", "🤖 AI Playbooks"])

# Market Movers
with tabs[0]:
    st.subheader("📊 Top Market Movers")
    try:
        g = requests.get(f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={POLYGON_KEY}").json()
        l = requests.get(f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/losers?apiKey={POLYGON_KEY}").json()
        gainers = g.get("tickers", [])[:5]
        losers = l.get("tickers", [])[:5]
        movers = []
        for d in gainers+losers:
            movers.append({
                "Ticker": d["ticker"],
                "Change%": f"{d['todaysChangePerc']:.2f}%",
                "Last": d["lastTrade"]["p"] if "lastTrade" in d else 0
            })
        st.dataframe(pd.DataFrame(movers))
    except Exception as e:
        st.error(f"Error loading movers: {e}")

# Catalysts
with tabs[1]:
    st.subheader("📰 Market Catalysts & News")
    try:
        news = requests.get(f"https://api.polygon.io/v2/reference/news?limit=10&apiKey={POLYGON_KEY}").json()
        if "results" in news:
            for n in news["results"]:
                st.markdown(f"**{n['title']}** — {n['published_utc']}")
                st.caption(n.get("description",""))
        else:
            st.info("No news available.")
    except Exception as e:
        st.error(f"News error: {e}")

# AI Playbooks
with tabs[2]:
    st.subheader("🤖 AI Trading Playbooks")
    t = st.selectbox("Select Symbol", tickers)
    catalyst = st.text_input("Catalyst (optional)")
    if st.button("Generate Playbook"):
        q = get_quote(t)
        prev = get_previous_close(t)
        change = ((q["last"]-prev)/prev*100) if prev>0 else 0
        df = get_intraday_sparkline(t)
        vol = df["volume"].sum() if df is not None else 0
        relvol = calculate_rel_volume(t, vol)
        pb = ai_playbook(t, change, relvol, catalyst)
        st.markdown(pb)

st.markdown("---")
st.caption("🔥 AI Radar Pro | Polygon.io + OpenAI")
