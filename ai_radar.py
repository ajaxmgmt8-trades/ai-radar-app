import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
from polygon import RESTClient
from zoneinfo import ZoneInfo
import openai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
TZ_CT = ZoneInfo("America/Chicago")
WATCHLIST_FILE = "watchlists.json"

# 🔑 API KEYS
POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
polygon_client = RESTClient(POLYGON_KEY)
openai.api_key = OPENAI_KEY

# ---------------- WATCHLIST PERSISTENCE ----------------
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
if "show_sparklines" not in st.session_state:
    st.session_state.show_sparklines = True

# ---------------- HELPERS ----------------
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

def get_intraday_sparkline(ticker):
    """Fetch today’s intraday candles for sparkline."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        bars = polygon_client.get_aggs(
            ticker, multiplier=5, timespan="minute",
            from_=today, to=today
        )
        df = pd.DataFrame(bars)
        if df.empty:
            return None
        df["t"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ_CT)
        return df
    except:
        return None

def render_sparkline(df, change):
    color = "green" if change >= 0 else "red"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba(0,255,0,0.1)" if change >= 0 else f"rgba(255,0,0,0.1)",
        hoverinfo="skip"
    ))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=40, width=120, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def ai_playbook(ticker, change, relvol, catalyst=""):
    prompt = f"""
    You are an expert trader. Given:
    - Ticker: {ticker}
    - Change %: {change}
    - RelVol: {relvol}
    - Catalyst: {catalyst}

    Generate:
    1. A sentiment label (Bullish, Neutral, or Bearish) with confidence %.
    2. A scalp setup (1–5m).
    3. A daytrade setup (15–30m).
    4. A swing setup (4H–1D).
    Include Entry, Target, Stop, and Bias.
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI Error: {e}"

# ---------------- SIDEBAR (WATCHLIST) ----------------
with st.sidebar:
    st.header("📌 Watchlist")

    # Watchlist selector
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name
    tickers = st.session_state.watchlists[list_name]

    # Add/remove
    new_ticker = st.text_input("Add Symbol").upper()
    if st.button("➕ Add"):
        if new_ticker and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)

    for t in tickers:
        col1, col2 = st.columns([4,1])
        col1.write(t)
        if col2.button("❌", key=f"remove_{t}"):
            tickers.remove(t)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)

    # Sparkline toggle
    st.session_state.show_sparklines = st.checkbox("⚡ Show Sparklines", value=st.session_state.show_sparklines)

    st.markdown("---")

    # Watchlist table
    rows = []
    for t in tickers:
        q = get_quote(t)
        change = 0.0
        relvol = 1.0  # placeholder until relvol logic
        sentiment = "⚪ Neutral (50%)"  # placeholder until AI summary
        rows.append({
            "Ticker": t,
            "Last": q["last"],
            "Change %": change,
            "RelVol": relvol,
            "Sentiment": sentiment
        })
    df_watch = pd.DataFrame(rows)

    for idx, row in df_watch.iterrows():
        c1, c2, c3 = st.columns([2,2,3])
        c1.write(f"**{row['Ticker']}**")
        c1.write(f"{row['Last']:.2f}")
        c2.write(f"{row['Change %']:+.2f}%")
        c2.write(f"RelVol: {row['RelVol']:.2f}x")
        c3.write(row["Sentiment"])
        if st.session_state.show_sparklines:
            df = get_intraday_sparkline(row["Ticker"])
            if df is not None:
                st.plotly_chart(render_sparkline(df, row["Change %"]), use_container_width=False)

# ---------------- MAIN AREA ----------------
st.title("🔥 AI Radar Pro — Market Scanner")

tabs = st.tabs(["📊 Market Movers", "📰 Catalysts & News", "🤖 AI Playbooks"])

with tabs[0]:
    st.subheader("Market Movers")
    st.info("TODO: scan market-wide movers by session")

with tabs[1]:
    st.subheader("Catalysts & News")
    st.info("TODO: fetch headlines from Polygon/Finnhub and show market-wide")

with tabs[2]:
    st.subheader("AI Playbooks")
    ticker_choice = st.selectbox("Select ticker for AI Playbook", tickers)
    if st.button("Generate Playbook"):
        st.write(ai_playbook(ticker_choice, 0, 1.0, "No catalyst"))
