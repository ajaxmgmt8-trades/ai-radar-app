import yfinance as yf
import pandas as pd
import streamlit as st
import requests
import openai
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar Dashboard", layout="wide")

# UI controls
default_watchlist = "QMMM,NVDA,ORCL,MDB,SPY"
watchlist = st.text_input("Watchlist (comma-separated tickers)", value=default_watchlist)
TICKERS = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
LOOKBACK = st.number_input("Avg volume lookback (days)", min_value=5, max_value=60, value=20, step=1)

# Secrets (set these in Streamlit Cloud under Settings → Secrets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")  # Finnhub

if not OPENAI_API_KEY or not NEWS_API_KEY:
    st.warning("Add your OPENAI_API_KEY and NEWS_API_KEY in App settings → Secrets to enable playbooks and news.")
openai.api_key = OPENAI_API_KEY

@st.cache_data(show_spinner=False, ttl=300)
def avg_volume(ticker: str, lookback: int = 20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

@st.cache_data(show_spinner=False, ttl=120)
def premarket_scan(ticker: str, lookback: int = 20):
    data = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)
    if data is None or data.empty:
        return None
    try:
        pre = data.between_time("04:00", "09:30")
    except Exception:
        return None
    if pre.empty:
        return None

    try:
        open_price = float(pre["Open"].iloc[0])
        last_close = float(pre["Close"].iloc[-1])
        pre_vol_sum = float(pre["Volume"].sum())
    except Exception:
        return None

    avg_vol = avg_volume(ticker, lookback)
    if avg_vol is None or avg_vol == 0:
        rel_vol = 0.0
    else:
        rel_vol = pre_vol_sum / avg_vol

    pct_gap = ((last_close - open_price) / open_price) * 100 if open_price else 0.0
    return pct_gap, rel_vol

@st.cache_data(show_spinner=False, ttl=300)
def get_news_headline(ticker: str):
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={today}&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, list) and js:
            # return the most recent non-empty headline
            for item in sorted(js, key=lambda x: x.get('datetime', 0), reverse=True):
                headline = item.get("headline") or item.get("title")
                if headline:
                    return headline
        return "No major news"
    except Exception:
        return "News fetch error"

def ai_playbook(ticker: str, gap: float, relvol: float, catalyst: str) -> str:
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY to generate AI playbooks."
    prompt = f"""
    Ticker: {ticker}
    Premarket Gap: {gap:.2f}%
    Relative Volume: {relvol:.2f}x
    Catalyst: {catalyst}

    Create a concise 3-sentence trading plan:
    1) Bias (long/short) with an entry idea (e.g., above premarket high or VWAP pullback).
    2) Expected duration (minutes vs multi-day) based on typical behavior for this catalyst/float.
    3) Key risks and a suggested stop guideline (e.g., VWAP loss, 1-1.5x ATR).
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.strip()}],
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI error: {e}"

st.title("🔥 AI Radar Dashboard")
st.caption("Premarket predictive scanner with catalysts & AI trading playbooks")

rows = []
for ticker in TICKERS:
    scan = premarket_scan(ticker, LOOKBACK)
    if scan:
        gap, relvol = scan
        catalyst = get_news_headline(ticker)
        playbook = ai_playbook(ticker, gap, relvol, catalyst)
        rows.append(
            {
                "Ticker": ticker,
                "Gap %": round(gap, 2),
                "RelVol": round(relvol, 2),
                "Catalyst": catalyst,
                "AI Playbook": playbook,
            }
        )

if rows:
    df = pd.DataFrame(rows).sort_values(["Gap %", "RelVol"], ascending=[False, False])
    st.dataframe(df, use_container_width=True, height=520)
else:
    st.info("No premarket data available yet. Try adding tickers or check closer to the open (4:00–9:30 ET).")

st.markdown("---")
st.write("Tip: Deploy on Streamlit Cloud → App settings → **Secrets**:")
st.code('''
OPENAI_API_KEY = "sk-..."
NEWS_API_KEY   = "YOUR_FINNHUB_KEY"
'''.strip(), language="toml")
