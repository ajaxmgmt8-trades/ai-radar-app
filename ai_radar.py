import yfinance as yf
import pandas as pd
import streamlit as st
import requests
import snscrape.modules.x as sntwitter
from openai import OpenAI
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Radar v3.1", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")  # Finnhub

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def avg_volume(ticker, lookback=20):
    hist = yf.download(ticker, period=f"{lookback}d", interval="1d", progress=False)
    if hist.empty or "Volume" not in hist:
        return None
    return float(hist["Volume"].mean())

def scan_24h(ticker):
    """Scan last 24h price action (premarket + intraday + postmarket)."""
    data = yf.download(ticker, period="2d", interval="5m", prepost=True, progress=False)
    if data.empty:
        return None
    last_price = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    pct_change = (last_price - prev_close) / prev_close * 100
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return pct_change, rel_vol

# =========================
# NEWS SOURCES
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def get_finnhub_news(ticker: str):
    """Get company news from Finnhub."""
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={today}&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, list) and js:
            for item in sorted(js, key=lambda x: x.get("datetime", 0), reverse=True):
                headline = item.get("headline") or item.get("title")
                if headline:
                    return headline
        return "No major Finnhub news"
    except Exception:
        return "Finnhub error"

def get_twitter_news(ticker, limit=2):
    """Scrape tweets mentioning a ticker like $TSLA."""
    try:
        query = f"${ticker} since:{(datetime.utcnow() - timedelta(days=1)).date()}"
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit: break
            tweets.append(tweet.content)
        return tweets if tweets else ["No fresh Twitter chatter"]
    except Exception as e:
        return [f"Twitter error: {e}"]

def get_twitter_accounts(accounts, limit=2):
    """Scrape tweets from specific accounts (list of usernames)."""
    tweets = []
    try:
        for account in accounts:
            query = f"from:{account} since:{(datetime.utcnow() - timedelta(days=1)).date()}"
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= limit: break
                tweets.append(f"{account}: {tweet.content}")
        return tweets if tweets else ["No fresh tweets from watchlist"]
    except Exception as e:
        return [f"Twitter account error: {e}"]

def get_combined_news(ticker, use_finnhub=True, use_twitter=True, use_accounts=False, accounts=[]):
    """Merge chosen news sources into one catalyst string."""
    sources = []
    if use_finnhub:
        sources.append(f"Finnhub: {get_finnhub_news(ticker)}")
    if use_twitter:
        tw = get_twitter_news(ticker, limit=1)
        sources.append(f"Twitter: {tw[0]}")
    if use_accounts and accounts:
        acct_tweets = get_twitter_accounts(accounts, limit=1)
        sources.append(f"Accounts: {acct_tweets[0]}")
    return " | ".join(sources) if sources else "No news selected"

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, change, relvol, catalyst):
    if not OPENAI_API_KEY:
        return "Add OPENAI_API_KEY in Secrets."

    change = float(change) if change is not None else 0.0
    relvol = float(relvol) if relvol is not None else 0.0

    prompt = f"""
    Ticker: {ticker}
    24h Change: {change:.2f}%
    RelVol: {relvol:.2f}x
    Catalyst: {catalyst}

    Generate a 3-sentence trading playbook:
    1) Bias (long/short).
    2) Expected duration (scalp vs swing).
    3) Risks (fade, IV crush, market pullback).
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
# TOP MOVERS
# =========================
@st.cache_data(show_spinner=True, ttl=300)
def get_top_movers(limit=10):
    movers = []
    try:
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        symbols = r.json()[:200]
        tickers = [s["symbol"] for s in symbols]

        for t in tickers:
            try:
                q = requests.get(f"https://finnhub.io/api/v1/quote?symbol={t}&token={NEWS_API_KEY}").json()
                pct = ((q["c"] - q["pc"]) / q["pc"]) * 100 if q.get("pc") else 0
                vol = q.get("v", 0)
                if abs(pct) >= 5 and vol > 500000:
                    movers.append((t, pct, vol))
            except Exception:
                continue
    except Exception:
        return ["AAPL","NVDA","TSLA","SPY"]

    movers = sorted(movers, key=lambda x: abs(x[1]), reverse=True)[:limit]
    return [m[0] for m in movers]

def scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts):
    rows = []
    for t in tickers:
        scan = scan_24h(t)
        if not scan: continue
        change, relvol = scan
        catalyst = get_combined_news(t, use_finnhub, use_twitter, use_accounts, accounts)
        playbook = ai_playbook(t, change, relvol, catalyst)
        rows.append([t, round(change, 2), round(relvol, 2), catalyst, playbook])
    return pd.DataFrame(rows, columns=["Ticker", "Change %", "RelVol", "Catalyst", "AI Playbook"])

# =========================
# STREAMLIT UI
# =========================
st.title("🔥 AI Radar v3.1 — 24h Scanner + Multi-Source News")

# --- News source toggles ---
use_finnhub = st.sidebar.checkbox("Use Finnhub", value=True)
use_twitter = st.sidebar.checkbox("Use Twitter by Ticker", value=True)
use_accounts = st.sidebar.checkbox("Use Twitter Accounts", value=False)

accounts_list = []
if use_accounts:
    accounts_input = st.sidebar.text_area("Enter Twitter accounts (comma separated)", "tradytics, unusual_whales, benzinga")
    accounts_list = [a.strip() for a in accounts_input.split(",")]

# --- Search box ---
search_ticker = st.text_input("🔍 Search a ticker for instant analysis (e.g. TSLA, NVDA, SPY)")
if search_ticker:
    st.subheader(f"Search Results for {search_ticker.upper()}")
    df = scan_list([search_ticker.upper()], use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df, use_container_width=True)

# --- Auto scanner tabs ---
tabs = st.tabs(["📊 Premarket", "💥 Intraday", "🌙 Postmarket"])
tickers = get_top_movers(limit=10)

with tabs[0]:
    st.subheader("Premarket Movers (4am–9:30am)")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.subheader("Intraday Explosives (9:30am–4pm)")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.subheader("Postmarket Movers + Earnings")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df, use_container_width=True)
