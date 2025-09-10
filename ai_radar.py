import yfinance as yf
import pandas as pd
import streamlit as st
import requests
import importlib
from openai import OpenAI
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Radar Pro",
    layout="wide",
    page_icon="🔥"
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")  # Finnhub

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# SNSCRAPE IMPORT (bulletproof)
# =========================
sntwitter = None
for mod in ["snscrape.modules.twitter", "snscrape.modules.x"]:
    try:
        sntwitter = importlib.import_module(mod)
        break
    except ImportError:
        continue
if not sntwitter:
    raise ImportError("snscrape is not installed correctly. Check requirements.txt")

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
    data = yf.download(ticker, period="2d", interval="5m", prepost=True, progress=False)
    if data.empty: return None
    last_price = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    pct_change = (last_price - prev_close) / prev_close * 100
    rel_vol = data["Volume"].sum() / (avg_volume(ticker) or 1)
    return pct_change, rel_vol

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
            headline = r[0].get("headline") or r[0].get("title")
            return headline
        return "No major Finnhub news"
    except:
        return "Finnhub error"

def get_twitter_news(ticker, limit=2):
    try:
        query = f"${ticker} since:{(datetime.utcnow() - timedelta(days=1)).date()}"
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit: break
            tweets.append(tweet.content)
        return tweets if tweets else ["No fresh Twitter chatter"]
    except Exception as e:
        return [f"Twitter error: {e}"]

def get_twitter_accounts(accounts, limit=3):
    tweets = []
    try:
        for account in accounts:
            query = f"from:{account} since:{(datetime.utcnow() - timedelta(days=1)).date()}"
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= limit: break
                tweets.append(f"@{account}: {tweet.content}")
        return tweets if tweets else ["No fresh tweets from accounts"]
    except Exception as e:
        return [f"Twitter account error: {e}"]

def get_combined_news(ticker, use_finnhub=True, use_twitter=True, use_accounts=False, accounts=[]):
    sources = []
    if use_finnhub: sources.append(f"Finnhub: {get_finnhub_news(ticker)}")
    if use_twitter: sources.append(f"Twitter: {get_twitter_news(ticker,1)[0]}")
    if use_accounts and accounts:
        acct_tweet = get_twitter_accounts(accounts,1)
        sources.append(f"Accounts: {acct_tweet[0]}")
    return " | ".join(sources) if sources else "No news selected"

# =========================
# AI PLAYBOOK
# =========================
def ai_playbook(ticker, change, relvol, catalyst):
    if not OPENAI_API_KEY: return "Add OPENAI_API_KEY in Secrets."
    change = float(change or 0.0)
    relvol = float(relvol or 0.0)

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
def get_top_movers(limit=10):
    # Placeholder: static list until Polygon/Benzinga added
    return ["AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG"]

def scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts):
    rows = []
    for t in tickers:
        scan = scan_24h(t)
        if not scan: continue
        change, relvol = scan
        catalyst = get_combined_news(t, use_finnhub, use_twitter, use_accounts, accounts)
        playbook = ai_playbook(t, change, relvol, catalyst)
        rows.append([t, round(change,2), round(relvol,2), catalyst, playbook])
    return pd.DataFrame(rows, columns=["Ticker","Change %","RelVol","Catalyst","AI Playbook"])

# =========================
# STREAMLIT UI
# =========================
st.markdown(
    "<h1 style='text-align: center; color: orange;'>🔥 AI Radar Pro — Market Scanner</h1>",
    unsafe_allow_html=True
)
st.caption("Premarket, Intraday, Postmarket, Twitter Feed, and AI Playbooks")

# Sidebar
st.sidebar.header("⚙️ News Settings")
use_finnhub = st.sidebar.checkbox("Use Finnhub", value=True)
use_twitter = st.sidebar.checkbox("Use Twitter by Ticker", value=True)
use_accounts = st.sidebar.checkbox("Use Twitter Accounts", value=False)

accounts_list = []
if use_accounts:
    accounts_input = st.sidebar.text_area(
        "Enter Twitter accounts (comma separated)",
        "tradytics, unusual_whales, benzinga"
    )
    accounts_list = [a.strip() for a in accounts_input.split(",")]

# Search box
search_ticker = st.text_input("🔍 Search a ticker (e.g. TSLA, NVDA, SPY)")
if search_ticker:
    st.subheader(f"Search Results for {search_ticker.upper()}")
    df = scan_list([search_ticker.upper()], use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df.style.format({
        "Change %": "{:+.2f}%",
        "RelVol": "{:.2f}x"
    }).background_gradient(subset=["Change %"], cmap="RdYlGn"), use_container_width=True)

# Tabs
tabs = st.tabs(["📊 Premarket","💥 Intraday","🌙 Postmarket","🐦 Twitter Feed"])
tickers = get_top_movers(limit=8)

with tabs[0]:
    st.subheader("Premarket Movers")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df.style.format({
        "Change %": "{:+.2f}%",
        "RelVol": "{:.2f}x"
    }).background_gradient(subset=["Change %"], cmap="RdYlGn"), use_container_width=True)

with tabs[1]:
    st.subheader("Intraday Movers")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df.style.format({
        "Change %": "{:+.2f}%",
        "RelVol": "{:.2f}x"
    }).background_gradient(subset=["Change %"], cmap="RdYlGn"), use_container_width=True)

with tabs[2]:
    st.subheader("Postmarket Movers")
    df = scan_list(tickers, use_finnhub, use_twitter, use_accounts, accounts_list)
    st.dataframe(df.style.format({
        "Change %": "{:+.2f}%",
        "RelVol": "{:.2f}x"
    }).background_gradient(subset=["Change %"], cmap="RdYlGn"), use_container_width=True)

with tabs[3]:
    st.subheader("🐦 Twitter Feed (Watchlist Accounts)")
    tweets = get_twitter_accounts(accounts_list if accounts_list else ["tradytics","unusual_whales"], limit=5)
    for tw in tweets:
        st.markdown(f"<p style='font-size:14px'>👉 {tw}</p>", unsafe_allow_html=True)
