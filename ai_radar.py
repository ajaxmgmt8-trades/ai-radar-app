import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
import yfinance as yf
from typing import Dict, List, Optional
import numpy as np
import time
import threading
from zoneinfo import ZoneInfo  # For timezone support
import concurrent.futures

# Configure page
st.set_page_config(page_title="AI Radar Pro", layout="wide")

# Core tickers for selection
CORE_TICKERS = [
    "AAPL","NVDA","TSLA","SPY","AMD","MSFT","META","ORCL","MDB","GOOG",
    "NFLX","SPX","APP","NDX","SMCI","QUBT","IONQ","QBTS","SOFI","IBM",
    "COST","MSTR","COIN","OSCR","LYFT","JOBY","ACHR","LLY","UNH","OPEN",
    "UPST","NOW","ISRG","RR","FIG","HOOD","IBIT","WULF","WOLF","OKLO",
    "APLD","HUT","SNPS","SE","ETHU","TSM","AVGO","BITF","HIMS","BULL",
    "SPOT","LULU","CRCL","SOUN","QMMM","BMNR","SBET","GEMI","CRWV","KLAR",
    "BABA","INTC","CMG","UAMY","IREN","BBAI","BRKB","TEM","GLD","IWM","LMND",
    "CELH","PDD"
]

# ETF list for sector tracking (including SPX and NDX)
ETF_TICKERS = [
    "SPY", "QQQ", "XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC",
    "SPX", "NDX"
]

# Initialize session state
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"Default": ["AAPL", "NVDA", "TSLA", "SPY", "AMD", "MSFT"]}
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"
if "show_sparklines" not in st.session_state:
    st.session_state.show_sparklines = True
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"  # Default to ET

# API Keys
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")

openai_client = None
if OPENAI_KEY:
    try:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        st.warning(f"Could not initialize OpenAI client. Check API key and library: {e}")

# Helper function to add ticker to watchlist
def add_ticker_to_watchlist(ticker: str):
    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
    if ticker not in current_list:
        current_list.append(ticker)
        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
        st.success(f"✅ Added {ticker} to watchlist!")
        st.rerun()
    else:
        st.warning(f"{ticker} is already in the watchlist.")

# Data functions
@st.cache_data(ttl=60)
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        hist_1d = stock.history(period="1d", interval="1m")
        hist_2d = stock.history(period="2d", interval="1m")
        
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        previous_close = float(info.get('previousClose', hist_2d['Close'].iloc[-2] if len(hist_2d) >= 2 else 0))
        regular_market_open = float(info.get('regularMarketOpen', 0))
        
        premarket_change = ((regular_market_open - previous_close) / previous_close) * 100 if regular_market_open and previous_close else 0
        intraday_change = ((current_price - regular_market_open) / regular_market_open) * 100 if current_price and regular_market_open else 0
        total_change = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
        
        tz_label = "ET" if tz == "ET" else "CT"
        return {
            "last": current_price,
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "change_percent": total_change,
            "premarket_change": premarket_change,
            "intraday_change": intraday_change,
            "previous_close": previous_close,
            "market_open": regular_market_open,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None
        }
    except Exception as e:
        tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
        return {
            "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
            "change": 0.0, "change_percent": 0.0, "premarket_change": 0.0,
            "intraday_change": 0.0, "previous_close": 0.0, "market_open": 0.0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": str(e)
        }

@st.cache_data(ttl=600)
def get_finnhub_news(symbol: Optional[str] = None) -> List[Dict]:
    if not FINNHUB_KEY:
        return []
    
    try:
        if symbol:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()}&to={datetime.date.today()}&token={FINNHUB_KEY}"
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()[:10]
    except Exception as e:
        st.warning(f"Finnhub API error: {e}")
    return []

# The rest of the functions (get_polygon_news, get_all_news, etc.) remain largely the same.
# We will focus on improving the main app flow and how these functions are called.

# Main app
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Timezone toggle
col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1, label_visibility="collapsed", help="Select Timezone (ET/CT)")

# Get current time in selected TZ
tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
tz_label = st.session_state.selected_tz

# Auto-refresh controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

with col2:
    st.session_state.refresh_interval = st.selectbox("Interval", [10, 30, 60], index=1)

with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

with col4:
    current_time = current_tz.strftime("%I:%M:%S %p")
    market_open = 9 <= current_tz.hour < 16
    status = "🟢 Open" if market_open else "🔴 Closed"
    st.write(f"**{status}** | {current_time} {tz_label}")

data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
st.markdown(f"<div style='text-align: center; color: #888; font-size: 12px;'>Last Updated: {data_timestamp}</div>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    
    with st.container():
        st.markdown("### Your Watchlist")
        tickers = st.session_state.watchlists[st.session_state.active_watchlist]
        
        if not tickers:
            st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab.")
        else:
            # Use concurrent futures to fetch quotes in parallel
            quotes = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(get_live_quote, ticker, tz_label): ticker for ticker in tickers}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        quotes[ticker] = future.result()
                    except Exception as exc:
                        st.error(f"Error fetching quote for {ticker}: {exc}")
            
            for ticker in tickers:
                quote = quotes.get(ticker)
                if quote and not quote["error"]:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                    col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.write("**Bid/Ask**")
                    col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                    col3.write("**Volume**")
                    col3.write(f"{quote['volume']:,}")
                    col3.caption(f"Updated: {quote['last_updated']}")
                    
                    if abs(quote['change_percent']) >= 2.0:
                        if col4.button(f"🎯 AI Analysis", key=f"ai_{ticker}"):
                            with st.status(f"Analyzing {ticker}...", expanded=True) as status:
                                analysis = ai_playbook(ticker, quote['change_percent'])
                                status.update(label=f"🤖 {ticker} Analysis Complete", state="complete")
                                st.markdown(analysis)
                    
                    st.divider()

# TAB 2: Watchlist Manager
with tabs[1]:
    st.subheader("📋 Watchlist Manager")
    
    # Search and add
    st.markdown("### 🔍 Search & Add Stocks")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_add_ticker = st.text_input("Search stock to add", placeholder="Enter ticker", key="search_add").upper().strip()
    with col2:
        if st.button("Search & Add", key="search_add_btn") and search_add_ticker:
            quote = get_live_quote(search_add_ticker, tz_label)
            if not quote["error"]:
                add_ticker_to_watchlist(search_add_ticker)
            else:
                st.error(f"Invalid ticker: {search_add_ticker}")
    
    # The rest of the watchlist manager tab remains similar, using the add_ticker_to_watchlist function
    # for the popular tickers and other locations.

# The other tabs (Catalyst Scanner, Market Analysis, AI Playbooks) can also be updated
# to use the new `add_ticker_to_watchlist` function and the improved `st.status` UI.
# The AI functions themselves can be enhanced as discussed in the previous response,
# but the core logic remains similar.

# Auto refresh
if st.session_state.auto_refresh:
    time.sleep(0.1)
    if st.session_state.refresh_interval == 10:
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔥 AI Radar Pro | Live data: yfinance | News: Finnhub/Polygon | AI: OpenAI"
    "</div>",
    unsafe_allow_html=True
)
