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
from zoneinfo import ZoneInfo
import concurrent.futures
import google.generativeai as genai
import openai

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
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Initialize AI clients
openai_client = None
if OPENAI_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        st.warning(f"Could not initialize OpenAI client. Check API key and library: {e}")

gemini_model = None
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.warning(f"Could not initialize Gemini model. Check API key and library: {e}")

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

# Data functions (unchanged from previous version)
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
            "change": float(info.get('regularMarketChange', current_price - previous_close if previous_close else 0)),
            "change_percent": total_change,
            "premarket_change": premarket_change,
            "intraday_change": intraday_change,
            "postmarket_change": 0,
            "previous_close": previous_close,
            "market_open": regular_market_open,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None
        }
    except Exception as e:
        tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
        return {
            "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
            "change": 0.0, "change_percent": 0.0,
            "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
            "previous_close": 0.0, "market_open": 0.0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": str(e)
        }

@st.cache_data(ttl=600)
def get_finnhub_news(symbol: str = None) -> List[Dict]:
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

@st.cache_data(ttl=600)
def get_polygon_news() -> List[Dict]:
    if not POLYGON_KEY:
        return []
    
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/reference/news?published_utc.gte={today}&limit=20&apikey={POLYGON_KEY}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
    except Exception as e:
        st.warning(f"Polygon API error: {e}")
    
    return []

def get_all_news() -> List[Dict]:
    all_news = []
    
    finnhub_news = get_finnhub_news()
    for item in finnhub_news:
        all_news.append({
            "title": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": "Finnhub",
            "url": item.get("url", ""),
            "datetime": item.get("datetime", 0),
            "related": item.get("related", "")
        })
    
    polygon_news = get_polygon_news()
    for item in polygon_news:
        all_news.append({
            "title": item.get("title", ""),
            "summary": item.get("description", ""),
            "source": "Polygon",
            "url": item.get("article_url", ""),
            "datetime": item.get("published_utc", ""),
            "related": ",".join(item.get("tickers", []))
        })
    
    try:
        all_news.sort(key=lambda x: x["datetime"], reverse=True)
    except:
        pass
    
    return all_news[:15]

def analyze_news_sentiment(title: str, summary: str = "") -> tuple:
    text = (title + " " + summary).lower()
    
    explosive_keywords = ["surge", "soars", "jumps", "rocket", "breakthrough", "beats", "record", "acquisition", "merger"]
    bullish_keywords = ["up", "rise", "gain", "positive", "strong", "growth", "partnership", "approval", "bullish", "buy"]
    bearish_keywords = ["down", "fall", "drop", "weak", "decline", "loss", "warning", "delay", "bearish", "sell"]
    
    explosive_score = sum(2 for word in explosive_keywords if word in text)
    bullish_score = sum(1 for word in bullish_keywords if word in text)
    bearish_score = sum(1 for word in bearish_keywords if word in text)
    
    total_score = explosive_score + bullish_score + bearish_score
    
    if explosive_score >= 2:
        return "🚀 EXPLOSIVE", min(95, 60 + explosive_score * 10)
    elif explosive_score >= 1:
        return "📈 Bullish", min(85, 50 + explosive_score * 15)
    elif bearish_score >= 2:
        return "📉 Bearish", min(80, 40 + bearish_score * 15)
    elif bullish_score >= 2:
        return "📈 Bullish", min(75, 35 + bullish_score * 10)
    else:
        return "⚪ Neutral", max(10, min(50, total_score * 5))

def ai_playbook(ticker: str, change: float, catalyst: str = "", model_choice: str = "Gemini") -> str:
    if model_choice == "OpenAI" and not openai_client:
        return f"**{ticker} Analysis** (OpenAI API not configured)"
    if model_choice == "Gemini" and not gemini_model:
        return f"**{ticker} Analysis** (Gemini API not configured)"
    
    try:
        prompt = f"""
        Analyze {ticker} with {change:+.2f}% change today.
        Catalyst: {catalyst if catalyst else "Market movement"}
        
        Provide trading analysis with:
        1. Sentiment (Bullish/Bearish/Neutral) with confidence
        2. Scalp setup (1-5 min timeframe)
        3. Day trade setup (15-30 min)
        4. Swing setup (4H-Daily)
        5. Key levels to watch
        
        Keep concise and actionable, under 250 words.
        """
        
        if model_choice == "OpenAI":
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        else: # Gemini
            response = gemini_model.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"AI Error: {str(e)}"

def ai_market_analysis(news_items: List[Dict], movers: List[Dict], model_choice: str = "Gemini") -> str:
    if model_choice == "OpenAI" and not openai_client:
        return "OpenAI API not configured for AI analysis."
    if model_choice == "Gemini" and not gemini_model:
        return "Gemini API not configured for AI analysis."
    
    try:
        news_context = "\n".join([f"- {item['title']}" for item in news_items[:5]])
        movers_context = "\n".join([f"- {m['ticker']}: {m['change_pct']:+.2f}%" for m in movers[:5]])
        
        prompt = f"""
        Analyze current market conditions based on:

        Top News Headlines:
        {news_context}

        Top Market Movers:
        {movers_context}

        Provide a brief market analysis covering:
        1. Overall market sentiment
        2. Key themes driving movement
        3. Sectors to watch
        4. Trading opportunities

        Keep it under 200 words and actionable.
        """
        
        if model_choice == "OpenAI":
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        else: # Gemini
            response = gemini_model.generate_content(prompt)
            return response.text
    
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"

def ai_auto_generate_plays(tz: str, model_choice: str) -> List[Dict]:
    plays = []
    
    try:
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        candidates = []
        
        for ticker in scan_tickers:
            quote = get_live_quote(ticker, tz)
            if not quote["error"] and abs(quote["change_percent"]) >= 1.5:
                candidates.append({
                    "ticker": ticker,
                    "quote": quote,
                    "significance": abs(quote["change_percent"])
                })
        
        candidates.sort(key=lambda x: x["significance"], reverse=True)
        top_candidates = candidates[:5]
        
        for candidate in top_candidates:
            ticker = candidate["ticker"]
            quote = candidate["quote"]
            news = get_finnhub_news(ticker)
            catalyst = news[0].get('headline', '')[:100] + "..." if news else ""
            
            if (model_choice == "OpenAI" and openai_client) or \
               (model_choice == "Gemini" and gemini_model):
                try:
                    play_prompt = f"""
                    Generate a concise trading play for {ticker}:
                    
                    Current Price: ${quote['last']:.2f}
                    Change: {quote['change_percent']:+.2f}%
                    Premarket: {quote['premarket_change']:+.2f}%
                    Intraday: {quote['intraday_change']:+.2f}%
                    After Hours: {quote['postmarket_change']:+.2f}%
                    Volume: {quote['volume']:,}
                    Catalyst: {catalyst if catalyst else "Market movement"}
                    
                    Provide:
                    1. Play type (Scalp/Day/Swing)
                    2. Entry strategy and levels
                    3. Target and stop levels
                    4. Risk/reward ratio
                    5. Confidence (1-10)
                    
                    Keep under 200 words, be specific and actionable.
                    """
                    
                    if model_choice == "OpenAI":
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": play_prompt}],
                            temperature=0.3,
                            max_tokens=300
                        )
                        play_analysis = response.choices[0].message.content
                    else: # Gemini
                        response = gemini_model.generate_content(play_prompt)
                        play_analysis = response.text
                
                except Exception as ai_error:
                    play_analysis = f"""
                    **{ticker} Trading Opportunity**
                    
                    **Movement:** {quote['change_percent']:+.2f}% change with {quote['volume']:,} volume
                    
                    **Session Breakdown:**
                    • Premarket: {quote['premarket_change']:+.2f}%
                    • Intraday: {quote['intraday_change']:+.2f}%"
                    • After Hours: {quote['postmarket_change']:+.2f}%
                    
                    **Quick Setup:** Watch for continuation or reversal around current levels
                    
                    *AI analysis unavailable: {str(ai_error)[:50]}...*
                    """
            else:
                direction = "bullish" if quote['change_percent'] > 0 else "bearish"
                play_analysis = f"""
                **{ticker} Trading Setup**
                
                **Movement:** {quote['change_percent']:+.2f}% change showing {direction} momentum
                
                **Session Analysis:**
                • Premarket: {quote['premarket_change']:+.2f}%
                • Intraday: {quote['intraday_change']:+.2f}%
                • After Hours: {quote['postmarket_change']:+.2f}%
                
                **Volume:** {quote['volume']:,} shares
                
                **Setup:** Monitor for continuation or reversal. Consider risk management around current price levels.
                
                *Configure API for detailed AI analysis*
                """
            
            play = {
                "ticker": ticker,
                "current_price": quote['last'],
                "change_percent": quote['change_percent'],
                "session_data": {
                    "premarket": quote['premarket_change'],
                    "intraday": quote['intraday_change'],
                    "afterhours": quote['postmarket_change']
                },
                "catalyst": catalyst if catalyst else f"Market movement: {quote['change_percent']:+.2f}%",
                "play_analysis": play_analysis,
                "volume": quote['volume'],
                "timestamp": quote['last_updated']
            }
            plays.append(play)
        return plays
    except Exception as e:
        st.error(f"Error generating auto plays: {str(e)}")
        return []

# Main app
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Sidebar for AI model selection
with st.sidebar:
    st.header("AI Settings")
    ai_options = []
    if gemini_model:
        ai_options.append("Gemini")
    if openai_client:
        ai_options.append("OpenAI")
        
    if not ai_options:
        st.error("No AI models available. Please configure your API keys in `.streamlit/secrets.toml`.")
        selected_ai_model = None
    else:
        selected_ai_model = st.selectbox("Select AI Model", ai_options)

col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed", help="Select Timezone (ET/CT)")

tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
tz_label = st.session_state.selected_tz

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

tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks"])

data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
st.markdown(f"<div style='text-align: center; color: #888; font-size: 12px;'>Last Updated: {data_timestamp}</div>", unsafe_allow_html=True)

with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    
    current_tz_hour = current_tz.hour
    if 4 <= current_tz_hour < 9:
        session_status = "🌅 Premarket"
    elif 9 <= current_tz_hour < 16:
        session_status = "🟢 Market Open"
    else:
        session_status = "🌆 After Hours"
    
    st.markdown(f"**Trading Session ({tz_label}):** {session_status}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Search Individual Stock", placeholder="Enter ticker", key="search_quotes").upper().strip()
    with col2:
        search_quotes = st.button("Get Quote", key="search_quotes_btn")
    
    if search_quotes and search_ticker:
        with st.status(f"Getting quote for {search_ticker}...", expanded=True) as status:
            quote = get_live_quote(search_ticker, tz_label)
            if not quote["error"]:
                status.update(label=f"Quote for {search_ticker} - Updated: {quote['last_updated']}", state="complete")
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                col1.metric(search_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.metric("Volume", f"{quote['volume']:,}")
                
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                if col4.button(f"Add {search_ticker} to Watchlist", key="add_searched"):
                    add_ticker_to_watchlist(search_ticker)
                st.divider()
            else:
                status.update(label=f"Could not get quote for {search_ticker}: {quote['error']}", state="error")
                st.error(f"Could not get quote for {search_ticker}: {quote['error']}")
    
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab.")
    else:
        st.markdown("### Your Watchlist")
        
        quotes = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tickers)) as executor:
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
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                    col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.write("**Bid/Ask**")
                    col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                    col3.write("**Volume**")
                    col3.write(f"{quote['volume']:,}")
                    col3.caption(f"Updated: {quote['last_updated']}")
                    
                    if abs(quote['change_percent']) >= 2.0 and selected_ai_model:
                        if col4.button(f"🎯 AI Analysis", key=f"ai_{ticker}"):
                            with st.status(f"Analyzing {ticker} with {selected_ai_model}...", expanded=True) as status:
                                analysis = ai_playbook(ticker, quote['change_percent'], model_choice=selected_ai_model)
                                status.update(label=f"🤖 {ticker} Analysis Complete", state="complete")
                                st.markdown(analysis)
                    
                    sess_col1, sess_col2, sess_col3, sess_col4 = st.columns([2, 2, 2, 4])
                    sess_col1.caption(f"**PM:** {quote['premarket_change']:+.2f}%")
                    sess_col2.caption(f"**Day:** {quote['intraday_change']:+.2f}%")
                    sess_col3.caption(f"**AH:** {quote['postmarket_change']:+.2f}%")
                    
                    with st.expander(f"🔎 Expand {ticker}"):
                        news = get_finnhub_news(ticker)
                        if news:
                            st.write("### 📰 Catalysts (last 24h)")
                            for n in news:
                                st.write(f"- [{n.get('headline', 'No title')}]({n.get('url', '#')}) ({n.get('source', 'Finnhub')})")
                        else:
                            st.info("No recent news.")
                        
                        st.markdown("### 🎯 AI Playbook")
                        if selected_ai_model:
                            catalyst_title = news[0].get('headline', '') if news else ""
                            st.markdown(ai_playbook(ticker, quote['change_percent'], catalyst_title, model_choice=selected_ai_model))
                        else:
                            st.info("Please select an AI model from the sidebar.")
                    
                    st.divider()
            else:
                st.error(f"Could not get quote for {ticker}: {quote['error']}")


with tabs[1]:
    st.subheader("📋 Watchlist Manager")
    
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
    
    st.markdown("### 📋 Manage Watchlists")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_watchlist = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
        st.session_state.active_watchlist = selected_watchlist
    
    with col2:
        new_watchlist = st.text_input("New Watchlist Name")
        if st.button("Create Watchlist") and new_watchlist:
            st.session_state.watchlists[new_watchlist] = []
            st.session_state.active_watchlist = new_watchlist
            st.rerun()
    
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist].copy()
    unique_current_tickers = list(dict.fromkeys(current_tickers))
    if len(unique_current_tickers) != len(current_tickers):
        st.session_state.watchlists[st.session_state.active_watchlist] = unique_current_tickers
        current_tickers = unique_current_tickers
        st.rerun()
    
    st.markdown("### ⭐ Popular Tickers")
    cols = st.columns(6)
    for i, ticker in enumerate(CORE_TICKERS):
        with cols[i % 6]:
            if st.button(f"+ {ticker}", key=f"add_{ticker}"):
                add_ticker_to_watchlist(ticker)
    
    st.markdown("### 📊 Current Watchlist")
    if current_tickers:
        for i in range(0, len(current_tickers), 5):
            cols = st.columns(5)
            for j, ticker in enumerate(current_tickers[i:i+5]):
                with cols[j]:
                    st.write(f"**{ticker}**")
                    if st.button(f"Remove", key=f"remove_{ticker}"):
                        current_tickers.remove(ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                        st.rerun()
    else:
        st.info("Watchlist is empty. Search for stocks above or add popular tickers.")

with tabs[2]:
    st.subheader("🔥 Real-Time Catalyst Scanner")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_catalyst_ticker = st.text_input("🔍 Search catalysts for stock", placeholder="Enter ticker", key="search_catalyst").upper().strip()
    with col2:
        search_catalyst = st.button("Search Catalysts", key="search_catalyst_btn")
    
    if search_catalyst and search_catalyst_ticker:
        with st.status(f"Searching catalysts for {search_catalyst_ticker}...", expanded=True) as status:
            specific_news = get_finnhub_news(search_catalyst_ticker)
            quote = get_live_quote(search_catalyst_ticker, tz_label)
            
            if not quote["error"]:
                status.update(label=f"Catalyst Analysis for {search_catalyst_ticker} - Updated: {quote['last_updated']}", state="complete")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                if col3.button(f"Add {search_catalyst_ticker} to WL", key="add_catalyst_search"):
                    add_ticker_to_watchlist(search_catalyst_ticker)
                
                if specific_news:
                    st.markdown("### News Catalysts")
                    for news_item in specific_news[:5]:
                        sentiment, confidence = analyze_news_sentiment(news_item.get("headline", ""), news_item.get("summary", ""))
                        
                        with st.expander(f"{sentiment} ({confidence}%) - {news_item.get('headline', 'No title')[:80]}..."):
                            st.write(f"**Summary:** {news_item.get('summary', 'No summary')}")
                            if news_item.get('url'):
                                st.markdown(f"[Read Article]({news_item['url']})")
                else:
                    st.info(f"No recent news found for {search_catalyst_ticker}")
                
                st.divider()
            else:
                status.update(label=f"Could not get quote for {search_catalyst_ticker}: {quote['error']}", state="error")
                st.error(f"Could not get quote for {search_catalyst_ticker}: {quote['error']}")
    
    if st.button("🔍 Scan for Market Catalysts", type="primary"):
        with st.status("Scanning for catalysts...", expanded=True) as status:
            all_news = get_all_news()
            movers = []
            for ticker in CORE_TICKERS[:20]:
                quote = get_live_quote(ticker, tz_label)
                if not quote["error"] and abs(quote["change_percent"]) >= 1.5:
                    movers.append({"ticker": ticker,"change_pct": quote["change_percent"],"price": quote["last"],"volume": quote["volume"]})
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            status.update(label="Scanning for catalysts... Complete", state="complete")
            
            st.markdown("### 📰 News Catalysts")
            for news in all_news[:10]:
                sentiment, confidence = analyze_news_sentiment(news["title"], news["summary"])
                with st.expander(f"{sentiment} ({confidence}%) - {news['title'][:100]}..."):
                    st.write(f"**Source:** {news['source']}")
                    st.write(f"**Summary:** {news['summary']}")
                    if news["related"]:
                        st.write(f"**Related Tickers:** {news['related']}")
                    if news["url"]:
                        st.markdown(f"[Read Article]({news['url']})")
            
            st.markdown("### 📊 Significant Market Moves")
            for mover in movers[:10]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    direction = "🚀" if mover["change_pct"] > 0 else "📉"
                    st.metric(f"{direction} {mover['ticker']}", f"${mover['price']:.2f}",f"{mover['change_pct']:+.2f}%")
                with col2:
                    if st.button(f"Add to WL", key=f"add_mover_{mover['ticker']}"):
                        add_ticker_to_watchlist(mover['ticker'])

with tabs[3]:
    st.subheader("📈 AI Market Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_analysis_ticker = st.text_input("🔍 Analyze specific stock", placeholder="Enter ticker", key="search_analysis").upper().strip()
    with col2:
        search_analysis = st.button("Analyze Stock", key="search_analysis_btn")
    
    if search_analysis and search_analysis_ticker:
        if selected_ai_model:
            with st.status(f"AI analyzing {search_analysis_ticker} with {selected_ai_model}...", expanded=True) as status:
                quote = get_live_quote(search_analysis_ticker, tz_label)
                if not quote["error"]:
                    news = get_finnhub_news(search_analysis_ticker)
                    catalyst = news[0].get('headline', '') if news else "Recent market movement"
                    analysis = ai_playbook(search_analysis_ticker, quote["change_percent"], catalyst, model_choice=selected_ai_model)
                    status.update(label=f"🤖 AI Analysis: {search_analysis_ticker} - Updated: {quote['last_updated']}", state="complete")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.metric("Volume", f"{quote['volume']:,}")
                    col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                    if col4.button(f"Add {search_analysis_ticker} to WL", key="add_analysis_search"):
                        add_ticker_to_watchlist(search_analysis_ticker)
                    
                    st.markdown("#### Session Performance")
                    sess_col1, sess_col2, sess_col3 = st.columns(3)
                    sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                    sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                    sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                    
                    st.markdown("### 🎯 AI Analysis")
                    st.markdown(analysis)
                    
                    if news:
                        with st.expander(f"📰 Recent News Context"):
                            for item in news[:3]:
                                st.write(f"**{item.get('headline', 'No title')}**")
                                st.write(item.get('summary', 'No summary')[:200] + "...")
                                st.write("---")
                    st.divider()
                else:
                    status.update(label=f"Could not analyze {search_analysis_ticker}: {quote['error']}", state="error")
                    st.error(f"Could not analyze {search_analysis_ticker}: {quote['error']}")
        else:
            st.info("Please select an AI model from the sidebar to analyze a stock.")
    
    if st.button("🤖 Generate Market Analysis", type="primary"):
        if selected_ai_model:
            with st.status(f"AI analyzing market conditions with {selected_ai_model}...", expanded=True) as status:
                news_items = get_all_news()
                movers = []
                for ticker in CORE_TICKERS[:15]:
                    quote = get_live_quote(ticker, tz_label)
                    if not quote["error"]:
                        movers.append({"ticker": ticker,"change_pct": quote["change_percent"],"price": quote["last"]})
                
                analysis = ai_market_analysis(news_items, movers, model_choice=selected_ai_model)
                status.update(label="🤖 AI Market Analysis Complete", state="complete")
                
                st.markdown(analysis)
                
                with st.expander("📊 Supporting Data"):
                    st.write("**Top Market Movers:**")
                    for mover in sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]:
                        st.write(f"• {mover['ticker']}: {mover['change_pct']:+.2f}%")
                    st.write("**Key News Headlines:**")
                    for news in news_items[:3]:
                        st.write(f"• {news['title']}")
        else:
            st.info("Please select an AI model from the sidebar to generate a market analysis.")

with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    
    st.markdown("### 🎯 Auto-Generated Trading Plays")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("AI automatically scans your watchlist and market movers to suggest trading opportunities")
    with col2:
        if st.button("🚀 Generate Auto Plays", type="primary"):
            if selected_ai_model:
                with st.status(f"AI generating trading plays from market scan with {selected_ai_model}...", expanded=True) as status:
                    auto_plays = ai_auto_generate_plays(tz_label, model_choice=selected_ai_model)
                    if auto_plays:
                        status.update(label=f"🤖 Generated {len(auto_plays)} Trading Plays", state="complete")
                        for i, play in enumerate(auto_plays):
                            with st.expander(f"🎯 {play['ticker']} - ${play['current_price']:.2f} ({play['change_percent']:+.2f}%)"):
                                sess_col1, sess_col2, sess_col3 = st.columns(3)
                                sess_col1.metric("Premarket", f"{play['session_data']['premarket']:+.2f}%")
                                sess_col2.metric("Intraday", f"{play['session_data']['intraday']:+.2f}%")
                                sess_col3.metric("After Hours", f"{play['session_data']['afterhours']:+.2f}%")
                                if play['catalyst']:
                                    st.write(f"**Catalyst:** {play['catalyst']}")
                                st.markdown("**AI Trading Play:**")
                                st.markdown(play['play_analysis'])
                                if st.button(f"Add {play['ticker']} to Watchlist", key=f"add_auto_play_{i}"):
                                    add_ticker_to_watchlist(play['ticker'])
                    else:
                        st.info("No significant trading opportunities detected at this time. Market conditions may be consolidating.")
                        status.update(label="No significant trading opportunities detected.", state="complete")
            else:
                st.info("Please select an AI model from the sidebar to generate plays.")
    
    st.divider()
    
    st.markdown("### 🔍 Custom Stock Analysis")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_playbook_ticker = st.text_input("🔍 Generate playbook for any stock", placeholder="Enter ticker", key="search_playbook").upper().strip()
    with col2:
        search_playbook = st.button("Generate Playbook", key="search_playbook_btn")
    
    if search_playbook and search_playbook_ticker:
        if selected_ai_model:
            with st.status(f"AI generating playbook for {search_playbook_ticker} with {selected_ai_model}...", expanded=True) as status:
                quote = get_live_quote(search_playbook_ticker, tz_label)
                if not quote["error"]:
                    news = get_finnhub_news(search_playbook_ticker)
                    catalyst = news[0].get('headline', '') if news else ""
                    playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst, model_choice=selected_ai_model)
                    
                    status.update(label=f"✅ {search_playbook_ticker} Trading Playbook - Updated: {quote['last_updated']}", state="complete")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                    col3.metric("Volume", f"{quote['volume']:,}")
                    if col4.button(f"Add {search_playbook_ticker} to WL", key="add_playbook_search"):
                        add_ticker_to_watchlist(search_playbook_ticker)
                    
                    st.markdown("#### Session Performance")
                    sess_col1, sess_col2, sess_col3 = st.columns(3)
                    sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                    sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                    sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                    
                    st.markdown("### 🎯 AI Trading Playbook")
                    st.markdown(playbook)
                    
                    if news:
                        with st.expander(f"📰 Recent News for {search_playbook_ticker}"):
                            for item in news[:3]:
                                st.write(f"**{item.get('headline', 'No title')}**")
                                st.write(item.get('summary', 'No summary')[:200] + "...")
                                st.write("---")
                    st.divider()
                else:
                    status.update(label=f"Could not get data for {search_playbook_ticker}: {quote['error']}", state="error")
                    st.error(f"Could not get data for {search_playbook_ticker}: {quote['error']}")
        else:
            st.info("Please select an AI model from the sidebar to generate a playbook.")
    
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if tickers:
        st.markdown("### 📋 Watchlist Playbooks")
        selected_ticker = st.selectbox("Select from watchlist", tickers, key="watchlist_playbook")
        catalyst_input = st.text_input("Catalyst (optional)", placeholder="News event, etc.", key="catalyst_input")
        
        if st.button("🤖 Generate Watchlist Playbook", type="secondary"):
            if selected_ai_model:
                with st.status(f"AI analyzing {selected_ticker} with {selected_ai_model}...", expanded=True) as status:
                    quote = get_live_quote(selected_ticker, tz_label)
                    
                    if not quote["error"]:
                        playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input, model_choice=selected_ai_model)
                        
                        status.update(label=f"✅ {selected_ticker} Trading Playbook - Updated: {quote['last_updated']}", state="complete")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                        col3.metric("Volume", f"{quote['volume']:,}")
                        
                        st.markdown("#### Session Breakdown")
                        sess_col1, sess_col2, sess_col3 = st.columns(3)
                        sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                        sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                        sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                        
                        st.markdown("### 🎯 AI Analysis")
                        st.markdown(playbook)
                        
                        news = get_finnhub_news(selected_ticker)
                        if news:
                            with st.expander(f"📰 Recent News for {selected_ticker}"):
                                for item in news[:3]:
                                    st.write(f"**{item.get('headline', 'No title')}**")
                                    st.write(item.get('summary', 'No summary')[:200] + "...")
                                    st.write("---")
                    else:
                        status.update(label=f"Could not get data for {selected_ticker}: {quote['error']}", state="error")
            else:
                st.info("Please select an AI model from the sidebar to generate a playbook.")
    else:
        st.info("Add stocks to watchlist or use search above.")
    
    with st.expander("💡 About Auto-Generated Plays"):
        st.markdown("""
        **Auto-Generated Plays** scan your watchlist and top market movers to identify:
        - Stocks with significant price movements (>1.5%)
        - High relative volume situations
        - Recent news catalysts
        - Technical setups with good risk/reward ratios
        
        Each play includes:
        - Specific entry, target, and stop levels
        - Play type (scalp, day trade, swing)
        - Risk/reward analysis
        - AI confidence rating
        
        **Note:** These are AI-generated suggestions for educational purposes. Always do your own research and risk management.
        """)

if st.session_state.auto_refresh:
    time.sleep(0.1)
    if st.session_state.refresh_interval == 10:
        st.rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔥 AI Radar Pro | Live data: yfinance | News: Finnhub/Polygon | AI: OpenAI/Gemini"
    "</div>",
    unsafe_allow_html=True
)
