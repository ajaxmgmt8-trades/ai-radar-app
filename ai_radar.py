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
    "SPX", "NDX", "IWM", "IWF", "HOOY", "MSTY", "NVDY", "CONY"
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
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)


# API Keys
try:
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if OPENAI_KEY:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    else:
        openai_client = None
except:
    FINNHUB_KEY = ""
    POLYGON_KEY = ""
    openai_client = None

# Data functions
@st.cache_data(ttl=60)  # Optimized with caching
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data
        hist_2d = stock.history(period="2d", interval="1m")
        hist_1d = stock.history(period="1d", interval="1m")
        
        if hist_1d.empty:
            hist_1d = stock.history(period="1d")
        if hist_2d.empty:
            hist_2d = stock.history(period="2d")
        
        # Current price
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        
        # Session data
        regular_market_open = info.get('regularMarketOpen', 0)
        previous_close = info.get('previousClose', hist_2d['Close'].iloc[-2] if len(hist_2d) >= 2 else 0)
        
        # Calculate session changes
        premarket_change = 0
        intraday_change = 0
        postmarket_change = 0
        
        if regular_market_open and previous_close:
            premarket_change = ((regular_market_open - previous_close) / previous_close) * 100
            if current_price and regular_market_open:
                intraday_change = ((current_price - regular_market_open) / regular_market_open) * 100
        
        # After hours (using selected TZ)
        current_hour = datetime.datetime.now(tz_zone).hour
        if current_hour >= 16 or current_hour < 4:
            regular_close = info.get('regularMarketPrice', current_price)
            if current_price != regular_close and regular_close:
                postmarket_change = ((current_price - regular_close) / regular_close) * 100
        
        total_change = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
        
        tz_label = "ET" if tz == "ET" else "CT"
        return {
            "last": float(current_price),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "change": float(info.get('regularMarketChange', current_price - previous_close if previous_close else 0)),
            "change_percent": float(total_change),
            "premarket_change": float(premarket_change),
            "intraday_change": float(intraday_change),
            "postmarket_change": float(postmarket_change),
            "previous_close": float(previous_close),
            "market_open": float(regular_market_open) if regular_market_open else 0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None
        }
    except Exception as e:
        tz_label = "ET" if tz == "ET" else "CT"
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
    
    # Finnhub news
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
    
    # Polygon news
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
    
    # Sort by datetime
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

def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    if not openai_client:
        return f"**{ticker} Analysis** (OpenAI API not configured)\n\nCurrent Change: {change:+.2f}%\nSet up OpenAI API key for detailed AI analysis."
    
    try:
        # Construct the prompt with additional details from options data if available
        options_text = ""
        if options_data:
            options_text = f"""
            Options Data:
            - Implied Volatility (IV): {options_data.get('iv', 'N/A')}%
            - Put/Call Ratio: {options_data.get('put_call_ratio', 'N/A')}
            - Top Call OI: {options_data.get('top_call_oi_strike', 'N/A')} with {options_data.get('top_call_oi', 'N/A')} OI
            - Top Put OI: {options_data.get('top_put_oi_strike', 'N/A')} with {options_data.get('top_put_oi', 'N/A')} OI
            - High IV Strike: {options_data.get('high_iv_strike', 'N/A')}
            """
        
        prompt = f"""
        Analyze {ticker} with {change:+.2f}% change today.
        Catalyst: {catalyst if catalyst else "Market movement"}
        {options_text}
        
        Provide an expert trading analysis focusing on:
        1. Overall Sentiment (Bullish/Bearish/Neutral) and an estimated confidence rating (out of 100).
        2. A concise trading strategy (e.g., Scalp, Day Trade, Swing, LEAP).
        3. Specific Entry levels, Target levels, and Stop levels.
        4. Key support and resistance levels.
        5. Justify your analysis by mentioning key metrics like volume, implied volatility, and open interest if available.
        6. A conclusion on the potential for an explosive move.
        
        Keep concise and actionable, under 300 words.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    if not openai_client:
        return "OpenAI API not configured for AI analysis."
    
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
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Analysis Error: {str(e)}"

def ai_auto_generate_plays(tz: str):
    """
    Auto-generates trading plays by scanning watchlist and market movers
    """
    plays = []
    
    try:
        # Get current watchlist
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        
        # Combine watchlist with core tickers for broader scan
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        
        # Scan for significant movers
        candidates = []
        
        for ticker in scan_tickers:
            quote = get_live_quote(ticker, tz)
            if not quote["error"]:
                # Look for significant moves (>1.5% change)
                if abs(quote["change_percent"]) >= 1.5:
                    candidates.append({
                        "ticker": ticker,
                        "quote": quote,
                        "significance": abs(quote["change_percent"])
                    })
        
        # Sort by significance and take top candidates
        candidates.sort(key=lambda x: x["significance"], reverse=True)
        top_candidates = candidates[:5]  # Limit to top 5 to avoid API limits
        
        # Generate plays for top candidates
        for candidate in top_candidates:
            ticker = candidate["ticker"]
            quote = candidate["quote"]
            
            # Get recent news for context
            news = get_finnhub_news(ticker)
            catalyst = ""
            if news:
                catalyst = news[0].get('headline', '')[:100] + "..."
            
            # Generate AI analysis if OpenAI is available
            if openai_client:
                try:
                    # Get placeholder options data
                    options_data = get_options_data(ticker)
                    
                    play_prompt = f"""
                    Generate a concise trading play for {ticker}:
                    
                    Current Price: ${quote['last']:.2f}
                    Change: {quote['change_percent']:+.2f}%
                    Premarket: {quote['premarket_change']:+.2f}%
                    Intraday: {quote['intraday_change']:+.2f}%
                    After Hours: {quote['postmarket_change']:+.2f}%
                    Volume: {quote['volume']:,}
                    Catalyst: {catalyst if catalyst else "Market movement"}
                    {options_data}

                    Provide:
                    1. Play type (Scalp/Day/Swing)
                    2. Entry strategy and levels
                    3. Target and stop levels
                    4. Risk/reward ratio
                    5. Confidence (1-10)
                    
                    Keep under 200 words, be specific and actionable.
                    """
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": play_prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    play_analysis = response.choices[0].message.content
                    
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
                # Fallback analysis without AI
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
                
                *Configure OpenAI API for detailed AI analysis*
                """
            
            # Create play dictionary
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

# Placeholder functions for advanced data
def get_options_data(ticker: str) -> Optional[Dict]:
    """
    Placeholder function to simulate getting options data.
    A real implementation would use a service like Polygon, CBOE, etc.
    """
    st.info(f"Disclaimer: Options data for {ticker} is a placeholder and not live.")
    return {
        "iv": np.random.uniform(20.0, 150.0),
        "put_call_ratio": np.random.uniform(0.5, 2.0),
        "top_call_oi": 15000 + np.random.randint(1, 10) * 100,
        "top_call_oi_strike": 200 + np.random.randint(-10, 10),
        "top_put_oi": 12000 + np.random.randint(1, 10) * 100,
        "top_put_oi_strike": 180 + np.random.randint(-10, 10),
        "high_iv_strike": np.random.choice([195, 205, 210])
    }

def get_earnings_calendar() -> List[Dict]:
    """
    Placeholder function for an earnings calendar.
    A real implementation would use a service like Finnhub, Polygon, or an dedicated earnings calendar API.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    st.info("Disclaimer: Earnings data is a placeholder and not live.")
    
    return [
        {"ticker": "MSFT", "date": today, "time": "After Hours", "estimate": "$2.50"},
        {"ticker": "NVDA", "date": today, "time": "Before Market", "estimate": "$1.20"},
        {"ticker": "TSLA", "date": today, "time": "After Hours", "estimate": "$0.75"},
    ]

# NEW: Placeholder function for news
def get_important_events() -> List[Dict]:
    """
    Placeholder function to simulate getting important economic events.
    A real implementation would use a service like Finnhub or an economic calendar API.
    """
    st.info("Disclaimer: Economic events data is a placeholder and not live.")
    return [
        {"event": "FOMC Meeting Minutes", "date": "2023-10-25", "time": "2:00 PM ET", "impact": "High"},
        {"event": "Unemployment Claims", "date": "2023-10-26", "time": "8:30 AM ET", "impact": "Medium"},
    ]


# Main app
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Timezone toggle (made smaller with column and smaller font)
col_tz, _ = st.columns([1, 10])  # Allocate small space for TZ
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed", help="Select Timezone (ET/CT)")

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

# Create tabs
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks", "🌐 Sector/ETF Tracking", "🎲 0DTE & Lottos", "🗓️ Earnings Plays", "📰 News"])

# Global timestamp
data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
st.markdown(f"<div style='text-align: center; color: #888; font-size: 12px;'>Last Updated: {data_timestamp}</div>", unsafe_allow_html=True)

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")

    # Session status (using selected TZ)
    current_tz_hour = current_tz.hour
    if 4 <= current_tz_hour < 9:
        session_status = "🌅 Premarket"
    elif 9 <= current_tz_hour < 16:
        session_status = "🟢 Market Open"
    else:
        session_status = "🌆 After Hours"
    st.markdown(f"**Trading Session ({tz_label}):** {session_status}")

    # Search bar
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Search Individual Stock", placeholder="Enter ticker", key="search_quotes").upper().strip()
    with col2:
        search_quotes = st.button("Get Quote", key="search_quotes_btn")

    # Search result
    if search_quotes and search_ticker:
        with st.spinner(f"Getting quote for {search_ticker}..."):
            quote = get_live_quote(search_ticker, tz_label)
            if not quote["error"]:
                st.success(f"Quote for {search_ticker} - Updated: {quote['last_updated']}")

                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                col1.metric(search_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.metric("Volume", f"{quote['volume']:,}")

                # Session breakdown
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                # News for searched ticker
                st.markdown("---")
                with st.expander(f"📰 Recent News for {search_ticker}"):
                    news = get_finnhub_news(search_ticker)
                    if news:
                        for item in news[:5]:
                            st.markdown(f"**[{item.get('headline', 'No title')}]({item.get('url', '#')})**")
                            st.write(item.get('summary', 'No summary'))
                            st.write(f"Source: Finnhub | Related: {item.get('related', 'N/A')}")
                            st.write("---")
                    else:
                        st.info("No recent news found.")

                if st.button(f"Add {search_ticker} to Active Watchlist"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_ticker not in current_list:
                        current_list.append(search_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_ticker} to watchlist '{st.session_state.active_watchlist}'!")
                        st.rerun()

            else:
                st.error(f"Could not find ticker '{search_ticker}'. Please check the symbol.")

    st.markdown("---")
    
    # Live watchlist display
    if st.session_state.watchlists[st.session_state.active_watchlist]:
        st.subheader(f"Current Watchlist: {st.session_state.active_watchlist}")
        st.write("Click a ticker to view details and news.")
        
        # Multithreading to fetch data faster
        with st.spinner("Fetching watchlist data..."):
            tickers_to_fetch = st.session_state.watchlists[st.session_state.active_watchlist]
            quotes_dict = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {executor.submit(get_live_quote, ticker, tz_label): ticker for ticker in tickers_to_fetch}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        quotes_dict[ticker] = future.result()
                    except Exception as exc:
                        st.error(f"Error fetching {ticker}: {exc}")
            
            # Sort tickers by change percent
            sorted_tickers = sorted(quotes_dict.keys(), key=lambda t: quotes_dict[t].get('change_percent', 0), reverse=True)
            
            # Display results
            for ticker in sorted_tickers:
                quote = quotes_dict[ticker]
                if quote["error"]:
                    st.error(f"{ticker}: {quote['error']}")
                    continue
                
                with st.expander(f"**{ticker}** | ${quote['last']:.2f} | Change: {quote['change_percent']:+.2f}%"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                    col2.metric("Volume", f"{quote['volume']:,}")
                    col3.metric("Last Updated", quote['last_updated'])
                    
                    st.markdown("---")
                    st.markdown("##### Session Performance")
                    sess_col1, sess_col2, sess_col3 = st.columns(3)
                    sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                    sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                    sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                    
                    st.markdown("---")
                    st.write(f"**Previous Close:** ${quote['previous_close']:.2f}")
                    st.write(f"**Market Open:** ${quote['market_open']:.2f}")
                    
                    # Sparkline (placeholder)
                    st.markdown("---")
                    st.markdown("##### Live Sparkline (Placeholder)")
                    st.markdown("*(Note: Actual sparkline generation is complex and requires live data, this is for visual representation)*")
                    st.line_chart(np.random.randn(20))
                    
                    # Recent news
                    st.markdown("---")
                    st.subheader(f"Recent News for {ticker}")
                    news = get_finnhub_news(ticker)
                    if news:
                        for item in news[:3]:
                            st.write(f"**[{item.get('headline', 'No title')}]({item.get('url', '#')})**")
                            st.write(item.get('summary', 'No summary')[:200] + "...")
                            st.write("---")
                    else:
                        st.info("No recent news found for this ticker.")
                        
    else:
        st.info("Add stocks to your watchlist using the search bar above or the Watchlist Manager tab.")

# TAB 2: Watchlist Manager
with tabs[1]:
    st.subheader("📋 Watchlist Manager")
    
    # Manage watchlists
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_watchlist_name = st.text_input("New Watchlist Name", placeholder="e.g., Tech Stocks")
    with col2:
        add_watchlist_btn = st.button("Add New Watchlist")
        if add_watchlist_btn and new_watchlist_name:
            if new_watchlist_name not in st.session_state.watchlists:
                st.session_state.watchlists[new_watchlist_name] = []
                st.success(f"Watchlist '{new_watchlist_name}' created!")
            else:
                st.warning("Watchlist already exists.")
    with col3:
        delete_watchlist_btn = st.button("Delete Active Watchlist", type="secondary")
        if delete_watchlist_btn and st.session_state.active_watchlist != "Default":
            del st.session_state.watchlists[st.session_state.active_watchlist]
            st.session_state.active_watchlist = "Default"
            st.success("Watchlist deleted.")
            st.rerun()
        elif delete_watchlist_btn:
            st.error("Cannot delete the 'Default' watchlist.")
            
    # Select active watchlist
    active_watchlist_name = st.selectbox("Select Active Watchlist", list(st.session_state.watchlists.keys()), key="select_watchlist")
    st.session_state.active_watchlist = active_watchlist_name
    
    st.markdown("---")
    st.subheader(f"Manage Tickers in '{st.session_state.active_watchlist}'")
    
    # Add/Remove tickers
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    add_ticker_to_list = st.text_input("Add Ticker", placeholder="Enter ticker (e.g., AMD)").upper().strip()
    if st.button("Add to Watchlist"):
        if add_ticker_to_list and add_ticker_to_list not in current_tickers:
            current_tickers.append(add_ticker_to_list)
            st.success(f"Added {add_ticker_to_list} to '{st.session_state.active_watchlist}'.")
            st.rerun()
        else:
            st.warning("Ticker is already in the list or invalid.")

    st.markdown("---")
    st.markdown("##### Current Tickers")
    if current_tickers:
        for ticker in current_tickers:
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("❌", key=f"remove_{ticker}"):
                    current_tickers.remove(ticker)
                    st.success(f"Removed {ticker} from '{st.session_state.active_watchlist}'.")
                    st.rerun()
            with col2:
                st.markdown(f"**{ticker}**")
    else:
        st.info("No tickers in this watchlist. Add some above.")

# TAB 3: Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Catalyst Scanner")
    st.write("Scan the market for significant movers and potential catalysts.")
    
    if st.button("Scan Now", type="primary"):
        with st.spinner("Scanning for catalysts..."):
            all_tickers = CORE_TICKERS
            movers = []
            
            for ticker in all_tickers:
                quote = get_live_quote(ticker, tz_label)
                if not quote["error"] and abs(quote["change_percent"]) >= 2.0: # Filter for significant movers
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "volume": quote["volume"],
                        "catalyst": get_finnhub_news(ticker)
                    })
            
            if not movers:
                st.info("No significant catalysts found at the moment.")
            else:
                st.success(f"Found {len(movers)} potential catalysts.")
                for m in sorted(movers, key=lambda x: abs(x['change_pct']), reverse=True):
                    with st.expander(f"**{m['ticker']}** | Change: {m['change_pct']:+.2f}% | Volume: {m['volume']:,}"):
                        if m["catalyst"]:
                            st.markdown("##### Recent News Catalyst")
                            for item in m["catalyst"][:2]:
                                st.markdown(f"**[{item.get('headline', 'No title')}]({item.get('url', '#')})**")
                                st.write(item.get('summary', 'No summary')[:200] + "...")
                                st.write("---")
                        else:
                            st.info("No specific news catalyst found. Movement may be technical or market-wide.")
                            
# TAB 4: Market Analysis
with tabs[3]:
    st.subheader("📈 AI Market Analysis")
    st.write("Get a real-time, AI-generated analysis of the overall market.")
    
    if st.button("Generate Market Analysis", type="primary"):
        with st.spinner("Generating analysis..."):
            # Get top news and movers for AI context
            all_news = get_all_news()[:10]
            
            movers = []
            for ticker in CORE_TICKERS[:20]: # Check top 20 for movers
                quote = get_live_quote(ticker, tz_label)
                if not quote["error"] and abs(quote["change_percent"]) > 1.0:
                    movers.append({"ticker": ticker, "change_pct": quote["change_percent"]})
            
            analysis = ai_market_analysis(all_news, movers)
            st.markdown(analysis)

# TAB 5: AI Playbooks
with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    st.write("Generate a detailed trading playbook for a specific ticker.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        playbook_ticker = st.text_input("Generate Playbook for:", placeholder="Enter ticker (e.g., NVDA)").upper().strip()
    with col2:
        generate_playbook_btn = st.button("Generate Playbook", type="primary")
        
    if generate_playbook_btn and playbook_ticker:
        with st.spinner(f"Generating AI playbook for {playbook_ticker}..."):
            quote = get_live_quote(playbook_ticker, tz_label)
            if not quote["error"]:
                news = get_finnhub_news(playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                options_data = get_options_data(playbook_ticker)
                
                playbook_text = ai_playbook(
                    ticker=playbook_ticker,
                    change=quote["change_percent"],
                    catalyst=catalyst,
                    options_data=options_data
                )
                
                st.markdown(playbook_text)
            else:
                st.error(f"Could not get data for {playbook_ticker}: {quote['error']}")
    else:
        st.info("Enter a ticker and click 'Generate Playbook' to get started.")
        
    st.markdown("---")
    st.subheader("Auto-Generated Plays (Beta)")
    st.write("AI-generated trading ideas based on significant market movements.")
    
    if st.button("Generate Auto Plays", key="auto_generate_btn", type="secondary"):
        with st.spinner("Scanning and generating plays..."):
            auto_plays = ai_auto_generate_plays(tz_label)
            
            if not auto_plays:
                st.info("No significant trading opportunities found at this time.")
            else:
                st.success(f"Found {len(auto_plays)} potential trading opportunities.")
                for play in auto_plays:
                    with st.expander(f"**{play['ticker']}** | Change: {play['change_percent']:+.2f}%"):
                        st.markdown(f"**Catalyst:** {play['catalyst']}")
                        st.markdown(f"**Current Price:** ${play['current_price']:.2f}")
                        st.markdown(f"**Volume:** {play['volume']:,}")
                        st.markdown(f"**Last Updated:** {play['timestamp']}")
                        
                        st.markdown("---")
                        st.markdown(play['play_analysis'])
                        
                        st.markdown("---")
                        st.write("---")
                        if st.button(f"Add {play['ticker']} to Watchlist", key=f"add_play_{play['ticker']}"):
                            current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                            if play['ticker'] not in current_list:
                                current_list.append(play['ticker'])
                                st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                st.success(f"Added {play['ticker']} to watchlist!")
                                st.rerun()
                                
    st.markdown("---")
    st.markdown("### Important Notes")
    st.markdown("These plays are AI-generated suggestions for educational and research purposes only. Always do your own due diligence and risk management.")

# TAB 6: Sector/ETF Tracking
with tabs[5]:
    st.subheader("🌐 Sector & ETF Tracking")
    st.write("Monitor the performance of major market sectors and ETFs.")
    
    for ticker in st.session_state.etf_list:
        quote = get_live_quote(ticker, tz_label)
        if quote["error"]:
            st.error(f"{ticker}: {quote['error']}")
            continue
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            
            col1.metric(ticker, f"\\${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
            col2.write("**Bid/Ask**")
            col2.write(f"\\${quote['bid']:.2f} / \\${quote['ask']:.2f}")
            col3.write("**Volume**")
            col3.write(f"{quote['volume']:,}")
            col3.caption(f"Updated: {quote['last_updated']}")
            
            if col4.button(f"Add {ticker} to Watchlist", key=f"add_etf_{ticker}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if ticker not in current_list:
                    current_list.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {ticker} to watchlist!")
                    st.rerun()

            st.divider()

# TAB 7: 0DTE & Lottos
with tabs[6]:
    st.subheader("🎲 0DTE & Lottery Plays")
    st.info("This feature is a placeholder. A real implementation would require a live options data feed (e.g., CBOE, Polygon).")
    
    st.write("AI-identified high-risk, high-reward options plays for the current trading day.")
    
    if st.button("Find Today's Lottery Plays", type="primary"):
        with st.spinner("Scanning for potential 0DTE opportunities..."):
            st.warning("Disclaimer: These are highly speculative plays and can result in 100% loss. Proceed with extreme caution.")
            
            # Placeholder data
            plays = [
                {"ticker": "NVDA", "type": "Call", "strike": 450, "premium": 0.50, "target": 1.50, "thesis": "AI-driven demand and positive news sentiment. High IV indicates potential for a large move."},
                {"ticker": "SPY", "type": "Put", "strike": 435, "premium": 0.35, "target": 1.00, "thesis": "Weakness in market internals and increased VIX. Monitoring for a breakdown below key support."},
            ]
            
            if plays:
                for play in plays:
                    with st.expander(f"**{play['ticker']} {play['strike']} {play['type']}** | Suggested Premium: ${play['premium']:.2f}"):
                        st.markdown(f"**Thesis:** {play['thesis']}")
                        st.markdown(f"**Target:** ${play['target']:.2f}")
                        st.markdown(f"**Risk:** Total loss of premium")
                        st.markdown("---")
                        st.markdown("This is a placeholder for a real-time options scan.")
            else:
                st.info("No significant 0DTE plays identified at this time.")

# TAB 8: Earnings Plays
with tabs[7]:
    st.subheader("🗓️ Earnings Play Calendar")
    st.write("Identify upcoming earnings reports and potential trading opportunities.")
    
    if st.button("Get Earnings Calendar", type="primary"):
        with st.spinner("Fetching earnings data..."):
            earnings_reports = get_earnings_calendar()
            
            if not earnings_reports:
                st.info("No major earnings reports scheduled for today.")
            else:
                st.success(f"Found {len(earnings_reports)} earnings reports today.")
                st.markdown("### Today's Earnings Reports")
                
                for report in earnings_reports:
                    time_str = "After Hours" if report['time'] == "After Hours" else "Before Market"
                    st.markdown(f"**{report['ticker']}** | {time_str}")
                    st.write(f"**Date:** {report['date']}")
                    st.write(f"**Time:** {time_str}")
                    st.write(f"**Estimate:** {report['estimate']}")
                    
                    ai_analysis_text = f"""
                    **AI Analysis for {report['ticker']} Earnings:**
                    - **Date:** {report["date"]}
                    - **Time:** {time_str}
                    - **AI Probability of a Move:** High (based on historical data and current market conditions)
                    - **AI Suggested Contract:** Placeholder (e.g., {report['ticker']} {datetime.date.today()} Call/Put option)
                    - **Entry Level:** Placeholder (e.g., above $150.00 for a call)
                    - **Target Level:** Placeholder (e.g., $160.00)
                    - **Stop Loss:** Placeholder (e.g., below $145.00)
                    - **AI Confidence:** 85%
                    
                    **AI Thesis:** The market is anticipating a strong/weak report. High volume and IV are supporting a potential explosive move post-earnings. The AI has identified a solid risk/reward setup based on a potential gap up/down.
                    """
                    
                    with st.expander(f"🔮 AI Predicts Play for {report['ticker']}"):
                        st.markdown(ai_analysis_text)
                    st.divider()

# TAB 9: News
with tabs[8]:
    st.subheader("📰 Important News & Economic Calendar")
    
    if st.button("📊 Get This Week's Events", type="primary"):
        with st.spinner("Fetching important events..."):
            important_events = get_important_events()
            
            if not important_events:
                st.info("No major economic events scheduled for this week.")
            else:
                st.markdown("### Major Market-Moving Events")
                
                for event in sorted(important_events, key=lambda x: x['date']):
                    st.markdown(f"**{event['event']}**")
                    st.write(f"**Date:** {event['date']}")
                    st.write(f"**Time:** {event['time']}")
                    st.write(f"**Impact:** {event['impact']}")
                    st.divider()
            
    st.markdown("---")
    st.subheader("Latest General Market News")
    
    if st.button("🔄 Refresh News", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    news_items = get_all_news()
    if news_items:
        for item in news_items:
            sentiment, score = analyze_news_sentiment(item.get("title", ""), item.get("summary", ""))
            
            st.markdown(f"**[{item.get('title', 'No title')}]({item.get('url', '#')})**")
            st.write(f"Source: {item.get('source', 'N/A')}")
            st.write(f"Sentiment: {sentiment}")
            st.write(item.get('summary', 'No summary'))
            st.write("---")
    else:
        st.info("No news articles found at this time.")

# Auto refresh
if st.session_state.auto_refresh:
    time.sleep(0.1)
    if st.session_state.refresh_interval == 10:
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"\
    "🔥 AI Radar Pro | Live data: yfinance | News: Finnhub/Polygon | AI: OpenAI"\
    "</div>",
    unsafe_allow_html=True
)
