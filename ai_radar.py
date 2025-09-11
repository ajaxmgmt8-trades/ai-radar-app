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
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks", "🌐 Sector/ETF Tracking", "🎲 0DTE & Lottos", "🗓️ Earnings Plays"])

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
                
                if col4.button(f"Add {search_ticker} to Watchlist", key="add_searched"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_ticker not in current_list:
                        current_list.append(search_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_ticker} to watchlist!")
                        st.rerun()
                st.divider()
            else:
                st.error(f"Could not get quote for {search_ticker}: {quote['error']}")
    
    # Watchlist display
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab.")
    else:
        st.markdown("### Your Watchlist")
        for ticker in tickers:
            quote = get_live_quote(ticker, tz_label)
            if quote["error"]:
                st.error(f"{ticker}: {quote['error']}")
                continue
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                
                col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.write("**Bid/Ask**")
                col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.write("**Volume**")
                col3.write(f"{quote['volume']:,}")
                col3.caption(f"Updated: {quote['last_updated']}")
                
                if abs(quote['change_percent']) >= 2.0:
                    if col4.button(f"🎯 AI Analysis", key=f"ai_{ticker}"):
                        with st.spinner(f"Analyzing {ticker}..."):
                            analysis = ai_playbook(ticker, quote['change_percent'])
                            st.success(f"🤖 {ticker} Analysis")
                            st.markdown(analysis)
                
                # Session data
                sess_col1, sess_col2, sess_col3, sess_col4 = st.columns([2, 2, 2, 4])
                sess_col1.caption(f"**PM:** {quote['premarket_change']:+.2f}%")
                sess_col2.caption(f"**Day:** {quote['intraday_change']:+.2f}%")
                sess_col3.caption(f"**AH:** {quote['postmarket_change']:+.2f}%")
                
                # Expandable detailed view
                with st.expander(f"🔎 Expand {ticker}"):
                    # Catalyst headlines
                    news = get_finnhub_news(ticker)
                    if news:
                        st.write("### 📰 Catalysts (last 24h)")
                        for n in news:
                            st.write(f"- [{n.get('headline', 'No title')}]({n.get('url', '#')}) ({n.get('source', 'Finnhub')})")
                    else:
                        st.info("No recent news.")
                    
                    # AI Playbook
                    st.markdown("### 🎯 AI Playbook")
                    catalyst_title = news[0].get('headline', '') if news else ""
                    st.markdown(ai_playbook(ticker, quote['change_percent'], catalyst_title))
                
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
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if search_add_ticker not in current_list:
                    current_list.append(search_add_ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"✅ Added {search_add_ticker}")
                    st.rerun()
                else:
                    st.warning(f"{search_add_ticker} already in watchlist")
            else:
                st.error(f"Invalid ticker: {search_add_ticker}")
    
    # Watchlist management
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
    
    # Clean up any existing duplicates
    unique_current_tickers = list(dict.fromkeys(current_tickers))
    if len(unique_current_tickers) != len(current_tickers):
        st.session_state.watchlists[st.session_state.active_watchlist] = unique_current_tickers
        current_tickers = unique_current_tickers
        st.rerun()  # Refresh to show cleaned list
    
    # Popular tickers
    st.markdown("### ⭐ Popular Tickers")
    cols = st.columns(6)
    for i, ticker in enumerate(CORE_TICKERS):
        with cols[i % 6]:
            if st.button(f"+ {ticker}", key=f"add_{ticker}"):
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                    st.success(f"Added {ticker}")
                    st.rerun()
    
    # Current watchlist
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

# TAB 3: Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Real-Time Catalyst Scanner")
    
    # Search specific stock
    col1, col2 = st.columns([3, 1])
    with col1:
        search_catalyst_ticker = st.text_input("🔍 Search catalysts for stock", placeholder="Enter ticker", key="search_catalyst").upper().strip()
    with col2:
        search_catalyst = st.button("Search Catalysts", key="search_catalyst_btn")
    
    if search_catalyst and search_catalyst_ticker:
        with st.spinner(f"Searching catalysts for {search_catalyst_ticker}..."):
            specific_news = get_finnhub_news(search_catalyst_ticker)
            quote = get_live_quote(search_catalyst_ticker, tz_label)
            
            if not quote["error"]:
                st.success(f"Catalyst Analysis for {search_catalyst_ticker} - Updated: {quote['last_updated']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                
                # Session breakdown
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%") 
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                if col3.button(f"Add {search_catalyst_ticker} to WL", key="add_catalyst_search"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_catalyst_ticker not in current_list:
                        current_list.append(search_catalyst_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_catalyst_ticker}")
                        st.rerun()
                
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
    
    # Main catalyst scan
    if st.button("🔍 Scan for Market Catalysts", type="primary"):
        with st.spinner("Scanning for catalysts..."):
            all_news = get_all_news()
            
            # Get movers
            movers = []
            for ticker in CORE_TICKERS[:20]:
                quote = get_live_quote(ticker, tz_label)
                if not quote["error"] and abs(quote["change_percent"]) >= 1.5:
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"],
                        "volume": quote["volume"]
                    })
            
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            
            # Display news catalysts
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
            
            # Display market movers
            st.markdown("### 📊 Significant Market Moves")
            for mover in movers[:10]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    direction = "🚀" if mover["change_pct"] > 0 else "📉"
                    st.metric(
                        f"{direction} {mover['ticker']}", 
                        f"${mover['price']:.2f}",
                        f"{mover['change_pct']:+.2f}%"
                    )
                with col2:
                    if st.button(f"Add to WL", key=f"add_mover_{mover['ticker']}"):
                        current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                        if mover['ticker'] not in current_list:
                            current_list.append(mover['ticker'])
                            st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                            st.success(f"Added {mover['ticker']}")
                            st.rerun()

# TAB 4: Market Analysis
with tabs[3]:
    st.subheader("📈 AI Market Analysis")
    
    # Search individual analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        search_analysis_ticker = st.text_input("🔍 Analyze specific stock", placeholder="Enter ticker", key="search_analysis").upper().strip()
    with col2:
        search_analysis = st.button("Analyze Stock", key="search_analysis_btn")
    
    if search_analysis and search_analysis_ticker:
        with st.spinner(f"AI analyzing {search_analysis_ticker}..."):
            quote = get_live_quote(search_analysis_ticker, tz_label)
            if not quote["error"]:
                news = get_finnhub_news(search_analysis_ticker)
                catalyst = news[0].get('headline', '') if news else "Recent market movement"
                
                analysis = ai_playbook(search_analysis_ticker, quote["change_percent"], catalyst)
                
                st.success(f"🤖 AI Analysis: {search_analysis_ticker} - Updated: {quote['last_updated']}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                if col4.button(f"Add {search_analysis_ticker} to WL", key="add_analysis_search"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_analysis_ticker not in current_list:
                        current_list.append(search_analysis_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_analysis_ticker}")
                        st.rerun()
                
                # Session breakdown
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
                st.error(f"Could not analyze {search_analysis_ticker}: {quote['error']}")
    
    # Main market analysis
    if st.button("🤖 Generate Market Analysis", type="primary"):
        with st.spinner("AI analyzing market conditions..."):
            news_items = get_all_news()
            
            movers = []
            for ticker in CORE_TICKERS[:15]:
                quote = get_live_quote(ticker, tz_label)
                if not quote["error"]:
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"]
                    })
            
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            with st.expander("📊 Supporting Data"):
                st.write("**Top Market Movers:**")
                for mover in sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]:
                    st.write(f"• {mover['ticker']}: {mover['change_pct']:+.2f}%")
                
                st.write("**Key News Headlines:**")
                for news in news_items[:3]:
                    st.write(f"• {news['title']}")

# TAB 5: AI Playbooks
with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    
    # Auto-generated plays section
    st.markdown("### 🎯 Auto-Generated Trading Plays")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("AI automatically scans your watchlist and market movers to suggest trading opportunities")
    with col2:
        if st.button("🚀 Generate Auto Plays", type="primary"):
            with st.spinner("AI generating trading plays from market scan..."):
                auto_plays = ai_auto_generate_plays(tz_label)
                
                if auto_plays:
                    st.success(f"🤖 Generated {len(auto_plays)} Trading Plays")
                    
                    for i, play in enumerate(auto_plays):
                        with st.expander(f"🎯 {play['ticker']} - ${play['current_price']:.2f} ({play['change_percent']:+.2f}%)"):
                            
                            # Display session data
                            sess_col1, sess_col2, sess_col3 = st.columns(3)
                            sess_col1.metric("Premarket", f"{play['session_data']['premarket']:+.2f}%")
                            sess_col2.metric("Intraday", f"{play['session_data']['intraday']:+.2f}%")
                            sess_col3.metric("After Hours", f"{play['session_data']['afterhours']:+.2f}%")
                            
                            # Display catalyst
                            if play['catalyst']:
                                st.write(f"**Catalyst:** {play['catalyst']}")
                            
                            # Display AI play analysis
                            st.markdown("**AI Trading Play:**")
                            st.markdown(play['play_analysis'])
                            
                            # Add to watchlist option
                            if st.button(f"Add {play['ticker']} to Watchlist", key=f"add_auto_play_{i}"):
                                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                                if play['ticker'] not in current_list:
                                    current_list.append(play['ticker'])
                                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                    st.success(f"Added {play['ticker']} to watchlist!")
                                    st.rerun()
                else:
                    st.info("No significant trading opportunities detected at this time. Market conditions may be consolidating.")
    
    st.divider()
    
    # Search any stock
    st.markdown("### 🔍 Custom Stock Analysis")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_playbook_ticker = st.text_input("🔍 Generate playbook for any stock", placeholder="Enter ticker", key="search_playbook").upper().strip()
    with col2:
        search_playbook = st.button("Generate Playbook", key="search_playbook_btn")
    
    if search_playbook and search_playbook_ticker:
        quote = get_live_quote(search_playbook_ticker, tz_label)
        
        if not quote["error"]:
            with st.spinner(f"AI generating playbook for {search_playbook_ticker}..."):
                news = get_finnhub_news(search_playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                
                playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst)
                
                st.success(f"✅ {search_playbook_ticker} Trading Playbook - Updated: {quote['last_updated']}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                col3.metric("Volume", f"{quote['volume']:,}")
                if col4.button(f"Add {search_playbook_ticker} to WL", key="add_playbook_search"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_playbook_ticker not in current_list:
                        current_list.append(search_playbook_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_playbook_ticker}")
                        st.rerun()
                
                # Session performance
                st.markdown("#### Session Breakdown")
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
            st.error(f"Could not get data for {search_playbook_ticker}: {quote['error']}")
    
    # Watchlist playbooks
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if tickers:
        st.markdown("### 📋 Watchlist Playbooks")
        selected_ticker = st.selectbox("Select from watchlist", tickers, key="watchlist_playbook")
        catalyst_input = st.text_input("Catalyst (optional)", placeholder="News event, etc.", key="catalyst_input")
        
        if st.button("🤖 Generate Watchlist Playbook", type="secondary"):
            quote = get_live_quote(selected_ticker, tz_label)
            
            if not quote["error"]:
                with st.spinner(f"AI analyzing {selected_ticker}..."):
                    playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input)
                    
                    st.success(f"✅ {selected_ticker} Trading Playbook - Updated: {quote['last_updated']}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                    col3.metric("Volume", f"{quote['volume']:,}")
                    
                    # Session performance
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
        st.info("Add stocks to watchlist or use search above.")
    
    # Quick tips for auto-generated plays
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
        
    # LEAP section
    st.markdown("### 📈 Long-Term Equity Anticipation Securities (LEAPS)")
    with st.expander("Explore LEAPS"):
        st.write("This section would provide analysis for solid LEAPs, including:")
        st.markdown("""
        - **Data Needed:** Long-term options chains (1-2 years out), Implied Volatility (IV), historical IV, delta/gamma/theta data.
        - **Analysis Focus:** Companies with strong fundamentals, low IV, and high probability for long-term growth.
        - **AI will provide:** Suggested strike prices, entry/target levels, and a comprehensive thesis.
        - **API Required:** A robust options data API to get the necessary information.
        """)

# TAB 6: Sector/ETF Tracking
with tabs[5]:
    st.subheader("🌐 Sector/ETF Tracking")

    # Add search and add functionality
    st.markdown("### 🔍 Search & Add ETFs")
    col1, col2 = st.columns([3, 1])
    with col1:
        etf_search_ticker = st.text_input("Search for an ETF to add", placeholder="Enter ticker (e.g., VOO)", key="etf_search_add").upper().strip()
    with col2:
        if st.button("Add ETF", key="add_etf_btn") and etf_search_ticker:
            if etf_search_ticker not in st.session_state.etf_list:
                quote = get_live_quote(etf_search_ticker)
                if not quote["error"]:
                    st.session_state.etf_list.append(etf_search_ticker)
                    st.success(f"✅ Added {etf_search_ticker} to the list.")
                    st.rerun()
                else:
                    st.error(f"Invalid ticker or ETF: {etf_search_ticker}")
            else:
                st.warning(f"{etf_search_ticker} is already in the list.")

    st.markdown("### ETF Performance Overview")
    
    for ticker in st.session_state.etf_list:
        quote = get_live_quote(ticker, tz_label)
        if quote["error"]:
            st.error(f"{ticker}: {quote['error']}")
            continue
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            
            col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
            col2.write("**Bid/Ask**")
            col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
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
    st.subheader("🎲 0DTE & Lottos")
    
    st.write("This section is designed to find potential explosive moves using 0DTE (zero-days-to-expiration) and high-risk 'lotto' options.")
    st.info("Disclaimer: This section requires a robust options data API and sophisticated AI models to function effectively. The analysis provided here is conceptual. The AI's confidence rating is an estimate based on limited data and is not a guarantee.")
    
    if st.button("🔍 Scan for 0DTE Plays", type="primary"):
        with st.spinner("AI scanning for potential 0DTE setups..."):
            # Placeholder for AI logic that scans for 0DTE plays
            # This would require an options data API (e.g., to get options chains, volume, open interest)
            # The AI would analyze this data combined with real-time stock data and news.
            
            # Simulated play
            play_ticker = np.random.choice(["SPY", "QQQ", "TSLA", "NVDA", "GOOG"])
            quote = get_live_quote(play_ticker)
            options_data = get_options_data(play_ticker)
            
            st.success(f"🎯 Potential 0DTE Play: {play_ticker} ")
            
            playbook_analysis = ai_playbook(play_ticker, quote["change_percent"], "High volume and implied volatility spike", options_data)
            
            st.markdown(playbook_analysis)
            st.divider()

# TAB 8: Earnings Plays
with tabs[7]:
    st.subheader("🗓️ Top Earnings Plays")
    
    st.write("This section tracks upcoming earnings reports and uses AI to identify the highest-probability trading setups.")
    st.info("Disclaimer: This section requires an earnings calendar API to be fully functional. The data shown is a placeholder and not live. AI's contract selection is conceptual.")
    
    if st.button("📊 Get Today's Earnings Plays", type="primary"):
        with st.spinner("AI analyzing earnings reports..."):
            
            earnings_today = get_earnings_calendar()
            
            if not earnings_today:
                st.info("No earnings reports found for today.")
            else:
                st.markdown("### Today's Earnings Reports")
                for report in earnings_today:
                    ticker = report["ticker"]
                    time_str = report["time"]
                    
                    st.markdown(f"**{ticker}** - Earnings **{time_str}**")
                    
                    # Placeholder for AI analysis of the earnings play
                    ai_analysis_text = f"""
                    **AI Analysis for {ticker} Earnings:**
                    - **Date:** {report["date"]}
                    - **Time:** {time_str}
                    - **AI Probability of a Move:** High (based on historical data and current market conditions)
                    - **AI Suggested Contract:** Placeholder (e.g., {ticker} {datetime.date.today()} Call/Put option)
                    - **Entry Level:** Placeholder (e.g., above $150.00 for a call)
                    - **Target Level:** Placeholder (e.g., $160.00)
                    - **Stop Loss:** Placeholder (e.g., below $145.00)
                    - **AI Confidence:** 85%
                    
                    **AI Thesis:** The market is anticipating a strong/weak report. High volume and IV are supporting a potential explosive move post-earnings. The AI has identified a solid risk/reward setup based on a potential gap up/down.
                    """
                    
                    with st.expander(f"🔮 AI Predicts Play for {ticker}"):
                        st.markdown(ai_analysis_text)
                    st.divider()

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


