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

# ---------------- CONFIG ----------------
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

# Initialize session state variables
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

# ---------------- LIVE DATA FUNCTIONS ----------------
@st.cache_data(ttl=10)  # 10 second cache for very live data
def get_live_quote(ticker: str) -> Dict:
    """Get real-time quote using yfinance with very short cache."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get real-time data
        info = stock.info
        
        # Try to get the most recent price
        hist = stock.history(period="1d", interval="1m")
        if not hist.empty:
            current_price = float(hist['Close'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
        else:
            # Fallback to info data
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            volume = info.get('volume', 0)
        
        return {
            "last": float(current_price),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(volume),
            "change": float(info.get('regularMarketChange', 0)),
            "change_percent": float(info.get('regularMarketChangePercent', 0)),
            "error": None
        }
    except Exception as e:
        return {
            "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
            "change": 0.0, "change_percent": 0.0, "error": str(e)
        }

@st.cache_data(ttl=300)
def get_previous_close(ticker: str) -> float:
    """Get previous trading day close price."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return float(info.get('previousClose', 0))
    except:
        return 0.0

@st.cache_data(ttl=600)  # 10 minute cache for news
def get_finnhub_news(symbol: str = None) -> List[Dict]:
    """Get news from Finnhub API."""
    if not FINNHUB_KEY:
        return []
    
    try:
        if symbol:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()}&to={datetime.date.today()}&token={FINNHUB_KEY}"
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()[:10]  # Limit to 10 articles
    except Exception as e:
        st.warning(f"Finnhub API error: {e}")
    
    return []

@st.cache_data(ttl=600)
def get_polygon_news() -> List[Dict]:
    """Get market news from Polygon API."""
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

@st.cache_data(ttl=300)
def get_benzinga_news() -> List[Dict]:
    """Get news from Benzinga (free tier)."""
    try:
        # Using RSS feed as free alternative
        url = "https://www.benzinga.com/feed"
        response = requests.get(url, timeout=10)
        # This would need XML parsing in a real implementation
        return []
    except:
        return []

def get_all_news() -> List[Dict]:
    """Aggregate news from all sources."""
    all_news = []
    
    # Get Finnhub general news
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
    
    # Get Polygon news
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
    
    # Sort by datetime (most recent first)
    try:
        all_news.sort(key=lambda x: x["datetime"], reverse=True)
    except:
        pass
    
    return all_news[:15]  # Return top 15 articles

def analyze_news_sentiment(title: str, summary: str = "") -> tuple:
    """Analyze news sentiment for catalyst detection."""
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

# ---------------- AUTO REFRESH FUNCTIONALITY ----------------
def auto_refresh_data():
    """Auto refresh data in background."""
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.cache_data.clear()
        st.rerun()

# ---------------- AI FUNCTIONS ----------------
def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    """Generate AI market analysis from news and movers."""
    if not openai_client:
        return "OpenAI API not configured for AI analysis."
    
    try:
        # Prepare context
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

def ai_playbook(ticker: str, change: float, catalyst: str = "") -> str:
    """Generate AI trading playbook."""
    if not openai_client:
        return f"**{ticker} Analysis** (OpenAI API not configured)\n\nCurrent Change: {change:+.2f}%\nSet up OpenAI API key for detailed AI analysis."
    
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
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- MAIN APP ----------------
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Auto-refresh controls in header
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
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p ET")
    market_open = 9 <= datetime.datetime.now().hour < 16
    status = "🟢 Open" if market_open else "🔴 Closed"
    st.write(f"**{status}** | {current_time}")

# Create tabs
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab.")
    else:
        # Display live quotes
        for ticker in tickers:
            quote = get_live_quote(ticker)
            if quote["error"]:
                st.error(f"{ticker}: {quote['error']}")
                continue
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                
                # Price and change
                col1.metric(
                    ticker,
                    f"${quote['last']:.2f}",
                    f"{quote['change_percent']:.2f}%"
                )
                
                # Bid/Ask
                col2.write("**Bid/Ask**")
                col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                
                # Volume
                col3.write("**Volume**")
                col3.write(f"{quote['volume']:,}")
                
                # Quick actions
                if abs(quote['change_percent']) >= 2.0:
                    if col4.button(f"🎯 AI Analysis", key=f"ai_{ticker}"):
                        with st.spinner(f"Analyzing {ticker}..."):
                            analysis = ai_playbook(ticker, quote['change_percent'])
                            st.success(f"🤖 {ticker} Analysis")
                            st.markdown(analysis)
                
                st.divider()

# TAB 2: Watchlist Manager
with tabs[1]:
    st.subheader("📋 Watchlist Manager")
    
    # Watchlist selection
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
    
    # Current watchlist
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist].copy()
    
    # Add symbols section
    st.markdown("### Add Symbols")
    
    # Quick add from core tickers
    st.write("**Popular Tickers:**")
    
    # Display core tickers in columns
    cols = st.columns(6)
    for i, ticker in enumerate(CORE_TICKERS):
        with cols[i % 6]:
            if st.button(f"+ {ticker}", key=f"add_{ticker}"):
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                    st.success(f"Added {ticker}")
                    st.rerun()
    
    # Manual add
    st.markdown("**Add Custom Symbol:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_ticker = st.text_input("Enter Symbol", placeholder="AAPL").upper().strip()
    with col2:
        if st.button("Add Symbol") and custom_ticker:
            if custom_ticker not in current_tickers:
                current_tickers.append(custom_ticker)
                st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                st.success(f"Added {custom_ticker}")
                st.rerun()
    
    # Current watchlist management
    st.markdown("### Current Watchlist")
    if current_tickers:
        # Display in groups of 5
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
        st.info("Watchlist is empty. Add some symbols above.")

# TAB 3: Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Real-Time Catalyst Scanner")
    
    if st.button("🔍 Scan for Catalysts", type="primary"):
        with st.spinner("Scanning news and market data for catalysts..."):
            # Get news
            all_news = get_all_news()
            
            # Get market movers
            movers = []
            for ticker in CORE_TICKERS[:20]:  # Check top 20 tickers for moves
                quote = get_live_quote(ticker)
                if not quote["error"] and abs(quote["change_percent"]) >= 1.5:
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"],
                        "volume": quote["volume"]
                    })
            
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            
            # Display catalysts from news
            st.markdown("### 📰 News Catalysts")
            for news in all_news[:10]:
                sentiment, confidence = analyze_news_sentiment(news["title"], news["summary"])
                
                with st.expander(f"{sentiment} ({confidence}%) - {news['title'][:100]}..."):
                    st.write(f"**Source:** {news['source']}")
                    st.write(f"**Summary:** {news['summary']}")
                    if news["related"]:
                        st.write(f"**Related Tickers:** {news['related']}")
                    if news["url"]:
                        st.markdown(f"[Read Full Article]({news['url']})")
            
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
    
    # Search bar for individual stock analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        search_analysis_ticker = st.text_input("🔍 Analyze specific stock", placeholder="Enter ticker for detailed analysis", key="search_analysis").upper().strip()
    with col2:
        search_analysis = st.button("Analyze Stock", key="search_analysis_btn")
    
    # Display individual stock analysis if requested
    if search_analysis and search_analysis_ticker:
        with st.spinner(f"AI analyzing {search_analysis_ticker}..."):
            quote = get_live_quote(search_analysis_ticker)
            if not quote["error"]:
                # Get news for context
                news = get_finnhub_news(search_analysis_ticker)
                catalyst = news[0].get('headline', '') if news else "Recent market movement"
                
                analysis = ai_playbook(search_analysis_ticker, quote["change_percent"], catalyst)
                
                st.success(f"🤖 AI Analysis: {search_analysis_ticker}")
                
                # Stock data
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
                
                st.markdown("### 🎯 AI Analysis")
                st.markdown(analysis)
                
                # Recent news context
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
        with st.spinner("AI analyzing current market conditions..."):
            news_items = get_all_news()
            
            # Get market movers
            movers = []
            for ticker in CORE_TICKERS[:15]:
                quote = get_live_quote(ticker)
                if not quote["error"]:
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"]
                    })
            
            # Generate AI analysis
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            # Show supporting data
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
    
    # Search bar for any stock playbook
    col1, col2 = st.columns([3, 1])
    with col1:
        search_playbook_ticker = st.text_input("🔍 Generate playbook for any stock", placeholder="Enter ticker for trading playbook", key="search_playbook").upper().strip()
    with col2:
        search_playbook = st.button("Generate Playbook", key="search_playbook_btn")
    
    # Display search result playbook
    if search_playbook and search_playbook_ticker:
        quote = get_live_quote(search_playbook_ticker)
        
        if not quote["error"]:
            with st.spinner(f"AI generating playbook for {search_playbook_ticker}..."):
                # Get news for context
                news = get_finnhub_news(search_playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                
                playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst)
                
                st.success(f"✅ {search_playbook_ticker} Trading Playbook")
                
                # Current data
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
                
                # AI analysis
                st.markdown("### 🎯 AI Trading Playbook")
                st.markdown(playbook)
                
                # Recent news
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
        selected_ticker = st.selectbox("Select from your watchlist", tickers, key="watchlist_playbook")
        catalyst_input = st.text_input("Catalyst (optional)", placeholder="News event, technical pattern, etc.", key="catalyst_input")
        
        if st.button("🤖 Generate Watchlist Playbook", type="primary"):
            quote = get_live_quote(selected_ticker)
            
            if not quote["error"]:
                with st.spinner(f"AI analyzing {selected_ticker}..."):
                    playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input)
                    
                    st.success(f"✅ {selected_ticker} Trading Playbook - Updated: {quote['last_updated']}")
                    
                    # Current data
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
                    
                    # AI analysis
                    st.markdown("### 🎯 AI Analysis")
                    st.markdown(playbook)
                    
                    # Recent news
                    news = get_finnhub_news(selected_ticker)
                    if news:
                        with st.expander(f"📰 Recent News for {selected_ticker}"):
                            for item in news[:3]:
                                st.write(f"**{item.get('headline', 'No title')}**")
                                st.write(item.get('summary', 'No summary')[:200] + "...")
                                st.write("---")
    else:
        st.info("Add stocks to your watchlist or use the search above to generate playbooks.")"price": quote["last"]
                    })
            
            # Generate AI analysis
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            # Show supporting data
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
    
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if not tickers:
        st.warning("Add symbols to your watchlist first.")
    else:
        selected_ticker = st.selectbox("Select Symbol", tickers)
        catalyst_input = st.text_input("Catalyst (optional)", placeholder="News event, technical pattern, etc.")
        
        if st.button("🤖 Generate Playbook", type="primary"):
            quote = get_live_quote(selected_ticker)
            
            if not quote["error"]:
                with st.spinner(f"AI analyzing {selected_ticker}..."):
                    playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input)
                    
                    st.success(f"✅ {selected_ticker} Trading Playbook")
                    
                    # Current data
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                    col3.metric("Volume", f"{quote['volume']:,}")
                    
                    # AI analysis
                    st.markdown("### 🎯 AI Analysis")
                    st.markdown(playbook)
                    
                    # Recent news
                    news = get_finnhub_news(selected_ticker)
                    if news:
                        with st.expander(f"📰 Recent News for {selected_ticker}"):
                            for item in news[:3]:
                                st.write(f"**{item.get('headline', 'No title')}**")
                                st.write(item.get('summary', 'No summary')[:200] + "...")
                                st.write("---")

# Auto refresh functionality
if st.session_state.auto_refresh:
    # Use streamlit's experimental_rerun for auto refresh
    import time
    time.sleep(0.1)  # Small delay
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
