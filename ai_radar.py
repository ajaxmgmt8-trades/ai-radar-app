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
    st.session_state.selected_tz = "ET"
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "OpenAI"

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

# NEW FUNCTION TO FETCH IMPORTANT EVENTS
@st.cache_data(ttl=3600)
def get_important_events():
    # Placeholder for a real API call (e.g., using a calendar API or scraping a financial news site)
    # This list can be updated manually or via a different data source
    today = datetime.date.today()
    return [
        {"date": today.strftime("%Y-%m-%d"), "time": "14:00 ET", "event": "FOMC Meeting Announcement", "impact": "High"},
        {"date": (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d"), "time": "08:30 ET", "event": "CPI Data Release", "impact": "High"},
        {"date": (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d"), "time": "10:00 ET", "event": "Jerome Powell Speech", "impact": "High"},
        {"date": (today + datetime.timedelta(days=3)).strftime("%Y-%m-%d"), "time": "10:30 ET", "event": "EIA Petroleum Status Report", "impact": "Medium"},
        {"date": (today + datetime.timedelta(days=4)).strftime("%Y-%m-%d"), "time": "08:30 ET", "event": "Non-Farm Payrolls", "impact": "High"}
    ]

# AI functions (already present in the previous version, just ensuring they are here)
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
                    4. Risk/reward analysis
                    5. AI confidence rating (out of 10)
                    
                    Keep it professional, concise, and easy to read.
                    """
                    
                    if model_choice == "OpenAI":
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": play_prompt}],
                            temperature=0.5,
                            max_tokens=400
                        )
                        plays.append({"ticker": ticker, "play": response.choices[0].message.content})
                    else: # Gemini
                        response = gemini_model.generate_content(play_prompt)
                        plays.append({"ticker": ticker, "play": response.text})
                except Exception as e:
                    plays.append({"ticker": ticker, "play": f"AI Error: {str(e)}"})
    except Exception as e:
        st.error(f"Failed to auto-generate plays: {str(e)}")
    return plays

# Helper function to display movers
def display_movers(movers_list):
    if not movers_list:
        st.info("No significant movers to display.")
    else:
        for mover in movers_list:
            ticker = mover['ticker']
            quote = mover['quote']
            col1, col2, col3 = st.columns([1.5, 1, 2])
            with col1:
                st.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
            with col2:
                st.write(f"Vol: {quote['volume']:,}")
            with col3:
                if st.button("Generate Playbook", key=f"mover_ai_{ticker}_{np.random.rand()}"):
                    with st.spinner("Generating AI Playbook..."):
                        news = get_finnhub_news(ticker)
                        catalyst = news[0].get('headline', '')[:100] + "..." if news else ""
                        playbook_text = ai_playbook(ticker, quote['change_percent'], catalyst, st.session_state.ai_model)
                        st.markdown(f"### {ticker} AI Playbook")
                        st.markdown(playbook_text)
            st.divider()

# Get top movers with multithreading
def get_top_gainers_losers_volume_multithreaded():
    tickers_to_check = list(set(CORE_TICKERS + ETF_TICKERS))
    gainers, losers, high_volume = [], [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_live_quote, ticker, "ET"): ticker for ticker in tickers_to_check}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                quote = future.result()
                if quote and not quote.get("error"):
                    change_pct = quote.get("change_percent", 0)
                    volume = quote.get("volume", 0)
                    
                    if change_pct > 2.0:
                        gainers.append({"ticker": ticker, "change_pct": change_pct, "quote": quote})
                    if change_pct < -2.0:
                        losers.append({"ticker": ticker, "change_pct": change_pct, "quote": quote})
                    if volume > 500000:  # Adjust threshold as needed
                        high_volume.append({"ticker": ticker, "volume": volume, "quote": quote})
            except Exception as e:
                pass
    
    gainers.sort(key=lambda x: x['change_pct'], reverse=True)
    losers.sort(key=lambda x: x['change_pct'])
    high_volume.sort(key=lambda x: x['volume'], reverse=True)
    
    return gainers[:10], losers[:10], high_volume[:10]

# Main application layout
st.title("🔥 AI Radar Pro")

st.markdown("""
_A powerful, all-in-one AI-powered trading companion. Get real-time data, AI-generated trading plays,
and market analysis._
""")

# Sidebar for controls
st.sidebar.header("Controls")
st.sidebar.markdown("**AI Model Selection**")
st.session_state.ai_model = st.sidebar.selectbox("AI Model:", ["OpenAI", "Gemini"], index=0 if st.session_state.ai_model == "OpenAI" else 1)

# Timezone selector
st.sidebar.markdown("---")
st.sidebar.markdown("**Timezone**")
st.session_state.selected_tz = st.sidebar.selectbox("Select Timezone:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1)
tz_label = st.session_state.selected_tz

# Watchlist management
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist Management")
new_watchlist_name = st.sidebar.text_input("New Watchlist Name:")
if st.sidebar.button("Create Watchlist", use_container_width=True):
    if new_watchlist_name and new_watchlist_name not in st.session_state.watchlists:
        st.session_state.watchlists[new_watchlist_name] = []
        st.session_state.active_watchlist = new_watchlist_name
        st.sidebar.success(f"Watchlist '{new_watchlist_name}' created!")

active_list = st.sidebar.selectbox(
    "Active Watchlist:",
    list(st.session_state.watchlists.keys()),
    index=list(st.session_state.watchlists.keys()).index(st.session_state.active_watchlist)
)
st.session_state.active_watchlist = active_list

if st.sidebar.button("Delete Active Watchlist", use_container_width=True):
    if active_list != "Default":
        del st.session_state.watchlists[active_list]
        st.session_state.active_watchlist = "Default"
        st.sidebar.success(f"Watchlist '{active_list}' deleted.")
        st.rerun()
    else:
        st.sidebar.warning("Cannot delete the 'Default' watchlist.")

# Search bar to add tickers
st.sidebar.markdown("---")
st.sidebar.subheader("Add Ticker")
search_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., QQQ):").upper()
if st.sidebar.button("Add to Watchlist", use_container_width=True):
    if search_ticker:
        add_ticker_to_watchlist(search_ticker)

st.sidebar.markdown("---")
st.session_state.show_sparklines = st.sidebar.checkbox("Show Sparklines", value=st.session_state.show_sparklines)
st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
if st.session_state.auto_refresh:
    st.session_state.refresh_interval = st.sidebar.radio("Refresh Interval:", [10, 30, 60], index=1, format_func=lambda x: f"{x}s")

# Main content tabs
tabs = st.tabs(["Dashboard", "Market Movers", "My Watchlist", "ETF Tracker", "Auto Plays", "Important News", "Settings"])

# TAB 1: Dashboard
with tabs[0]:
    st.subheader("📊 Market Dashboard")
    
    # AI Market Analysis
    st.markdown("---")
    st.markdown(f"**AI Market Analysis** powered by **{st.session_state.ai_model}**")
    if st.button("Generate Market Analysis", use_container_width=True, type="primary"):
        with st.spinner(f"Generating market analysis with {st.session_state.ai_model}..."):
            news_items = get_all_news()
            gainers, losers, high_volume = get_top_gainers_losers_volume_multithreaded()
            movers = gainers + losers
            analysis_text = ai_market_analysis(news_items, movers, st.session_state.ai_model)
            st.markdown(analysis_text)

# TAB 2: Market Movers
with tabs[1]:
    st.subheader("📈 Top Market Movers")
    gainers, losers, high_volume = get_top_gainers_losers_volume_multithreaded()
    mover_tab1, mover_tab2, mover_tab3 = st.tabs(["🚀 Top Gainers", "🔻 Top Losers", "🌊 High Volume"])
    with mover_tab1:
        display_movers(gainers)
    with mover_tab2:
        display_movers(losers)
    with mover_tab3:
        display_movers(high_volume)

# TAB 3: My Watchlist
with tabs[2]:
    st.subheader(f"👀 My Watchlist: {st.session_state.active_watchlist}")
    current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
    if not current_watchlist:
        st.info("Your watchlist is empty. Add some tickers to get started!")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            ticker_quotes = {ticker: executor.submit(get_live_quote, ticker, tz_label) for ticker in current_watchlist}
            for ticker, future in ticker_quotes.items():
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 2, 2])
                    try:
                        quote = future.result()
                        if quote["error"]:
                            st.error(f"{ticker}: {quote['error']}")
                            continue
                        if col5.button("Remove", key=f"remove_{ticker}", type="secondary"):
                            st.session_state.watchlists[st.session_state.active_watchlist].remove(ticker)
                            st.rerun()
                        if col5.button(f"Generate {st.session_state.ai_model} Playbook", key=f"ai_{ticker}", type="primary"):
                            with st.spinner("Generating AI Playbook..."):
                                news = get_finnhub_news(ticker)
                                catalyst = news[0].get('headline', '')[:100] + "..." if news else ""
                                playbook_text = ai_playbook(ticker, quote['change_percent'], catalyst, st.session_state.ai_model)
                                st.markdown(f"### {ticker} AI Playbook")
                                st.markdown(playbook_text)
                        col1.metric(label=ticker, value=f"${quote['last']:.2f}", delta=f"{quote['change_percent']:+.2f}%")
                        col2.write(f"**Bid/Ask**")
                        col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                        col3.write(f"**Volume**")
                        col3.write(f"{quote['volume']:,}")
                        if st.session_state.show_sparklines:
                            try:
                                history = yf.Ticker(ticker).history(period="1d", interval="5m")
                                if not history.empty:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=history.index, y=history['Close'], mode='lines', name='Price'))
                                    fig.update_layout(
                                        height=100,
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        xaxis=dict(visible=False),
                                        yaxis=dict(visible=False),
                                        showlegend=False,
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)'
                                    )
                                    with col4:
                                        st.plotly_chart(fig, use_container_width=True)
                            except:
                                col4.write("Sparkline not available.")
                        with st.expander(f"📰 Recent News for {ticker}"):
                            news = get_finnhub_news(ticker)
                            if news:
                                for item in news[:3]:
                                    st.write(f"**{item.get('headline', 'No title')}**")
                                    st.write(item.get('summary', 'No summary')[:200] + "...")
                                    st.write("---")
                            else:
                                st.info(f"No recent news found for {ticker}.")
                    except Exception as e:
                        st.error(f"Failed to fetch data for {ticker}: {e}")

# TAB 4: ETF Tracker
with tabs[3]:
    st.subheader("🌐 ETF & Sector Tracker")
    st.info("Track major ETFs to gauge sector and market-wide performance.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        etf_quotes = {ticker: executor.submit(get_live_quote, ticker, tz_label) for ticker in ETF_TICKERS}
        for ticker, future in etf_quotes.items():
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                try:
                    quote = future.result()
                    if quote["error"]:
                        st.error(f"{ticker}: {quote['error']}")
                        continue
                    col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.write("**Bid/Ask**")
                    col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                    col3.write("**Volume**")
                    col3.write(f"{quote['volume']:,}")
                    col3.caption(f"Updated: {quote['last_updated']}")
                    if col4.button(f"Add {ticker} to Watchlist", key=f"add_etf_{ticker}"):
                        add_ticker_to_watchlist(ticker)
                except Exception as e:
                    st.error(f"Failed to fetch data for {ticker}: {e}")

# TAB 5: Auto Plays
with tabs[4]:
    st.subheader("✨ Auto-Generated Plays")
    st.info("AI-powered plays generated by scanning your watchlist and the broader market for opportunities.")
    if st.button("Generate Auto Plays Now", use_container_width=True, type="primary"):
        with st.spinner(f"Generating auto-plays with {st.session_state.ai_model}..."):
            plays = ai_auto_generate_plays(tz_label, st.session_state.ai_model)
            if plays:
                for play_item in plays:
                    st.markdown(f"### 📈 {play_item['ticker']} - AI Trading Play")
                    st.markdown(play_item['play'])
                    st.markdown("---")
            else:
                st.info("No significant movers found to generate plays for. Try again later.")

# TAB 6: Important News & Events
with tabs[5]:
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

# TAB 7: Settings
with tabs[6]:
    st.subheader("⚙️ Settings")
    st.info("Adjust application settings and API configurations.")
    st.markdown("### API Keys")
    st.warning("Note: Storing keys in plain text is not recommended for production apps. Use Streamlit Secrets for a secure way to manage API keys.")
    st.text_input("Finnhub API Key:", value=FINNHUB_KEY, type="password", help="Enter your Finnhub API key")
    st.text_input("Polygon API Key:", value=POLYGON_KEY, type="password", help="Enter your Polygon API key")
    st.text_input("OpenAI API Key:", value=OPENAI_KEY, type="password", help="Enter your OpenAI API key")
    st.text_input("Gemini API Key:", value=GEMINI_KEY, type="password", help="Enter your Gemini API key")
    st.markdown("### Display Options")
    st.session_state.show_sparklines = st.checkbox("Show Sparklines", value=st.session_state.show_sparklines)
    st.markdown("### Auto Refresh")
    st.session_state.auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.radio("Refresh Interval (seconds):", [10, 30, 60], index=1, format_func=lambda x: f"{x}s")

# Auto refresh
if st.session_state.auto_refresh:
    time.sleep(0.1)
    if st.session_state.refresh_interval == 10:
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"\
    "🔥 AI Radar Pro | Live data: yfinance | News: Finnhub/Polygon | AI: OpenAI/Gemini"\
    "</div>",
    unsafe_allow_html=True
)
