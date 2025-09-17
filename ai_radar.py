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
import google.generativeai as genai
import openai
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
    st.session_state.refresh_interval = 15  # Faster default
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Unusual Whales"  # UW as default
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"

# API Keys
try:
    UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROK_API_KEY = st.secrets.get("GROK_API_KEY", "")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    TWELVEDATA_KEY = st.secrets.get("TWELVEDATA_API_KEY", "")

    # Initialize AI clients
    openai_client = None
    gemini_model = None
    grok_client = None
    
    if OPENAI_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    
    if GROK_API_KEY:
        # Initialize Grok client (using OpenAI-compatible API)
        grok_client = openai.OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )

except Exception as e:
    st.error(f"Error loading API keys: {e}")
    openai_client = None
    gemini_model = None
    grok_client = None

# ===== UNUSUAL WHALES INTEGRATION (PRIMARY) =====
def get_stock_state(ticker: str) -> dict:
    """Get stock state from UW"""
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/stock-state"
    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
        "accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        result = r.json()
        if not result or 'ticker' not in result:
            raise ValueError("Invalid UW response")
        return result
    except Exception as e:
        st.warning(f"UW error: {e} — fallback triggered.")
        return get_stock_data_fallback(ticker)

def get_option_chains(ticker: str) -> dict:
    """Get options chain from UW"""
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/option-chains"
    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
        "accept": "application/json"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"UW options error: {e}")
        return {}

def get_unusual_trades(ticker: str) -> dict:
    """Get unusual trades from UW"""
    url = f"https://api.unusualwhales.com/api/historic_chains/{ticker}"
    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
        "accept": "application/json"
    }
    params = {
        "limit": 30,
        "direction": "all",
        "order": "desc"
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("chains", [])
    except Exception as e:
        st.warning(f"UW trades error: {e}")
        return []

def get_stock_data_fallback(ticker: str) -> Dict:
    """Fallback to yfinance when UW fails"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_1d = stock.history(period="1d", interval="1m", prepost=True)
        
        if hist_1d.empty:
            hist_1d = stock.history(period="1d", prepost=True)
        
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        previous_close = info.get('previousClose', 0)
        
        return {
            "ticker": ticker,
            "price": current_price,
            "change": current_price - previous_close if previous_close else 0,
            "change_percent": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "previous_close": previous_close,
            "open": float(info.get('regularMarketOpen', 0)),
            "premarket_change_percent": 0,  # Simplified
            "intraday_change_percent": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            "afterhours_change_percent": 0,
            "data_source": "Yahoo Finance (Fallback)"
        }
    except Exception as e:
        return {"error": f"Fallback error: {str(e)}"}

class GrokClient:
    """Enhanced Grok client for trading analysis"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    
    def analyze_trading_setup(self, prompt: str) -> str:
        """Generate trading analysis using Grok"""
        try:
            response = self.client.chat.completions.create(
                model="grok-3",
                messages=[{
                    "role": "system", 
                    "content": "You are an expert trading analyst. Provide concise, actionable trading analysis with specific entry/exit levels and risk management."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Grok Analysis Error: {str(e)}"
    
    def get_twitter_market_sentiment(self, ticker: str = None) -> str:
        """Get Twitter sentiment analysis"""
        if ticker:
            prompt = f"Analyze recent Twitter sentiment and discussion for ${ticker} stock. What are traders saying?"
        else:
            prompt = "Analyze overall market sentiment on Twitter/X. What are the main themes in trading discussions today?"
        
        return self.analyze_trading_setup(prompt)
    
    def analyze_social_catalyst(self, ticker: str, timeframe: str = "24h") -> str:
        """Analyze social media catalysts"""
        prompt = f"Analyze social media catalysts and rumors for ${ticker} over the last {timeframe}. What news or events are driving discussion?"
        return self.analyze_trading_setup(prompt)
        
# Initialize enhanced Grok client
grok_enhanced = GrokClient(GROK_API_KEY) if GROK_API_KEY else None

# Alpha Vantage Client
class AlphaVantageClient:
    """Enhanced Alpha Vantage client for real-time stock data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Radar-Pro/1.0"
        })
        
    def get_quote(self, symbol: str) -> Dict:
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if "Global Quote" in data:
                    quote_data = data["Global Quote"]
                    
                    price = float(quote_data.get("05. price", 0))
                    change = float(quote_data.get("09. change", 0))
                    change_percent = float(quote_data.get("10. change percent", "0%").replace("%", ""))
                    volume = int(quote_data.get("06. volume", 0))
                    
                    return {
                        "last": price,
                        "bid": price - 0.01,  # Approximate
                        "ask": price + 0.01,  # Approximate
                        "volume": volume,
                        "change": change,
                        "change_percent": change_percent,
                        "premarket_change": 0,
                        "intraday_change": change_percent,
                        "postmarket_change": 0,
                        "previous_close": price - change,
                        "market_open": price - change,
                        "last_updated": datetime.datetime.now().isoformat(),
                        "data_source": "Alpha Vantage",
                        "error": None
                    }
                else:
                    return {"error": f"No data found for {symbol}", "data_source": "Alpha Vantage"}
            else:
                return {"error": f"API error: {response.status_code}", "data_source": "Alpha Vantage"}
                
        except Exception as e:
            return {"error": f"Alpha Vantage error: {str(e)}", "data_source": "Alpha Vantage"}

# Twelve Data Client
class TwelveDataClient:
    """Twelve Data client for real-time stock data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        
    def get_quote(self, symbol: str) -> Dict:
        try:
            params = {
                "symbol": symbol,
                "interval": "1min",
                "outputsize": "1",
                "apikey": self.api_key
            }
            
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "status" in data and data["status"] == "error":
                    return {"error": f"Twelve Data API Error: {data.get('message', 'Unknown error')}", "data_source": "Twelve Data", "raw_data": data}
                
                if "values" in data and len(data["values"]) > 0:
                    latest = data["values"][0]
                    
                    price = float(latest.get("close", 0))
                    open_price = float(latest.get("open", price))
                    high_price = float(latest.get("high", price))
                    low_price = float(latest.get("low", price))
                    volume = int(latest.get("volume", 0))
                    
                    change = price - open_price
                    change_percent = ((price - open_price) / open_price * 100) if open_price > 0 else 0
                    
                    if price > 0:
                        return {
                            "last": price,
                            "bid": low_price,
                            "ask": high_price,
                            "volume": volume,
                            "change": change,
                            "change_percent": change_percent,
                            "premarket_change": 0,
                            "intraday_change": change_percent,
                            "postmarket_change": 0,
                            "previous_close": open_price,
                            "market_open": open_price,
                            "last_updated": datetime.datetime.now().isoformat(),
                            "data_source": "Twelve Data",
                            "error": None,
                            "raw_data": data
                        }
                
                # Try with exchange specified
                params["symbol"] = f"{symbol}:NASDAQ"
                response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if "values" in data and len(data["values"]) > 0:
                        latest = data["values"][0]
                        
                        price = float(latest.get("close", 0))
                        open_price = float(latest.get("open", price))
                        high_price = float(latest.get("high", price))
                        low_price = float(latest.get("low", price))
                        volume = int(latest.get("volume", 0))
                        
                        change = price - open_price
                        change_percent = ((price - open_price) / open_price * 100) if open_price > 0 else 0
                        
                        if price > 0:
                            return {
                                "last": price,
                                "bid": low_price,
                                "ask": high_price,
                                "volume": volume,
                                "change": change,
                                "change_percent": change_percent,
                                "premarket_change": 0,
                                "intraday_change": change_percent,
                                "postmarket_change": 0,
                                "previous_close": open_price,
                                "market_open": open_price,
                                "last_updated": datetime.datetime.now().isoformat(),
                                "data_source": "Twelve Data",
                                "error": None,
                                "raw_data": data
                            }
            else:
                return {"error": f"API error: {response.status_code}", "data_source": "Twelve Data"}
                
        except Exception as e:
            return {"error": f"Twelve Data error: {str(e)}", "data_source": "Twelve Data"}

# Initialize data clients
alpha_vantage_client = AlphaVantageClient(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else None
twelvedata_client = TwelveDataClient(TWELVEDATA_KEY) if TWELVEDATA_KEY else None

# Enhanced primary data function - UW FIRST, then original fallbacks
@st.cache_data(ttl=15)  # Faster refresh: 15 seconds
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """
    Get live stock quote using UW FIRST, then original fallbacks
    """
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # TRY UNUSUAL WHALES FIRST (PRIMARY)
    if UNUSUAL_WHALES_KEY:
        try:
            uw_data = get_stock_state(ticker)
            if not uw_data.get("error") and uw_data.get("price", 0) > 0:
                # Transform UW data to your format
                price = float(uw_data.get('price', 0))
                change = float(uw_data.get('change', 0))
                change_percent = float(uw_data.get('change_percent', 0))
                
                return {
                    "last": price,
                    "bid": float(uw_data.get('bid', price - 0.01)),
                    "ask": float(uw_data.get('ask', price + 0.01)),
                    "volume": int(uw_data.get('volume', 0)),
                    "change": change,
                    "change_percent": change_percent,
                    "premarket_change": float(uw_data.get('premarket_change_percent', 0)),
                    "intraday_change": change_percent,
                    "postmarket_change": float(uw_data.get('afterhours_change_percent', 0)),
                    "previous_close": float(uw_data.get('previous_close', price - change)),
                    "market_open": float(uw_data.get('open', 0)),
                    "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
                    "error": None,
                    "data_source": "Unusual Whales"
                }
        except Exception as e:
            print(f"UW error for {ticker}: {str(e)}")
    
    # Try Twelve Data second (if available)
    if twelvedata_client:
        try:
            twelve_quote = twelvedata_client.get_quote(ticker)
            if not twelve_quote.get("error") and twelve_quote.get("last", 0) > 0:
                twelve_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return twelve_quote
        except Exception as e:
            print(f"Twelve Data error for {ticker}: {str(e)}")
    
    # Try Alpha Vantage third (if available)
    if alpha_vantage_client:
        try:
            alpha_quote = alpha_vantage_client.get_quote(ticker)
            if not alpha_quote.get("error") and alpha_quote.get("last", 0) > 0:
                alpha_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return alpha_quote
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {str(e)}")
    
    # Fall back to Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data with extended hours
        hist_2d = stock.history(period="2d", interval="1m", prepost=True)
        hist_1d = stock.history(period="1d", interval="1m", prepost=True)
        
        if hist_1d.empty:
            hist_1d = stock.history(period="1d", prepost=True)
        if hist_2d.empty:
            hist_2d = stock.history(period="2d", prepost=True)
        
        # Current price
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        
        # Session data
        regular_market_open = info.get('regularMarketOpen', 0)
        previous_close = info.get('previousClose', hist_2d['Close'].iloc[-2] if len(hist_2d) >= 2 else 0)
        
        # Calculate session changes
        premarket_change = 0
        intraday_change = 0
        postmarket_change = 0
        
        # Enhanced session tracking
        if not hist_1d.empty and len(hist_1d) > 0:
            try:
                # Convert to timezone-aware
                hist_1d_tz = hist_1d.copy()
                if hist_1d_tz.index.tz is None:
                    hist_1d_tz.index = hist_1d_tz.index.tz_localize('America/New_York')
                else:
                    hist_1d_tz.index = hist_1d_tz.index.tz_convert('America/New_York')
                
                # Filter for different sessions
                market_hours = hist_1d_tz.between_time('09:30', '16:00')
                
                if not market_hours.empty:
                    market_open_price = market_hours['Open'].iloc[0]
                    market_close_price = market_hours['Close'].iloc[-1]
                    
                    # Premarket change
                    if previous_close and market_open_price:
                        premarket_change = ((market_open_price - previous_close) / previous_close) * 100
                    
                    # Intraday change
                    if market_open_price and market_close_price:
                        intraday_change = ((market_close_price - market_open_price) / market_open_price) * 100
                    
                    # After hours change
                    current_hour = datetime.datetime.now(tz_zone).hour
                    if (current_hour >= 16 or current_hour < 4) and current_price != market_close_price:
                        postmarket_change = ((current_price - market_close_price) / market_close_price) * 100
                        
            except Exception:
                # Fallback to basic calculation
                if regular_market_open and previous_close:
                    premarket_change = ((regular_market_open - previous_close) / previous_close) * 100
                    if current_price and regular_market_open:
                        intraday_change = ((current_price - regular_market_open) / regular_market_open) * 100
        
        # Total change
        total_change = ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
        change_dollar = current_price - previous_close if previous_close else 0
        
        return {
            "last": float(current_price),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "change": float(change_dollar),
            "change_percent": float(total_change),
            "premarket_change": float(premarket_change),
            "intraday_change": float(intraday_change),
            "postmarket_change": float(postmarket_change),
            "previous_close": float(previous_close),
            "market_open": float(regular_market_open) if regular_market_open else 0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None,
            "data_source": "Yahoo Finance"
        }
        
    except Exception as e:
        tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
        return {
            "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
            "change": 0.0, "change_percent": 0.0,
            "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
            "previous_close": 0.0, "market_open": 0.0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": str(e),
            "data_source": "Yahoo Finance"
        }

# Enhanced Technical Analysis using multiple data sources
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_comprehensive_technical_analysis(ticker: str) -> Dict:
    """Enhanced technical analysis with multiple indicators and timeframes"""
    try:
        # Try Twelve Data first for more comprehensive data
        if twelvedata_client:
            # Get multiple timeframes from Twelve Data
            timeframes = ["1day", "4h", "1h"]
            analysis_data = {}
            
            for tf in timeframes:
                try:
                    params = {
                        "symbol": ticker,
                        "interval": tf,
                        "outputsize": "50",
                        "apikey": TWELVEDATA_KEY
                    }
                    
                    response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if "values" in data and len(data["values"]) > 0:
                            analysis_data[tf] = data["values"]
                except Exception:
                    continue
            
            if analysis_data:
                # Calculate comprehensive indicators
                indicators = calculate_advanced_indicators(analysis_data, ticker)
                return indicators
        
        # Fallback to yfinance with enhanced analysis
        stock = yf.Ticker(ticker)
        hist_1d = stock.history(period="1d", interval="5m")
        hist_5d = stock.history(period="5d", interval="15m") 
        hist_1mo = stock.history(period="1mo")
        hist_3mo = stock.history(period="3mo")
        
        if hist_3mo.empty:
            return {"error": "No historical data available"}
        
        # Calculate multiple timeframe analysis
        indicators = {
            "short_term": calculate_indicators(hist_1d, "1D"),
            "medium_term": calculate_indicators(hist_5d, "5D"), 
            "long_term": calculate_indicators(hist_3mo, "3M"),
            "trend_analysis": analyze_trend_strength(hist_3mo),
            "volatility": calculate_volatility_metrics(hist_3mo),
            "momentum": calculate_momentum_indicators(hist_3mo),
            "support_resistance": find_support_resistance_levels(hist_3mo),
            "signal_strength": calculate_signal_strength(hist_3mo)
        }
        
        return indicators
        
    except Exception as e:
        return {"error": f"Technical analysis error: {str(e)}"}

def calculate_advanced_indicators(data_dict: Dict, ticker: str) -> Dict:
    """Calculate advanced technical indicators from Twelve Data"""
    try:
        # Convert Twelve Data format to DataFrame for analysis
        daily_data = data_dict.get("1day", [])
        if not daily_data:
            return {"error": "No daily data available"}
        
        df = pd.DataFrame(daily_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_index()
        
        # Calculate comprehensive indicators
        indicators = {}
        
        # Trend Indicators
        indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        indicators['macd_histogram'] = (macd - signal).iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        indicators['bb_upper'] = (sma20 + (std20 * 2)).iloc[-1]
        indicators['bb_lower'] = (sma20 - (std20 * 2)).iloc[-1]
        indicators['bb_position'] = (df['close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume indicators
        indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift())
        low_close_prev = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(14).mean().iloc[-1]
        
        # Price levels
        indicators['current_price'] = df['close'].iloc[-1]
        indicators['price_change_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        
        # Support/Resistance
        recent_highs = df['high'].rolling(10).max()
        recent_lows = df['low'].rolling(10).min()
        indicators['resistance'] = recent_highs.iloc[-5:].max()
        indicators['support'] = recent_lows.iloc[-5:].min()
        
        # Trend analysis
        if indicators['sma_20'] and indicators['sma_50']:
            if indicators['current_price'] > indicators['sma_20'] > indicators['sma_50']:
                indicators['trend_analysis'] = "Strong Bullish"
            elif indicators['current_price'] > indicators['sma_20']:
                indicators['trend_analysis'] = "Bullish"
            elif indicators['current_price'] < indicators['sma_20'] < indicators['sma_50']:
                indicators['trend_analysis'] = "Strong Bearish"
            else:
                indicators['trend_analysis'] = "Bearish"
        else:
            indicators['trend_analysis'] = "Neutral"
        
        return indicators
        
    except Exception as e:
        return {"error": f"Advanced indicators calculation error: {str(e)}"}

def calculate_indicators(df: pd.DataFrame, timeframe: str) -> Dict:
    """Calculate indicators for a specific timeframe"""
    if df.empty:
        return {"error": f"No data for {timeframe}"}
    
    try:
        # RSI
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Moving Averages
        sma_20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].mean()
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_val = macd.iloc[-1] - signal.iloc[-1] if len(macd) > 0 else 0
        
        # Volume analysis
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
        volume_ratio = df['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        return {
            "timeframe": timeframe,
            "rsi": rsi,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "macd": macd_val,
            "volume_ratio": volume_ratio,
            "current_price": df['Close'].iloc[-1]
        }
    except Exception as e:
        return {"error": f"Indicator calculation error for {timeframe}: {str(e)}"}

def analyze_trend_strength(df: pd.DataFrame) -> str:
    """Analyze trend strength"""
    if df.empty or len(df) < 50:
        return "Insufficient data"
    
    try:
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "Strong Bullish"
        elif current_price > sma_20:
            return "Bullish"
        elif current_price < sma_20 < sma_50:
            return "Strong Bearish"
        elif current_price < sma_20:
            return "Bearish"
        else:
            return "Sideways"
    except:
        return "Unknown"

def calculate_volatility_metrics(df: pd.DataFrame) -> Dict:
    """Calculate volatility metrics"""
    if df.empty:
        return {"error": "No data"}
    
    try:
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return {
            "daily_volatility": returns.std(),
            "annualized_volatility": volatility,
            "volatility_level": "High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low"
        }
    except:
        return {"error": "Volatility calculation failed"}

def calculate_momentum_indicators(df: pd.DataFrame) -> Dict:
    """Calculate momentum indicators"""
    if df.empty or len(df) < 14:
        return {"error": "Insufficient data"}
    
    try:
        # Rate of Change
        roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-14]) / df['Close'].iloc[-14]) * 100
        
        # Momentum
        momentum = df['Close'].iloc[-1] - df['Close'].iloc[-10] if len(df) >= 10 else 0
        
        return {
            "roc_14": roc,
            "momentum_10": momentum,
            "momentum_signal": "Strong" if abs(roc) > 5 else "Weak"
        }
    except:
        return {"error": "Momentum calculation failed"}

def find_support_resistance_levels(df: pd.DataFrame) -> Dict:
    """Find support and resistance levels"""
    if df.empty or len(df) < 20:
        return {"error": "Insufficient data"}
    
    try:
        # Recent highs and lows
        recent_high = df['High'].rolling(20).max().iloc[-1]
        recent_low = df['Low'].rolling(20).min().iloc[-1]
        
        # Pivot points (simplified)
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        return {
            "support": min(recent_low, s1),
            "resistance": max(recent_high, r1),
            "pivot": pivot
        }
    except:
        return {"error": "Support/Resistance calculation failed"}

def calculate_signal_strength(df: pd.DataFrame) -> str:
    """Calculate overall signal strength"""
    if df.empty:
        return "No Signal"
    
    try:
        # Simple signal strength based on multiple factors
        signals = 0
        
        # Price vs SMA
        if len(df) >= 20:
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            if df['Close'].iloc[-1] > sma_20:
                signals += 1
            else:
                signals -= 1
        
        # Volume
        if len(df) >= 20:
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            if df['Volume'].iloc[-1] > avg_volume * 1.5:
                signals += 1
        
        # RSI
        if len(df) >= 14:
            delta = df['Close'].diff(1)
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            if 30 <= rsi <= 70:  # Not overbought/oversold
                signals += 1
        
        if signals >= 2:
            return "Strong"
        elif signals >= 1:
            return "Moderate"
        else:
            return "Weak"
    except:
        return "Unknown"

# Enhanced Fundamental Analysis
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_fundamental_analysis(ticker: str) -> Dict:
    """Comprehensive fundamental analysis using yfinance and external APIs"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Financial metrics
        fundamentals = {
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', None),
            "forward_pe": info.get('forwardPE', None),
            "peg_ratio": info.get('pegRatio', None),
            "price_to_book": info.get('priceToBook', None),
            "price_to_sales": info.get('priceToSalesTrailing12Months', None),
            "debt_to_equity": info.get('debtToEquity', None),
            "roe": info.get('returnOnEquity', None),
            "roa": info.get('returnOnAssets', None),
            "profit_margin": info.get('profitMargins', None),
            "operating_margin": info.get('operatingMargins', None),
            "current_ratio": info.get('currentRatio', None),
            "quick_ratio": info.get('quickRatio', None),
            "revenue_growth": info.get('revenueGrowth', None),
            "earnings_growth": info.get('earningsGrowth', None),
            "beta": info.get('beta', None),
            "dividend_yield": info.get('dividendYield', None),
            "payout_ratio": info.get('payoutRatio', None),
            "free_cashflow": info.get('freeCashflow', None),
            "operating_cashflow": info.get('operatingCashflow', None),
            "total_cash": info.get('totalCash', None),
            "total_debt": info.get('totalDebt', None),
            "book_value": info.get('bookValue', None),
            "shares_outstanding": info.get('sharesOutstanding', None),
            "float_shares": info.get('floatShares', None),
            "insider_ownership": info.get('heldPercentInsiders', None),
            "institutional_ownership": info.get('heldPercentInstitutions', None),
            "short_ratio": info.get('shortRatio', None),
            "short_percent": info.get('shortPercentOfFloat', None),
            "analyst_rating": info.get('recommendationMean', None),
            "target_price": info.get('targetMeanPrice', None),
            "52_week_high": info.get('fiftyTwoWeekHigh', None),
            "52_week_low": info.get('fiftyTwoWeekLow', None),
            "sector": info.get('sector', None),
            "industry": info.get('industry', None)
        }
        
        # Calculate derived metrics
        current_price = info.get('currentPrice', 0)
        if fundamentals['52_week_high'] and fundamentals['52_week_low']:
            fundamentals['price_position'] = ((current_price - fundamentals['52_week_low']) / 
                                           (fundamentals['52_week_high'] - fundamentals['52_week_low'])) * 100
        
        # Financial health score
        fundamentals['financial_health'] = calculate_financial_health_score(fundamentals)
        
        # Valuation assessment
        fundamentals['valuation_assessment'] = assess_valuation(fundamentals)
        
        return fundamentals
        
    except Exception as e:
        return {"error": f"Fundamental analysis error: {str(e)}"}

def calculate_financial_health_score(fundamentals: Dict) -> str:
    """Calculate overall financial health score"""
    score = 0
    max_score = 0
    
    # Profitability metrics
    if fundamentals.get('roe') is not None:
        max_score += 20
        if fundamentals['roe'] > 0.15: score += 20
        elif fundamentals['roe'] > 0.10: score += 15
        elif fundamentals['roe'] > 0.05: score += 10
    
    # Liquidity metrics
    if fundamentals.get('current_ratio') is not None:
        max_score += 15
        if fundamentals['current_ratio'] > 2: score += 15
        elif fundamentals['current_ratio'] > 1.5: score += 12
        elif fundamentals['current_ratio'] > 1: score += 8
    
    # Debt metrics
    if fundamentals.get('debt_to_equity') is not None:
        max_score += 15
        if fundamentals['debt_to_equity'] < 0.3: score += 15
        elif fundamentals['debt_to_equity'] < 0.6: score += 10
        elif fundamentals['debt_to_equity'] < 1: score += 5
    
    # Growth metrics
    if fundamentals.get('revenue_growth') is not None:
        max_score += 15
        if fundamentals['revenue_growth'] > 0.2: score += 15
        elif fundamentals['revenue_growth'] > 0.1: score += 12
        elif fundamentals['revenue_growth'] > 0.05: score += 8
    
    # Valuation metrics
    if fundamentals.get('pe_ratio') is not None:
        max_score += 10
        if 10 <= fundamentals['pe_ratio'] <= 25: score += 10
        elif 5 <= fundamentals['pe_ratio'] <= 35: score += 7
    
    if max_score > 0:
        health_score = (score / max_score) * 100
        if health_score >= 80: return "Excellent"
        elif health_score >= 60: return "Good"
        elif health_score >= 40: return "Fair"
        else: return "Poor"
    
    return "Insufficient Data"

def assess_valuation(fundamentals: Dict) -> str:
    """Assess if stock is undervalued, fairly valued, or overvalued"""
    valuation_signals = []
    
    # P/E Analysis
    pe = fundamentals.get('pe_ratio')
    if pe is not None:
        if pe < 15: valuation_signals.append("Undervalued")
        elif pe > 25: valuation_signals.append("Overvalued")
        else: valuation_signals.append("Fair")
    
    # P/B Analysis
    pb = fundamentals.get('price_to_book')
    if pb is not None:
        if pb < 1.5: valuation_signals.append("Undervalued")
        elif pb > 3: valuation_signals.append("Overvalued")
        else: valuation_signals.append("Fair")
    
    # PEG Analysis
    peg = fundamentals.get('peg_ratio')
    if peg is not None:
        if peg < 1: valuation_signals.append("Undervalued")
        elif peg > 1.5: valuation_signals.append("Overvalued")
        else: valuation_signals.append("Fair")
    
    if not valuation_signals:
        return "Insufficient Data"
    
    # Majority vote
    undervalued = valuation_signals.count("Undervalued")
    overvalued = valuation_signals.count("Overvalued")
    fair = valuation_signals.count("Fair")
    
    if undervalued > overvalued and undervalued > fair:
        return "Undervalued"
    elif overvalued > undervalued and overvalued > fair:
        return "Overvalued"
    else:
        return "Fairly Valued"

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

# New function to fetch option chain data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_option_chain(ticker: str, tz: str = "ET") -> Optional[Dict]:
    """Fetch 0DTE or nearest expiration option chain using UW first, then yfinance"""
    
    # Try UW first for options
    if UNUSUAL_WHALES_KEY:
        try:
            uw_options = get_option_chains(ticker)
            if uw_options and not uw_options.get("error"):
                return {
                    "data": uw_options,
                    "data_source": "Unusual Whales",
                    "error": None
                }
        except Exception as e:
            st.warning(f"UW options error for {ticker}: {e}")
    
    # Fallback to yfinance
    try:
        stock = yf.Ticker(ticker)
        # Get all expiration dates
        expirations = stock.options
        if not expirations:
            return {"error": f"No options data available for {ticker}"}

        # Find today's expiration or closest future date
        today = datetime.datetime.now(ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')).date()
        expiration_dates = [datetime.datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
        valid_expirations = [exp for exp in expiration_dates if exp >= today]
        if not valid_expirations:
            return {"error": f"No valid expirations found for {ticker}"}

        # Select 0DTE or closest expiration
        target_expiration = min(valid_expirations, key=lambda x: (x - today).days)
        expiration_str = target_expiration.strftime('%Y-%m-%d')

        # Fetch option chain
        option_chain = stock.option_chain(expiration_str)
        calls = option_chain.calls
        puts = option_chain.puts

        # Clean and format data
        calls = calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        puts = puts[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        
        # Determine moneyness
        current_price = get_live_quote(ticker, tz).get('last', 0)
        calls['moneyness'] = calls['strike'].apply(lambda x: 'ITM' if x < current_price else 'OTM')
        puts['moneyness'] = puts['strike'].apply(lambda x: 'ITM' if x > current_price else 'OTM')

        # Convert IV to percentage
        calls['impliedVolatility'] = calls['impliedVolatility'] * 100
        puts['impliedVolatility'] = puts['impliedVolatility'] * 100

        return {
            "calls": calls,
            "puts": puts,
            "expiration": expiration_str,
            "current_price": current_price,
            "data_source": "Yahoo Finance",
            "error": None
        }
    except Exception as e:
        return {"error": f"Error fetching option chain for {ticker}: {str(e)}"}

# Modified get_options_data to use UW first
def get_options_data(ticker: str) -> Optional[Dict]:
    """Fetch options data using UW first, then fallback"""
    
    # Try UW first
    if UNUSUAL_WHALES_KEY:
        try:
            uw_options = get_option_chains(ticker)
            if uw_options and not uw_options.get("error"):
                # Process UW options data and return simplified metrics
                return {
                    "iv": 0,  # UW data structure may differ
                    "put_call_ratio": 0,
                    "top_call_oi": 0,
                    "top_call_oi_strike": 0,
                    "top_put_oi": 0,
                    "top_put_oi_strike": 0,
                    "high_iv_strike": 0,
                    "total_calls": 0,
                    "total_puts": 0,
                    "data_source": "Unusual Whales",
                    "raw_data": uw_options
                }
        except Exception as e:
            st.warning(f"UW options error: {e}")
    
    # Fallback to yfinance
    option_chain = get_option_chain(ticker, st.session_state.selected_tz)
    if option_chain.get("error"):
        return {"error": option_chain["error"]}

    calls = option_chain.get("calls", pd.DataFrame())
    puts = option_chain.get("puts", pd.DataFrame())
    current_price = option_chain.get("current_price", 0)

    if calls.empty or puts.empty:
        return {"error": "No option chain data available"}

    # Find high IV strike
    high_iv_call = calls[calls['impliedVolatility'] == calls['impliedVolatility'].max()] if not calls.empty else pd.DataFrame()
    high_iv_put = puts[puts['impliedVolatility'] == puts['impliedVolatility'].max()] if not puts.empty else pd.DataFrame()

    return {
        "iv": calls['impliedVolatility'].mean() if not calls.empty else 0,
        "put_call_ratio": puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0,
        "top_call_oi": calls['openInterest'].max() if not calls.empty else 0,
        "top_call_oi_strike": calls[calls['openInterest'] == calls['openInterest'].max()]['strike'].iloc[0] if not calls.empty and calls['openInterest'].max() > 0 else 0,
        "top_put_oi": puts['openInterest'].max() if not puts.empty else 0,
        "top_put_oi_strike": puts[puts['openInterest'] == puts['openInterest'].max()]['strike'].iloc[0] if not puts.empty and puts['openInterest'].max() > 0 else 0,
        "high_iv_strike": high_iv_call['strike'].iloc[0] if not high_iv_call.empty else (high_iv_put['strike'].iloc[0] if not high_iv_put.empty else 0),
        "total_calls": calls['volume'].sum() if not calls.empty else 0,
        "total_puts": puts['volume'].sum() if not puts.empty else 0,
        "data_source": option_chain.get("data_source", "Yahoo Finance")
    }

def get_earnings_calendar() -> List[Dict]:
    """
    Placeholder function for earnings calendar
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    return [
        {"ticker": "MSFT", "date": today, "time": "After Hours", "estimate": "$2.50"},
        {"ticker": "NVDA", "date": today, "time": "Before Market", "estimate": "$1.20"},
        {"ticker": "TSLA", "date": today, "time": "After Hours", "estimate": "$0.75"},
    ]

# Enhanced AI analysis functions with UW integration
def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook with UW data integration"""
    
    # Get UW-enhanced data
    uw_data = None
    unusual_activity = []
    
    if UNUSUAL_WHALES_KEY:
        try:
            uw_stock_data = get_stock_state(ticker)
            if not uw_stock_data.get("error"):
                uw_data = uw_stock_data
            
            unusual_activity = get_unusual_trades(ticker)
        except Exception as e:
            st.warning(f"UW data error: {e}")
    
    # Construct enhanced prompt
    uw_context = ""
    if uw_data:
        uw_context = f"""
**UNUSUAL WHALES DATA:**
- Current Price: ${uw_data.get('price', 0):.2f}
- Change: {uw_data.get('change_percent', 0):+.2f}%
- Volume: {uw_data.get('volume', 0):,}
- Unusual Trades Detected: {len(unusual_activity)}
- Data Source: Unusual Whales
"""
    
    # Get comprehensive analysis data
    with st.spinner(f"Gathering comprehensive data for {ticker}..."):
        quote = get_live_quote(ticker, st.session_state.selected_tz)
        technical_analysis = get_comprehensive_technical_analysis(ticker)
        fundamental_analysis = get_fundamental_analysis(ticker)
        
        # Get news context
        news = get_finnhub_news(ticker)
        news_context = ""
        if news:
            news_context = f"Recent News: {news[0].get('headline', '')[:100]}..."
    
    # Technical summary
    tech_summary = "Technical Analysis:\n"
    if technical_analysis.get("error"):
        tech_summary += f"Technical Error: {technical_analysis['error']}\n"
    else:
        if "short_term" in technical_analysis:
            tech_summary += f"- RSI: {technical_analysis['short_term'].get('rsi', 'N/A'):.1f}\n"
            tech_summary += f"- SMA20: ${technical_analysis['short_term'].get('sma_20', 0):.2f}\n"
            tech_summary += f"- MACD: {technical_analysis['short_term'].get('macd', 0):.3f}\n"
        if "trend_analysis" in technical_analysis:
            tech_summary += f"- Trend: {technical_analysis.get('trend_analysis', 'Unknown')}\n"
        if "support_resistance" in technical_analysis:
            tech_summary += f"- Support: ${technical_analysis.get('support_resistance', {}).get('support', 0):.2f}\n"
            tech_summary += f"- Resistance: ${technical_analysis.get('support_resistance', {}).get('resistance', 0):.2f}\n"
    
    # Fundamental summary
    fund_summary = "Fundamental Analysis:\n"
    if fundamental_analysis.get("error"):
        fund_summary += f"Fundamental Error: {fundamental_analysis['error']}\n"
    else:
        fund_summary += f"- P/E Ratio: {fundamental_analysis.get('pe_ratio', 'N/A')}\n"
        fund_summary += f"- Market Cap: ${fundamental_analysis.get('market_cap', 0):,.0f}\n"
        fund_summary += f"- Financial Health: {fundamental_analysis.get('financial_health', 'Unknown')}\n"
        fund_summary += f"- Valuation: {fundamental_analysis.get('valuation_assessment', 'Unknown')}\n"
        fund_summary += f"- Sector: {fundamental_analysis.get('sector', 'Unknown')}\n"
    
    # Options summary
    options_summary = "Options Analysis:\n"
    if options_data and not options_data.get("error"):
        options_summary += f"- Put/Call Ratio: {options_data.get('put_call_ratio', 0):.2f}\n"
        options_summary += f"- IV: {options_data.get('iv', 0):.1f}%\n"
        options_summary += f"- Data Source: {options_data.get('data_source', 'Unknown')}\n"
    else:
        options_summary += "- Options data unavailable\n"
    
    prompt = f"""
Comprehensive Trading Analysis for {ticker}:

**Current Price:** ${quote['last']:.2f} ({quote['change_percent']:+.2f}%)
**Volume:** {quote['volume']:,}
**Data Source:** {quote.get('data_source', 'Unknown')}

{uw_context}

{tech_summary}

{fund_summary}

{options_summary}

**Catalyst:** {catalyst if catalyst else "Market movement analysis"}

**News Context:** {news_context}

Based on this analysis (enhanced with Unusual Whales data when available), provide:

1. **Overall Assessment** (Bullish/Bearish/Neutral) with confidence rating (1-100)
2. **Trading Strategy** (Scalp/Day Trade/Swing/Position/Avoid) with specific timeframe
3. **Entry Strategy**: Exact price levels and conditions
4. **Profit Targets**: 3 realistic target levels with rationale
5. **Risk Management**: Stop loss levels and position sizing guidance
6. **Technical Outlook**: Key levels to watch and breakout scenarios
7. **Fundamental Justification**: How fundamentals support the technical setup
8. **Options Strategy**: Specific options plays if applicable
9. **Unusual Activity**: Analysis of any UW unusual trades detected
10. **Risk Factors**: What could invalidate this analysis

Keep analysis under 400 words but be specific and actionable.
"""
    
    if st.session_state.ai_model == "Multi-AI":
        # Use multi-AI consensus
        analyses = {}
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400
                )
                analyses["OpenAI"] = response.choices[0].message.content
            except Exception as e:
                analyses["OpenAI"] = f"OpenAI Error: {str(e)}"
        
        if gemini_model:
            try:
                response = gemini_model.generate_content(prompt)
                analyses["Gemini"] = response.text
            except Exception as e:
                analyses["Gemini"] = f"Gemini Error: {str(e)}"
        
        if grok_enhanced:
            analyses["Grok"] = grok_enhanced.analyze_trading_setup(prompt)
        
        if analyses:
            result = f"## 🐋 UW-Enhanced Multi-AI Analysis for {ticker}\n\n"
            result += f"**Data Sources:** {quote.get('data_source', 'Unknown')} | Updated: {quote['last_updated']}\n\n"
            
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            
            return result
        else:
            return f"**{ticker} Analysis** - No AI models available for multi-AI analysis."
    
    elif st.session_state.ai_model == "OpenAI":
        if not openai_client:
            return f"**{ticker} Analysis** (OpenAI API not configured)"
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    elif st.session_state.ai_model == "Gemini":
        if not gemini_model:
            return f"**{ticker} Analysis** (Gemini API not configured)"
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    
    elif st.session_state.ai_model == "Grok":
        if not grok_enhanced:
            return f"**{ticker} Analysis** (Grok API not configured)"
        return grok_enhanced.analyze_trading_setup(prompt)
    
    else:
        return "No AI model selected or configured."

# Enhanced auto-generation with UW data
def ai_auto_generate_plays_enhanced(tz: str):
    """Enhanced auto-generation with UW data integration"""
    plays = []
    
    try:
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        
        # Scan for significant movers with enhanced criteria
        candidates = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(get_live_quote, ticker, tz): ticker for ticker in scan_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    quote = future.result()
                    if not quote.get("error"):
                        # Enhanced criteria for significance
                        volume_significant = quote["volume"] > 1000000  # Minimum volume threshold
                        price_significant = abs(quote["change_percent"]) >= 1.5
                        spread_reasonable = (quote["ask"] - quote["bid"]) / quote["last"] < 0.02  # Max 2% spread
                        
                        if price_significant and volume_significant and spread_reasonable:
                            candidates.append({
                                "ticker": ticker,
                                "quote": quote,
                                "significance": abs(quote["change_percent"]) * (quote["volume"] / 1000000)  # Volume-weighted significance
                            })
                except Exception as exc:
                    print(f'{ticker} generated an exception: {exc}')
        
        # Sort by significance and take top candidates
        candidates.sort(key=lambda x: x["significance"], reverse=True)
        top_candidates = candidates[:5]
        
        # Generate enhanced plays for top candidates
        for candidate in top_candidates:
            ticker = candidate["ticker"]
            quote = candidate["quote"]
            
            # Get UW unusual activity
            unusual_activity = []
            if UNUSUAL_WHALES_KEY:
                try:
                    unusual_activity = get_unusual_trades(ticker)
                except:
                    pass
            
            # Get recent news for context
            news = get_finnhub_news(ticker)
            catalyst = ""
            if news:
                catalyst = news[0].get('headline', '')[:100] + "..."
            elif unusual_activity:
                catalyst = f"Unusual options activity detected: {len(unusual_activity)} trades"
            else:
                catalyst = f"Significant market movement: {quote['change_percent']:+.2f}%"
            
            # Generate AI analysis using selected model
            options_data = get_options_data(ticker)
            play_analysis = ai_playbook(ticker, quote["change_percent"], catalyst, options_data)

            # Create enhanced play dictionary
            play = {
                "ticker": ticker,
                "current_price": quote['last'],
                "change_percent": quote['change_percent'],
                "session_data": {
                    "premarket": quote['premarket_change'],
                    "intraday": quote['intraday_change'],
                    "afterhours": quote['postmarket_change']
                },
                "catalyst": catalyst,
                "play_analysis": play_analysis,
                "volume": quote['volume'],
                "timestamp": quote['last_updated'],
                "data_source": quote.get('data_source', 'Unknown'),
                "unusual_trades": len(unusual_activity),
                "significance_score": candidate["significance"]
            }
            plays.append(play)
        
        return plays
    except Exception as e:
        st.error(f"Error generating enhanced auto plays: {str(e)}")
        return []

# Enhanced market analysis
def ai_market_analysis_enhanced(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis with UW data integration"""
    
    # Gather market-wide technical data
    market_technical = {}
    key_indices = ["SPY", "QQQ", "IWM"]
    
    for index in key_indices:
        try:
            quote = get_live_quote(index)
            technical = get_comprehensive_technical_analysis(index)
            market_technical[index] = {
                "price": quote['last'],
                "change": quote['change_percent'],
                "technical": technical,
                "data_source": quote.get('data_source', 'Unknown')
            }
        except:
            continue
    
    # Construct enhanced market analysis prompt
    market_context = f"""
Market Technical Overview:
{chr(10).join([f"- {symbol}: ${data['price']:.2f} ({data['change']:+.2f}%) - Source: {data['data_source']}" for symbol, data in market_technical.items()])}

Top News Headlines:
{chr(10).join([f"- {item['title']}" for item in news_items[:5]])}

Top Market Movers:
{chr(10).join([f"- {m['ticker']}: {m['change_pct']:+.2f}% - Source: {m.get('data_source', 'Unknown')}" for m in movers[:5]])}

**Enhanced with Unusual Whales data where available**

Provide comprehensive market analysis covering:
1. Overall market sentiment and direction
2. Key technical levels for major indices
3. Risk-on vs risk-off positioning
4. Trading opportunities and strategies
5. Key events and catalysts to watch
6. Impact of unusual options activity

Keep analysis under 300 words but be specific and actionable.
"""
    
    # Use selected AI model for analysis
    if st.session_state.ai_model == "Multi-AI":
        analyses = {}
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": market_context}],
                    temperature=0.3,
                    max_tokens=300
                )
                analyses["OpenAI"] = response.choices[0].message.content
            except Exception as e:
                analyses["OpenAI"] = f"OpenAI Error: {str(e)}"
        
        if gemini_model:
            try:
                response = gemini_model.generate_content(market_context)
                analyses["Gemini"] = response.text
            except Exception as e:
                analyses["Gemini"] = f"Gemini Error: {str(e)}"
        
        if grok_enhanced:
            analyses["Grok"] = grok_enhanced.analyze_trading_setup(market_context)
        
        if analyses:
            result = "## 🐋 UW-Enhanced Multi-AI Market Analysis\n\n"
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            return result
    else:
        # Use individual AI model
        if openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": market_context}],
                    temperature=0.3,
                    max_tokens=300
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"
        else:
            return "AI analysis not available"

def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis using selected AI model"""
    return ai_market_analysis_enhanced(news_items, movers)

# Function to get important economic events using AI
def get_important_events() -> List[Dict]:
    if not openai_client and not gemini_model and not grok_enhanced:
        return []
    
    try:
        prompt = f"""
        Provide a list of the most important economic events for the current week.
        Focus on events that are known to move the market, such as CPI, FOMC meetings,
        Unemployment Reports, and major Fed speeches.
        
        Format the response as a JSON array of objects, with each object having the following keys:
        - "event": (string) The name of the event.
        - "date": (string) The date of the event (e.g., "Monday, June 17").
        - "time": (string) The time of the event (e.g., "10:00 AM ET").
        - "impact": (string) The expected market impact (e.g., "High", "Medium", "Low").
        
        Do not include any text, notes, or explanations outside of the JSON.
        """
        
        if st.session_state.ai_model == "Multi-AI":
            # Use first available model for events
            if openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400
                )
                events = json.loads(response.choices[0].message.content)
                return events
            elif gemini_model:
                response = gemini_model.generate_content(prompt)
                events = json.loads(response.text)
                return events
            elif grok_enhanced:
                response = grok_enhanced.analyze_trading_setup(prompt)
                events = json.loads(response)
                return events
            else:
                return []
        elif st.session_state.ai_model == "OpenAI" and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            events = json.loads(response.choices[0].message.content)
            return events
        elif st.session_state.ai_model == "Gemini" and gemini_model:
            response = gemini_model.generate_content(prompt)
            events = json.loads(response.text)
            return events
        elif st.session_state.ai_model == "Grok" and grok_enhanced:
            response = grok_enhanced.analyze_trading_setup(prompt)
            events = json.loads(response)
            return events
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching economic events: {str(e)}")
        return []

# Multi-AI Analysis System
class MultiAIAnalyzer:
    """Enhanced multi-AI analysis system"""
    
    def __init__(self):
        self.openai_client = openai_client
        self.gemini_model = gemini_model
        self.grok_client = grok_enhanced
        
    def get_available_models(self) -> List[str]:
        """Get list of available AI models"""
        models = []
        if self.openai_client:
            models.append("OpenAI")
        if self.gemini_model:
            models.append("Gemini")
        if self.grok_client:
            models.append("Grok")
        return models

# Initialize Multi-AI Analyzer
multi_ai = MultiAIAnalyzer()

# Main app
st.title("🔥 AI Radar Pro — UW Enhanced Trading Assistant")

# Timezone toggle (made smaller with column and smaller font)
col_tz, _ = st.columns([1, 10])  # Allocate small space for TZ
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed", help="Select Timezone (ET/CT)")

# Get current time in selected TZ
tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
tz_label = st.session_state.selected_tz

# Enhanced AI Settings
st.sidebar.subheader("🤖 AI Configuration")
available_models = ["Multi-AI"] + multi_ai.get_available_models()
st.session_state.ai_model = st.sidebar.selectbox("AI Model", available_models, 
                                                  index=available_models.index(st.session_state.ai_model) if st.session_state.ai_model in available_models else 0)

# Show AI model status
st.sidebar.subheader("AI Models Status")
if openai_client:
    st.sidebar.success("✅ OpenAI Connected")
else:
    st.sidebar.warning("⚠️ OpenAI Not Connected")

if gemini_model:
    st.sidebar.success("✅ Gemini Connected")
else:
    st.sidebar.warning("⚠️ Gemini Not Connected")

if grok_enhanced:
    st.sidebar.success("✅ Grok Connected")
else:
    st.sidebar.warning("⚠️ Grok Not Connected")

# Data Source Status
st.sidebar.subheader("📊 Data Sources")

# UW Status (PRIMARY)
if UNUSUAL_WHALES_KEY:
    st.sidebar.success("🐋 Unusual Whales Connected (PRIMARY)")
else:
    st.sidebar.error("❌ Unusual Whales Not Connected")

# Other sources
if twelvedata_client:
    st.sidebar.success("✅ Twelve Data Connected")
else:
    st.sidebar.warning("⚠️ Twelve Data Not Connected")

if alpha_vantage_client:
    st.sidebar.success("✅ Alpha Vantage Connected")
else:
    st.sidebar.warning("⚠️ Alpha Vantage Not Connected")

st.sidebar.success("✅ Yahoo Finance Connected")

if FINNHUB_KEY:
    st.sidebar.success("✅ Finnhub API Connected")
else:
    st.sidebar.warning("⚠️ Finnhub API Not Found")

if POLYGON_KEY:
    st.sidebar.success("✅ Polygon API Connected")
else:
    st.sidebar.warning("⚠️ Polygon API Not Found")

# Auto-refresh controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

with col2:
    st.session_state.refresh_interval = st.selectbox("Interval", [5, 10, 15, 30], index=2)  # Default 15s

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
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks", "🌐 Sector/ETF Tracking", "🎲 0DTE & Lottos", "🗓️ Earnings Plays", "📰 Important News","🐦 Twitter/X Market Sentiment & Rumors"])

# Global timestamp
data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
data_sources = []
if UNUSUAL_WHALES_KEY:
    data_sources.append("🐋 Unusual Whales (PRIMARY)")
if alpha_vantage_client:
    data_sources.append("Alpha Vantage")
if twelvedata_client:
    data_sources.append("Twelve Data")
data_sources.append("Yahoo Finance")
data_source_info = " + ".join(data_sources)

# AI model info
ai_info = f"AI: {st.session_state.ai_model}"
if st.session_state.ai_model == "Multi-AI":
    active_models = multi_ai.get_available_models()
    ai_info += f" ({'+'.join(active_models)})" if active_models else " (None Available)"

# TAB 1: Live Quotes (Enhanced with UW)
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist & Market Movers")
    
    # Show data source priority
    st.info("🐋 **Primary:** Unusual Whales | **Fallback:** Twelve Data → Alpha Vantage → Yahoo Finance")
    
    # Session status (using selected TZ)
    current_tz_hour = current_tz.hour
    if 4 <= current_tz_hour < 9:
        session_status = "🌅 Premarket"
    elif 9 <= current_tz_hour < 16:
        session_status = "🟢 Market Open"
    else:
        session_status = "🌆 After Hours"
    
    st.markdown(f"**Trading Session ({tz_label}):** {session_status}")
    
    # Search bar for any ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Search Any Stock", placeholder="Enter any ticker (e.g., AAPL, SPY, GME)", key="search_quotes").upper().strip()
    with col2:
        search_quotes = st.button("Get Quote", key="search_quotes_btn")
    
    # Search result for any ticker
    if search_quotes and search_ticker:
        with st.spinner(f"Getting UW-enhanced quote for {search_ticker}..."):
            quote = get_live_quote(search_ticker, tz_label)
            if not quote.get("error"):
                st.success(f"Quote for {search_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Unknown')}")
                
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
                
                # UW Enhanced Analysis Button
                if col4.button(f"🐋 UW Analysis", key=f"quotes_uw_{search_ticker}"):
                    with st.spinner(f"Running UW-enhanced analysis for {search_ticker}..."):
                        options_data = get_options_data(search_ticker)
                        analysis = ai_playbook(search_ticker, quote['change_percent'], "User search query", options_data)
                        st.markdown("### 🐋 UW-Enhanced AI Analysis")
                        st.markdown(analysis)
                
                if col4.button(f"Add {search_ticker} to Watchlist", key="quotes_add_searched_ticker"):
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
    st.markdown("### Your Watchlist")
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab or check Market Movers below.")
    else:
        for ticker in tickers:
            quote = get_live_quote(ticker, tz_label)
            if quote.get("error"):
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
                col3.caption(f"Source: {quote.get('data_source', 'Unknown')}")
                
                if abs(quote['change_percent']) >= 2.0:
                    if col4.button(f"🐋 UW Analysis", key=f"quotes_ai_{ticker}"):
                        with st.spinner(f"Analyzing {ticker}..."):
                            options_data = get_options_data(ticker)
                            analysis = ai_playbook(ticker, quote['change_percent'], "", options_data)
                            st.success(f"🐋 {ticker} Analysis")
                            st.markdown(analysis)
                
                # Session data
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
                    catalyst_title = news[0].get('headline', '') if news else ""
                    options_data = get_options_data(ticker)
                    if options_data:
                        st.write("**Options Metrics:**")
                        opt_col1, opt_col2, opt_col3 = st.columns(3)
                        opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                        opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                        opt_col3.metric("Total Contracts", f"{options_data.get('total_calls', 0) + options_data.get('total_puts', 0):,}")
                        st.caption(f"Source: {options_data.get('data_source', 'Unknown')}")
                    st.markdown(ai_playbook(ticker, quote['change_percent'], catalyst_title, options_data))
                
                st.divider()

    # Top Market Movers
    st.markdown("### 🌟 Top Market Movers")
    st.caption("Stocks with significant intraday movement from CORE_TICKERS")
    movers = []
    for ticker in CORE_TICKERS[:20]:  # Limit to top 20 for performance
        quote = get_live_quote(ticker, tz_label)
        if not quote.get("error"):
            movers.append({
                "ticker": ticker,
                "change_pct": quote["change_percent"],
                "price": quote["last"],
                "volume": quote["volume"],
                "data_source": quote.get("data_source", "Unknown")
            })
    movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    top_movers = movers[:10]  # Show top 10 movers

    for mover in top_movers:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            direction = "🚀" if mover["change_pct"] > 0 else "📉"
            col1.metric(f"{direction} {mover['ticker']}", f"${mover['price']:.2f}", f"{mover['change_pct']:+.2f}%")
            col2.write("**Bid/Ask**")
            col2.write(f"N/A")  # Movers don't include bid/ask in this view
            col3.write("**Volume**")
            col3.write(f"{mover['volume']:,}")
            col3.caption(f"Source: {mover['data_source']}")
            if col4.button(f"Add {mover['ticker']} to Watchlist", key=f"quotes_mover_{mover['ticker']}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if mover['ticker'] not in current_list:
                    current_list.append(mover['ticker'])
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {mover['ticker']} to watchlist!")
                    st.rerun()
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
            if not quote.get("error"):
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
            if st.button(f"+ {ticker}", key=f"watchlist_add_{ticker}"):
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
                    if st.button(f"Remove", key=f"watchlist_remove_{ticker}"):
                        current_tickers.remove(ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                        st.rerun()

# [Continue with remaining tabs using your original code structure...]
# TAB 3: Catalyst Scanner (keeping your original)
with tabs[2]:
    st.subheader("🔥 Enhanced Real-Time Catalyst Scanner")
    st.caption("Enhanced with UW unusual activity detection")
    
    # Search specific stock catalysts
    col1, col2 = st.columns([3, 1])
    with col1:
        search_catalyst_ticker = st.text_input("🔍 Search catalysts for stock", placeholder="Enter ticker", key="search_catalyst").upper().strip()
    with col2:
        search_catalyst = st.button("🔍 Analyze Catalysts", key="search_catalyst_btn")
    
    if search_catalyst and search_catalyst_ticker:
        with st.spinner(f"Searching catalysts and UW data for {search_catalyst_ticker}..."):
            quote = get_live_quote(search_catalyst_ticker, tz_label)
            
            if not quote.get("error"):
                st.success(f"Catalyst Analysis for {search_catalyst_ticker} - Source: {quote.get('data_source', 'Unknown')}")
                
                # Price and volume info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Data Source", quote.get('data_source', 'Unknown'))
                
                # Get UW unusual activity
                unusual_trades = []
                if UNUSUAL_WHALES_KEY:
                    try:
                        unusual_trades = get_unusual_trades(search_catalyst_ticker)
                        col4.metric("UW Unusual Trades", len(unusual_trades))
                    except Exception as e:
                        col4.write("UW data unavailable")
                
                # Session breakdown
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%") 
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                # News analysis
                news = get_finnhub_news(search_catalyst_ticker)
                if news:
                    st.markdown("#### 📰 News Catalysts")
                    for i, item in enumerate(news[:5]):
                        with st.expander(f"📰 {item.get('headline', 'No title')[:80]}... - {item.get('source', 'Finnhub')}"):
                            st.write(f"**Summary:** {item.get('summary', 'No summary')}")
                            if item.get('url'):
                                st.markdown(f"[📖 Read Full Article]({item['url']})")
                
                # UW unusual activity
                if unusual_trades:
                    st.markdown(f"#### 🐋 Unusual Options Activity ({len(unusual_trades)} trades)")
                    for i, trade in enumerate(unusual_trades[:3]):
                        with st.expander(f"UW Trade #{i+1}"):
                            st.json(trade)
                
                # AI Analysis
                if st.button(f"🤖 Comprehensive Analysis", key=f"catalyst_ai_{search_catalyst_ticker}"):
                    catalyst_context = f"News catalysts: {len(news)}, UW unusual trades: {len(unusual_trades)}"
                    options_data = get_options_data(search_catalyst_ticker)
                    analysis = ai_playbook(search_catalyst_ticker, quote['change_percent'], catalyst_context, options_data)
                    st.markdown("### 🤖 AI Catalyst Analysis")
                    st.markdown(analysis)

# [Keep all your remaining tabs with the same structure, just ensuring they use the UW-enhanced get_live_quote function]

# Continue with tabs 4-10 using your original structure...
# [I'll include the rest of your tabs here to maintain your exact layout]

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
            if not quote.get("error"):
                news = get_finnhub_news(search_analysis_ticker)
                catalyst = news[0].get('headline', '') if news else "Recent market movement"
                
                # Get options data for enhanced analysis
                options_data = get_options_data(search_analysis_ticker)
                analysis = ai_playbook(search_analysis_ticker, quote["change_percent"], catalyst, options_data)
                
                st.success(f"🤖 AI Analysis: {search_analysis_ticker} - Source: {quote.get('data_source', 'Unknown')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                if col4.button(f"Add {search_analysis_ticker} to WL", key="analysis_add_searched_ticker"):
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
                
                # Show options data if available
                if options_data:
                    st.markdown("#### Options Metrics")
                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    opt_col1.metric("IV", f"{options_data.get('iv', 0):.1f}%")
                    opt_col2.metric("Put/Call", f"{options_data.get('put_call_ratio', 0):.2f}")
                    opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,}")
                    opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,}")
                    st.caption(f"Source: {options_data.get('data_source', 'Unknown')}")
                
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
                if not quote.get("error"):
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"],
                        "data_source": quote.get("data_source", "Unknown")
                    })
            
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            with st.expander("📊 Supporting Data"):
                st.write("**Top Market Movers:**")
                for mover in sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]:
                    st.write(f"• {mover['ticker']}: {mover['change_pct']:+.2f}% | Source: {mover.get('data_source', 'Unknown')}")
                
                st.write("**Key News Headlines:**")
                for news in news_items[:3]:
                    st.write(f"• {news['title']}")

# TAB 5: AI Playbooks
with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    
    # Show current AI configuration
    st.info(f"🤖 Current AI Mode: **{st.session_state.ai_model}** | Available Models: {', '.join(multi_ai.get_available_models()) if multi_ai.get_available_models() else 'None'}")
    
    # Auto-generated plays section
    st.markdown("### 🎯 Auto-Generated Trading Plays")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("AI automatically scans your watchlist and market movers to suggest trading opportunities")
    with col2:
        if st.button("🚀 Generate Auto Plays", type="primary"):
            with st.spinner("AI generating trading plays from market scan..."):
                auto_plays = ai_auto_generate_plays_enhanced(tz_label)
                
                if auto_plays:
                    st.success(f"🤖 Generated {len(auto_plays)} Trading Plays")
                    
                    for i, play in enumerate(auto_plays):
                        with st.expander(f"🎯 {play['ticker']} - ${play['current_price']:.2f} ({play['change_percent']:+.2f}%) | {play.get('data_source', 'Unknown')}"):
                            
                            # Display session data
                            sess_col1, sess_col2, sess_col3 = st.columns(3)
                            sess_col1.metric("Premarket", f"{play['session_data']['premarket']:+.2f}%")
                            sess_col2.metric("Intraday", f"{play['session_data']['intraday']:+.2f}%")
                            sess_col3.metric("After Hours", f"{play['session_data']['afterhours']:+.2f}%")
                            
                            # Display catalyst
                            if play['catalyst']:
                                st.write(f"**Catalyst:** {play['catalyst']}")
                            
                            # Display UW unusual activity if available
                            if play.get('unusual_trades', 0) > 0:
                                st.info(f"🐋 {play['unusual_trades']} unusual trades detected via UW")
                            
                            # Display AI play analysis
                            st.markdown("**AI Trading Play:**")
                            st.markdown(play['play_analysis'])
                            
                            st.caption(f"Data Source: {play.get('data_source', 'Unknown')} | Updated: {play['timestamp']}")
                            
                            # Add to watchlist option
                            if st.button(f"Add {play['ticker']} to Watchlist", key=f"playbook_auto_{i}_{play['ticker']}"):
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
        
        if not quote.get("error"):
            with st.spinner(f"AI generating playbook for {search_playbook_ticker}..."):
                news = get_finnhub_news(search_playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                
                # Get options data for enhanced playbook
                options_data = get_options_data(search_playbook_ticker)
                playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst, options_data)
                
                st.success(f"✅ {search_playbook_ticker} Trading Playbook - Source: {quote.get('data_source', 'Unknown')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                col3.metric("Volume", f"{quote['volume']:,}")
                if col4.button(f"Add {search_playbook_ticker} to WL", key="playbook_add_searched_ticker"):
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
                
                # Show options data if available
                if options_data:
                    st.markdown("#### Options Analysis")
                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                    opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                    opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,} @ ${options_data.get('top_call_oi_strike', 0)}")
                    opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,} @ ${options_data.get('top_put_oi_strike', 0)}")
                    st.caption(f"Source: {options_data.get('data_source', 'Unknown')}")
                
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
            
            if not quote.get("error"):
                with st.spinner(f"AI analyzing {selected_ticker}..."):
                    # Get options data for enhanced analysis
                    options_data = get_options_data(selected_ticker)
                    playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input, options_data)
                    
                    st.success(f"✅ {selected_ticker} Trading Playbook - Source: {quote.get('data_source', 'Unknown')}")
                    
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
                if not quote.get("error"):
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
        if quote.get("error"):
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
            col3.caption(f"Source: {quote.get('data_source', 'Unknown')}")
            
            if col4.button(f"Add {ticker} to Watchlist", key=f"sector_etf_add_{ticker}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if ticker not in current_list:
                    current_list.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {ticker} to watchlist!")
                    st.rerun()

            st.divider()

# TAB 7: 0DTE & Lottos
with tabs[6]:
    st.subheader("🎲 0DTE & Lotto Plays")
    st.markdown("**High-risk, high-reward options expiring today. Enhanced with UW flow data.**")

    # Ticker selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox("Select Ticker for 0DTE", options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], key="0dte_ticker")
    with col2:
        if st.button("Analyze 0DTE", key="analyze_0dte"):
            st.cache_data.clear()
            st.rerun()

    # Fetch option chain
    with st.spinner(f"Fetching option chain for {selected_ticker}..."):
        option_chain = get_option_chain(selected_ticker, st.session_state.selected_tz)
        quote = get_live_quote(selected_ticker, st.session_state.selected_tz)

    if option_chain.get("error"):
        st.error(option_chain["error"])
    else:
        current_price = quote['last']
        
        if option_chain.get("data_source") == "Unusual Whales":
            st.success(f"🐋 UW Options Data for {selected_ticker}")
            st.markdown(f"**Current Price:** ${current_price:.2f} | **Source:** {quote.get('data_source', 'Unknown')}")
            
            # Display UW options data
            with st.expander("UW Options Chain Data"):
                st.json(option_chain["data"])
        else:
            expiration = option_chain.get("expiration", "Unknown")
            is_0dte = (datetime.datetime.strptime(expiration, '%Y-%m-%d').date() == datetime.datetime.now(ZoneInfo('US/Eastern')).date()) if expiration != "Unknown" else False
            st.markdown(f"**Option Chain for {selected_ticker}** (Expiration: {expiration}{' - 0DTE' if is_0dte else ''})")
            st.markdown(f"**Current Price:** ${current_price:.2f} | **Source:** {quote.get('data_source', 'Unknown')}")

        # AI Analysis at the top
        st.markdown("### 🤖 AI 0DTE Playbook")
        with st.spinner("Generating AI analysis..."):
            # Get UW unusual activity
            unusual_trades = []
            if UNUSUAL_WHALES_KEY:
                try:
                    unusual_trades = get_unusual_trades(selected_ticker)
                except:
                    pass
            
            tech_analysis = get_comprehensive_technical_analysis(selected_ticker)
            catalyst = f"0DTE options analysis. UW unusual trades: {len(unusual_trades)}. Options source: {option_chain.get('data_source', 'Unknown')}"
            
            options_data = get_options_data(selected_ticker)
            playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst, options_data)
            st.markdown(playbook)

        # Display option chain if from yfinance
        if option_chain.get("calls") is not None:
            st.markdown("### Calls")
            calls = option_chain["calls"]
            if not calls.empty:
                display_calls = calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']].copy()
                display_calls.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest', 'IV (%)', 'Moneyness']
                display_calls['IV (%)'] = display_calls['IV (%)'].map('{:.2f}'.format)
                st.dataframe(display_calls, use_container_width=True)
            else:
                st.warning("No call options available.")

            st.markdown("### Puts")
            puts = option_chain["puts"]
            if not puts.empty:
                display_puts = puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']].copy()
                display_puts.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest', 'IV (%)', 'Moneyness']
                display_puts['IV (%)'] = display_puts['IV (%)'].map('{:.2f}'.format)
                st.dataframe(display_puts, use_container_width=True)
            else:
                st.warning("No put options available.")

        # UW Unusual Activity
        if unusual_trades:
            st.markdown(f"### 🐋 UW Unusual Activity ({len(unusual_trades)} trades)")
            for i, trade in enumerate(unusual_trades[:5]):
                with st.expander(f"UW Trade #{i+1}"):
                    st.json(trade)

# TAB 8: Earnings Plays
with tabs[7]:
    st.subheader("🗓️ Top Earnings Plays")
    
    st.write("This section tracks upcoming earnings reports and provides AI analysis for potential earnings plays.")
    st.info("Note: Enhanced with UW data when available. For live earnings calendar, consider integrating a financial calendar API.")
    
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
                    
                    # Get live quote and options data for earnings analysis
                    quote = get_live_quote(ticker)
                    
                    if not quote.get("error"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        col3.metric("Data Source", quote.get('data_source', 'Unknown'))
                        
                        # Get UW unusual activity for earnings context
                        unusual_trades = []
                        if UNUSUAL_WHALES_KEY:
                            try:
                                unusual_trades = get_unusual_trades(ticker)
                            except:
                                pass
                        
                        if unusual_trades:
                            st.info(f"🐋 {len(unusual_trades)} unusual trades detected via UW")
                        
                        options_data = get_options_data(ticker)
                        if options_data and not options_data.get("error"):
                            st.write("**Options Metrics:**")
                            opt_col1, opt_col2, opt_col3 = st.columns(3)
                            opt_col1.metric("IV", f"{options_data.get('iv', 0):.1f}%")
                            opt_col2.metric("Put/Call", f"{options_data.get('put_call_ratio', 0):.2f}")
                            opt_col3.metric("Total OI", f"{options_data.get('top_call_oi', 0) + options_data.get('top_put_oi', 0):,}")
                            st.caption(f"Source: {options_data.get('data_source', 'Unknown')}")
                    
                    if not options_data or options_data.get("error"):
                        ai_analysis = f"""
                        **AI Analysis for {ticker} Earnings:**
                        - **Date:** {report["date"]}
                        - **Time:** {time_str}
                        - **Current Price:** ${quote.get('last', 0):.2f}
                        - **Daily Change:** {quote.get('change_percent', 0):+.2f}%
                        - **Volume:** {quote.get('volume', 0):,}
                        - **Data Source:** {quote.get('data_source', 'Unknown')}
                        - **UW Unusual Trades:** {len(unusual_trades)}
                        
                        **Note:** Monitor for post-earnings volatility. Consider both upside and downside scenarios.
                        """
                    else:
                        earnings_context = f"Earnings {time_str}. UW unusual trades: {len(unusual_trades)}"
                        ai_analysis = ai_playbook(ticker, quote.get("change_percent", 0), earnings_context, options_data)
                    
                    with st.expander(f"🔮 AI Analysis for {ticker}"):
                        st.markdown(ai_analysis)
                    st.divider()

# TAB 9: Important News & Economic Calendar
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

# TAB 10: Twitter/X Market Sentiment & Rumors
with tabs[9]:
    st.subheader("🐦 Twitter/X Market Sentiment & Rumors")

    # Important disclaimer
    st.warning("⚠️ **Risk Disclaimer:** Social media content includes unverified rumors and speculation. "
               "Always verify information through official sources before making trading decisions. "
               "GROK analysis may include both verified news and unconfirmed rumors - trade responsibly.")

    if not grok_enhanced:
        st.error("🔴 Grok API not configured. This tab requires Grok API access for Twitter/X integration.")
        st.info("Please add your Grok API key to access real-time Twitter sentiment and social media catalysts.")
    else:
        st.success("✅ Grok connected for Twitter/X analysis")

        # Overall Market Sentiment
        st.markdown("### 📊 Overall Market Sentiment")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Get real-time Twitter/X sentiment analysis for the overall market")
        with col2:
            if st.button("🔍 Scan Market Sentiment", type="primary"):
                with st.spinner("Grok analyzing Twitter/X market sentiment..."):
                    market_sentiment = grok_enhanced.get_twitter_market_sentiment()
                    st.markdown("### 🐦 Twitter/X Market Analysis")
                    st.markdown(market_sentiment)
                    st.caption("Analysis powered by Grok with real-time Twitter/X access")

        st.divider()

        # Stock-Specific Analysis
        st.markdown("### 🎯 Stock-Specific Social Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            social_ticker = st.text_input(
                "🔍 Analyze Twitter sentiment for stock",
                placeholder="Enter ticker (e.g., TSLA)",
                key="social_ticker"
            ).upper().strip()
        with col2:
            analyze_social = st.button("Analyze Sentiment", key="analyze_social_btn")

        if analyze_social and social_ticker:
            with st.spinner(f"Grok analyzing Twitter/X sentiment for {social_ticker}..."):
                try:
                    # Get current quote for context
                    quote = get_live_quote(social_ticker, tz_label)

                    col1, col2, col3 = st.columns(3)
                    if not quote.get("error"):
                        col1.metric(f"{social_ticker} Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        col3.metric("Data Source", quote.get('data_source', 'Unknown'))

                    # Get Twitter sentiment
                    sentiment_analysis = grok_enhanced.get_twitter_market_sentiment(social_ticker)
                    st.markdown(f"### 🐦 Twitter/X Sentiment for {social_ticker}")
                    st.markdown(sentiment_analysis)

                    # Get social catalysts
                    st.markdown(f"### 🔥 Social Media Catalysts for {social_ticker}")
                    with st.spinner("Scanning for social catalysts..."):
                        catalyst_analysis = grok_enhanced.analyze_social_catalyst(social_ticker)
                        st.markdown(catalyst_analysis)

                    # Add to watchlist option
                    if st.button(f"Add {social_ticker} to Watchlist", key="twitter_add_searched_ticker"):
                        current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                        if social_ticker not in current_list:
                            current_list.append(social_ticker)
                            st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                            st.success(f"Added {social_ticker} to watchlist!")
                            st.rerun()

                except Exception as e:
                    st.error(f"Error analyzing {social_ticker}: {str(e)}")

        st.divider()

        # Watchlist Social Scanning
        tickers = st.session_state.watchlists[st.session_state.active_watchlist]
        if tickers:
            st.markdown("### 📋 Watchlist Social Media Scan")
            selected_social_ticker = st.selectbox(
                "Select from watchlist for social analysis",
                [""] + tickers,
                key="watchlist_social"
            )

            col1, col2 = st.columns([2, 2])
            with col1:
                timeframe = st.selectbox("Timeframe", ["24h", "12h", "6h", "3h"], key="social_timeframe")
            with col2:
                if st.button("🔍 Scan Social Media", key="scan_watchlist_social") and selected_social_ticker:
                    with st.spinner(f"Grok scanning social media for {selected_social_ticker}..."):
                        try:
                            quote = get_live_quote(selected_social_ticker, tz_label)

                            if not quote.get("error"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric(f"{selected_social_ticker} Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                                col2.metric("Volume", f"{quote['volume']:,}")
                                col3.metric("Session", f"PM: {quote['premarket_change']:+.1f}% | "
                                                       f"Day: {quote['intraday_change']:+.1f}% | "
                                                       f"AH: {quote['postmarket_change']:+.1f}%")

                            # Get comprehensive social analysis
                            sentiment = grok_enhanced.get_twitter_market_sentiment(selected_social_ticker)
                            catalysts = grok_enhanced.analyze_social_catalyst(selected_social_ticker, timeframe)

                            st.markdown(f"### 🐦 Social Sentiment: {selected_social_ticker}")
                            st.markdown(sentiment)

                            st.markdown(f"### 🔥 Social Catalysts ({timeframe})")
                            st.markdown(catalysts)

                        except Exception as e:
                            st.error(f"Error scanning social media for {selected_social_ticker}: {str(e)}")
        else:
            st.info("Add stocks to your watchlist to enable watchlist social media scanning.")

        st.divider()

        # Quick Social Sentiment for Popular Tickers
        st.markdown("### ⭐ Popular Stocks Social Sentiment")
        popular_for_social = ["TSLA", "NVDA", "AAPL", "SPY", "QQQ", "MSFT", "META", "AMD"]
        cols = st.columns(4)

        for i, ticker in enumerate(popular_for_social):
            with cols[i % 4]:
                if st.button(f"📊 {ticker}", key=f"twitter_quick_social_{ticker}"):
                    with st.spinner(f"Getting {ticker} social sentiment..."):
                        try:
                            sentiment = grok_enhanced.get_twitter_market_sentiment(ticker)
                            quote = get_live_quote(ticker, tz_label)

                            st.markdown(f"**{ticker} Social Analysis**")
                            if not quote.get("error"):
                                st.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")

                            with st.expander(f"📱 {ticker} Twitter Analysis"):
                                st.markdown(sentiment)

                        except Exception as e:
                            st.error(f"Error getting {ticker} sentiment: {str(e)}")

        with st.expander("💡 Social Media Trading Guidelines"):
            st.markdown("""
            **Using Social Media for Trading Research:**
            
            ✅ Best Practices:
            - Verify information through multiple sources
            - Focus on verified accounts and credible sources
            - Look for consistent themes across multiple posts
            - Use sentiment as one factor among many in your analysis
            - Pay attention to unusual volume spikes mentioned on social media

            ❌ Avoid:
            - Trading based solely on rumors or unverified information
            - Following pump and dump schemes
            - FOMO trading based on viral posts
            - Ignoring fundamentals in favor of sentiment
            """)

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

# ===== FOOTER =====
st.markdown("---")
footer_sources = []
if UNUSUAL_WHALES_KEY:
    footer_sources.append("🐋 Unusual Whales (PRIMARY)")
if alpha_vantage_client:
    footer_sources.append("Alpha Vantage")
if twelvedata_client:
    footer_sources.append("Twelve Data")
footer_sources.append("Yahoo Finance")
footer_text = " + ".join(footer_sources)

available_ai_models = multi_ai.get_available_models()
ai_footer = f"AI: {st.session_state.ai_model}"
if st.session_state.ai_model == "Multi-AI" and available_ai_models:
    ai_footer += f" ({'+'.join(available_ai_models)})"

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🔥 AI Radar Pro Enhanced | Data: {footer_text} | {ai_footer} | Refresh: {st.session_state.refresh_interval}s"
    "</div>",
    unsafe_allow_html=True
)
