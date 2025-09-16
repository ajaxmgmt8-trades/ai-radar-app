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
    st.session_state.refresh_interval = 10  # Faster default refresh
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"  # Default to ET
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Unusual Whales"  # New primary source
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"  # Default to multi-AI

# API Keys
try:
    UNUSUAL_WHALES_KEY = st.secrets.get("UNUSUAL_WHALES_API_KEY", "")
    TRADING_ECONOMICS_KEY = st.secrets.get("TRADING_ECONOMICS_API_KEY", "")
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

# FIXED: Unusual Whales API Client using ACTUAL API endpoints
class UnusualWhalesClient:
    """Primary API client for Unusual Whales using real API endpoints"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Try different base URLs based on common patterns
        self.base_urls_to_try = [
            "https://api.unusualwhales.com",
            "https://unusualwhales.com/api",
            "https://api.unusualwhales.com/v1",
        ]
        self.session = requests.Session()
        # Try multiple authentication methods
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,  # Alternative auth method
            "apikey": api_key,     # Another common pattern
            "Accept": "application/json",
            "User-Agent": "AI-Radar-Pro/2.0",
            "Content-Type": "application/json"
        })
    
    def get_quote(self, symbol: str) -> Dict:
        """Get stock data from Unusual Whales - try multiple endpoints"""
        endpoints_to_try = [
            f"/api/stock/{symbol}",
            f"/api/v1/stock/{symbol}",
            f"/stock/{symbol}",
            f"/api/stocks/{symbol}",
        ]
        
        for base_url in self.base_urls_to_try:
            for endpoint in endpoints_to_try:
                try:
                    full_url = f"{base_url}{endpoint}"
                    # Try both header auth and query param auth
                    urls_to_try = [
                        full_url,  # Header auth
                        f"{full_url}?api_key={self.api_key}",  # Query param auth
                        f"{full_url}?apikey={self.api_key}",   # Alternative query param
                    ]
                    
                    for test_url in urls_to_try:
                        print(f"Trying UW URL: {test_url}")
                        
                        response = self.session.get(test_url, timeout=8)
                        
                        print(f"UW API Response Status: {response.status_code}")
                        print(f"UW API Response Headers: {dict(response.headers)}")
                        print(f"UW API Response Text: {response.text[:500]}")
                        
                        if response.status_code == 200:
                        try:
                            data = response.json()
                            print(f"UW API JSON Data Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                            
                            # Try to extract price data - be flexible about field names
                            price = 0
                            prev_close = 0
                            volume = 0
                            
                            # Try different field name patterns
                            price_fields = ["last_price", "price", "close", "last", "current_price"]
                            for field in price_fields:
                                if field in data and data[field]:
                                    price = float(data[field])
                                    break
                            
                            prev_close_fields = ["prev_close", "previous_close", "previousClose", "yesterday_close"]
                            for field in prev_close_fields:
                                if field in data and data[field]:
                                    prev_close = float(data[field])
                                    break
                            
                            volume_fields = ["volume", "total_volume", "day_volume"]
                            for field in volume_fields:
                                if field in data and data[field]:
                                    volume = int(data[field])
                                    break
                            
                            if price > 0:  # We got valid data
                                # Calculate changes
                                change = price - prev_close if prev_close > 0 else 0
                                change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                                
                                # Get bid/ask if available
                                bid = float(data.get("bid", price - 0.01))
                                ask = float(data.get("ask", price + 0.01))
                                
                                return {
                                    "last": price,
                                    "bid": bid,
                                    "ask": ask,
                                    "volume": volume,
                                    "change": change,
                                    "change_percent": change_percent,
                                    "premarket_change": 0,
                                    "intraday_change": change_percent,
                                    "postmarket_change": 0,
                                    "previous_close": prev_close,
                                    "market_open": float(data.get("open", price)),
                                    "last_updated": datetime.datetime.now().isoformat(),
                                    "data_source": "Unusual Whales",
                                    "error": None,
                                    "raw_data": data
                                }
                        except Exception as json_error:
                            print(f"UW JSON parsing error: {json_error}")
                            continue
                    elif response.status_code == 401:
                        return {"error": "UW API: Unauthorized - Check API key", "data_source": "Unusual Whales"}
                    elif response.status_code == 403:
                        return {"error": "UW API: Forbidden - Check API permissions", "data_source": "Unusual Whales"}
                    elif response.status_code == 404:
                        print(f"UW 404 for: {full_url}")
                        continue  # Try next endpoint
                        
                except requests.exceptions.ConnectionError:
                    print(f"UW Connection failed for: {base_url}")
                    continue  # Try next base URL
                except requests.exceptions.Timeout:
                    print(f"UW Timeout for: {base_url}")
                    continue
                except Exception as e:
                    print(f"UW Error for {base_url}{endpoint}: {str(e)}")
                    continue
        
        # If we get here, all attempts failed
        return {"error": "UW API: All endpoints failed - check API documentation", "data_source": "Unusual Whales"}
    
    def get_options_volume(self, symbol: str) -> Dict:
        """Get options volume data using /api/stock/{ticker}/options-volume"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/stock/{symbol}/options-volume",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "options_volume": data,
                    "data_source": "Unusual Whales",
                    "error": None
                }
            else:
                return {"error": f"Options volume error: {response.status_code}", "data_source": "Unusual Whales"}
                
        except Exception as e:
            return {"error": f"Options volume error: {str(e)}", "data_source": "Unusual Whales"}
    
    def get_flow_alerts(self, symbol: str) -> Dict:
        """Get unusual options flow alerts using /api/stock/{ticker}/flow-alerts"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/stock/{symbol}/flow-alerts",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "flow_alerts": data,
                    "data_source": "Unusual Whales",
                    "error": None
                }
            else:
                return {"error": f"Flow alerts error: {response.status_code}", "data_source": "Unusual Whales"}
                
        except Exception as e:
            return {"error": f"Flow alerts error: {str(e)}", "data_source": "Unusual Whales"}
    
    def get_greek_exposure(self, symbol: str) -> Dict:
        """Get Greek exposure data using /api/stock/{ticker}/greek-exposure"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/stock/{symbol}/greek-exposure",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "greek_exposure": data,
                    "data_source": "Unusual Whales",
                    "error": None
                }
            else:
                return {"error": f"Greek exposure error: {response.status_code}", "data_source": "Unusual Whales"}
                
        except Exception as e:
            return {"error": f"Greek exposure error: {str(e)}", "data_source": "Unusual Whales"}
    
    def get_market_tide(self) -> Dict:
        """Get overall market sentiment using /api/market/market-tide"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/market/market-tide",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "market_tide": data,
                    "data_source": "Unusual Whales",
                    "error": None
                }
            else:
                return {"error": f"Market tide error: {response.status_code}", "data_source": "Unusual Whales"}
                
        except Exception as e:
            return {"error": f"Market tide error: {str(e)}", "data_source": "Unusual Whales"}
    
    def get_institutional_activity(self, symbol: str) -> Dict:
        """Get institutional activity - try multiple endpoints"""
        # Try congressional trading data
        try:
            response = self.session.get(
                f"{self.base_url}/api/congress/{symbol}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "institutional_data": data,
                    "data_type": "congressional",
                    "data_source": "Unusual Whales",
                    "error": None
                }
        except Exception:
            pass
        
        # Try insider trading data as fallback
        try:
            response = self.session.get(
                f"{self.base_url}/api/insider/{symbol}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "institutional_data": data,
                    "data_type": "insider",
                    "data_source": "Unusual Whales", 
                    "error": None
                }
        except Exception:
            pass
        
        return {"error": "No institutional data available", "data_source": "Unusual Whales"}

# Trading Economics API Client
class TradingEconomicsClient:
    """Trading Economics API client for economic data and calendars"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
    
    def get_economic_calendar(self, days: int = 7) -> List[Dict]:
        """Get economic calendar for the next N days"""
        try:
            end_date = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
            
            response = self.session.get(
                f"{self.base_url}/calendar",
                params={
                    "c": self.api_key,
                    "f": "json",
                    "d1": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "d2": end_date
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            print(f"Trading Economics calendar error: {e}")
            return []
    
    def get_economic_indicators(self, country: str = "united states") -> Dict:
        """Get key economic indicators"""
        try:
            response = self.session.get(
                f"{self.base_url}/indicators",
                params={
                    "c": self.api_key,
                    "f": "json",
                    "country": country
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "indicators": data,
                    "data_source": "Trading Economics",
                    "error": None
                }
            else:
                return {"error": f"Indicators error: {response.status_code}", "data_source": "Trading Economics"}
                
        except Exception as e:
            return {"error": f"Indicators error: {str(e)}", "data_source": "Trading Economics"}

# Initialize API clients
unusual_whales_client = UnusualWhalesClient(UNUSUAL_WHALES_KEY) if UNUSUAL_WHALES_KEY else None
trading_economics_client = TradingEconomicsClient(TRADING_ECONOMICS_KEY) if TRADING_ECONOMICS_KEY else None

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

# Alpha Vantage Client (now fallback)
class AlphaVantageClient:
    """Alpha Vantage client for real-time stock data (fallback)"""
    
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
            
            response = self.session.get(self.base_url, params=params, timeout=8)
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

# Twelve Data Client (now secondary fallback)
class TwelveDataClient:
    """Twelve Data client for real-time stock data (secondary fallback)"""
    
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
            
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=8)
            
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
                response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=8)
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

# Initialize data clients (with new priority order)
alpha_vantage_client = AlphaVantageClient(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else None
twelvedata_client = TwelveDataClient(TWELVEDATA_KEY) if TWELVEDATA_KEY else None

# UPDATED: Enhanced primary data function - Unusual Whales FIRST, then fallbacks
@st.cache_data(ttl=15)  # MUCH faster refresh - 15 seconds instead of 60
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """
    Get live stock quote using Unusual Whales FIRST, then fallbacks
    """
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # Try Unusual Whales FIRST (Primary source)
    if unusual_whales_client:
        try:
            uw_quote = unusual_whales_client.get_quote(ticker)
            if not uw_quote.get("error") and uw_quote.get("last", 0) > 0:
                uw_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return uw_quote
        except Exception as e:
            print(f"Unusual Whales error for {ticker}: {str(e)}")
    
    # Try Alpha Vantage second (if available)
    if alpha_vantage_client:
        try:
            alpha_quote = alpha_vantage_client.get_quote(ticker)
            if not alpha_quote.get("error") and alpha_quote.get("last", 0) > 0:
                alpha_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return alpha_quote
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {str(e)}")
    
    # Try Twelve Data third (if available)
    if twelvedata_client:
        try:
            twelve_quote = twelvedata_client.get_quote(ticker)
            if not twelve_quote.get("error") and twelve_quote.get("last", 0) > 0:
                twelve_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return twelve_quote
        except Exception as e:
            print(f"Twelve Data error for {ticker}: {str(e)}")
    
    # Final fallback to Yahoo Finance
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
@st.cache_data(ttl=60)  # Faster refresh - reduced from 300 to 60 seconds
def get_comprehensive_technical_analysis(ticker: str) -> Dict:
    """Enhanced technical analysis with multiple indicators and timeframes"""
    try:
        # Try Unusual Whales first for enhanced data
        if unusual_whales_client:
            try:
                # Get enhanced data from UW including technical indicators
                quote = unusual_whales_client.get_quote(ticker)
                if not quote.get("error") and quote.get("raw_data"):
                    raw_data = quote["raw_data"]
                    
                    # Extract technical data if available from UW
                    technical_data = raw_data.get("technical", {})
                    
                    if technical_data:
                        return {
                            "rsi": technical_data.get("rsi", 0),
                            "macd": technical_data.get("macd", 0),
                            "macd_signal": technical_data.get("macd_signal", 0),
                            "sma_20": technical_data.get("sma_20", 0),
                            "sma_50": technical_data.get("sma_50", 0),
                            "ema_12": technical_data.get("ema_12", 0),
                            "ema_26": technical_data.get("ema_26", 0),
                            "bb_upper": technical_data.get("bb_upper", 0),
                            "bb_lower": technical_data.get("bb_lower", 0),
                            "support": technical_data.get("support", 0),
                            "resistance": technical_data.get("resistance", 0),
                            "current_price": quote["last"],
                            "trend_analysis": technical_data.get("trend", "Neutral"),
                            "data_source": "Unusual Whales Enhanced",
                            "error": None
                        }
            except Exception as e:
                print(f"UW technical analysis error: {e}")
        
        # Try Twelve Data for comprehensive data
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
                    
                    response = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=8)
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
@st.cache_data(ttl=900)  # 15 minutes cache for fundamental data
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

# Enhanced Options Flow Analysis with Unusual Whales REAL API
def get_advanced_options_analysis(ticker: str) -> Dict:
    """Comprehensive options analysis with Unusual Whales REAL API data"""
    try:
        # Try Unusual Whales first using REAL API endpoints
        if unusual_whales_client:
            try:
                # Get options volume data
                options_volume = unusual_whales_client.get_options_volume(ticker)
                flow_alerts = unusual_whales_client.get_flow_alerts(ticker)
                greek_exposure = unusual_whales_client.get_greek_exposure(ticker)
                
                # Process UW data if any endpoint succeeds
                if not options_volume.get("error") or not flow_alerts.get("error") or not greek_exposure.get("error"):
                    uw_data = {
                        "unusual_whales_data": True,
                        "options_volume": options_volume.get("options_volume", {}),
                        "flow_alerts": flow_alerts.get("flow_alerts", {}), 
                        "greek_exposure": greek_exposure.get("greek_exposure", {}),
                        "data_source": "Unusual Whales",
                        "error": None
                    }
                    
                    # Calculate summary metrics from UW data
                    volume_data = options_volume.get("options_volume", {})
                    alerts_data = flow_alerts.get("flow_alerts", {})
                    
                    if isinstance(volume_data, dict):
                        total_premium = volume_data.get("total_premium", 0)
                        call_volume = volume_data.get("call_volume", 0)
                        put_volume = volume_data.get("put_volume", 0)
                        
                        uw_data.update({
                            "total_premium": total_premium,
                            "call_volume": call_volume,
                            "put_volume": put_volume,
                            "put_call_ratio": put_volume / call_volume if call_volume > 0 else 0
                        })
                    
                    if isinstance(alerts_data, list):
                        uw_data["flow_count"] = len(alerts_data)
                        uw_data["unusual_activity"] = alerts_data
                    elif isinstance(alerts_data, dict):
                        uw_data["flow_count"] = len(alerts_data.get("alerts", []))
                        uw_data["unusual_activity"] = alerts_data.get("alerts", [])
                    
                    return uw_data
                    
            except Exception as e:
                print(f"UW options analysis error: {e}")
        
        # Fallback to yfinance option chain analysis
        option_chain = get_option_chain(ticker, st.session_state.selected_tz)
        if option_chain.get("error"):
            return {"error": option_chain["error"]}

        calls = option_chain["calls"]
        puts = option_chain["puts"]
        current_price = option_chain["current_price"]

        # Advanced analysis using yfinance data
        analysis = {
            "unusual_whales_data": False,
            "basic_metrics": calculate_basic_options_metrics(calls, puts),
            "flow_analysis": analyze_options_flow(calls, puts, current_price),
            "unusual_activity": detect_unusual_activity(calls, puts),
            "gamma_levels": calculate_gamma_levels(calls, puts, current_price),
            "sentiment_indicators": calculate_options_sentiment(calls, puts),
            "expiration": option_chain["expiration"],
            "current_price": current_price
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Advanced options analysis error: {str(e)}"}

def calculate_basic_options_metrics(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
    """Calculate basic options metrics"""
    total_call_volume = calls['volume'].sum()
    total_put_volume = puts['volume'].sum()
    total_call_oi = calls['openInterest'].sum()
    total_put_oi = puts['openInterest'].sum()
    
    return {
        "total_call_volume": total_call_volume,
        "total_put_volume": total_put_volume,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "put_call_volume_ratio": total_put_volume / total_call_volume if total_call_volume > 0 else 0,
        "put_call_oi_ratio": total_put_oi / total_call_oi if total_call_oi > 0 else 0,
        "avg_call_iv": calls['impliedVolatility'].mean(),
        "avg_put_iv": puts['impliedVolatility'].mean(),
        "iv_skew": puts['impliedVolatility'].mean() - calls['impliedVolatility'].mean()
    }

def analyze_options_flow(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> Dict:
    """Analyze options order flow"""
    # Volume/OI ratios indicate new positions
    calls['vol_oi_ratio'] = calls['volume'] / calls['openInterest'].replace(0, 1)
    puts['vol_oi_ratio'] = puts['volume'] / puts['openInterest'].replace(0, 1)
    
    # ITM vs OTM analysis
    itm_calls = calls[calls['strike'] < current_price]
    otm_calls = calls[calls['strike'] >= current_price]
    itm_puts = puts[puts['strike'] > current_price]
    otm_puts = puts[puts['strike'] <= current_price]
    
    return {
        "itm_call_volume": itm_calls['volume'].sum(),
        "otm_call_volume": otm_calls['volume'].sum(),
        "itm_put_volume": itm_puts['volume'].sum(),
        "otm_put_volume": otm_puts['volume'].sum(),
        "net_call_bias": otm_calls['volume'].sum() - itm_calls['volume'].sum(),
        "net_put_bias": itm_puts['volume'].sum() - otm_puts['volume'].sum(),
        "flow_sentiment": "Bullish" if otm_calls['volume'].sum() > itm_puts['volume'].sum() else "Bearish"
    }

def detect_unusual_activity(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
    """Detect unusual options activity"""
    # High volume relative to OI
    calls_unusual = calls[calls['volume'] > calls['openInterest'] * 2]
    puts_unusual = puts[puts['volume'] > puts['openInterest'] * 2]
    
    # High volume absolute
    high_vol_threshold = max(calls['volume'].quantile(0.8), puts['volume'].quantile(0.8))
    calls_high_vol = calls[calls['volume'] >= high_vol_threshold]
    puts_high_vol = puts[puts['volume'] >= high_vol_threshold]
    
    return {
        "unusual_calls": calls_unusual[['strike', 'volume', 'openInterest', 'lastPrice']].to_dict('records'),
        "unusual_puts": puts_unusual[['strike', 'volume', 'openInterest', 'lastPrice']].to_dict('records'),
        "high_volume_calls": calls_high_vol[['strike', 'volume', 'lastPrice']].to_dict('records'),
        "high_volume_puts": puts_high_vol[['strike', 'volume', 'lastPrice']].to_dict('records'),
        "total_unusual_contracts": len(calls_unusual) + len(puts_unusual)
    }

def calculate_gamma_levels(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> Dict:
    """Calculate gamma exposure levels (simplified)"""
    # Approximate gamma calculation
    calls['approx_gamma'] = calls['openInterest'] * 0.01  # Simplified
    puts['approx_gamma'] = puts['openInterest'] * -0.01  # Simplified
    
    # Find strikes with highest gamma exposure
    all_strikes = pd.concat([
        calls[['strike', 'approx_gamma', 'openInterest']].assign(type='call'),
        puts[['strike', 'approx_gamma', 'openInterest']].assign(type='put')
    ])
    
    gamma_by_strike = all_strikes.groupby('strike')['approx_gamma'].sum().sort_values(ascending=False)
    
    return {
        "max_gamma_strike": gamma_by_strike.index[0] if len(gamma_by_strike) > 0 else current_price,
        "max_gamma_level": gamma_by_strike.iloc[0] if len(gamma_by_strike) > 0 else 0,
        "gamma_strikes": gamma_by_strike.head(5).to_dict()
    }

def calculate_options_sentiment(calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
    """Calculate options-based sentiment indicators"""
    # Premium analysis
    call_premium = (calls['volume'] * calls['lastPrice']).sum()
    put_premium = (puts['volume'] * puts['lastPrice']).sum()
    
    # Aggressive vs defensive positioning
    aggressive_calls = calls[calls['volume'] > calls['volume'].quantile(0.7)]['volume'].sum()
    defensive_puts = puts[puts['volume'] > puts['volume'].quantile(0.7)]['volume'].sum()
    
    sentiment_score = (call_premium - put_premium) / (call_premium + put_premium) if (call_premium + put_premium) > 0 else 0
    
    return {
        "call_premium": call_premium,
        "put_premium": put_premium,
        "premium_ratio": call_premium / put_premium if put_premium > 0 else float('inf'),
        "sentiment_score": sentiment_score,
        "aggressive_positioning": aggressive_calls,
        "defensive_positioning": defensive_puts,
        "overall_sentiment": "Bullish" if sentiment_score > 0.1 else "Bearish" if sentiment_score < -0.1 else "Neutral"
    }

# New function to fetch option chain data
@st.cache_data(ttl=120)  # Faster refresh for options - 2 minutes
def get_option_chain(ticker: str, tz: str = "ET") -> Optional[Dict]:
    """Fetch 0DTE or nearest expiration option chain using yfinance"""
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
            "error": None
        }
    except Exception as e:
        return {"error": f"Error fetching option chain for {ticker}: {str(e)}"}

# New function to simulate order flow (placeholder for premium API integration)
@st.cache_data(ttl=120)  # Faster options flow refresh
def get_order_flow(ticker: str, option_chain: Dict) -> Dict:
    """Simulate order flow by analyzing option chain volume and open interest"""
    calls = option_chain.get('calls', pd.DataFrame())
    puts = option_chain.get('puts', pd.DataFrame())
    if calls.empty or puts.empty:
        return {"error": "No option chain data for order flow analysis"}

    try:
        # Calculate put/call volume ratio
        total_call_volume = calls['volume'].sum() if not calls.empty else 0
        total_put_volume = puts['volume'].sum() if not puts.empty else 0
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0

        # Identify unusual activity (high volume relative to open interest)
        calls['volume_oi_ratio'] = calls['volume'] / calls['openInterest'].replace(0, 1)
        puts['volume_oi_ratio'] = puts['volume'] / puts['openInterest'].replace(0, 1)

        # Top trades (simulated as high volume or high volume/OI ratio)
        top_calls = calls[calls['volume_oi_ratio'] > 1.5][['contractSymbol', 'strike', 'lastPrice', 'volume', 'moneyness']].head(3)
        top_puts = puts[puts['volume_oi_ratio'] > 1.5][['contractSymbol', 'strike', 'lastPrice', 'volume', 'moneyness']].head(3)

        # Sentiment based on volume
        sentiment = "Bullish" if total_call_volume > total_put_volume else "Bearish" if total_put_volume > total_call_volume else "Neutral"

        return {
            "put_call_ratio": put_call_ratio,
            "top_calls": top_calls.to_dict('records'),
            "top_puts": top_puts.to_dict('records'),
            "sentiment": sentiment,
            "error": None
        }
    except Exception as e:
        return {"error": f"Error analyzing order flow: {str(e)}"}

# Modified get_options_data to use real data
def get_options_data(ticker: str) -> Optional[Dict]:
    """Fetch real options data for a ticker"""
    option_chain = get_option_chain(ticker, st.session_state.selected_tz)
    if option_chain.get("error"):
        return {"error": option_chain["error"]}

    calls = option_chain["calls"]
    puts = option_chain["puts"]
    current_price = option_chain["current_price"]

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
        "total_puts": puts['volume'].sum() if not puts.empty else 0
    }

@st.cache_data(ttl=300)  # 5 minutes for news
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

@st.cache_data(ttl=300)
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

def get_comprehensive_news(ticker: str) -> List[Dict]:
    """Get comprehensive news from all sources for a specific ticker"""
    all_news = []
    
    # Get Finnhub news
    finnhub_news = get_finnhub_news(ticker)
    for item in finnhub_news:
        all_news.append({
            "title": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": "Finnhub",
            "url": item.get("url", ""),
            "datetime": item.get("datetime", 0),
            "related": item.get("related", ""),
            "provider": "Finnhub API"
        })
    
    # Get Yahoo Finance news
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        
        for item in yf_news[:10]:  # Limit to recent news
            all_news.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": "Yahoo Finance",
                "url": item.get("link", ""),
                "datetime": item.get("providerPublishTime", 0),
                "related": ticker,
                "provider": "Yahoo Finance"
            })
    except Exception as e:
        print(f"Yahoo Finance news error for {ticker}: {e}")
    
    return all_news

def get_yfinance_news() -> List[Dict]:
    """Get general market news from Yahoo Finance"""
    try:
        # Use a major index to get general market news
        import yfinance as yf
        spy = yf.Ticker("SPY")
        news = spy.news
        
        formatted_news = []
        for item in news[:15]:
            formatted_news.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": "Yahoo Finance",
                "url": item.get("link", ""),
                "datetime": item.get("providerPublishTime", ""),
                "related": "",
                "provider": "Yahoo Finance"
            })
        
        return formatted_news
    except Exception as e:
        print(f"Yahoo Finance general news error: {e}")
        return []

def analyze_catalyst_impact(title: str, summary: str = "") -> Dict:
    """Analyze the potential market impact of a news item"""
    text = (title + " " + summary).lower()
    
    # Enhanced keyword analysis for market impact
    high_impact_keywords = [
        "fed", "federal reserve", "rate", "inflation", "cpi", "ppi", "gdp", 
        "earnings", "guidance", "acquisition", "merger", "bankruptcy", "sec", 
        "fda approval", "clinical trial", "breakthrough", "partnership",
        "upgrade", "downgrade", "analyst", "target price", "revenue beat",
        "earnings miss", "layoffs", "restructuring", "ipo", "dings"
    ]
    
    bullish_keywords = [
        "beat", "exceed", "strong", "growth", "positive", "approval", "partnership",
        "acquisition", "upgrade", "buy", "outperform", "breakthrough", "expansion",
        "record", "surge", "jump", "rally", "bullish"
    ]
    
    bearish_keywords = [
        "miss", "disappoint", "weak", "decline", "negative", "rejection", "delay",
        "downgrade", "sell", "underperform", "warning", "concern", "drop",
        "fall", "crash", "bearish", "loss", "lawsuit", "investigation"
    ]
    
    category_keywords = {
        "earnings": ["earnings", "revenue", "eps", "guidance", "quarter"],
        "regulatory": ["fda", "sec", "regulation", "approval", "investigation"],
        "corporate": ["merger", "acquisition", "partnership", "ceo", "management"],
        "economic": ["fed", "inflation", "gdp", "unemployment", "rate"],
        "technical": ["breakthrough", "innovation", "patent", "technology"],
        "analyst": ["upgrade", "downgrade", "target", "rating", "analyst"]
    }
    
    # Calculate scores
    high_impact_score = sum(2 for word in high_impact_keywords if word in text)
    bullish_score = sum(1 for word in bullish_keywords if word in text)
    bearish_score = sum(1 for word in bearish_keywords if word in text)
    
    # Determine primary category
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for word in keywords if word in text)
        if score > 0:
            category_scores[category] = score
    
    primary_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "general"
    
    # Determine sentiment
    if bullish_score > bearish_score:
        sentiment = "positive"
    elif bearish_score > bullish_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Calculate overall catalyst strength (0-100)
    catalyst_strength = min(100, (high_impact_score * 15) + (bullish_score * 5) + (bearish_score * 5) + 
                           (max(category_scores.values()) * 10 if category_scores else 0))
    
    # Determine impact level
    if catalyst_strength >= 60:
        impact_level = "high"
    elif catalyst_strength >= 30:
        impact_level = "medium"
    else:
        impact_level = "low"
    
    return {
        "catalyst_strength": catalyst_strength,
        "sentiment": sentiment,
        "impact_level": impact_level,
        "primary_category": primary_category,
        "category_scores": category_scores
    }

def get_market_moving_news() -> List[Dict]:
    """Get market-moving news from all sources with catalyst analysis"""
    all_news = []
    
    # Get general market news from all sources
    finnhub_general = get_finnhub_news()  # General news
    polygon_general = get_polygon_news()
    yahoo_general = get_yfinance_news()
    
    # Process Finnhub news
    for item in finnhub_general:
        catalyst_analysis = analyze_catalyst_impact(
            item.get("headline", ""), 
            item.get("summary", "")
        )
        
        news_item = {
            "title": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": "Finnhub",
            "url": item.get("url", ""),
            "datetime": item.get("datetime", 0),
            "related": item.get("related", ""),
            "provider": "Finnhub API",
            "catalyst_analysis": catalyst_analysis
        }
        all_news.append(news_item)
    
    # Process Polygon news
    for item in polygon_general:
        catalyst_analysis = analyze_catalyst_impact(
            item.get("title", ""), 
            item.get("description", "")
        )
        
        news_item = {
            "title": item.get("title", ""),
            "summary": item.get("description", ""),
            "source": "Polygon",
            "url": item.get("article_url", ""),
            "datetime": item.get("published_utc", ""),
            "related": ",".join(item.get("tickers", [])),
            "provider": "Polygon API",
            "catalyst_analysis": catalyst_analysis
        }
        all_news.append(news_item)
    
    # Process Yahoo Finance news
    for item in yahoo_general:
        catalyst_analysis = analyze_catalyst_impact(
            item.get("title", ""), 
            item.get("summary", "")
        )
        
        news_item = {
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": "Yahoo Finance",
            "url": item.get("url", ""),
            "datetime": item.get("datetime", ""),
            "related": item.get("related", ""),
            "provider": "Yahoo Finance",
            "catalyst_analysis": catalyst_analysis
        }
        all_news.append(news_item)
    
    # Sort by catalyst strength and return top items
    all_news.sort(key=lambda x: x["catalyst_analysis"]["catalyst_strength"], reverse=True)
    return all_news

def get_stock_specific_catalysts(ticker: str) -> Dict:
    """Get comprehensive catalyst analysis for a specific stock"""
    try:
        # Get news from all sources
        all_news = get_comprehensive_news(ticker)
        
        # Analyze each news item
        analyzed_news = []
        for news_item in all_news:
            catalyst_analysis = analyze_catalyst_impact(
                news_item.get("title", ""),
                news_item.get("summary", "")
            )
            news_item["catalyst_analysis"] = catalyst_analysis
            analyzed_news.append(news_item)
        
        # Generate summary statistics
        total_catalysts = len(analyzed_news)
        positive_catalysts = len([n for n in analyzed_news if n["catalyst_analysis"]["sentiment"] == "positive"])
        negative_catalysts = len([n for n in analyzed_news if n["catalyst_analysis"]["sentiment"] == "negative"])
        
        # Find highest impact and primary categories
        highest_impact = max([n["catalyst_analysis"]["catalyst_strength"] for n in analyzed_news]) if analyzed_news else 0
        
        # Count categories
        category_counts = {}
        for news in analyzed_news:
            category = news["catalyst_analysis"]["primary_category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        primary_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate trading implications
        trading_implications = generate_trading_implications_text(
            ticker, analyzed_news, highest_impact, positive_catalysts, negative_catalysts
        )
        
        return {
            "news_items": analyzed_news,
            "catalyst_summary": {
                "total_catalysts": total_catalysts,
                "positive_catalysts": positive_catalysts,
                "negative_catalysts": negative_catalysts,
                "highest_impact": highest_impact,
                "primary_categories": primary_categories[:5]  # Top 5 categories
            },
            "trading_implications": trading_implications
        }
        
    except Exception as e:
        return {
            "error": f"Error analyzing catalysts for {ticker}: {str(e)}",
            "news_items": [],
            "catalyst_summary": {
                "total_catalysts": 0,
                "positive_catalysts": 0,
                "negative_catalysts": 0,
                "highest_impact": 0,
                "primary_categories": []
            },
            "trading_implications": f"Unable to analyze catalysts for {ticker} due to data error."
        }

def generate_trading_implications_text(ticker: str, news_items: List[Dict], 
                                     highest_impact: int, positive: int, negative: int) -> str:
    """Generate trading implications text based on catalyst analysis"""
    
    if not news_items:
        return f"No recent catalysts found for {ticker}. Monitor for new developments."
    
    # Determine overall sentiment bias
    if positive > negative:
        sentiment_bias = "bullish"
        sentiment_desc = f"with {positive} positive vs {negative} negative catalysts"
    elif negative > positive:
        sentiment_bias = "bearish" 
        sentiment_desc = f"with {negative} negative vs {positive} positive catalysts"
    else:
        sentiment_bias = "neutral"
        sentiment_desc = f"with balanced sentiment ({positive} positive, {negative} negative)"
    
    # Determine impact level
    if highest_impact >= 70:
        impact_desc = "HIGH IMPACT catalysts detected"
    elif highest_impact >= 40:
        impact_desc = "moderate impact catalysts present"
    else:
        impact_desc = "low impact catalysts only"
    
    # Generate implications
    implications = f"""
**Catalyst Overview for {ticker}:**
- Overall sentiment bias: **{sentiment_bias.upper()}** {sentiment_desc}
- Impact level: **{impact_desc}**
- Total catalysts analyzed: {len(news_items)}

**Trading Considerations:**
"""
    
    if highest_impact >= 70:
        implications += f"""
- **High volatility expected** due to significant catalysts
- Consider wider stops and position sizing adjustments
- Monitor for breakout/breakdown from key technical levels
- Options implied volatility likely elevated
"""
    elif highest_impact >= 40:
        implications += f"""
- **Moderate volatility possible** from catalyst developments
- Watch for confirmation of direction with volume
- Standard risk management appropriate
"""
    else:
        implications += f"""
- **Low catalyst impact** - focus on technical analysis
- Normal market conditions expected
- Standard trading approaches suitable
"""
    
    if sentiment_bias == "bullish":
        implications += f"""
- **Bullish bias** suggests looking for long opportunities on dips
- Watch for bullish continuation patterns
- Consider call options if volatility is reasonable
"""
    elif sentiment_bias == "bearish":
        implications += f"""
- **Bearish bias** suggests caution on long positions
- Look for short opportunities on rallies
- Consider protective puts for existing positions
"""
    
    implications += f"""
**Risk Management:**
- Set alerts for new catalyst developments
- Adjust position size based on catalyst volatility
- Monitor news flow throughout trading session
"""
    
    return implications.strip()

# Enhanced AI Analysis Prompt Construction
def construct_comprehensive_analysis_prompt(ticker: str, quote: Dict, technical: Dict, fundamental: Dict, options: Dict, news_context: str = "") -> str:
    """Construct comprehensive analysis prompt with all data including Unusual Whales"""
    
    # Technical summary
    tech_summary = "Technical Analysis:\n"
    if technical.get("error"):
        tech_summary += f"Technical Error: {technical['error']}\n"
    else:
        if "short_term" in technical:
            tech_summary += f"- RSI: {technical['short_term'].get('rsi', 'N/A'):.1f}\n"
            tech_summary += f"- SMA20: ${technical['short_term'].get('sma_20', 0):.2f}\n"
            tech_summary += f"- MACD: {technical['short_term'].get('macd', 0):.3f}\n"
        if "trend_analysis" in technical:
            tech_summary += f"- Trend: {technical.get('trend_analysis', 'Unknown')}\n"
        if "support_resistance" in technical:
            tech_summary += f"- Support: ${technical.get('support_resistance', {}).get('support', 0):.2f}\n"
            tech_summary += f"- Resistance: ${technical.get('support_resistance', {}).get('resistance', 0):.2f}\n"
        # Enhanced technical from Twelve Data or Unusual Whales
        if "rsi" in technical:
            tech_summary += f"- RSI: {technical.get('rsi', 'N/A'):.1f}\n"
            tech_summary += f"- Trend: {technical.get('trend_analysis', 'Unknown')}\n"
            tech_summary += f"- Support: ${technical.get('support', 0):.2f}\n"
            tech_summary += f"- Resistance: ${technical.get('resistance', 0):.2f}\n"
    
    # Fundamental summary
    fund_summary = "Fundamental Analysis:\n"
    if fundamental.get("error"):
        fund_summary += f"Fundamental Error: {fundamental['error']}\n"
    else:
        fund_summary += f"- P/E Ratio: {fundamental.get('pe_ratio', 'N/A')}\n"
        fund_summary += f"- Market Cap: ${fundamental.get('market_cap', 0):,.0f}\n"
        fund_summary += f"- Financial Health: {fundamental.get('financial_health', 'Unknown')}\n"
        fund_summary += f"- Valuation: {fundamental.get('valuation_assessment', 'Unknown')}\n"
        fund_summary += f"- Sector: {fundamental.get('sector', 'Unknown')}\n"
        revenue_growth = fundamental.get('revenue_growth', 'N/A')
        if revenue_growth != 'N/A' and revenue_growth is not None:
            fund_summary += f"- Revenue Growth: {revenue_growth:.1%}\n"
        else:
            fund_summary += "- Revenue Growth: N/A\n"
        fund_summary += f"- Debt/Equity: {fundamental.get('debt_to_equity', 'N/A')}\n"
    
    # Enhanced Options summary with Unusual Whales data
    options_summary = "Options Analysis:\n"
    if options.get("error"):
        options_summary += f"Options Error: {options['error']}\n"
    else:
        if options.get("unusual_whales_data"):
            # Real Unusual Whales data
            options_summary += f"- Data Source: Unusual Whales (LIVE FLOW)\n"
            options_summary += f"- Total Premium: ${options.get('total_premium', 0):,.0f}\n"
            options_summary += f"- Call/Put Premium: ${options.get('call_premium', 0):,.0f}/${options.get('put_premium', 0):,.0f}\n"
            options_summary += f"- Premium Ratio: {options.get('premium_ratio', 0):.2f}\n"
            options_summary += f"- Flow Count: {options.get('flow_count', 0)}\n"
            options_summary += f"- Bullish/Bearish Flows: {options.get('bullish_flows', 0)}/{options.get('bearish_flows', 0)}\n"
            options_summary += f"- Unusual Activity: {len(options.get('unusual_activity', []))} alerts\n"
        else:
            # Fallback options data
            basic = options.get('basic_metrics', {})
            flow = options.get('flow_analysis', {})
            unusual = options.get('unusual_activity', {})
            options_summary += f"- Put/Call Ratio: {basic.get('put_call_volume_ratio', 0):.2f}\n"
            options_summary += f"- Average IV: {basic.get('avg_call_iv', 0):.1f}%\n"
            options_summary += f"- IV Skew: {basic.get('iv_skew', 0):.1f}%\n"
            options_summary += f"- Flow Sentiment: {flow.get('flow_sentiment', 'Neutral')}\n"
            options_summary += f"- Unusual Activity: {unusual.get('total_unusual_contracts', 0)} contracts\n"
            options_summary += f"- Total Volume: {basic.get('total_call_volume', 0) + basic.get('total_put_volume', 0):,}\n"
    
    # Session data with enhanced source info
    session_summary = f"""Session Performance:
- Current: ${quote['last']:.2f} ({quote['change_percent']:+.2f}%)
- Premarket: {quote['premarket_change']:+.2f}%
- Intraday: {quote['intraday_change']:+.2f}%
- After Hours: {quote['postmarket_change']:+.2f}%
- Volume: {quote['volume']:,}
- Data Source: {quote.get('data_source', 'Yahoo Finance')}
"""
    
    news_section = f"\nNews Context:\n{news_context}" if news_context else ""
    
    prompt = f"""
COMPREHENSIVE HIGH-ACCURACY TRADING ANALYSIS for {ticker}:

{session_summary}

{tech_summary}

{fund_summary}

{options_summary}
{news_section}

**CRITICAL**: This analysis must provide 80-100% accuracy trading insights. Based on this comprehensive multi-source analysis, provide:

1. **Overall Assessment** (Bullish/Bearish/Neutral) with confidence rating (1-100)
2. **Trading Strategy** (Scalp/Day Trade/Swing/Position/Avoid) with specific timeframe
3. **Entry Strategy**: Exact price levels and conditions
4. **Profit Targets**: 3 realistic target levels with rationale
5. **Risk Management**: Stop loss levels and position sizing guidance
6. **Technical Outlook**: Key levels to watch and breakout scenarios
7. **Fundamental Justification**: How fundamentals support the technical setup
8. **Options Strategy**: Specific options plays if applicable (use Unusual Whales data if available)
9. **Catalyst Watch**: Events or levels that could trigger major moves
10. **Risk Factors**: What could invalidate this analysis

Keep analysis under 400 words but be HIGHLY SPECIFIC and ACTIONABLE for maximum trading accuracy.
"""
    
    return prompt

# Multi-AI Analysis System
class MultiAIAnalyzer:
    """Enhanced multi-AI analysis system with comprehensive data integration"""
    
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
    
    def analyze_with_openai(self, prompt: str) -> str:
        """OpenAI analysis"""
        if not self.openai_client:
            return "OpenAI not available"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temperature for higher accuracy
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def analyze_with_gemini(self, prompt: str) -> str:
        """Gemini analysis"""
        if not self.gemini_model:
            return "Gemini not available"
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    
    def analyze_with_grok(self, prompt: str) -> str:
        """Grok analysis"""
        if not self.grok_client:
            return "Grok not available"
        
        return self.grok_client.analyze_trading_setup(prompt)
    
    def multi_ai_consensus_enhanced(self, comprehensive_prompt: str) -> Dict[str, str]:
        """Get consensus analysis from all available AI models with enhanced prompts"""
        analyses = {}
        
        # Get analysis from each available model
        if self.openai_client:
            analyses["OpenAI"] = self.analyze_with_openai(comprehensive_prompt)
        
        if self.gemini_model:
            analyses["Gemini"] = self.analyze_with_gemini(comprehensive_prompt)
        
        if self.grok_client:
            analyses["Grok"] = self.analyze_with_grok(comprehensive_prompt)
        
        return analyses
    
    def synthesize_consensus(self, analyses: Dict[str, str], ticker: str) -> str:
        """Synthesize multiple AI analyses into a consensus view"""
        if not analyses:
            return "No AI models available for analysis."
        
        # Create synthesis prompt
        analysis_text = "\n\n".join([f"**{model} Analysis:**\n{analysis}" for model, analysis in analyses.items()])
        
        synthesis_prompt = f"""
        Based on the following AI analyses for {ticker}, provide a synthesized HIGH-ACCURACY consensus view:
        
        {analysis_text}
        
        Synthesize into:
        1. **Consensus Sentiment** and average confidence
        2. **Agreed Trading Strategy**
        3. **Consensus Price Levels** (entry, targets, stops)
        4. **Risk Assessment**
        5. **Key Points of Agreement/Disagreement**
        
        Prioritize areas where models agree and note any significant disagreements. Focus on HIGHEST PROBABILITY trades.
        """
        
        # Use the first available model for synthesis
        if self.openai_client:
            return self.analyze_with_openai(synthesis_prompt)
        elif self.gemini_model:
            return self.analyze_with_gemini(synthesis_prompt)
        elif self.grok_client:
            return self.analyze_with_grok(synthesis_prompt)
        
        return "No AI models available for synthesis."

# Initialize Multi-AI Analyzer
multi_ai = MultiAIAnalyzer()

# Enhanced AI analysis functions
def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook using comprehensive technical, fundamental, and options analysis"""
    
    # Get comprehensive analysis data
    with st.spinner(f"Gathering comprehensive data for {ticker}..."):
        quote = get_live_quote(ticker, st.session_state.selected_tz)
        technical_analysis = get_comprehensive_technical_analysis(ticker)
        fundamental_analysis = get_fundamental_analysis(ticker)
        options_analysis = get_advanced_options_analysis(ticker)
        
        # Get news context
        news = get_finnhub_news(ticker)
        news_context = ""
        if news:
            news_context = f"Recent News: {news[0].get('headline', '')[:100]}..."
    
    # Construct comprehensive prompt
    comprehensive_prompt = construct_comprehensive_analysis_prompt(
        ticker, quote, technical_analysis, fundamental_analysis, 
        options_analysis, news_context
    )
    
    if st.session_state.ai_model == "Multi-AI":
        # Use multi-AI consensus with enhanced data
        analyses = multi_ai.multi_ai_consensus_enhanced(comprehensive_prompt)
        if analyses:
            result = f"## 🤖 Enhanced Multi-AI Analysis for {ticker}\n\n"
            result += f"**Data Sources:** {quote.get('data_source', 'Yahoo Finance')} | Updated: {quote['last_updated']}\n\n"
            
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            
            # Add synthesis
            synthesis = multi_ai.synthesize_consensus(analyses, ticker)
            result += f"### 🎯 AI Consensus Summary:\n{synthesis}"
            return result
        else:
            return f"**{ticker} Analysis** - No AI models available for multi-AI analysis."
    
    elif st.session_state.ai_model == "OpenAI":
        if not openai_client:
            return f"**{ticker} Analysis** (OpenAI API not configured)"
        return multi_ai.analyze_with_openai(comprehensive_prompt)
    
    elif st.session_state.ai_model == "Gemini":
        if not gemini_model:
            return f"**{ticker} Analysis** (Gemini API not configured)"
        return multi_ai.analyze_with_gemini(comprehensive_prompt)
    
    elif st.session_state.ai_model == "Grok":
        if not grok_enhanced:
            return f"**{ticker} Analysis** (Grok API not configured)"
        return multi_ai.analyze_with_grok(comprehensive_prompt)
    
    else:
        return "No AI model selected or configured."

# Enhanced earnings calendar using Trading Economics
def get_earnings_calendar() -> List[Dict]:
    """Get earnings calendar using Trading Economics API"""
    if trading_economics_client:
        try:
            # Get economic calendar which includes earnings
            calendar = trading_economics_client.get_economic_calendar(days=7)
            
            # Filter for earnings-related events
            earnings_events = []
            for event in calendar:
                if any(keyword in event.get("event", "").lower() for keyword in ["earnings", "results", "report"]):
                    earnings_events.append({
                        "ticker": event.get("currency", "N/A"),  # May need adjustment based on API response
                        "date": event.get("date", ""),
                        "time": event.get("time", ""),
                        "estimate": event.get("forecast", ""),
                        "importance": event.get("importance", "Medium")
                    })
            
            if earnings_events:
                return earnings_events
        except Exception as e:
            print(f"Trading Economics earnings error: {e}")
    
    # Fallback to simulated data
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    return [
        {"ticker": "MSFT", "date": today, "time": "After Hours", "estimate": "$2.50", "importance": "High"},
        {"ticker": "NVDA", "date": today, "time": "Before Market", "estimate": "$1.20", "importance": "High"},
        {"ticker": "TSLA", "date": today, "time": "After Hours", "estimate": "$0.75", "importance": "High"},
    ]

# Function to get important economic events using AI and Trading Economics
def get_important_events() -> List[Dict]:
    # Try Trading Economics first
    if trading_economics_client:
        try:
            calendar = trading_economics_client.get_economic_calendar(days=7)
            
            # Filter for high-impact events
            important_events = []
            for event in calendar:
                importance = event.get("importance", "").lower()
                if "high" in importance or event.get("event", "").lower() in [
                    "cpi", "ppi", "gdp", "unemployment", "fed", "fomc", "nonfarm", "retail"
                ]:
                    important_events.append({
                        "event": event.get("event", "Unknown Event"),
                        "date": event.get("date", ""),
                        "time": event.get("time", ""),
                        "impact": event.get("importance", "High")
                    })
            
            if important_events:
                return important_events[:10]  # Return top 10
                
        except Exception as e:
            print(f"Trading Economics events error: {e}")
    
    # Fallback to AI generation if Trading Economics not available
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
                response = multi_ai.analyze_with_openai(prompt)
            elif gemini_model:
                response = multi_ai.analyze_with_gemini(prompt)
            elif grok_enhanced:
                response = multi_ai.analyze_with_grok(prompt)
            else:
                return []
        elif st.session_state.ai_model == "OpenAI" and openai_client:
            response = multi_ai.analyze_with_openai(prompt)
        elif st.session_state.ai_model == "Gemini" and gemini_model:
            response = multi_ai.analyze_with_gemini(prompt)
        elif st.session_state.ai_model == "Grok" and grok_enhanced:
            response = multi_ai.analyze_with_grok(prompt)
        else:
            return []
        
        events = json.loads(response)
        return events
    except Exception as e:
        st.error(f"Error fetching economic events: {str(e)}")
        return []

def generate_technical_summary(technical: Dict) -> str:
    """Generate concise technical summary"""
    if technical.get("error"):
        return f"Technical Error: {technical['error']}"
    
    if "short_term" in technical:
        rsi = technical['short_term'].get('rsi', 0)
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        return f"RSI: {rsi:.1f} ({rsi_status}), Trend: {technical.get('trend_analysis', 'Unknown')}"
    elif "rsi" in technical:
        rsi = technical.get('rsi', 0)
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        return f"RSI: {rsi:.1f} ({rsi_status}), Trend: {technical.get('trend_analysis', 'Unknown')}"
    
    return "Technical analysis pending..."

def generate_fundamental_summary(fundamental: Dict) -> str:
    """Generate concise fundamental summary"""
    if fundamental.get("error"):
        return f"Fundamental Error: {fundamental['error']}"
    
    health = fundamental.get('financial_health', 'Unknown')
    valuation = fundamental.get('valuation_assessment', 'Unknown')
    pe = fundamental.get('pe_ratio', 'N/A')
    
    return f"Health: {health}, Valuation: {valuation}, P/E: {pe}"

def generate_options_summary(options: Dict) -> str:
    """Generate concise options summary"""
    if options.get("error"):
        return f"Options Error: {options['error']}"
    
    if options.get("unusual_whales_data"):
        return f"UW Premium: ${options.get('total_premium', 0):,.0f}, Flows: {options.get('flow_count', 0)}"
    
    basic = options.get('basic_metrics', {})
    flow = options.get('flow_analysis', {})
    
    pc_ratio = basic.get('put_call_volume_ratio', 0)
    pc_ratio_str = f"{pc_ratio:.2f}" if pc_ratio is not None else "N/A"
    
    sentiment = flow.get('flow_sentiment', 'Neutral')
    
    return f"P/C Ratio: {pc_ratio_str}, Flow: {sentiment}"

# Enhanced auto-generation with comprehensive analysis
def ai_auto_generate_plays_enhanced(tz: str):
    """Enhanced auto-generation with comprehensive analysis"""
    plays = []
    
    try:
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        
        # Scan for significant movers with enhanced criteria
        candidates = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # Increased workers for speed
            future_to_ticker = {executor.submit(get_live_quote, ticker, tz): ticker for ticker in scan_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    quote = future.result()
                    if not quote["error"]:
                        # Enhanced criteria for significance
                        volume_significant = quote["volume"] > 500000  # Lower threshold for more opportunities
                        price_significant = abs(quote["change_percent"]) >= 1.0  # Lower threshold
                        spread_reasonable = (quote["ask"] - quote["bid"]) / quote["last"] < 0.03  # Max 3% spread
                        
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
        top_candidates = candidates[:8]  # Increased from 5 to 8
        
        # Generate enhanced plays for top candidates
        for candidate in top_candidates:
            ticker = candidate["ticker"]
            quote = candidate["quote"]
            
            # Get comprehensive analysis data
            technical_analysis = get_comprehensive_technical_analysis(ticker)
            fundamental_analysis = get_fundamental_analysis(ticker)
            options_analysis = get_advanced_options_analysis(ticker)
            
            # Get recent news for context
            news = get_finnhub_news(ticker)
            catalyst = ""
            if news:
                catalyst = news[0].get('headline', '')[:100] + "..."
            
            # Generate comprehensive analysis
            comprehensive_prompt = construct_comprehensive_analysis_prompt(
                ticker, quote, technical_analysis, fundamental_analysis, 
                options_analysis, catalyst
            )
            
            # Generate AI analysis using selected model
            if st.session_state.ai_model == "Multi-AI":
                analyses = multi_ai.multi_ai_consensus_enhanced(comprehensive_prompt)
                if analyses:
                    play_analysis = f"## Enhanced Multi-AI Analysis\n\n"
                    for model, analysis in analyses.items():
                        play_analysis += f"**{model}:** {analysis[:200]}...\n\n"
                    synthesis = multi_ai.synthesize_consensus(analyses, ticker)
                    play_analysis += f"**Consensus:** {synthesis}"
                else:
                    play_analysis = f"No AI models available for {ticker} analysis"
            else:
                play_analysis = ai_playbook(ticker, quote["change_percent"], catalyst, options_analysis)

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
                "catalyst": catalyst if catalyst else f"Market movement: {quote['change_percent']:+.2f}%",
                "play_analysis": play_analysis,
                "volume": quote['volume'],
                "timestamp": quote['last_updated'],
                "data_source": quote.get('data_source', 'Yahoo Finance'),
                "technical_summary": generate_technical_summary(technical_analysis),
                "fundamental_summary": generate_fundamental_summary(fundamental_analysis),
                "options_summary": generate_options_summary(options_analysis),
                "significance_score": candidate["significance"]
            }
            plays.append(play)
        
        return plays
    except Exception as e:
        st.error(f"Error generating enhanced auto plays: {str(e)}")
        return []

# Enhanced market analysis
def ai_market_analysis_enhanced(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis with comprehensive data"""
    
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
                "technical": technical
            }
        except:
            continue
    
    # Analyze sector rotation
    sector_data = analyze_sector_rotation()
    
    # Construct enhanced market analysis prompt
    market_context = f"""
Market Technical Overview:
{format_market_technical(market_technical)}

Sector Analysis:
{sector_data}

Top News Headlines:
{chr(10).join([f"- {item['title']}" for item in news_items[:5]])}

Top Market Movers:
{chr(10).join([f"- {m['ticker']}: {m['change_pct']:+.2f}%" for m in movers[:5]])}

Provide comprehensive market analysis covering:
1. Overall market sentiment and direction
2. Key technical levels for major indices
3. Sector rotation patterns and opportunities
4. Risk-on vs risk-off positioning
5. Trading opportunities and strategies
6. Key events and catalysts to watch

Keep analysis under 300 words but be specific and actionable.
"""
    
    # Use selected AI model for analysis
    if st.session_state.ai_model == "Multi-AI":
        analyses = {}
        if openai_client:
            analyses["OpenAI"] = multi_ai.analyze_with_openai(market_context)
        if gemini_model:
            analyses["Gemini"] = multi_ai.analyze_with_gemini(market_context)
        if grok_enhanced:
            analyses["Grok"] = multi_ai.analyze_with_grok(market_context)
        
        if analyses:
            result = "## 🤖 Enhanced Multi-AI Market Analysis\n\n"
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            
            synthesis = multi_ai.synthesize_consensus(analyses, "Market")
            result += f"### 🎯 Market Consensus:\n{synthesis}"
            return result
    else:
        # Use individual AI model
        return multi_ai.analyze_with_openai(market_context) if openai_client else "AI analysis not available"

def format_market_technical(market_tech: Dict) -> str:
    """Format market technical data for prompt"""
    formatted = ""
    for symbol, data in market_tech.items():
        formatted += f"{symbol}: ${data['price']:.2f} ({data['change']:+.2f}%)\n"
    return formatted

def analyze_sector_rotation() -> str:
    """Analyze sector rotation patterns"""
    sector_etfs = ["XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC"]
    sector_performance = {}
    
    for etf in sector_etfs:
        try:
            quote = get_live_quote(etf)
            if not quote.get("error"):
                sector_performance[etf] = quote['change_percent']
        except:
            continue
    
    # Sort by performance
    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
    
    result = "Sector Performance Today:\n"
    for etf, perf in sorted_sectors[:5]:
        result += f"- {etf}: {perf:+.2f}%\n"
    
    return result

def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis using selected AI model"""
    return ai_market_analysis_enhanced(news_items, movers)

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

# UPDATED: Data Source Configuration - Unusual Whales Primary
st.sidebar.subheader("📊 Data Configuration")
available_sources = ["Unusual Whales"]
if alpha_vantage_client:
    available_sources.append("Alpha Vantage")
if twelvedata_client:
    available_sources.append("Twelve Data")
available_sources.append("Yahoo Finance")
st.session_state.data_source = st.sidebar.selectbox("Primary Data Source", available_sources, index=0)

# UPDATED: Data source status with Unusual Whales and Trading Economics
st.sidebar.subheader("Data Sources")

# Debug toggle and API test
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show API response details")
st.session_state.debug_mode = debug_mode

if debug_mode:
    st.sidebar.subheader("🔬 Enhanced Data Debug")
    debug_ticker = st.sidebar.selectbox("Debug Ticker", CORE_TICKERS[:10])
    
    if st.sidebar.button("🧪 Test Enhanced Analysis"):
        with st.sidebar:
            st.write("**Testing Enhanced Functions:**")
            
            # Test Unusual Whales
            if unusual_whales_client:
                uw_result = unusual_whales_client.get_quote(debug_ticker)
                st.write(f"Unusual Whales: {'✅' if not uw_result.get('error') else '❌'}")
            
            # Test technical analysis
            tech_result = get_comprehensive_technical_analysis(debug_ticker)
            st.write(f"Technical: {'✅' if not tech_result.get('error') else '❌'}")
            
            # Test fundamental analysis  
            fund_result = get_fundamental_analysis(debug_ticker)
            st.write(f"Fundamental: {'✅' if not fund_result.get('error') else '❌'}")
            
            # Test options analysis
            opt_result = get_advanced_options_analysis(debug_ticker)
            st.write(f"Options: {'✅' if not opt_result.get('error') else '❌'}")
            
            if st.checkbox("Show Raw Data"):
                st.json({"uw": uw_result if unusual_whales_client else None, 
                        "tech": tech_result, "fund": fund_result, "opts": opt_result})

if debug_mode and st.sidebar.button("🧪 Test All APIs"):
    st.sidebar.write("**Testing Data APIs:**")
    
    if unusual_whales_client:
        with st.spinner("Testing Unusual Whales API..."):
            test_response = unusual_whales_client.get_quote("AAPL")
            st.sidebar.json(test_response)
    
    if trading_economics_client:
        with st.spinner("Testing Trading Economics API..."):
            test_calendar = trading_economics_client.get_economic_calendar(days=1)
            st.sidebar.write(f"Trading Economics: {len(test_calendar)} events found")
    
    if twelvedata_client:
        with st.spinner("Testing Twelve Data API..."):
            test_response = twelvedata_client.get_quote("AAPL")
            st.sidebar.json(test_response)
    
    st.sidebar.write("**Testing AI APIs:**")
    test_prompt = "Test connection - respond with 'OK'"
    
    if openai_client:
        openai_test = multi_ai.analyze_with_openai(test_prompt)
        st.sidebar.write(f"OpenAI: {openai_test[:50]}...")
    
    if gemini_model:
        gemini_test = multi_ai.analyze_with_gemini(test_prompt)
        st.sidebar.write(f"Gemini: {gemini_test[:50]}...")
    
    if grok_enhanced:
        grok_test = multi_ai.analyze_with_grok(test_prompt)
        st.sidebar.write(f"Grok: {grok_test[:50]}...")

# UPDATED: Show available data sources with new APIs
if unusual_whales_client:
    st.sidebar.success("✅ Unusual Whales Connected (PRIMARY)")
else:
    st.sidebar.error("❌ Unusual Whales Not Connected")

if trading_economics_client:
    st.sidebar.success("✅ Trading Economics Connected")
else:
    st.sidebar.warning("⚠️ Trading Economics Not Connected")

if twelvedata_client:
    st.sidebar.success("✅ Twelve Data Connected")
else:
    st.sidebar.warning("⚠️ Twelve Data Not Connected")

if alpha_vantage_client:
    st.sidebar.success("✅ Alpha Vantage Connected")
else:
    st.sidebar.warning("⚠️ Alpha Vantage Not Connected")

st.sidebar.success("✅ Yahoo Finance Connected (Fallback)")

if FINNHUB_KEY:
    st.sidebar.success("✅ Finnhub API Connected")
else:
    st.sidebar.warning("⚠️ Finnhub API Not Found")

if POLYGON_KEY:
    st.sidebar.success("✅ Polygon API Connected (News)")
else:
    st.sidebar.warning("⚠️ Polygon API Not Found")

# UPDATED: Auto-refresh controls with faster intervals
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

with col2:
    st.session_state.refresh_interval = st.selectbox("Interval", [5, 10, 15, 30], index=1)  # Faster intervals

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
if unusual_whales_client:
    data_sources.append("Unusual Whales")
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

# TAB 1: Live Quotes - ENHANCED with Unusual Whales
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist & Market Movers")
    
    # Enhanced session status showing data source priority
    current_tz_hour = current_tz.hour
    if 4 <= current_tz_hour < 9:
        session_status = "🌅 Premarket"
    elif 9 <= current_tz_hour < 16:
        session_status = "🟢 Market Open"
    else:
        session_status = "🌆 After Hours"
    
    st.markdown(f"**Trading Session ({tz_label}):** {session_status}")
    
    # Show data source hierarchy
    if unusual_whales_client:
        st.info("🐋 **PRIMARY**: Unusual Whales → Alpha Vantage → Twelve Data → Yahoo Finance")
    else:
        st.warning("⚠️ **Unusual Whales not connected** - Using fallback sources")
    
    # Search bar for any ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Search Any Stock", placeholder="Enter any ticker (e.g., AAPL, SPY, GME)", key="search_quotes").upper().strip()
    with col2:
        search_quotes = st.button("Get Quote", key="search_quotes_btn")
    
    # Search result for any ticker
    if search_quotes and search_ticker:
        with st.spinner(f"Getting quote for {search_ticker}..."):
            quote = get_live_quote(search_ticker, tz_label)
            if not quote["error"]:
                st.success(f"Quote for {search_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
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
                
                # Enhanced Analysis Button with Unusual Whales data
                if col4.button(f"📊 Enhanced Analysis", key=f"quotes_enhanced_{search_ticker}"):
                    with st.spinner(f"Running comprehensive analysis for {search_ticker}..."):
                        technical = get_comprehensive_technical_analysis(search_ticker)
                        fundamental = get_fundamental_analysis(search_ticker)
                        options = get_advanced_options_analysis(search_ticker)
                        
                        # Show Unusual Whales institutional data if available
                        if unusual_whales_client:
                            institutional = unusual_whales_client.get_institutional_activity(search_ticker)
                            if not institutional.get("error"):
                                st.success("🐋 Unusual Whales Institutional Activity")
                                inst_data = institutional.get("institutional_data", [])
                                data_type = institutional.get("data_type", "institutional")
                                
                                if isinstance(inst_data, list) and inst_data:
                                    for i, activity in enumerate(inst_data[:3]):
                                        if isinstance(activity, dict):
                                            activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                            value = activity.get("transaction_value", activity.get("value", 0))
                                            st.write(f"• {activity_type}: ${value:,.0f}")
                                elif isinstance(inst_data, dict):
                                    st.write(f"**{data_type.title()} data available**")
                        
                        # Display technical summary
                        if not technical.get("error"):
                            st.success("✅ Technical Analysis Complete")
                            tech_col1, tech_col2, tech_col3 = st.columns(3)
                            if "short_term" in technical:
                                tech_col1.metric("RSI", f"{technical['short_term'].get('rsi', 0):.1f}")
                                tech_col2.metric("Trend", technical.get('trend_analysis', 'Unknown'))
                                tech_col3.metric("Signal", technical.get('signal_strength', 'Unknown'))
                            elif "rsi" in technical:
                                tech_col1.metric("RSI", f"{technical.get('rsi', 0):.1f}")
                                tech_col2.metric("Trend", technical.get('trend_analysis', 'Unknown'))
                                tech_col3.metric("BB Position", f"{technical.get('bb_position', 0):.2f}")
                        
                        # Display fundamental summary  
                        if not fundamental.get("error"):
                            st.success("✅ Fundamental Analysis Complete")
                            fund_col1, fund_col2, fund_col3 = st.columns(3)
                            fund_col1.metric("Health", fundamental.get('financial_health', 'Unknown'))
                            fund_col2.metric("Valuation", fundamental.get('valuation_assessment', 'Unknown'))
                            fund_col3.metric("P/E Ratio", fundamental.get('pe_ratio', 'N/A'))
                        
                        # Display enhanced options summary with UW data
                        if not options.get("error"):
                            if options.get("unusual_whales_data"):
                                st.success("🐋 Unusual Whales Options Flow Complete")
                                opt_col1, opt_col2, opt_col3 = st.columns(3)
                                opt_col1.metric("Total Premium", f"${options.get('total_premium', 0):,.0f}")
                                opt_col2.metric("Flow Count", options.get('flow_count', 0))
                                opt_col3.metric("Unusual Alerts", len(options.get('unusual_activity', [])))
                            else:
                                st.success("✅ Options Analysis Complete")
                                opt_col1, opt_col2, opt_col3 = st.columns(3)
                                basic = options.get('basic_metrics', {})
                                flow = options.get('flow_analysis', {})
                                opt_col1.metric("P/C Ratio", f"{basic.get('put_call_volume_ratio', 0):.2f}")
                                opt_col2.metric("Flow Sentiment", flow.get('flow_sentiment', 'Neutral'))
                                opt_col3.metric("Unusual Activity", f"{options.get('unusual_activity', {}).get('total_unusual_contracts', 0)}")
                
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
    
    # Watchlist display with enhanced data
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    st.markdown("### Your Watchlist")
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab or check Market Movers below.")
    else:
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
                col3.caption(f"Source: {quote.get('data_source', 'Yahoo Finance')}")
                
                # Show data source indicator
                source = quote.get('data_source', 'Yahoo Finance')
                if source == "Unusual Whales":
                    col3.success("🐋 UW")
                elif source in ["Alpha Vantage", "Twelve Data"]:
                    col3.info(f"📊 {source[:2]}")
                else:
                    col3.warning("📈 YF")
                
                if abs(quote['change_percent']) >= 1.0:  # Lowered threshold
                    if col4.button(f"🎯 AI Analysis", key=f"quotes_ai_{ticker}"):
                        with st.spinner(f"Analyzing {ticker}..."):
                            options_data = get_options_data(ticker)
                            analysis = ai_playbook(ticker, quote['change_percent'], "", options_data)
                            st.success(f"🤖 {ticker} Analysis")
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
                    
                    # Get Unusual Whales Options Flow if available
                    if unusual_whales_client:
                        uw_volume = unusual_whales_client.get_options_volume(ticker)
                        uw_alerts = unusual_whales_client.get_flow_alerts(ticker)
                        
                        if not uw_volume.get("error") or not uw_alerts.get("error"):
                            st.write("### 🐋 Unusual Whales Options Data")
                            
                            if not uw_volume.get("error"):
                                volume_data = uw_volume.get("options_volume", {})
                                if isinstance(volume_data, dict):
                                    st.write(f"**Options Volume Data Available**")
                                    if volume_data.get("call_volume"):
                                        st.write(f"• Call Volume: {volume_data.get('call_volume', 0):,}")
                                    if volume_data.get("put_volume"):
                                        st.write(f"• Put Volume: {volume_data.get('put_volume', 0):,}")
                            
                            if not uw_alerts.get("error"):
                                alerts_data = uw_alerts.get("flow_alerts", [])
                                if alerts_data:
                                    st.write(f"**Flow Alerts:** {len(alerts_data)} detected")
                                    # Show first few alerts if they exist
                                    for i, alert in enumerate(alerts_data[:3]):
                                        if isinstance(alert, dict):
                                            st.write(f"• Alert {i+1}: {alert.get('description', 'Flow detected')}")
                        else:
                            st.info("UW options data not available for this ticker")
                    
                    st.markdown("### 🎯 AI Playbook")
                    catalyst_title = news[0].get('headline', '') if news else ""
                    options_data = get_options_data(ticker)
                    if options_data:
                        st.write("**Options Metrics:**")
                        opt_col1, opt_col2, opt_col3 = st.columns(3)
                        opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                        opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                        opt_col3.metric("Total Contracts", f"{options_data.get('total_calls', 0) + options_data.get('total_puts', 0):,}")
                        st.caption("Note: Options data from yfinance + Unusual Whales")
                    st.markdown(ai_playbook(ticker, quote['change_percent'], catalyst_title, options_data))
                
                st.divider()

    # Top Market Movers - Enhanced with faster scanning
    st.markdown("### 🌟 Top Market Movers")
    st.caption("Stocks with significant intraday movement from CORE_TICKERS (Enhanced with Unusual Whales)")
    
    if st.button("🔄 Refresh Movers", key="refresh_movers"):
        with st.spinner("Scanning market movers..."):
            movers = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ticker = {executor.submit(get_live_quote, ticker, tz_label): ticker for ticker in CORE_TICKERS[:25]}  # Increased from 20 to 25
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        quote = future.result()
                        if not quote["error"]:
                            movers.append({
                                "ticker": ticker,
                                "change_pct": quote["change_percent"],
                                "price": quote["last"],
                                "volume": quote["volume"],
                                "data_source": quote.get("data_source", "Yahoo Finance")
                            })
                    except Exception as exc:
                        print(f'{ticker} generated an exception: {exc}')
            
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            top_movers = movers[:12]  # Show top 12 movers

            for mover in top_movers:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    direction = "🚀" if mover["change_pct"] > 0 else "📉"
                    col1.metric(f"{direction} {mover['ticker']}", f"${mover['price']:.2f}", f"{mover['change_pct']:+.2f}%")
                    col2.write("**Volume**")
                    col2.write(f"{mover['volume']:,}")
                    col3.write("**Source**")
                    source = mover['data_source']
                    if source == "Unusual Whales":
                        col3.success("🐋 UW")
                    else:
                        col3.write(source)
                    
                    if col4.button(f"Add {mover['ticker']} to Watchlist", key=f"quotes_mover_{mover['ticker']}"):
                        current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                        if mover['ticker'] not in current_list:
                            current_list.append(mover['ticker'])
                            st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                            st.success(f"Added {mover['ticker']} to watchlist!")
                            st.rerun()
                    st.divider()
    else:
        st.info("Click 'Refresh Movers' to scan the market for significant moves")

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

# TAB 3: Enhanced Catalyst Scanner with Unusual Whales Integration
with tabs[2]:
    st.subheader("🔥 Enhanced Real-Time Catalyst Scanner")
    st.caption("Comprehensive news analysis + Unusual Whales institutional activity")
    
    # Show data sources status including UW
    sources_status = []
    if unusual_whales_client:
        sources_status.append("✅ Unusual Whales")
    else:
        sources_status.append("❌ Unusual Whales")
    if FINNHUB_KEY:
        sources_status.append("✅ Finnhub")
    else:
        sources_status.append("❌ Finnhub")
    if POLYGON_KEY:
        sources_status.append("✅ Polygon")
    else:
        sources_status.append("❌ Polygon")
    sources_status.append("✅ Yahoo Finance")
    
    st.info(f"**Data Sources:** {' | '.join(sources_status)}")
    
    # Search specific stock catalysts with UW integration
    col1, col2 = st.columns([3, 1])
    with col1:
        search_catalyst_ticker = st.text_input("🔍 Search catalysts for stock", placeholder="Enter ticker", key="search_catalyst").upper().strip()
    with col2:
        search_catalyst = st.button("🔍 Analyze Catalysts", key="search_catalyst_btn")
    
    if search_catalyst and search_catalyst_ticker:
        with st.spinner(f"Searching all sources for {search_catalyst_ticker} catalysts..."):
            # Get comprehensive catalyst analysis
            catalyst_data = get_stock_specific_catalysts(search_catalyst_ticker)
            quote = get_live_quote(search_catalyst_ticker, tz_label)
            
            if not quote["error"]:
                st.success(f"Catalyst Analysis for {search_catalyst_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
                # Price and volume info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Total Catalysts", catalyst_data["catalyst_summary"]["total_catalysts"])
                col4.metric("Highest Impact", f"{catalyst_data['catalyst_summary']['highest_impact']}")
                
                # Get institutional data if available
                institutional_data = None
                if unusual_whales_client:
                    institutional_data = unusual_whales_client.get_institutional_activity(search_catalyst_ticker)
                
                # Show institutional activity if available
                if institutional_data and not institutional_data.get("error"):
                    st.markdown("#### 🐋 Unusual Whales Institutional Activity")
                    inst_data = institutional_data.get("institutional_data", [])
                    data_type = institutional_data.get("data_type", "institutional")
                    
                    if isinstance(inst_data, list) and inst_data:
                        inst_col1, inst_col2, inst_col3 = st.columns(3)
                        for i, activity in enumerate(inst_data[:3]):
                            with [inst_col1, inst_col2, inst_col3][i]:
                                if isinstance(activity, dict):
                                    activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                    value = activity.get("transaction_value", activity.get("value", 0))
                                    st.metric(activity_type, f"${value:,.0f}")
                                else:
                                    st.metric(f"Activity {i+1}", "Data Available")
                    elif isinstance(inst_data, dict):
                        st.write(f"**{data_type.title()} data available for {search_catalyst_ticker}**")
                    else:
                        st.info("Institutional activity data structure not recognized")
                
                # Session breakdown
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%") 
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                # Catalyst Summary
                st.markdown("#### 📊 Catalyst Summary")
                summary = catalyst_data["catalyst_summary"]
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Positive", summary["positive_catalysts"], help="Bullish catalysts")
                summary_col2.metric("Negative", summary["negative_catalysts"], help="Bearish catalysts")
                summary_col3.metric("Categories", len(summary["primary_categories"]), help="Types of catalysts found")
                
                # Primary Categories
                if summary["primary_categories"]:
                    st.write("**Main Catalyst Categories:**")
                    for category, count in summary["primary_categories"]:
                        st.write(f"• {category.replace('_', ' ').title()}: {count} items")
                
                # Trading Implications
                if catalyst_data["trading_implications"]:
                    st.markdown("#### 🎯 Trading Implications")
                    st.markdown(catalyst_data["trading_implications"])
                
                # Individual News Items
                if catalyst_data["news_items"]:
                    st.markdown("#### 📰 Individual Catalysts")
                    
                    # Sort by catalyst strength
                    sorted_news = sorted(catalyst_data["news_items"], 
                                       key=lambda x: x["catalyst_analysis"]["catalyst_strength"], 
                                       reverse=True)
                    
                    for i, news_item in enumerate(sorted_news[:10]):  # Show top 10
                        analysis = news_item["catalyst_analysis"]
                        
                        # Create impact indicator
                        if analysis["impact_level"] == "high":
                            impact_emoji = "🚀"
                        elif analysis["impact_level"] == "medium":
                            impact_emoji = "📈"
                        else:
                            impact_emoji = "📊"
                        
                        # Sentiment indicator
                        sentiment_emoji = "📈" if analysis["sentiment"] == "positive" else "📉" if analysis["sentiment"] == "negative" else "⚪"
                        
                        with st.expander(f"{impact_emoji} {sentiment_emoji} {analysis['catalyst_strength']}/100 - {news_item['title'][:80]}... | {news_item['source']}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Summary:** {news_item.get('summary', 'No summary available')}")
                                st.write(f"**Source:** {news_item['source']} | **Provider:** {news_item.get('provider', 'Unknown')}")
                                if news_item.get('url'):
                                    st.markdown(f"[📖 Read Full Article]({news_item['url']})")
                            
                            with col2:
                                st.metric("Impact", f"{analysis['catalyst_strength']}/100")
                                st.write(f"**Category:** {analysis['primary_category'].replace('_', ' ').title()}")
                                st.write(f"**Sentiment:** {analysis['sentiment'].title()}")
                                st.write(f"**Level:** {analysis['impact_level'].title()}")
                
                # Add to watchlist button
                if st.button(f"Add {search_catalyst_ticker} to Watchlist", key="catalyst_add_searched_ticker"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_catalyst_ticker not in current_list:
                        current_list.append(search_catalyst_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_catalyst_ticker} to watchlist!")
                        st.rerun()
                
                st.divider()
            else:
                st.error(f"Could not get quote for {search_catalyst_ticker}: {quote['error']}")
    
    # Main market catalyst scan
    st.markdown("### 🌐 Market-Wide Catalyst Scanner")
    
    scan_col1, scan_col2 = st.columns([2, 1])
    with scan_col1:
        st.caption("Scan all news sources + institutional activity for market-moving catalysts")
    with scan_col2:
        scan_type = st.selectbox("Scan Type", ["All Catalysts", "High Impact Only", "By Category"], key="catalyst_scan_type")
    
    if st.button("🔍 Scan Market Catalysts", type="primary"):
        with st.spinner("Scanning all sources for market catalysts..."):
            # Get market-moving news
            market_news = get_market_moving_news()
            
            # Get significant movers for correlation with faster scanning
            movers = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ticker = {executor.submit(get_live_quote, ticker, tz_label): ticker for ticker in CORE_TICKERS[:25]}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        quote = future.result()
                        if not quote["error"] and abs(quote["change_percent"]) >= 1.0:  # Lowered threshold
                            movers.append({
                                "ticker": ticker,
                                "change_pct": quote["change_percent"],
                                "price": quote["last"],
                                "volume": quote["volume"],
                                "data_source": quote.get("data_source", "Yahoo Finance")
                            })
                    except Exception as exc:
                        print(f'{ticker} generated an exception: {exc}')
            
            movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
            
            # Display results based on scan type
            if scan_type == "High Impact Only":
                filtered_news = [n for n in market_news if n["catalyst_analysis"]["impact_level"] == "high"]
            elif scan_type == "By Category":
                # Group by category
                category_groups = {}
                for n in market_news:
                    cat = n["catalyst_analysis"]["primary_category"]
                    if cat not in category_groups:
                        category_groups[cat] = []
                    category_groups[cat].append(n)
                
                st.markdown("### 📊 Catalysts by Category")
                for category, news_items in category_groups.items():
                    with st.expander(f"📂 {category.replace('_', ' ').title()} ({len(news_items)} items)"):
                        for news in news_items[:5]:  # Show top 5 per category
                            analysis = news["catalyst_analysis"]
                            sentiment_emoji = "📈" if analysis["sentiment"] == "positive" else "📉" if analysis["sentiment"] == "negative" else "⚪"
                            
                            st.write(f"{sentiment_emoji} **{news['title']}** ({news['source']})")
                            st.write(f"Impact: {analysis['catalyst_strength']}/100 | Sentiment: {analysis['sentiment'].title()}")
                            if news.get('related'):
                                st.write(f"Related: {news['related']}")
                            st.write("---")
                filtered_news = []  # Don't show main list for category view
            else:
                filtered_news = market_news
            
            # Display main catalyst list
            if filtered_news:
                st.markdown("### 🔥 Market-Moving Catalysts")
                st.caption(f"Found {len(filtered_news)} significant catalysts from all sources")
                
                # Summary metrics
                high_impact = len([n for n in filtered_news if n["catalyst_analysis"]["impact_level"] == "high"])
                positive_news = len([n for n in filtered_news if n["catalyst_analysis"]["sentiment"] == "positive"])
                negative_news = len([n for n in filtered_news if n["catalyst_analysis"]["sentiment"] == "negative"])
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Total Catalysts", len(filtered_news))
                metric_col2.metric("High Impact", high_impact)
                metric_col3.metric("Positive", positive_news)
                metric_col4.metric("Negative", negative_news)
                
                # Display news items
                for i, news in enumerate(filtered_news[:15]):  # Show top 15
                    analysis = news["catalyst_analysis"]
                    
                    # Impact and sentiment indicators
                    if analysis["impact_level"] == "high":
                        impact_emoji = "🚀"
                    elif analysis["impact_level"] == "medium":
                        impact_emoji = "📈"
                    else:
                        impact_emoji = "📊"
                    
                    sentiment_emoji = "📈" if analysis["sentiment"] == "positive" else "📉" if analysis["sentiment"] == "negative" else "⚪"
                    
                    with st.expander(f"{impact_emoji} {sentiment_emoji} {analysis['catalyst_strength']}/100 - {news['title'][:100]}... | {news['source']}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Summary:** {news['summary'][:300]}{'...' if len(news['summary']) > 300 else ''}")
                            st.write(f"**Source:** {news['source']} | **Provider:** {news.get('provider', 'Unknown')}")
                            if news.get('related'):
                                st.write(f"**Related Tickers:** {news['related']}")
                            if news.get('url'):
                                st.markdown(f"[📖 Read Full Article]({news['url']})")
                        
                        with col2:
                            st.metric("Impact Score", f"{analysis['catalyst_strength']}/100")
                            st.write(f"**Category:** {analysis['primary_category'].replace('_', ' ').title()}")
                            st.write(f"**Sentiment:** {analysis['sentiment'].title()}")
                            st.write(f"**Impact Level:** {analysis['impact_level'].title()}")
                            
                            # Category breakdown
                            if analysis["category_scores"]:
                                st.write("**Categories:**")
                                for cat, score in list(analysis["category_scores"].items())[:3]:
                                    st.write(f"• {cat}: {score}")
            
            # Display significant market movers with enhanced data
            if movers:
                st.markdown("### 📊 Significant Market Moves")
                st.caption("Stocks with major price movements that may be catalyst-driven")
                
                for mover in movers[:12]:  # Increased from 10 to 12
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        direction = "🚀" if mover["change_pct"] > 0 else "📉"
                        st.metric(
                            f"{direction} {mover['ticker']}", 
                            f"${mover['price']:.2f}",
                            f"{mover['change_pct']:+.2f}%"
                        )
                    with col2:
                        st.write(f"Volume: {mover['volume']:,}")
                        source = mover.get('data_source', 'Yahoo Finance')
                        if source == "Unusual Whales":
                            st.success(f"🐋 {source}")
                        else:
                            st.caption(f"Source: {source}")
                    with col3:
                        if st.button(f"📰 News", key=f"catalyst_news_{mover['ticker']}"):
                            # Quick news lookup for this ticker
                            ticker_news = get_comprehensive_news(mover['ticker'])
                            if ticker_news:
                                st.write(f"**Recent news for {mover['ticker']}:**")
                                for news in ticker_news[:3]:
                                    st.write(f"• {news['title'][:80]}... ({news['source']})")
                            else:
                                st.write(f"No recent news found for {mover['ticker']}")
                        
                        if st.button(f"Add", key=f"catalyst_add_mover_{mover['ticker']}"):
                            current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                            if mover['ticker'] not in current_list:
                                current_list.append(mover['ticker'])
                                st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                st.success(f"Added {mover['ticker']}")
                                st.rerun()

# TAB 4: Market Analysis - Enhanced with comprehensive data
with tabs[3]:
    st.subheader("📈 AI Market Analysis")
    
    # Search individual analysis with enhanced data sources
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
                
                # Get enhanced options data including UW flow
                options_data = get_options_data(search_analysis_ticker)
                
                # Get UW institutional data if available
                institutional_data = None
                if unusual_whales_client:
                    institutional_data = unusual_whales_client.get_institutional_activity(search_analysis_ticker)
                
                analysis = ai_playbook(search_analysis_ticker, quote["change_percent"], catalyst, options_data)
                
                st.success(f"🤖 AI Analysis: {search_analysis_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
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
                
                # Show institutional activity if available
                if institutional_data and not institutional_data.get("error"):
                    st.markdown("#### 🐋 Institutional Activity")
                    inst_data = institutional_data.get("institutional_data", [])
                    data_type = institutional_data.get("data_type", "institutional")
                    
                    if isinstance(inst_data, list) and inst_data:
                        inst_cols = st.columns(min(3, len(inst_data)))
                        for i, activity in enumerate(inst_data[:3]):
                            with inst_cols[i]:
                                if isinstance(activity, dict):
                                    activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                    value = activity.get("transaction_value", activity.get("value", 0))
                                    st.metric(activity_type, f"${value:,.0f}")
                    elif isinstance(inst_data, dict):
                        st.write(f"**{data_type.title()} data available**")
                    else:
                        st.info("Institutional data available")
                
                # Show options data if available
                if options_data:
                    st.markdown("#### Options Metrics")
                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    opt_col1.metric("IV", f"{options_data.get('iv', 0):.1f}%")
                    opt_col2.metric("Put/Call", f"{options_data.get('put_call_ratio', 0):.2f}")
                    opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,}")
                    opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,}")
                    st.caption("Options data: yfinance + Unusual Whales")
                
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
            
            # Enhanced movers scanning with increased speed
            movers = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ticker = {executor.submit(get_live_quote, ticker, tz_label): ticker for ticker in CORE_TICKERS[:20]}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        quote = future.result()
                        if not quote["error"]:
                            movers.append({
                                "ticker": ticker,
                                "change_pct": quote["change_percent"],
                                "price": quote["last"],
                                "data_source": quote.get("data_source", "Yahoo Finance")
                            })
                    except Exception as exc:
                        print(f'{ticker} generated an exception: {exc}')
            
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            with st.expander("📊 Supporting Data"):
                st.write("**Top Market Movers:**")
                for mover in sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]:
                    source_emoji = "🐋" if mover.get('data_source') == "Unusual Whales" else "📊"
                    st.write(f"• {source_emoji} {mover['ticker']}: {mover['change_pct']:+.2f}% | Source: {mover.get('data_source', 'Yahoo Finance')}")
                
                st.write("**Key News Headlines:**")
                for news in news_items[:3]:
                    st.write(f"• {news['title']}")

# TAB 5: AI Playbooks - Enhanced for 80-100% accuracy
with tabs[4]:
    st.subheader("🤖 AI Trading Playbooks")
    
    # Show current AI configuration with accuracy focus
    st.info(f"🤖 Current AI Mode: **{st.session_state.ai_model}** | Available Models: {', '.join(multi_ai.get_available_models()) if multi_ai.get_available_models() else 'None'}")
    st.success("🎯 **Targeting 80-100% Trading Accuracy** with enhanced data sources")
    
    # Auto-generated plays section - Enhanced with UW data
    st.markdown("### 🎯 Auto-Generated High-Accuracy Trading Plays")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("AI scans comprehensive data sources including Unusual Whales for institutional-grade trading opportunities")
    with col2:
        if st.button("🚀 Generate Auto Plays", type="primary"):
            with st.spinner("AI generating high-accuracy trading plays..."):
                auto_plays = ai_auto_generate_plays_enhanced(tz_label)
                
                if auto_plays:
                    st.success(f"🤖 Generated {len(auto_plays)} High-Accuracy Trading Plays")
                    
                    for i, play in enumerate(auto_plays):
                        with st.expander(f"🎯 {play['ticker']} - ${play['current_price']:.2f} ({play['change_percent']:+.2f}%) | {play.get('data_source', 'Yahoo Finance')}"):
                            
                            # Display session data
                            sess_col1, sess_col2, sess_col3 = st.columns(3)
                            sess_col1.metric("Premarket", f"{play['session_data']['premarket']:+.2f}%")
                            sess_col2.metric("Intraday", f"{play['session_data']['intraday']:+.2f}%")
                            sess_col3.metric("After Hours", f"{play['session_data']['afterhours']:+.2f}%")
                            
                            # Display catalyst
                            if play['catalyst']:
                                st.write(f"**Catalyst:** {play['catalyst']}")
                            
                            # Display enhanced summaries
                            st.write(f"**Technical:** {play['technical_summary']}")
                            st.write(f"**Fundamental:** {play['fundamental_summary']}")
                            st.write(f"**Options:** {play['options_summary']}")
                            st.write(f"**Significance Score:** {play['significance_score']:.2f}")
                            
                            # Display AI play analysis with accuracy focus
                            st.markdown("**🎯 High-Accuracy Trading Play:**")
                            st.markdown(play['play_analysis'])
                            
                            st.caption(f"Data Source: {play.get('data_source', 'Yahoo Finance')} | Updated: {play['timestamp']}")
                            
                            # Add to watchlist option
                            if st.button(f"Add {play['ticker']} to Watchlist", key=f"playbook_auto_{i}_{play['ticker']}"):
                                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                                if play['ticker'] not in current_list:
                                    current_list.append(play['ticker'])
                                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                    st.success(f"Added {play['ticker']} to watchlist!")
                                    st.rerun()
                else:
                    st.info("No high-accuracy trading opportunities detected. Market conditions may require patience.")
    
    st.divider()
    
    # Search any stock with comprehensive analysis
    st.markdown("### 🔍 Custom Stock Analysis")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_playbook_ticker = st.text_input("🔍 Generate high-accuracy playbook for any stock", placeholder="Enter ticker", key="search_playbook").upper().strip()
    with col2:
        search_playbook = st.button("Generate Playbook", key="search_playbook_btn")
    
    if search_playbook and search_playbook_ticker:
        quote = get_live_quote(search_playbook_ticker, tz_label)
        
        if not quote["error"]:
            with st.spinner(f"AI generating high-accuracy playbook for {search_playbook_ticker}..."):
                news = get_finnhub_news(search_playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                
                # Get comprehensive options data including UW
                options_data = get_options_data(search_playbook_ticker)
                
                # Get institutional data if available
                institutional_data = None
                if unusual_whales_client:
                    institutional_data = unusual_whales_client.get_institutional_activity(search_playbook_ticker)
                
                playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst, options_data)
                
                st.success(f"✅ {search_playbook_ticker} High-Accuracy Trading Playbook - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
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
                
                # Show institutional data if available
                if institutional_data and not institutional_data.get("error"):
                    st.markdown("#### 🐋 Institutional Activity")
                    inst_data = institutional_data.get("institutional_data", [])
                    data_type = institutional_data.get("data_type", "institutional")
                    
                    if isinstance(inst_data, list) and inst_data:
                        for i, activity in enumerate(inst_data[:2]):
                            if isinstance(activity, dict):
                                activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                value = activity.get("transaction_value", activity.get("value", 0))
                                st.write(f"• {activity_type}: ${value:,.0f}")
                            else:
                                st.write(f"• Activity {i+1}: Data available")
                    elif isinstance(inst_data, dict):
                        st.write(f"• {data_type.title()} data available")
                    else:
                        st.write("• Institutional activity detected")
                
                # Show options data if available
                if options_data:
                    st.markdown("#### Options Analysis")
                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                    opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                    opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,} @ ${options_data.get('top_call_oi_strike', 0)}")
                    opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,} @ ${options_data.get('top_put_oi_strike', 0)}")
                    st.caption("Options data: yfinance + Unusual Whales flow")
                
                st.markdown("### 🎯 High-Accuracy AI Trading Playbook")
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
                    # Get comprehensive data including UW
                    options_data = get_options_data(selected_ticker)
                    institutional_data = None
                    if unusual_whales_client:
                        institutional_data = unusual_whales_client.get_institutional_activity(selected_ticker)
                    
                    playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst_input, options_data)
                    
                    st.success(f"✅ {selected_ticker} Trading Playbook - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                    
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
                    
                    # Show institutional data if available
                    if institutional_data and not institutional_data.get("error"):
                        st.markdown("#### 🐋 Institutional Activity")
                        inst_data = institutional_data.get("institutional_data", [])
                        data_type = institutional_data.get("data_type", "institutional")
                        
                        if isinstance(inst_data, list) and inst_data:
                            inst_cols = st.columns(min(3, len(inst_data)))
                            for i, activity in enumerate(inst_data[:3]):
                                with inst_cols[i]:
                                    if isinstance(activity, dict):
                                        activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                        value = activity.get("transaction_value", activity.get("value", 0))
                                        st.metric(activity_type, f"${value:,.0f}")
                                    else:
                                        st.metric(f"Activity {i+1}", "Available")
                        elif isinstance(inst_data, dict):
                            st.write(f"**{data_type.title()} data available**")
                        else:
                            st.write("**Institutional activity detected**")
                    
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
            
            # Show data source with enhanced indicators
            source = quote.get('data_source', 'Yahoo Finance')
            if source == "Unusual Whales":
                col3.success("🐋 UW")
            elif source in ["Alpha Vantage", "Twelve Data"]:
                col3.info(f"📊 {source}")
            else:
                col3.caption(f"Source: {source}")
            
            if col4.button(f"Add {ticker} to Watchlist", key=f"sector_etf_add_{ticker}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if ticker not in current_list:
                    current_list.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {ticker} to watchlist!")
                    st.rerun()

            st.divider()

# TAB 7: 0DTE & Lottos - Enhanced with real-time options flow
with tabs[6]:
    st.subheader("🎲 0DTE & Lotto Plays")
    st.markdown("**High-risk, high-reward options expiring today. Enhanced with Unusual Whales options flow.**")

    # Ticker selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox("Select Ticker for 0DTE", options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], key="0dte_ticker")
    with col2:
        if st.button("Analyze 0DTE", key="analyze_0dte"):
            st.cache_data.clear()
            st.rerun()

    # Fetch option chain and UW flow data
    with st.spinner(f"Fetching option chain and flow data for {selected_ticker}..."):
        option_chain = get_option_chain(selected_ticker, st.session_state.selected_tz)
        quote = get_live_quote(selected_ticker, st.session_state.selected_tz)
        
        # Get UW options flow if available
        uw_flow = None
        if unusual_whales_client:
            uw_volume = unusual_whales_client.get_options_volume(selected_ticker)
            uw_alerts = unusual_whales_client.get_flow_alerts(selected_ticker)
            
            # Combine UW data if available
            if not uw_volume.get("error") or not uw_alerts.get("error"):
                uw_flow = {
                    "volume_data": uw_volume.get("options_volume", {}),
                    "flow_alerts": uw_alerts.get("flow_alerts", [])
                }

    if option_chain.get("error"):
        st.error(option_chain["error"])
    else:
        current_price = quote['last']
        expiration = option_chain["expiration"]
        is_0dte = (datetime.datetime.strptime(expiration, '%Y-%m-%d').date() == datetime.datetime.now(ZoneInfo('US/Eastern')).date())
        st.markdown(f"**Option Chain for {selected_ticker}** (Expiration: {expiration}{' - 0DTE' if is_0dte else ''})")
        st.markdown(f"**Current Price:** ${current_price:.2f} | **Source:** {quote.get('data_source', 'Yahoo Finance')}")

        # Show UW flow data if available
        if uw_flow and (uw_flow.get("volume_data") or uw_flow.get("flow_alerts")):
            st.markdown("### 🐋 Unusual Whales Options Flow")
            
            volume_data = uw_flow.get("volume_data", {})
            alerts_data = uw_flow.get("flow_alerts", [])
            
            if volume_data and isinstance(volume_data, dict):
                flow_col1, flow_col2, flow_col3 = st.columns(3)
                
                call_vol = volume_data.get("call_volume", 0)
                put_vol = volume_data.get("put_volume", 0)
                total_premium = volume_data.get("total_premium", 0)
                
                flow_col1.metric("Call Volume", f"{call_vol:,}")
                flow_col2.metric("Put Volume", f"{put_vol:,}")
                flow_col3.metric("Total Premium", f"${total_premium:,.0f}")
            
            if alerts_data and len(alerts_data) > 0:
                st.write("**Recent Flow Activity:**")
                for i, alert in enumerate(alerts_data[:5]):
                    if isinstance(alert, dict):
                        st.write(f"• {alert.get('description', f'Flow alert {i+1}')}")
                    else:
                        st.write(f"• Flow alert {i+1}")
        else:
            st.info("UW options data not available - using yfinance data")

        # AI Analysis at the top with enhanced data
        st.markdown("### 🤖 AI 0DTE Playbook")
        with st.spinner("Generating high-accuracy 0DTE analysis..."):
            tech_analysis = get_comprehensive_technical_analysis(selected_ticker)
            options_analysis = get_advanced_options_analysis(selected_ticker)
            order_flow = get_order_flow(selected_ticker, option_chain)
            
            # Enhanced catalyst with UW flow data
            catalyst = f"0DTE options activity. Technical: {generate_technical_summary(tech_analysis)}."
            if uw_flow and (uw_flow.get("volume_data") or uw_flow.get("flow_alerts")):
                volume_data = uw_flow.get("volume_data", {})
                alerts_data = uw_flow.get("flow_alerts", [])
                
                if volume_data:
                    total_premium = volume_data.get("total_premium", 0)
                    catalyst += f" UW Flow: ${total_premium:,.0f} premium"
                if alerts_data:
                    catalyst += f", {len(alerts_data)} flows."
            catalyst += f" Order Flow Sentiment: {order_flow.get('sentiment', 'Neutral')}, P/C Ratio: {order_flow.get('put_call_ratio', 0):.2f}."
            
            # Summarize option chain for AI
            calls = option_chain["calls"]
            puts = option_chain["puts"]
            top_calls = calls.sort_values('volume', ascending=False).head(3)[['strike', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False)
            top_puts = puts.sort_values('volume', ascending=False).head(3)[['strike', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False)
            option_summary = f"Top Calls:\n{top_calls}\nTop Puts:\n{top_puts}"
            catalyst += f" Option Chain: {option_summary}"
            
            playbook = ai_playbook(
                selected_ticker,
                quote["change_percent"],
                catalyst,
                options_analysis
            )
            st.markdown(playbook)

        # Display option chain
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

        # Order flow analysis
        st.markdown("### Order Flow Analysis")
        order_flow = get_order_flow(selected_ticker, option_chain)
        if order_flow.get("error"):
            st.error(order_flow["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Put/Call Volume Ratio", f"{order_flow['put_call_ratio']:.2f}")
            col2.metric("Sentiment", order_flow["sentiment"])
            col3.metric("Total Volume", f"{int(calls['volume'].sum() + puts['volume'].sum()):,}")

            st.markdown("#### Top Trades (Unusual Activity)")
            if order_flow["top_calls"]:
                st.markdown("**Top Call Trades**")
                for trade in order_flow["top_calls"]:
                    st.write(f"- Strike ${trade['strike']:.2f} ({trade['moneyness']}): ${trade['lastPrice']:.2f}, Volume: {trade['volume']:,}")
            if order_flow["top_puts"]:
                st.markdown("**Top Put Trades**")
                for trade in order_flow["top_puts"]:
                    st.write(f"- Strike ${trade['strike']:.2f} ({trade['moneyness']}): ${trade['lastPrice']:.2f}, Volume: {trade['volume']:,}")
            if not order_flow["top_calls"] and not order_flow["top_puts"]:
                st.info("No unusual activity detected.")

# TAB 8: Earnings Plays - Enhanced with Trading Economics
with tabs[7]:
    st.subheader("🗓️ Enhanced Earnings Plays")
    
    st.write("This section tracks upcoming earnings reports using Trading Economics API and provides AI analysis for potential earnings plays.")
    
    # Show Trading Economics status
    if trading_economics_client:
        st.success("✅ Trading Economics connected for enhanced earnings calendar")
    else:
        st.info("📊 Using simulated earnings data - connect Trading Economics API for live calendar")
    
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
                    importance = report.get("importance", "Medium")
                    
                    # Importance indicator
                    importance_emoji = "🚀" if importance == "High" else "📈" if importance == "Medium" else "📊"
                    
                    st.markdown(f"{importance_emoji} **{ticker}** - Earnings **{time_str}** | Importance: **{importance}**")
                    
                    # Get live quote and comprehensive options data for earnings analysis
                    quote = get_live_quote(ticker)
                    options_analysis = get_advanced_options_analysis(ticker)
                    
                    # Get UW institutional data if available
                    institutional_data = None
                    if unusual_whales_client:
                        institutional_data = unusual_whales_client.get_institutional_activity(ticker)
                    
                    if not quote.get("error"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        
                        # Show data source
                        source = quote.get('data_source', 'Yahoo Finance')
                        if source == "Unusual Whales":
                            col3.success("🐋 UW Data")
                        else:
                            col3.info(f"📊 {source}")
                        
                        col4.metric("Estimate", report.get("estimate", "N/A"))
                        
                        # Show institutional activity if available
                        if institutional_data and not institutional_data.get("error"):
                            st.write("🐋 **Institutional Activity:**")
                            inst_data = institutional_data.get("institutional_data", [])
                            data_type = institutional_data.get("data_type", "institutional")
                            
                            if isinstance(inst_data, list) and inst_data:
                                inst_cols = st.columns(min(3, len(inst_data)))
                                for i, activity in enumerate(inst_data[:3]):
                                    with inst_cols[i]:
                                        if isinstance(activity, dict):
                                            activity_type = activity.get("transaction_type", activity.get("type", "Trade"))
                                            value = activity.get("transaction_value", activity.get("value", 0))
                                            st.metric(activity_type, f"${value:,.0f}")
                        
                        if not options_analysis.get("error"):
                            st.write("**Enhanced Options Metrics:**")
                            opt_col1, opt_col2, opt_col3 = st.columns(3)
                            
                            if options_analysis.get("unusual_whales_data"):
                                # UW options data
                                opt_col1.metric("Premium Flow", f"${options_analysis.get('total_premium', 0):,.0f}")
                                opt_col2.metric("Flow Count", options_analysis.get('flow_count', 0))
                                opt_col3.metric("UW Alerts", len(options_analysis.get('unusual_activity', [])))
                            else:
                                # Fallback options data
                                basic = options_analysis.get('basic_metrics', {})
                                opt_col1.metric("IV", f"{basic.get('avg_call_iv', 0):.1f}%")
                                opt_col2.metric("Put/Call", f"{basic.get('put_call_volume_ratio', 0):.2f}")
                                opt_col3.metric("Total OI", f"{basic.get('total_call_oi', 0) + basic.get('total_put_oi', 0):,}")
                    
                    # Generate enhanced earnings analysis
                    if not options_analysis.get("error"):
                        ai_analysis = ai_playbook(ticker, quote.get("change_percent", 0), f"Earnings {time_str} | Importance: {importance}", options_analysis)
                    else:
                        ai_analysis = f"""
                        **Enhanced AI Analysis for {ticker} Earnings:**
                        - **Date:** {report["date"]}
                        - **Time:** {time_str}
                        - **Importance:** {importance}
                        - **Current Price:** ${quote.get('last', 0):.2f}
                        - **Daily Change:** {quote.get('change_percent', 0):+.2f}%
                        - **Volume:** {quote.get('volume', 0):,}
                        - **Data Source:** {quote.get('data_source', 'Yahoo Finance')}
                        - **Estimate:** {report.get("estimate", "N/A")}
                        
                        **Enhanced Trading Notes:**
                        - Monitor for post-earnings volatility and gap moves
                        - Consider straddle/strangle strategies for high IV
                        - Watch for unusual options activity leading up to announcement
                        """
                    
                    with st.expander(f"🔮 Enhanced AI Analysis for {ticker}"):
                        st.markdown(ai_analysis)
                    st.divider()

# TAB 9: Important News & Economic Calendar - Enhanced with Trading Economics
with tabs[8]:
    st.subheader("📰 Important News & Economic Calendar")
    
    # Show Trading Economics integration status
    if trading_economics_client:
        st.success("✅ Trading Economics connected for live economic calendar")
    else:
        st.info("📊 Using AI-generated events - connect Trading Economics API for live data")

    if st.button("📊 Get This Week's Events", type="primary"):
        with st.spinner("Fetching important events..."):
            important_events = get_important_events()

            if not important_events:
                st.info("No major economic events scheduled for this week.")
            else:
                st.markdown("### Major Market-Moving Events")
                
                # Sort events by impact level
                high_impact = [e for e in important_events if e.get('impact', '').lower() == 'high']
                medium_impact = [e for e in important_events if e.get('impact', '').lower() == 'medium']
                low_impact = [e for e in important_events if e.get('impact', '').lower() == 'low']
                
                # Display high impact events first
                if high_impact:
                    st.markdown("#### 🚀 High Impact Events")
                    for event in high_impact:
                        st.markdown(f"**{event['event']}**")
                        st.write(f"**Date:** {event['date']}")
                        st.write(f"**Time:** {event['time']}")
                        st.write(f"**Impact:** {event['impact']}")
                        st.divider()
                
                if medium_impact:
                    st.markdown("#### 📈 Medium Impact Events")
                    for event in medium_impact:
                        st.markdown(f"**{event['event']}**")
                        st.write(f"**Date:** {event['date']}")
                        st.write(f"**Time:** {event['time']}")
                        st.write(f"**Impact:** {event['impact']}")
                        st.divider()
                
                if low_impact:
                    with st.expander("📊 Low Impact Events"):
                        for event in low_impact:
                            st.markdown(f"**{event['event']}**")
                            st.write(f"**Date:** {event['date']}")
                            st.write(f"**Time:** {event['time']}")
                            st.write(f"**Impact:** {event['impact']}")
                            st.write("---")

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
                    # Get current quote for context with enhanced data
                    quote = get_live_quote(social_ticker, tz_label)

                    col1, col2, col3 = st.columns(3)
                    if not quote.get("error"):
                        col1.metric(f"{social_ticker} Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        
                        # Enhanced data source display
                        source = quote.get('data_source', 'Yahoo Finance')
                        if source == "Unusual Whales":
                            col3.success("🐋 UW Data")
                        else:
                            col3.info(f"📊 {source}")

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

# ===== ENHANCED FOOTER =====
st.markdown("---")
footer_sources = []
if unusual_whales_client:
    footer_sources.append("🐋 Unusual Whales")
if alpha_vantage_client:
    footer_sources.append("Alpha Vantage")
if twelvedata_client:
    footer_sources.append("Twelve Data")
footer_sources.append("Yahoo Finance")

if trading_economics_client:
    footer_sources.append("📊 Trading Economics")

footer_text = " + ".join(footer_sources)

available_ai_models = multi_ai.get_available_models()
ai_footer = f"AI: {st.session_state.ai_model}"
if st.session_state.ai_model == "Multi-AI" and available_ai_models:
    ai_footer += f" ({'+'.join(available_ai_models)})"

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🔥 AI Radar Pro v2.0 | Data: {footer_text} | {ai_footer} | 🎯 Targeting 80-100% Accuracy"
    "</div>",
    unsafe_allow_html=True
)

# Auto-refresh functionality with faster intervals
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
