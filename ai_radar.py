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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

# Performance tracking
class PerformanceTracker:
    def __init__(self):
        self.api_times = {}
        self.success_rates = {}
        
    def record_call(self, source: str, duration: float, success: bool):
        if source not in self.api_times:
            self.api_times[source] = []
            self.success_rates[source] = []
        
        self.api_times[source].append(duration)
        self.success_rates[source].append(success)
        
        # Keep only last 100 calls
        if len(self.api_times[source]) > 100:
            self.api_times[source] = self.api_times[source][-100:]
            self.success_rates[source] = self.success_rates[source][-100:]
    
    def get_best_source(self):
        best_source = None
        best_score = 0
        
        for source in self.api_times:
            if len(self.api_times[source]) < 5:
                continue
                
            avg_time = sum(self.api_times[source][-10:]) / min(10, len(self.api_times[source]))
            success_rate = sum(self.success_rates[source][-10:]) / min(10, len(self.success_rates[source]))
            
            # Score: prioritize success rate, then speed
            score = (success_rate * 100) - (avg_time * 10)
            
            if score > best_score:
                best_score = score
                best_source = source
                
        return best_source

# Global performance tracker
perf_tracker = PerformanceTracker()

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
if "data_source" not in st.session_state:
    st.session_state.data_source = "Yahoo Finance"  # Default data source
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"  # Default to multi-AI
if "quote_cache" not in st.session_state:
    st.session_state.quote_cache = {}
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = {}

# Dynamic cache TTL based on market hours and volatility
def get_cache_ttl(volatility_score: float = 1.0) -> int:
    """Dynamic TTL: shorter during market hours, longer after hours"""
    current_tz = datetime.datetime.now(ZoneInfo('US/Eastern'))
    
    if 9 <= current_tz.hour < 16:  # Market hours
        base_ttl = 15  # 15 seconds during market hours
    elif 4 <= current_tz.hour < 9 or 16 <= current_tz.hour < 20:  # Pre/post market
        base_ttl = 45  # 45 seconds during extended hours
    else:  # After hours
        base_ttl = 300  # 5 minutes after hours
    
    # Adjust for volatility (higher volatility = shorter cache)
    adjusted_ttl = max(5, int(base_ttl / volatility_score))
    return adjusted_ttl

# API Keys
try:
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROK_API_KEY = st.secrets.get("GROK_API_KEY", "")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    TWELVEDATA_KEY = st.secrets.get("TWELVEDATA_API_KEY", "")
    UNUSUAL_WHALES_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")

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

# Optimized HTTP session with connection pooling
def create_optimized_session():
    """Create HTTP session with connection pooling and retry strategy"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=2,  # Reduced from default 3
        backoff_factor=0.1,  # Faster backoff
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,  # Connection pooling
        pool_maxsize=50,
        pool_block=False
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Faster timeouts
    session.timeout = (2, 5)  # (connect, read) - much faster than 10s
    
    return session

# Global optimized session
http_session = create_optimized_session()

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

# ===== SPEED-OPTIMIZED UNUSUAL WHALES API CLIENT =====
class UnusualWhalesClient:
    """Speed-optimized Unusual Whales client with connection pooling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com"
        self.session = create_optimized_session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json, text/plain",
            "User-Agent": "AI-Radar-Pro/2.1-Speed"
        })
        
        # Optimized endpoints from CSV
        self.endpoints = {
            "quote": "/api/stock/{symbol}/quote",
            "stock_state": "/api/stock/{symbol}/stock-state", 
            "batch_quotes": "/api/stocks/quotes",  # For batch requests
            "options_flow": "/api/stock/{symbol}/options-flow",
            "social_sentiment": "/api/stock/{symbol}/social-sentiment",
            "market_overview": "/api/market/overview",
            "trending_tickers": "/api/market/trending-tickers"
        }
    
    def get_quote_fast(self, symbol: str) -> Dict:
        """Ultra-fast quote retrieval with performance tracking"""
        start_time = time.time()
        
        try:
            # Try primary quote endpoint first
            url = f"{self.base_url}{self.endpoints['quote'].format(symbol=symbol.upper())}"
            response = self._make_fast_request(url)
            
            duration = time.time() - start_time
            success = response and not response.get("error")
            perf_tracker.record_call("UW", duration, success)
            
            if response and not response.get("error"):
                return self._format_quote_response(response, symbol)
            
            # Fast fallback to stock-state endpoint
            url = f"{self.base_url}{self.endpoints['stock_state'].format(symbol=symbol.upper())}"
            response = self._make_fast_request(url)
            
            if response and not response.get("error"):
                return self._format_stock_state_response(response, symbol)
            
            return {"error": f"UW: No valid quote data for {symbol}", "data_source": "🐋 Unusual Whales"}
            
        except Exception as e:
            duration = time.time() - start_time
            perf_tracker.record_call("UW", duration, False)
            return {"error": f"UW error: {str(e)}", "data_source": "🐋 Unusual Whales"}
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch quote retrieval for multiple symbols"""
        try:
            # If batch endpoint available
            if len(symbols) > 5 and "batch_quotes" in self.endpoints:
                url = f"{self.base_url}{self.endpoints['batch_quotes']}"
                payload = {"symbols": symbols}
                response = self._make_fast_request(url, json_data=payload, method="POST")
                
                if response and not response.get("error"):
                    return self._format_batch_response(response)
            
            # Fallback to parallel individual requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {executor.submit(self.get_quote_fast, symbol): symbol for symbol in symbols}
                results = {}
                
                for future in concurrent.futures.as_completed(future_to_symbol, timeout=10):
                    symbol = future_to_symbol[future]
                    try:
                        results[symbol] = future.result()
                    except Exception as exc:
                        results[symbol] = {"error": f"Batch error: {str(exc)}", "data_source": "🐋 Unusual Whales"}
                
                return results
                
        except Exception as e:
            # Return individual errors for each symbol
            return {symbol: {"error": f"Batch failed: {str(e)}", "data_source": "🐋 Unusual Whales"} for symbol in symbols}
    
    def _make_fast_request(self, url: str, params: Dict = None, json_data: Dict = None, method: str = "GET") -> Dict:
        """Ultra-fast HTTP request with aggressive timeouts"""
        try:
            if method == "POST":
                response = self.session.post(url, params=params, json=json_data, timeout=(1, 3))
            else:
                response = self.session.get(url, params=params, timeout=(1, 3))
            
            if response.status_code == 429:
                return {"error": "Rate limited", "status_code": 429}
            
            if response.status_code != 200:
                return {"error": f"API error {response.status_code}", "status_code": response.status_code}
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
    
    def _format_batch_response(self, data: Dict) -> Dict[str, Dict]:
        """Format batch response from UW"""
        results = {}
        batch_data = data.get("data", {})
        
        for symbol, quote_data in batch_data.items():
            results[symbol] = self._format_quote_response({"data": quote_data}, symbol)
        
        return results
    
    def _format_quote_response(self, data: Dict, symbol: str) -> Dict:
        """Format quote endpoint response"""
        try:
            quote_data = data.get("data", data)
            
            last_price = self._safe_float(quote_data.get("last", quote_data.get("price", 0)))
            if not last_price or last_price <= 0:
                return {"error": f"Invalid price data for {symbol}", "data_source": "🐋 Unusual Whales"}
            
            open_price = self._safe_float(quote_data.get("open", last_price))
            high_price = self._safe_float(quote_data.get("high", last_price))
            low_price = self._safe_float(quote_data.get("low", last_price))
            volume = self._safe_int(quote_data.get("volume", 0))
            prev_close = self._safe_float(quote_data.get("previous_close", quote_data.get("prev_close", last_price)))
            
            change_dollar = last_price - prev_close if prev_close > 0 else 0
            change_percent = (change_dollar / prev_close * 100) if prev_close > 0 else 0
            
            return {
                "last": float(last_price),
                "bid": self._safe_float(quote_data.get("bid", last_price - 0.01)),
                "ask": self._safe_float(quote_data.get("ask", last_price + 0.01)),
                "volume": int(volume),
                "change": float(change_dollar),
                "change_percent": float(change_percent),
                "premarket_change": self._safe_float(quote_data.get("premarket_change", 0)),
                "intraday_change": float(change_percent),
                "postmarket_change": self._safe_float(quote_data.get("afterhours_change", 0)),
                "previous_close": float(prev_close),
                "market_open": float(open_price),
                "last_updated": datetime.datetime.now().isoformat(),
                "error": None,
                "data_source": "🐋 Unusual Whales"
            }
            
        except Exception as e:
            return {"error": f"UW quote formatting error: {str(e)}", "data_source": "🐋 Unusual Whales"}
    
    def _format_stock_state_response(self, data: Dict, symbol: str) -> Dict:
        """Format stock-state endpoint response (fallback)"""
        try:
            # Handle nested data structure if present
            if isinstance(data, dict) and 'data' in data:
                stock_data = data['data']
            else:
                stock_data = data
            
            # Extract price - UW typically has 'price' or 'last' field
            last_price = None
            for field in ['price', 'last', 'current_price', 'close']:
                if field in stock_data and stock_data[field] is not None:
                    try:
                        last_price = float(stock_data[field])
                        if last_price > 0:
                            break
                    except (ValueError, TypeError):
                        continue
            
            if not last_price or last_price <= 0:
                return {"error": f"Invalid price data for {symbol}", "data_source": "🐋 Unusual Whales"}
            
            # Extract other fields with fallbacks
            open_price = self._safe_float(stock_data.get('open', stock_data.get('open_price', last_price)))
            high_price = self._safe_float(stock_data.get('high', stock_data.get('day_high', last_price)))
            low_price = self._safe_float(stock_data.get('low', stock_data.get('day_low', last_price)))
            volume = self._safe_int(stock_data.get('volume', stock_data.get('day_volume', 0)))
            prev_close = self._safe_float(stock_data.get('previous_close', stock_data.get('prev_close', last_price)))
            
            # Calculate changes
            change_dollar = last_price - prev_close if prev_close > 0 else 0
            change_percent = (change_dollar / prev_close * 100) if prev_close > 0 else 0
            
            # Session changes (simplified - UW may have more detailed session data)
            premarket_change = self._safe_float(stock_data.get('premarket_change', 0))
            postmarket_change = self._safe_float(stock_data.get('afterhours_change', 0))
            
            return {
                "last": float(last_price),
                "bid": float(last_price - 0.01),  # UW may have actual bid/ask
                "ask": float(last_price + 0.01),  # UW may have actual bid/ask
                "volume": int(volume),
                "change": float(change_dollar),
                "change_percent": float(change_percent),
                "premarket_change": float(premarket_change),
                "intraday_change": float(change_percent),
                "postmarket_change": float(postmarket_change),
                "previous_close": float(prev_close),
                "market_open": float(open_price),
                "last_updated": datetime.datetime.now().isoformat(),
                "error": None,
                "data_source": "🐋 Unusual Whales"
            }
            
        except Exception as e:
            return {"error": f"UW stock-state formatting error: {str(e)}", "data_source": "🐋 Unusual Whales"}
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0

# Speed-optimized Alpha Vantage Client
class AlphaVantageClient:
    """Speed-optimized Alpha Vantage client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = create_optimized_session()
        
    def get_quote_fast(self, symbol: str) -> Dict:
        start_time = time.time()
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=(1, 4))
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if "Global Quote" in data:
                    quote_data = data["Global Quote"]
                    
                    price = float(quote_data.get("05. price", 0))
                    change = float(quote_data.get("09. change", 0))
                    change_percent = float(quote_data.get("10. change percent", "0%").replace("%", ""))
                    volume = int(quote_data.get("06. volume", 0))
                    
                    perf_tracker.record_call("Alpha Vantage", duration, True)
                    
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
                    perf_tracker.record_call("Alpha Vantage", duration, False)
                    return {"error": f"No data found for {symbol}", "data_source": "Alpha Vantage"}
            else:
                perf_tracker.record_call("Alpha Vantage", duration, False)
                return {"error": f"API error: {response.status_code}", "data_source": "Alpha Vantage"}
                
        except Exception as e:
            duration = time.time() - start_time
            perf_tracker.record_call("Alpha Vantage", duration, False)
            return {"error": f"Alpha Vantage error: {str(e)}", "data_source": "Alpha Vantage"}

# Speed-optimized Twelve Data Client
class TwelveDataClient:
    """Speed-optimized Twelve Data client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = create_optimized_session()
        
    def get_quote_fast(self, symbol: str) -> Dict:
        start_time = time.time()
        
        try:
            params = {
                "symbol": symbol,
                "interval": "1min",
                "outputsize": "1",
                "apikey": self.api_key
            }
            
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=(1, 4))
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if "status" in data and data["status"] == "error":
                    perf_tracker.record_call("Twelve Data", duration, False)
                    return {"error": f"Twelve Data API Error: {data.get('message', 'Unknown error')}", "data_source": "Twelve Data"}
                
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
                        perf_tracker.record_call("Twelve Data", duration, True)
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
                            "error": None
                        }
            
            perf_tracker.record_call("Twelve Data", duration, False)
            return {"error": f"API error: {response.status_code}", "data_source": "Twelve Data"}
                
        except Exception as e:
            duration = time.time() - start_time
            perf_tracker.record_call("Twelve Data", duration, False)
            return {"error": f"Twelve Data error: {str(e)}", "data_source": "Twelve Data"}

# Initialize speed-optimized data clients
unusual_whales_client = UnusualWhalesClient(UNUSUAL_WHALES_KEY) if UNUSUAL_WHALES_KEY else None
alpha_vantage_client = AlphaVantageClient(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else None
twelvedata_client = TwelveDataClient(TWELVEDATA_KEY) if TWELVEDATA_KEY else None

# ===== SPEED-OPTIMIZED QUOTE FETCHING =====
def is_cache_valid(ticker: str, cache_entry: Dict) -> bool:
    """Check if cache entry is still valid based on dynamic TTL"""
    if not cache_entry or "timestamp" not in cache_entry:
        return False
    
    cache_time = cache_entry["timestamp"]
    current_time = time.time()
    
    # Calculate volatility score from price change
    volatility_score = 1.0
    if "change_percent" in cache_entry:
        volatility_score = max(1.0, abs(cache_entry["change_percent"]) / 2.0)
    
    ttl = get_cache_ttl(volatility_score)
    
    return (current_time - cache_time) < ttl

@st.cache_data(ttl=None)  # We handle caching manually for dynamic TTL
def get_live_quote_cached(ticker: str, tz: str = "ET", force_refresh: bool = False) -> Dict:
    """Smart cached quote fetching with dynamic TTL"""
    
    # Check cache first (unless force refresh)
    cache_key = f"{ticker}_{tz}"
    if not force_refresh and cache_key in st.session_state.quote_cache:
        cached_quote = st.session_state.quote_cache[cache_key]
        if is_cache_valid(ticker, cached_quote):
            return cached_quote
    
    # Fetch new quote
    quote = get_live_quote_fast(ticker, tz)
    
    # Update cache with timestamp
    if not quote.get("error"):
        quote["timestamp"] = time.time()
        st.session_state.quote_cache[cache_key] = quote
    
    return quote

def get_live_quote_fast(ticker: str, tz: str = "ET") -> Dict:
    """
    Ultra-fast live quote fetching with intelligent source selection
    """
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # Smart source selection based on performance
    best_source = perf_tracker.get_best_source()
    sources_to_try = []
    
    # Reorder sources based on performance
    if best_source == "UW" and unusual_whales_client:
        sources_to_try = ["UW", "Twelve Data", "Alpha Vantage", "Yahoo Finance"]
    elif best_source == "Twelve Data" and twelvedata_client:
        sources_to_try = ["Twelve Data", "UW", "Alpha Vantage", "Yahoo Finance"]
    elif best_source == "Alpha Vantage" and alpha_vantage_client:
        sources_to_try = ["Alpha Vantage", "UW", "Twelve Data", "Yahoo Finance"]
    else:
        # Default order
        sources_to_try = ["UW", "Twelve Data", "Alpha Vantage", "Yahoo Finance"]
    
    # Try sources in order with fast failover
    for source in sources_to_try:
        try:
            if source == "UW" and unusual_whales_client:
                result = unusual_whales_client.get_quote_fast(ticker)
                if not result.get("error") and result.get("last", 0) > 0:
                    result["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                    return result
                    
            elif source == "Twelve Data" and twelvedata_client:
                result = twelvedata_client.get_quote_fast(ticker)
                if not result.get("error") and result.get("last", 0) > 0:
                    result["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                    return result
                    
            elif source == "Alpha Vantage" and alpha_vantage_client:
                result = alpha_vantage_client.get_quote_fast(ticker)
                if not result.get("error") and result.get("last", 0) > 0:
                    result["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                    return result
                    
            elif source == "Yahoo Finance":
                # Fast Yahoo Finance fallback
                return get_yfinance_quote_fast(ticker, tz_zone, tz_label)
                
        except Exception as e:
            continue  # Quick failover to next source
    
    # If all sources failed
    return {
        "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
        "change": 0.0, "change_percent": 0.0,
        "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
        "previous_close": 0.0, "market_open": 0.0,
        "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
        "error": "All data sources failed",
        "data_source": "None Available"
    }

def get_yfinance_quote_fast(ticker: str, tz_zone, tz_label: str) -> Dict:
    """Fast Yahoo Finance quote with minimal processing"""
    start_time = time.time()
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get only essential data quickly
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
        if current_price == 0:
            hist = stock.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
        
        previous_close = float(info.get('previousClose', current_price))
        volume = int(info.get('volume', 0))
        
        # Quick calculations
        change_dollar = current_price - previous_close if previous_close > 0 else 0
        change_percent = (change_dollar / previous_close * 100) if previous_close > 0 else 0
        
        duration = time.time() - start_time
        perf_tracker.record_call("Yahoo Finance", duration, True)
        
        return {
            "last": float(current_price),
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": volume,
            "change": float(change_dollar),
            "change_percent": float(change_percent),
            "premarket_change": 0.0,  # Simplified for speed
            "intraday_change": float(change_percent),
            "postmarket_change": 0.0,  # Simplified for speed
            "previous_close": float(previous_close),
            "market_open": float(info.get('regularMarketOpen', previous_close)),
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None,
            "data_source": "Yahoo Finance"
        }
        
    except Exception as e:
        duration = time.time() - start_time
        perf_tracker.record_call("Yahoo Finance", duration, False)
        return {"error": f"Yahoo Finance error: {str(e)}", "data_source": "Yahoo Finance"}

# ===== PARALLEL QUOTE FETCHING =====
def get_multiple_quotes_parallel(tickers: List[str], tz: str = "ET", max_workers: int = 10) -> Dict[str, Dict]:
    """Fetch multiple quotes in parallel for maximum speed"""
    
    # Check if we can use batch API
    if unusual_whales_client and len(tickers) > 5:
        try:
            batch_results = unusual_whales_client.get_batch_quotes(tickers)
            if batch_results and not all(result.get("error") for result in batch_results.values()):
                # Add timezone info
                tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
                tz_label = "ET" if tz == "ET" else "CT"
                timestamp = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                
                for ticker, result in batch_results.items():
                    if not result.get("error"):
                        result["last_updated"] = timestamp
                
                return batch_results
        except Exception:
            pass  # Fall back to parallel individual requests
    
    # Parallel individual requests
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all quote requests
        future_to_ticker = {
            executor.submit(get_live_quote_cached, ticker, tz): ticker 
            for ticker in tickers
        }
        
        # Collect results as they complete (faster than waiting for all)
        for future in concurrent.futures.as_completed(future_to_ticker, timeout=8):  # Reduced timeout
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                results[ticker] = {
                    "error": f"Parallel fetch error: {str(exc)}", 
                    "data_source": "Error",
                    "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
                    "change": 0.0, "change_percent": 0.0,
                    "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
                    "previous_close": 0.0, "market_open": 0.0,
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
    
    return results

# ===== SELECTIVE ANALYSIS LOADING =====
def should_trigger_ai_analysis(change_percent: float, volume: int = 0) -> bool:
    """Determine if AI analysis should be triggered based on significance"""
    # Only trigger AI for significant moves to save processing time
    significant_move = abs(change_percent) >= 2.0
    high_volume = volume > 1000000
    
    return significant_move or high_volume

# Lightweight analysis functions for speed
@st.cache_data(ttl=300)  # 5 minute cache for analysis
def get_lightweight_technical_analysis(ticker: str) -> Dict:
    """Lightweight technical analysis for speed"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1h")  # Reduced data for speed
        
        if hist.empty:
            return {"error": "No data available"}
        
        # Calculate only essential indicators
        current_price = hist['Close'].iloc[-1]
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
        
        # Simple RSI calculation
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(hist) >= 14 else 50
        
        # Simple trend
        if current_price > sma_20:
            trend = "Bullish"
        else:
            trend = "Bearish"
        
        return {
            "rsi": rsi,
            "sma_20": sma_20,
            "current_price": current_price,
            "trend_analysis": trend,
            "support": hist['Low'].rolling(10).min().iloc[-1],
            "resistance": hist['High'].rolling(10).max().iloc[-1]
        }
        
    except Exception as e:
        return {"error": f"Technical analysis error: {str(e)}"}

def get_lightweight_fundamental_analysis(ticker: str) -> Dict:
    """Lightweight fundamental analysis for speed"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get only essential fundamentals
        fundamentals = {
            "pe_ratio": info.get('trailingPE', None),
            "market_cap": info.get('marketCap', 0),
            "sector": info.get('sector', 'Unknown'),
            "beta": info.get('beta', None),
            "revenue_growth": info.get('revenueGrowth', None)
        }
        
        # Quick health assessment
        pe = fundamentals.get('pe_ratio')
        if pe and 10 <= pe <= 25:
            fundamentals['valuation_assessment'] = "Fair"
        elif pe and pe < 15:
            fundamentals['valuation_assessment'] = "Undervalued"
        else:
            fundamentals['valuation_assessment'] = "Overvalued"
        
        fundamentals['financial_health'] = "Good"  # Simplified for speed
        
        return fundamentals
        
    except Exception as e:
        return {"error": f"Fundamental analysis error: {str(e)}"}

# ===== NEWS AND DATA FUNCTIONS (OPTIMIZED) =====
@st.cache_data(ttl=600)  # 10 minute cache for news
def get_finnhub_news_fast(symbol: str = None) -> List[Dict]:
    if not FINNHUB_KEY:
        return []
    
    try:
        if symbol:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today()}&to={datetime.date.today()}&token={FINNHUB_KEY}"
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
        
        response = http_session.get(url, timeout=(2, 5))
        if response.status_code == 200:
            return response.json()[:5]  # Limit to 5 for speed
    except Exception as e:
        pass  # Fail silently for speed
    
    return []

@st.cache_data(ttl=600)
def get_polygon_news_fast() -> List[Dict]:
    if not POLYGON_KEY:
        return []
    
    try:
        today = datetime.date.today().strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/reference/news?published_utc.gte={today}&limit=10&apikey={POLYGON_KEY}"
        
        response = http_session.get(url, timeout=(2, 5))
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])[:5]  # Limit to 5 for speed
    except Exception as e:
        pass  # Fail silently for speed
    
    return []

# ===== OPTIONS DATA (LIGHTWEIGHT) =====
@st.cache_data(ttl=300)
def get_options_data_fast(ticker: str) -> Optional[Dict]:
    """Fast options data retrieval"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": f"No options data available for {ticker}"}

        # Get nearest expiration only for speed
        today = datetime.date.today()
        exp_dates = [datetime.datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
        valid_exps = [exp for exp in exp_dates if exp >= today]
        if not valid_exps:
            return {"error": "No valid expirations"}

        target_exp = min(valid_exps, key=lambda x: (x - today).days)
        exp_str = target_exp.strftime('%Y-%m-%d')

        option_chain = stock.option_chain(exp_str)
        calls = option_chain.calls
        puts = option_chain.puts

        # Get essential metrics only
        total_call_volume = calls['volume'].sum() if not calls.empty else 0
        total_put_volume = puts['volume'].sum() if not puts.empty else 0
        
        return {
            "iv": calls['impliedVolatility'].mean() * 100 if not calls.empty else 0,
            "put_call_ratio": total_put_volume / total_call_volume if total_call_volume > 0 else 0,
            "total_calls": total_call_volume,
            "total_puts": total_put_volume,
            "expiration": exp_str
        }
        
    except Exception as e:
        return {"error": f"Options error: {str(e)}"}

# ===== AI ANALYSIS (OPTIMIZED) =====
class MultiAIAnalyzer:
    """Speed-optimized multi-AI analysis system"""
    
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
    
    def analyze_with_openai_fast(self, prompt: str) -> str:
        """Fast OpenAI analysis with reduced tokens"""
        if not self.openai_client:
            return "OpenAI not available"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Reduced for faster generation
                max_tokens=200    # Reduced for speed
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def get_fast_analysis(self, ticker: str, quote: Dict, significant_move: bool = False) -> str:
        """Get fast analysis based on significance"""
        
        if not significant_move:
            # Quick summary for normal moves
            return f"""**{ticker} Quick Analysis:**
- Price: ${quote['last']:.2f} ({quote['change_percent']:+.2f}%)
- Volume: {quote['volume']:,}
- Status: Normal trading activity
- Data Source: {quote.get('data_source', 'Unknown')}
"""
        
        # Full analysis for significant moves
        prompt = f"""
Quick trading analysis for {ticker}:
- Current: ${quote['last']:.2f} ({quote['change_percent']:+.2f}%)
- Volume: {quote['volume']:,}
- Source: {quote.get('data_source', 'Unknown')}

Provide brief (100 words max):
1. Sentiment (Bullish/Bearish/Neutral)
2. Key levels to watch
3. Risk assessment
"""
        
        if self.openai_client:
            return self.analyze_with_openai_fast(prompt)
        else:
            return f"**{ticker} Significant Move Detected** - {quote['change_percent']:+.2f}% move with {quote['volume']:,} volume. Monitor for continuation or reversal."

# Initialize speed-optimized Multi-AI Analyzer
multi_ai = MultiAIAnalyzer()

# ===== MAIN APP WITH SPEED OPTIMIZATIONS =====
st.title("🚀 AI Radar Pro – Ultra-Fast Live Trading")

# Performance metrics in sidebar
if st.sidebar.checkbox("⚡ Performance Monitor"):
    if perf_tracker.api_times:
        st.sidebar.subheader("API Performance")
        for source, times in perf_tracker.api_times.items():
            if times:
                avg_time = sum(times[-10:]) / min(10, len(times))
                success_rate = sum(perf_tracker.success_rates[source][-10:]) / min(10, len(perf_tracker.success_rates[source]))
                st.sidebar.metric(source, f"{avg_time:.2f}s", f"{success_rate:.1%} success")

# Timezone toggle
col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed", help="Select Timezone (ET/CT)")

# Get current time in selected TZ
tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
tz_label = st.session_state.selected_tz

# Speed-optimized settings
st.sidebar.subheader("⚡ Speed Settings")
max_workers = st.sidebar.slider("Parallel Workers", 5, 20, 10, help="More workers = faster but more resources")
enable_ai = st.sidebar.checkbox("🤖 Enable AI Analysis", value=True, help="Disable for maximum speed")
auto_refresh_speed = st.sidebar.selectbox("Auto Refresh", [5, 10, 15, 30], index=1, help="Seconds between refreshes")

# Enhanced AI Settings
st.sidebar.subheader("🤖 AI Configuration")
available_models = ["Multi-AI"] + multi_ai.get_available_models()
st.session_state.ai_model = st.sidebar.selectbox("AI Model", available_models, 
                                                  index=available_models.index(st.session_state.ai_model) if st.session_state.ai_model in available_models else 0)

# Data source status
st.sidebar.subheader("Data Sources")
if unusual_whales_client:
    st.sidebar.success("✅ Enhanced UW (Primary)")
else:
    st.sidebar.warning("⚠️ UW Not Connected")

if twelvedata_client:
    st.sidebar.success("✅ Twelve Data")
else:
    st.sidebar.warning("⚠️ Twelve Data Not Connected")

if alpha_vantage_client:
    st.sidebar.success("✅ Alpha Vantage")
else:
    st.sidebar.warning("⚠️ Alpha Vantage Not Connected")

st.sidebar.success("✅ Yahoo Finance (Fallback)")

# Initialize session state for speed optimizations
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"Default": ["AAPL", "NVDA", "TSLA", "SPY", "AMD", "MSFT"]}
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"

# Auto-refresh controls with speed optimizations
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    auto_refresh = st.checkbox("🔄 Ultra Fast Refresh", value=False)

with col2:
    if st.button("⚡ Quick Refresh"):
        # Force refresh cache
        st.session_state.quote_cache = {}
        st.cache_data.clear()
        st.rerun()

with col3:
    if st.button("🧹 Clear Cache"):
        st.session_state.quote_cache = {}
        st.cache_data.clear()
        st.success("Cache cleared!")

with col4:
    current_time = current_tz.strftime("%I:%M:%S %p")
    market_open = 9 <= current_tz.hour < 16
    status = "🟢 LIVE" if market_open else "🔴 CLOSED"
    st.write(f"**{status}** | {current_time} {tz_label}")

# Create speed-optimized tabs
tabs = st.tabs(["⚡ Live Quotes", "📋 Watchlist", "🔥 Catalysts", "📈 Analysis", "🤖 AI Plays"])

# TAB 1: Ultra-Fast Live Quotes
with tabs[0]:
    st.subheader("⚡ Ultra-Fast Live Quotes")
    
    # Trading session status
    current_tz_hour = current_tz.hour
    if 4 <= current_tz_hour < 9:
        session_status = "🌅 Premarket"
    elif 9 <= current_tz_hour < 16:
        session_status = "🟢 Market Open"
    else:
        session_status = "🌆 After Hours"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Session:** {session_status} | **Speed Mode:** {'AI Enabled' if enable_ai else 'Maximum Speed'}")
    with col2:
        if st.button("🚀 Refresh All Quotes", type="primary"):
            with st.spinner("Ultra-fast refresh..."):
                # Clear cache for fresh data
                st.session_state.quote_cache = {}
                st.rerun()
    
    # Search bar for any ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Quick Search", placeholder="Enter ticker (e.g., AAPL)", key="search_quotes").upper().strip()
    with col2:
        search_quotes = st.button("Get Quote", key="search_quotes_btn")
    
    # Handle search
    if search_quotes and search_ticker:
        with st.spinner(f"Ultra-fast quote for {search_ticker}..."):
            quote = get_live_quote_cached(search_ticker, tz_label, force_refresh=True)
            if not quote["error"]:
                st.success(f"⚡ {search_ticker} - {quote.get('data_source', 'Unknown')} - {quote['last_updated']}")
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                col1.metric(search_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                
                # AI analysis for significant moves
                if enable_ai and should_trigger_ai_analysis(quote['change_percent'], quote['volume']):
                    with col4:
                        if st.button("🤖 AI Analysis", key=f"ai_{search_ticker}"):
                            analysis = multi_ai.get_fast_analysis(search_ticker, quote, significant_move=True)
                            st.success("AI Analysis")
                            st.markdown(analysis)
                else:
                    col4.write("Normal activity")
                
                # Quick add to watchlist
                if st.button(f"➕ Add {search_ticker}", key="add_searched"):
                    current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                    if search_ticker not in current_list:
                        current_list.append(search_ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                        st.success(f"Added {search_ticker}!")
                        st.rerun()
                
                st.divider()
            else:
                st.error(f"Quote error: {quote['error']}")
    
    # Ultra-fast watchlist display
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    if not tickers:
        st.warning("Add stocks to your watchlist for ultra-fast monitoring!")
    else:
        st.markdown(f"### ⚡ Watchlist ({len(tickers)} stocks)")
        
        # Parallel quote fetching for entire watchlist
        with st.spinner("Fetching quotes in parallel..."):
            start_time = time.time()
            
            # Get all quotes in parallel
            all_quotes = get_multiple_quotes_parallel(tickers, tz_label, max_workers=max_workers)
            
            fetch_time = time.time() - start_time
            st.caption(f"⚡ Fetched {len(tickers)} quotes in {fetch_time:.2f}s")
        
        # Display quotes in optimized format
        for ticker in tickers:
            quote = all_quotes.get(ticker, {"error": "Not fetched"})
            
            if quote.get("error"):
                st.error(f"{ticker}: {quote['error']}")
                continue
            
            # Compact display for speed
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
            
            # Color coding for performance
            color = "green" if quote['change_percent'] > 0 else "red" if quote['change_percent'] < 0 else "gray"
            
            col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
            col2.write(f"Vol: {quote['volume']:,}")
            col3.write(f"Src: {quote.get('data_source', 'Unknown')[:8]}")
            col4.write(f"Updated: {quote['last_updated'][-8:]}")  # Just time
            
            # Quick actions
            with col5:
                significant_move = should_trigger_ai_analysis(quote['change_percent'], quote['volume'])
                
                if enable_ai and significant_move:
                    if st.button("🤖", key=f"ai_quick_{ticker}", help="AI Analysis"):
                        analysis = multi_ai.get_fast_analysis(ticker, quote, significant_move=True)
                        st.info(f"🤖 {ticker} Analysis")
                        st.write(analysis)
                else:
                    st.write("📊" if abs(quote['change_percent']) > 1 else "⚪")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(auto_refresh_speed)
            st.rerun()

# TAB 2: Speed-Optimized Watchlist Manager
with tabs[1]:
    st.subheader("📋 Ultra-Fast Watchlist Manager")
    
    # Quick add section
    col1, col2 = st.columns([3, 1])
    with col1:
        quick_add = st.text_input("⚡ Quick Add", placeholder="Ticker", key="quick_add").upper().strip()
    with col2:
        if st.button("Add", key="quick_add_btn") and quick_add:
            current_list = st.session_state.watchlists[st.session_state.active_watchlist]
            if quick_add not in current_list:
                current_list.append(quick_add)
                st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                st.success(f"✅ Added {quick_add}")
                st.rerun()
            else:
                st.warning(f"{quick_add} already in watchlist")
    
    # Watchlist management
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_watchlist = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
        st.session_state.active_watchlist = selected_watchlist
    
    with col2:
        new_watchlist = st.text_input("New List")
        if st.button("Create") and new_watchlist:
            st.session_state.watchlists[new_watchlist] = []
            st.session_state.active_watchlist = new_watchlist
            st.rerun()
    
    # Popular tickers - quick add
    st.markdown("### ⭐ Popular (Click to Add)")
    cols = st.columns(8)  # More columns for compact display
    for i, ticker in enumerate(CORE_TICKERS[:24]):  # Show more for quick access
        with cols[i % 8]:
            if st.button(f"{ticker}", key=f"pop_{ticker}", help=f"Add {ticker}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if ticker not in current_list:
                    current_list.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {ticker}")
                    st.rerun()
    
    # Current watchlist with quick removal
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    if current_tickers:
        st.markdown(f"### 📊 Current Watchlist ({len(current_tickers)})")
        
        # Display in compact grid
        cols = st.columns(6)
        for i, ticker in enumerate(current_tickers):
            with cols[i % 6]:
                col_inner1, col_inner2 = st.columns([2, 1])
                with col_inner1:
                    st.write(f"**{ticker}**")
                with col_inner2:
                    if st.button("❌", key=f"rem_{ticker}", help=f"Remove {ticker}"):
                        current_tickers.remove(ticker)
                        st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                        st.rerun()

# TAB 3: Fast Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Fast Catalyst Scanner")
    
    # Quick catalyst search
    col1, col2 = st.columns([3, 1])
    with col1:
        catalyst_ticker = st.text_input("🔍 Quick Catalyst Search", placeholder="Ticker", key="catalyst_search").upper().strip()
    with col2:
        search_catalyst = st.button("Scan", key="catalyst_btn")
    
    if search_catalyst and catalyst_ticker:
        with st.spinner(f"Fast scanning {catalyst_ticker}..."):
            # Get quote and basic news
            quote = get_live_quote_cached(catalyst_ticker, tz_label)
            news = get_finnhub_news_fast(catalyst_ticker)
            
            if not quote.get("error"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Source", quote.get('data_source', 'Unknown'))
                
                if news:
                    st.markdown("#### 📰 Recent Catalysts")
                    for item in news[:3]:  # Limit for speed
                        st.write(f"• **{item.get('headline', 'No title')}**")
                        if item.get('url'):
                            st.markdown(f"  [Read more]({item['url']})")
                else:
                    st.info("No recent catalysts found")
                
                # AI analysis for significant moves
                if enable_ai and should_trigger_ai_analysis(quote['change_percent'], quote['volume']):
                    st.markdown("#### 🤖 AI Catalyst Analysis")
                    catalyst_context = news[0].get('headline', '') if news else "Price movement"
                    analysis = multi_ai.get_fast_analysis(catalyst_ticker, quote, significant_move=True)
                    st.markdown(analysis)
            else:
                st.error(f"Error: {quote['error']}")
    
    # Market movers as potential catalysts
    if st.button("🚀 Scan Market Movers", type="primary"):
        with st.spinner("Scanning for market-moving catalysts..."):
            # Get quotes for core tickers in parallel
            mover_quotes = get_multiple_quotes_parallel(CORE_TICKERS[:20], tz_label, max_workers=max_workers)
            
            # Find significant movers
            movers = []
            for ticker, quote in mover_quotes.items():
                if not quote.get("error") and abs(quote.get('change_percent', 0)) >= 2.0:
                    movers.append((ticker, quote))
            
            # Sort by absolute change
            movers.sort(key=lambda x: abs(x[1]['change_percent']), reverse=True)
            
            if movers:
                st.markdown("### 🔥 Significant Market Movers")
                for ticker, quote in movers[:10]:  # Top 10
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    direction = "🚀" if quote['change_percent'] > 0 else "📉"
                    col1.metric(f"{direction} {ticker}", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.write(f"Vol: {quote['volume']:,}")
                    col3.write(f"Src: {quote.get('data_source', 'Unknown')[:10]}")
                    
                    with col4:
                        if st.button(f"Add {ticker}", key=f"add_mover_{ticker}"):
                            current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                            if ticker not in current_list:
                                current_list.append(ticker)
                                st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                st.success(f"Added {ticker}!")
                                st.rerun()
            else:
                st.info("No significant market movers detected")

# TAB 4: Fast Analysis
with tabs[3]:
    st.subheader("📈 Fast Analysis")
    
    # Quick analysis search
    col1, col2 = st.columns([3, 1])
    with col1:
        analysis_ticker = st.text_input("🔍 Quick Analysis", placeholder="Ticker", key="analysis_search").upper().strip()
    with col2:
        run_analysis = st.button("Analyze", key="analysis_btn")
    
    if run_analysis and analysis_ticker:
        with st.spinner(f"Fast analysis for {analysis_ticker}..."):
            # Get data in parallel
            quote = get_live_quote_cached(analysis_ticker, tz_label)
            
            if not quote.get("error"):
                # Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                col4.metric("Source", quote.get('data_source', 'Unknown'))
                
                # Lightweight analysis only if enabled
                if enable_ai:
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### 📊 Quick Technical")
                        tech = get_lightweight_technical_analysis(analysis_ticker)
                        if not tech.get("error"):
                            st.write(f"**RSI:** {tech.get('rsi', 0):.1f}")
                            st.write(f"**Trend:** {tech.get('trend_analysis', 'Unknown')}")
                            st.write(f"**Support:** ${tech.get('support', 0):.2f}")
                            st.write(f"**Resistance:** ${tech.get('resistance', 0):.2f}")
                        else:
                            st.write("Technical data unavailable")
                    
                    with analysis_col2:
                        st.markdown("#### 📈 Quick Fundamental")
                        fund = get_lightweight_fundamental_analysis(analysis_ticker)
                        if not fund.get("error"):
                            st.write(f"**P/E:** {fund.get('pe_ratio', 'N/A')}")
                            st.write(f"**Sector:** {fund.get('sector', 'Unknown')}")
                            st.write(f"**Valuation:** {fund.get('valuation_assessment', 'Unknown')}")
                            st.write(f"**Market Cap:** ${fund.get('market_cap', 0):,.0f}")
                        else:
                            st.write("Fundamental data unavailable")
                    
                    # AI analysis for significant moves
                    if should_trigger_ai_analysis(quote['change_percent'], quote['volume']):
                        st.markdown("#### 🤖 AI Analysis")
                        analysis = multi_ai.get_fast_analysis(analysis_ticker, quote, significant_move=True)
                        st.markdown(analysis)
                    else:
                        st.info("Normal trading activity - no AI analysis triggered")
                else:
                    st.info("AI analysis disabled for maximum speed")
            else:
                st.error(f"Analysis error: {quote['error']}")

# TAB 5: Fast AI Playbooks (WITH HORIZONTAL LAYOUT)
with tabs[4]:
    st.subheader("🤖 Ultra-Fast AI Playbooks")
    
    if not enable_ai:
        st.warning("⚡ AI analysis is disabled for maximum speed. Enable in sidebar to use AI features.")
    else:
        # Quick AI analysis
        col1, col2 = st.columns([3, 1])
        with col1:
            ai_ticker = st.text_input("🤖 Quick AI Analysis", placeholder="Ticker", key="ai_search").upper().strip()
        with col2:
            run_ai = st.button("AI Analyze", key="ai_btn")
        
        if run_ai and ai_ticker:
            with st.spinner(f"AI analyzing {ai_ticker}..."):
                quote = get_live_quote_cached(ai_ticker, tz_label)
                
                if not quote.get("error"):
                    # Show quote info
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                    col2.metric("Volume", f"{quote['volume']:,}")
                    col3.metric("Source", quote.get('data_source', 'Unknown'))
                    
                    # AI analysis
                    significant_move = should_trigger_ai_analysis(quote['change_percent'], quote['volume'])
                    analysis = multi_ai.get_fast_analysis(ai_ticker, quote, significant_move)
                    
                    st.markdown("### 🤖 AI Analysis")
                    st.markdown(analysis)
                    
                    # Quick add to watchlist
                    if st.button(f"Add {ai_ticker} to Watchlist", key="ai_add_ticker"):
                        current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                        if ai_ticker not in current_list:
                            current_list.append(ai_ticker)
                            st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                            st.success(f"Added {ai_ticker}!")
                            st.rerun()
                else:
                    st.error(f"AI analysis error: {quote['error']}")
        
        # Auto AI playbook generation
        if st.button("🚀 Generate Fast AI Plays", type="primary"):
            with st.spinner("AI scanning for trading opportunities..."):
                # Get quotes for watchlist + some core tickers
                scan_tickers = list(set(st.session_state.watchlists[st.session_state.active_watchlist] + CORE_TICKERS[:15]))
                
                quotes = get_multiple_quotes_parallel(scan_tickers, tz_label, max_workers=max_workers)
                
                # Find significant moves for AI analysis
                ai_candidates = []
                for ticker, quote in quotes.items():
                    if not quote.get("error") and should_trigger_ai_analysis(quote['change_percent'], quote['volume']):
                        ai_candidates.append((ticker, quote))
                
                # Sort by significance
                ai_candidates.sort(key=lambda x: abs(x[1]['change_percent']) * (x[1]['volume'] / 1000000), reverse=True)
                
                if ai_candidates:
                    st.success(f"🤖 Found {len(ai_candidates)} AI trading opportunities")
                    
                    for ticker, quote in ai_candidates[:5]:  # Top 5 for speed
                        with st.expander(f"🎯 {ticker} - ${quote['last']:.2f} ({quote['change_percent']:+.2f}%) | {quote.get('data_source', 'Unknown')}"):
                            
                            # Session data horizontally (FIXED LAYOUT)
                            sess_col1, sess_col2, sess_col3 = st.columns(3)
                            sess_col1.metric("Current Move", f"{quote['change_percent']:+.2f}%")
                            sess_col2.metric("Volume", f"{quote['volume']:,}")
                            sess_col3.metric("Significance", "HIGH" if abs(quote['change_percent']) > 5 else "MEDIUM")
                            
                            # Get lightweight analysis for summaries
                            tech = get_lightweight_technical_analysis(ticker)
                            fund = get_lightweight_fundamental_analysis(ticker)
                            options = get_options_data_fast(ticker)
                            
                            # 🔥 HORIZONTAL LAYOUT FIX APPLIED HERE 🔥
                            # Display summaries horizontally
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            
                            with summary_col1:
                                st.write("**Technical:**")
                                if not tech.get("error"):
                                    st.write(f"RSI: {tech.get('rsi', 0):.1f}, Trend: {tech.get('trend_analysis', 'Unknown')}")
                                else:
                                    st.write("Data unavailable")
                            
                            with summary_col2:
                                st.write("**Fundamental:**")
                                if not fund.get("error"):
                                    st.write(f"P/E: {fund.get('pe_ratio', 'N/A')}, Sector: {fund.get('sector', 'Unknown')}")
                                else:
                                    st.write("Data unavailable")
                            
                            with summary_col3:
                                st.write("**Options:**")
                                if not options.get("error"):
                                    st.write(f"IV: {options.get('iv', 0):.1f}%, P/C: {options.get('put_call_ratio', 0):.2f}")
                                else:
                                    st.write("Data unavailable")
                            
                            # AI analysis
                            st.markdown("**🤖 AI Trading Analysis:**")
                            analysis = multi_ai.get_fast_analysis(ticker, quote, significant_move=True)
                            st.markdown(analysis)
                            
                            # Quick action
                            if st.button(f"Add {ticker} to Watchlist", key=f"ai_play_add_{ticker}"):
                                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                                if ticker not in current_list:
                                    current_list.append(ticker)
                                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                                    st.success(f"Added {ticker}!")
                                    st.rerun()
                else:
                    st.info("⚡ No significant moves detected. Market in normal trading range.")

# Auto-refresh for live data
if auto_refresh:
    time.sleep(auto_refresh_speed)
    st.rerun()

# Footer with performance info
st.markdown("---")
footer_sources = []
if unusual_whales_client:
    footer_sources.append("🐋 Enhanced UW")
if twelvedata_client:
    footer_sources.append("Twelve Data")
if alpha_vantage_client:
    footer_sources.append("Alpha Vantage")
footer_sources.append("Yahoo Finance")

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🚀 AI Radar Pro - Speed Optimized | Sources: {' → '.join(footer_sources)} | Workers: {max_workers} | AI: {'Enabled' if enable_ai else 'Disabled'}"
    "</div>",
    unsafe_allow_html=True
)
