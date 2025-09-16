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
import google.generativeai as genai
import openai
import concurrent.futures
from dataclasses import dataclass
import logging

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

# ETF list for sector tracking
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
    st.session_state.refresh_interval = 10
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Unusual Whales"
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"

# API Keys - Enhanced error handling
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

    # Initialize AI clients with better error handling
    openai_client = None
    gemini_model = None
    grok_client = None
    
    if OPENAI_KEY:
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_KEY)
        except Exception as e:
            st.error(f"OpenAI initialization error: {e}")
    
    if GEMINI_KEY:
        try:
            genai.configure(api_key=GEMINI_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception as e:
            st.error(f"Gemini initialization error: {e}")
    
    if GROK_API_KEY:
        try:
            grok_client = openai.OpenAI(
                api_key=GROK_API_KEY,
                base_url="https://api.x.ai/v1"
            )
        except Exception as e:
            st.error(f"Grok initialization error: {e}")

except Exception as e:
    st.error(f"Error loading API keys: {e}")
    openai_client = None
    gemini_model = None
    grok_client = None

@dataclass
class QuoteData:
    """Standardized quote data structure"""
    symbol: str
    last: float
    bid: float
    ask: float
    volume: int
    change: float
    change_percent: float
    premarket_change: float
    intraday_change: float
    postmarket_change: float
    previous_close: float
    market_open: float
    last_updated: str
    data_source: str
    error: Optional[str] = None
    raw_data: Optional[Dict] = None

class UnusualWhalesClient:
    """FIXED Unusual Whales API client with proper authentication and endpoints"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "AI-Radar-Pro/2.0",
            "Content-Type": "application/json"
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache for recent requests
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key"""
        return f"{endpoint}_{hash(str(sorted((params or {}).items())))}"
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return True
            else:
                del self.cache[cache_key]
        return False
    
    def _set_cache(self, cache_key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[cache_key] = (time.time(), data)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """FIXED: Make authenticated request to UW API with proper error handling"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(endpoint, params)
            if self._is_cached(cache_key):
                return self.cache[cache_key][1]
            
            # Rate limiting
            self._rate_limit()
            
            url = f"{self.base_url}{endpoint}"
            
            # Ensure API key is in params
            if params is None:
                params = {}
            
            # Some endpoints may require apikey in params instead of header
            if 'apikey' not in params:
                params['apikey'] = self.api_key
            
            response = self.session.get(url, params=params, timeout=15)
            
            # Enhanced status code handling
            if response.status_code == 200:
                try:
                    data = response.json()
                    result = {
                        "data": data,
                        "error": None,
                        "status_code": response.status_code,
                        "endpoint": endpoint
                    }
                    self._set_cache(cache_key, result)
                    return result
                except json.JSONDecodeError:
                    return {
                        "error": "Invalid JSON response from UW API",
                        "status_code": response.status_code,
                        "endpoint": endpoint,
                        "raw_response": response.text[:200]
                    }
            elif response.status_code == 401:
                return {
                    "error": "Unauthorized - Check API key",
                    "status_code": 401,
                    "endpoint": endpoint
                }
            elif response.status_code == 403:
                return {
                    "error": "Forbidden - Insufficient permissions or subscription level",
                    "status_code": 403,
                    "endpoint": endpoint
                }
            elif response.status_code == 429:
                return {
                    "error": "Rate limit exceeded - Please wait",
                    "status_code": 429,
                    "endpoint": endpoint
                }
            elif response.status_code == 404:
                return {
                    "error": f"Endpoint not found: {endpoint}",
                    "status_code": 404,
                    "endpoint": endpoint
                }
            else:
                return {
                    "error": f"API error: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "endpoint": endpoint,
                    "response_text": response.text[:200]
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout - UW API slow to respond",
                "status_code": None,
                "endpoint": endpoint
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Connection error - Check internet connection",
                "status_code": None,
                "endpoint": endpoint
            }
        except Exception as e:
            return {
                "error": f"Request failed: {str(e)}",
                "status_code": None,
                "endpoint": endpoint
            }
    
    def get_quote(self, symbol: str) -> Dict:
        """FIXED: Get stock quote using correct UW endpoint"""
        try:
            # Try the stock-state endpoint first (more comprehensive)
            result = self._make_request(f"/api/stock/{symbol}/stock-state")
            
            if result.get("error"):
                # Fallback to basic stock endpoint
                result = self._make_request(f"/api/stock/{symbol}")
            
            if result.get("error"):
                return {"error": result["error"], "data_source": "Unusual Whales"}
            
            data = result["data"]
            
            # FIXED: Enhanced data parsing for actual UW response structure
            if isinstance(data, dict):
                # Extract price data with multiple possible field names
                current_price = self._extract_field(data, [
                    "price", "last_price", "lastPrice", "currentPrice", 
                    "close", "close_price", "mark", "last"
                ])
                
                prev_close = self._extract_field(data, [
                    "prev_close_price", "previousClose", "prev_close", 
                    "prevClose", "yesterday_close"
                ])
                
                volume = self._extract_field(data, [
                    "volume", "totalVolume", "total_volume", "dayVolume"
                ], int)
                
                if current_price and current_price > 0:
                    change = current_price - prev_close if prev_close else 0
                    change_percent = (change / prev_close * 100) if prev_close else 0
                    
                    return {
                        "last": current_price,
                        "bid": self._extract_field(data, ["bid", "bidPrice", "bid_price"], float) or current_price - 0.01,
                        "ask": self._extract_field(data, ["ask", "askPrice", "ask_price"], float) or current_price + 0.01,
                        "volume": volume or 0,
                        "change": change,
                        "change_percent": change_percent,
                        "premarket_change": self._extract_premarket_data(data),
                        "intraday_change": change_percent,
                        "postmarket_change": self._extract_postmarket_data(data),
                        "previous_close": prev_close or current_price,
                        "market_open": self._extract_field(data, ["open", "openPrice", "open_price"], float) or current_price,
                        "last_updated": datetime.datetime.now().isoformat(),
                        "data_source": "Unusual Whales",
                        "error": None,
                        "raw_data": data
                    }
            
            return {"error": "No valid price data in UW response", "data_source": "Unusual Whales"}
            
        except Exception as e:
            return {"error": f"UW quote error: {str(e)}", "data_source": "Unusual Whales"}
    
    def _extract_field(self, data: Dict, field_names: List[str], data_type=float):
        """Extract field from data with multiple possible names"""
        for field in field_names:
            if field in data and data[field] is not None:
                try:
                    if data_type == int:
                        return int(float(data[field]))
                    elif data_type == float:
                        return float(data[field])
                    else:
                        return data[field]
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_premarket_data(self, data: Dict) -> float:
        """Extract premarket change data"""
        pm_fields = [
            "premarket_change_percent", "preMarketChangePercent", 
            "premarket_change", "preMarketChange", "pm_change"
        ]
        return self._extract_field(data, pm_fields) or 0.0
    
    def _extract_postmarket_data(self, data: Dict) -> float:
        """Extract after hours change data"""
        ah_fields = [
            "afterhours_change_percent", "afterHoursChangePercent",
            "postmarket_change", "ah_change", "extended_hours_change"
        ]
        return self._extract_field(data, ah_fields) or 0.0
    
    def get_options_volume(self, symbol: str) -> Dict:
        """FIXED: Get options volume data"""
        result = self._make_request(f"/api/stock/{symbol}/options-volume")
        
        if result.get("error"):
            return {"error": result["error"], "data_source": "Unusual Whales"}
        
        return {
            "options_volume": result["data"],
            "data_source": "Unusual Whales",
            "error": None
        }
    
    def get_flow_alerts(self, symbol: str) -> Dict:
        """FIXED: Get options flow alerts"""
        result = self._make_request(f"/api/option-trades/flow-alerts", {"ticker": symbol})
        
        if result.get("error"):
            return {"error": result["error"], "data_source": "Unusual Whales"}
        
        return {
            "flow_alerts": result["data"],
            "data_source": "Unusual Whales",
            "error": None
        }
    
    def get_greek_exposure(self, symbol: str) -> Dict:
        """FIXED: Get Greek exposure data"""
        result = self._make_request(f"/api/stock/{symbol}/greeks")
        
        if result.get("error"):
            return {"error": result["error"], "data_source": "Unusual Whales"}
        
        return {
            "greek_exposure": result["data"],
            "data_source": "Unusual Whales",
            "error": None
        }
    
    def get_market_tide(self) -> Dict:
        """FIXED: Get market sentiment"""
        result = self._make_request("/api/market/ticker/etf-tide")
        
        if result.get("error"):
            return {"error": result["error"], "data_source": "Unusual Whales"}
        
        return {
            "market_tide": result["data"],
            "data_source": "Unusual Whales",
            "error": None
        }
    
    def get_institutional_activity(self, symbol: str) -> Dict:
        """FIXED: Get institutional activity"""
        # Try multiple endpoints for institutional data
        endpoints = [
            f"/api/institution/{symbol}/activity",
            f"/api/institution/name/holdings?ticker={symbol}",
            f"/api/congress/recent-trades?ticker={symbol}"
        ]
        
        for endpoint in endpoints:
            if "recent-trades" in endpoint:
                result = self._make_request(endpoint)
            else:
                result = self._make_request(endpoint.split('?')[0], 
                                          {"ticker": symbol} if "?" in endpoint else {})
            
            if not result.get("error") and result.get("data"):
                return {
                    "institutional_data": result["data"],
                    "data_type": "congressional" if "congress" in endpoint else "institutional",
                    "data_source": "Unusual Whales",
                    "error": None
                }
        
        return {"error": "No institutional data available", "data_source": "Unusual Whales"}
    
    def test_connection(self) -> Dict:
        """FIXED: Test API connection"""
        # Test with a simple endpoint
        result = self._make_request("/api/stock/AAPL/stock-state")
        
        if result.get("error"):
            return {
                "connected": False, 
                "error": result["error"],
                "status_code": result.get("status_code")
            }
        else:
            return {
                "connected": True, 
                "message": "UW API connection successful",
                "endpoint_tested": "/api/stock/AAPL/stock-state"
            }

class TradingEconomicsClient:
    """FIXED Trading Economics API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Radar-Pro/2.0"
        })
    
    def get_economic_calendar(self, days: int = 7) -> List[Dict]:
        """Get economic calendar"""
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
            st.warning(f"Trading Economics calendar error: {e}")
            return []

class AlphaVantageClient:
    """FIXED Alpha Vantage client with better error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        
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
                
                # Check for API limit message
                if "Note" in data:
                    return {"error": "API rate limit exceeded", "data_source": "Alpha Vantage"}
                
                if "Global Quote" in data:
                    quote_data = data["Global Quote"]
                    
                    if not quote_data:
                        return {"error": "No data returned", "data_source": "Alpha Vantage"}
                    
                    price = float(quote_data.get("05. price", 0))
                    change = float(quote_data.get("09. change", 0))
                    change_percent = float(quote_data.get("10. change percent", "0%").replace("%", ""))
                    volume = int(quote_data.get("06. volume", 0))
                    
                    if price > 0:
                        return {
                            "last": price,
                            "bid": price - 0.01,
                            "ask": price + 0.01,
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
                
                return {"error": "No quote data found", "data_source": "Alpha Vantage"}
            else:
                return {"error": f"API error: {response.status_code}", "data_source": "Alpha Vantage"}
                
        except Exception as e:
            return {"error": f"Alpha Vantage error: {str(e)}", "data_source": "Alpha Vantage"}

class TwelveDataClient:
    """FIXED Twelve Data client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        
    def get_quote(self, symbol: str) -> Dict:
        try:
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            # Try real-time price first
            response = self.session.get(f"{self.base_url}/price", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "status" in data and data["status"] == "error":
                    return {"error": f"API Error: {data.get('message', 'Unknown error')}", "data_source": "Twelve Data"}
                
                if "price" in data:
                    price = float(data["price"])
                    
                    # Get additional data
                    quote_params = params.copy()
                    quote_response = self.session.get(f"{self.base_url}/quote", params=quote_params, timeout=10)
                    
                    if quote_response.status_code == 200:
                        quote_data = quote_response.json()
                        
                        if "status" not in quote_data or quote_data["status"] != "error":
                            open_price = float(quote_data.get("open", price))
                            prev_close = float(quote_data.get("previous_close", open_price))
                            volume = int(quote_data.get("volume", 0))
                            
                            change = price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                            
                            return {
                                "last": price,
                                "bid": float(quote_data.get("low", price)),
                                "ask": float(quote_data.get("high", price)),
                                "volume": volume,
                                "change": change,
                                "change_percent": change_percent,
                                "premarket_change": 0,
                                "intraday_change": change_percent,
                                "postmarket_change": 0,
                                "previous_close": prev_close,
                                "market_open": open_price,
                                "last_updated": datetime.datetime.now().isoformat(),
                                "data_source": "Twelve Data",
                                "error": None
                            }
            
            return {"error": "No valid data received", "data_source": "Twelve Data"}
                
        except Exception as e:
            return {"error": f"Twelve Data error: {str(e)}", "data_source": "Twelve Data"}

# Initialize API clients with proper error handling
unusual_whales_client = None
trading_economics_client = None
alpha_vantage_client = None
twelvedata_client = None

if UNUSUAL_WHALES_KEY:
    try:
        unusual_whales_client = UnusualWhalesClient(UNUSUAL_WHALES_KEY)
    except Exception as e:
        st.error(f"Error initializing Unusual Whales client: {e}")

if TRADING_ECONOMICS_KEY:
    try:
        trading_economics_client = TradingEconomicsClient(TRADING_ECONOMICS_KEY)
    except Exception as e:
        st.error(f"Error initializing Trading Economics client: {e}")

if ALPHA_VANTAGE_KEY:
    try:
        alpha_vantage_client = AlphaVantageClient(ALPHA_VANTAGE_KEY)
    except Exception as e:
        st.error(f"Error initializing Alpha Vantage client: {e}")

if TWELVEDATA_KEY:
    try:
        twelvedata_client = TwelveDataClient(TWELVEDATA_KEY)
    except Exception as e:
        st.error(f"Error initializing Twelve Data client: {e}")

@st.cache_data(ttl=15)
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """FIXED: Get live stock quote with proper fallback order and error handling"""
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # Try Unusual Whales FIRST (Primary source)
    if unusual_whales_client:
        try:
            uw_quote = unusual_whales_client.get_quote(ticker)
            if not uw_quote.get("error") and uw_quote.get("last", 0) > 0:
                uw_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return uw_quote
            else:
                print(f"UW failed for {ticker}: {uw_quote.get('error', 'No valid price')}")
        except Exception as e:
            print(f"UW exception for {ticker}: {str(e)}")
    
    # Try Alpha Vantage second
    if alpha_vantage_client:
        try:
            alpha_quote = alpha_vantage_client.get_quote(ticker)
            if not alpha_quote.get("error") and alpha_quote.get("last", 0) > 0:
                alpha_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return alpha_quote
            else:
                print(f"Alpha Vantage failed for {ticker}: {alpha_quote.get('error', 'No valid price')}")
        except Exception as e:
            print(f"Alpha Vantage exception for {ticker}: {str(e)}")
    
    # Try Twelve Data third
    if twelvedata_client:
        try:
            twelve_quote = twelvedata_client.get_quote(ticker)
            if not twelve_quote.get("error") and twelve_quote.get("last", 0) > 0:
                twelve_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return twelve_quote
            else:
                print(f"Twelve Data failed for {ticker}: {twelve_quote.get('error', 'No valid price')}")
        except Exception as e:
            print(f"Twelve Data exception for {ticker}: {str(e)}")
    
    # Final fallback to Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data with extended hours
        hist_1d = stock.history(period="1d", interval="1m", prepost=True)
        if hist_1d.empty:
            hist_1d = stock.history(period="1d", prepost=True)
        
        # Current price from multiple sources
        current_price = None
        price_sources = [
            info.get('currentPrice'),
            info.get('regularMarketPrice'),
            info.get('bid'),
            info.get('ask')
        ]
        
        for price in price_sources:
            if price and price > 0:
                current_price = float(price)
                break
        
        if not current_price and not hist_1d.empty:
            current_price = float(hist_1d['Close'].iloc[-1])
        
        if not current_price or current_price <= 0:
            return {
                "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
                "change": 0.0, "change_percent": 0.0,
                "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
                "previous_close": 0.0, "market_open": 0.0,
                "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
                "error": f"No valid price data found for {ticker}",
                "data_source": "Yahoo Finance"
            }
        
        # Calculate other metrics
        previous_close = float(info.get('previousClose', current_price))
        regular_market_open = float(info.get('regularMarketOpen', current_price))
        volume = int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0))
        
        # Calculate changes
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close > 0 else 0
        
        # Session data (simplified for Yahoo Finance)
        premarket_change = 0.0
        intraday_change = change_percent
        postmarket_change = 0.0
        
        # Try to get extended hours data
        try:
            if not hist_1d.empty and len(hist_1d) > 0:
                market_hours = hist_1d.between_time('09:30', '16:00') if hasattr(hist_1d.index, 'time') else hist_1d
                if not market_hours.empty:
                    market_open_price = float(market_hours['Open'].iloc[0])
                    if previous_close and market_open_price:
                        premarket_change = ((market_open_price - previous_close) / previous_close) * 100
        except:
            pass
        
        return {
            "last": current_price,
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": volume,
            "change": change,
            "change_percent": change_percent,
            "premarket_change": premarket_change,
            "intraday_change": intraday_change,
            "postmarket_change": postmarket_change,
            "previous_close": previous_close,
            "market_open": regular_market_open,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None,
            "data_source": "Yahoo Finance"
        }
        
    except Exception as e:
        return {
            "last": 0.0, "bid": 0.0, "ask": 0.0, "volume": 0,
            "change": 0.0, "change_percent": 0.0,
            "premarket_change": 0.0, "intraday_change": 0.0, "postmarket_change": 0.0,
            "previous_close": 0.0, "market_open": 0.0,
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": f"Yahoo Finance error: {str(e)}",
            "data_source": "Yahoo Finance"
        }

# Enhanced AI Analysis System
class MultiAIAnalyzer:
    """FIXED Multi-AI analysis system"""
    
    def __init__(self):
        self.openai_client = openai_client
        self.gemini_model = gemini_model
        self.grok_client = grok_client
        
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
        """OpenAI analysis with better error handling"""
        if not self.openai_client:
            return "OpenAI not available"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are an expert trading analyst. Provide concise, actionable analysis with specific entry/exit levels."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2,
                max_tokens=500,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def analyze_with_gemini(self, prompt: str) -> str:
        """Gemini analysis with better error handling"""
        if not self.gemini_model:
            return "Gemini not available"
        
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    
    def analyze_with_grok(self, prompt: str) -> str:
        """Grok analysis with better error handling"""
        if not self.grok_client:
            return "Grok not available"
        
        try:
            response = self.grok_client.chat.completions.create(
                model="grok-beta",
                messages=[{
                    "role": "system",
                    "content": "You are an expert trading analyst with access to real-time market data and social sentiment."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=500,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Grok Error: {str(e)}"

# Initialize Multi-AI Analyzer
multi_ai = MultiAIAnalyzer()

def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """FIXED AI playbook with enhanced data integration"""
    
    try:
        # Get comprehensive data
        quote = get_live_quote(ticker, st.session_state.selected_tz)
        
        # Get UW options data if available
        uw_options_data = ""
        if unusual_whales_client:
            try:
                uw_volume = unusual_whales_client.get_options_volume(ticker)
                uw_alerts = unusual_whales_client.get_flow_alerts(ticker)
                uw_greeks = unusual_whales_client.get_greek_exposure(ticker)
                
                if not uw_volume.get("error"):
                    uw_options_data += f"UW Options Volume: {uw_volume.get('options_volume', {})} "
                if not uw_alerts.get("error"):
                    alerts = uw_alerts.get('flow_alerts', [])
                    uw_options_data += f"UW Flow Alerts: {len(alerts) if isinstance(alerts, list) else 0} detected "
                if not uw_greeks.get("error"):
                    uw_options_data += f"UW Greeks: {uw_greeks.get('greek_exposure', {})} "
            except Exception as e:
                uw_options_data = f"UW Options Error: {str(e)}"
        
        # Construct enhanced prompt
        prompt = f"""
        COMPREHENSIVE TRADING ANALYSIS for {ticker}:
        
        Price Data:
        - Current: ${quote['last']:.2f} ({quote['change_percent']:+.2f}%)
        - Volume: {quote['volume']:,}
        - Source: {quote.get('data_source', 'Unknown')}
        - Session: PM {quote['premarket_change']:+.2f}% | Day {quote['intraday_change']:+.2f}% | AH {quote['postmarket_change']:+.2f}%
        
        Catalyst: {catalyst or 'Market movement'}
        
        Unusual Whales Data: {uw_options_data or 'Not available'}
        
        Options Data: {options_data if options_data else 'Standard yfinance data'}
        
        Provide a concise trading analysis with:
        1. Direction bias (Bullish/Bearish/Neutral) with confidence %
        2. Entry strategy and price levels
        3. Profit targets (3 levels)
        4. Stop loss placement
        5. Risk factors
        6. Time horizon (scalp/day/swing)
        
        Focus on ACTIONABLE insights. Keep under 300 words.
        """
        
        # Use selected AI model
        if st.session_state.ai_model == "Multi-AI":
            analyses = {}
            if multi_ai.openai_client:
                analyses["OpenAI"] = multi_ai.analyze_with_openai(prompt)
            if multi_ai.gemini_model:
                analyses["Gemini"] = multi_ai.analyze_with_gemini(prompt)
            if multi_ai.grok_client:
                analyses["Grok"] = multi_ai.analyze_with_grok(prompt)
            
            if analyses:
                result = f"## Multi-AI Analysis for {ticker}\n\n"
                for model, analysis in analyses.items():
                    result += f"### {model}:\n{analysis}\n\n---\n\n"
                return result
            else:
                return f"No AI models available for {ticker} analysis"
        
        elif st.session_state.ai_model == "OpenAI":
            return multi_ai.analyze_with_openai(prompt)
        elif st.session_state.ai_model == "Gemini":
            return multi_ai.analyze_with_gemini(prompt)
        elif st.session_state.ai_model == "Grok":
            return multi_ai.analyze_with_grok(prompt)
        else:
            return "No AI model selected"
            
    except Exception as e:
        return f"Error generating playbook for {ticker}: {str(e)}"

# Main Streamlit App
st.title("AI Radar Pro — Fixed Trading Assistant")

# Timezone selector
col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], 
                                                index=0 if st.session_state.selected_tz == "ET" else 1)

# Current time display
tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
current_time = current_tz.strftime("%I:%M:%S %p")
market_open = 9 <= current_tz.hour < 16
status = "🟢 Open" if market_open else "🔴 Closed"

# Enhanced Sidebar with API Status
st.sidebar.subheader("🤖 AI Configuration")
available_models = ["Multi-AI"] + multi_ai.get_available_models()
st.session_state.ai_model = st.sidebar.selectbox("AI Model", available_models)

# API Status Display
st.sidebar.subheader("📊 Data Sources")
if unusual_whales_client:
    if st.sidebar.button("Test UW Connection"):
        with st.spinner("Testing Unusual Whales API..."):
            result = unusual_whales_client.test_connection()
            if result.get("connected"):
                st.sidebar.success(f"✅ {result['message']}")
            else:
                st.sidebar.error(f"❌ {result['error']}")
    st.sidebar.success("✅ Unusual Whales Configured")
else:
    st.sidebar.error("❌ Unusual Whales Not Configured")

if alpha_vantage_client:
    st.sidebar.success("✅ Alpha Vantage Connected")
else:
    st.sidebar.warning("⚠️ Alpha Vantage Not Connected")

if twelvedata_client:
    st.sidebar.success("✅ Twelve Data Connected")
else:
    st.sidebar.warning("⚠️ Twelve Data Not Connected")

st.sidebar.success("✅ Yahoo Finance (Fallback)")

# Main interface
st.markdown(f"**Status:** {status} | **Time:** {current_time} {st.session_state.selected_tz}")

# Test section
st.subheader("🧪 Test Fixed Integration")

col1, col2 = st.columns([3, 1])
with col1:
    test_ticker = st.text_input("Test ticker", value="AAPL", key="test_ticker").upper()
with col2:
    if st.button("Test Quote"):
        with st.spinner(f"Testing quote for {test_ticker}..."):
            quote = get_live_quote(test_ticker, st.session_state.selected_tz)
            
            if not quote.get("error"):
                st.success(f"✅ Quote successful - Source: {quote.get('data_source')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(test_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.3f}")
                col4.metric("Source", quote.get('data_source', 'Unknown'))
                
                # Test UW options data if available
                if unusual_whales_client:
                    st.write("**Testing UW Options Data:**")
                    with st.spinner("Getting UW options data..."):
                        uw_volume = unusual_whales_client.get_options_volume(test_ticker)
                        uw_alerts = unusual_whales_client.get_flow_alerts(test_ticker)
                        
                        opt_col1, opt_col2 = st.columns(2)
                        with opt_col1:
                            if not uw_volume.get("error"):
                                st.success("✅ UW Options Volume")
                                st.json(uw_volume.get("options_volume", {}) if isinstance(uw_volume.get("options_volume"), dict) else "Data received")
                            else:
                                st.error(f"❌ UW Volume: {uw_volume['error']}")
                        
                        with opt_col2:
                            if not uw_alerts.get("error"):
                                st.success("✅ UW Flow Alerts")
                                alerts = uw_alerts.get("flow_alerts", [])
                                st.write(f"Alerts: {len(alerts) if isinstance(alerts, list) else 'Data received'}")
                            else:
                                st.error(f"❌ UW Alerts: {uw_alerts['error']}")
                
                # Test AI analysis
                if st.button("Test AI Analysis"):
                    with st.spinner("Testing AI analysis..."):
                        analysis = ai_playbook(test_ticker, quote['change_percent'], "Test analysis")
                        st.markdown("### 🤖 AI Analysis")
                        st.markdown(analysis)
                
            else:
                st.error(f"❌ Quote failed: {quote['error']}")

# Footer
st.markdown("---")
data_sources = []
if unusual_whales_client:
    data_sources.append("🐋 Unusual Whales")
if alpha_vantage_client:
    data_sources.append("Alpha Vantage")
if twelvedata_client:
    data_sources.append("Twelve Data")
data_sources.append("Yahoo Finance")

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🔥 AI Radar Pro v2.0 FIXED | Data: {' + '.join(data_sources)} | AI: {st.session_state.ai_model}"
    "</div>",
    unsafe_allow_html=True
)
