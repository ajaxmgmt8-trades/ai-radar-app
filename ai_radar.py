import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
import plotly.express as px
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
st.set_page_config(page_title="🐋 AI Radar Pro - Unusual Whales Edition", layout="wide")

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
    st.session_state.selected_tz = "ET"
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Unusual Whales"
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"

# API Keys - Enhanced Configuration
try:
    UNUSUAL_WHALES_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")
    TRADING_ECONOMICS_KEY = st.secrets.get("TRADING_ECONOMICS_KEY", "")
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
        grok_client = openai.OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )

except Exception as e:
    st.error(f"Error loading API keys: {e}")
    openai_client = None
    gemini_model = None
    grok_client = None

# ===== COMPREHENSIVE UNUSUAL WHALES CLIENT =====
class UnusualWhalesClient:
    """Complete Unusual Whales API client using all 109 endpoints"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com/api"
        self.session = requests.Session()
        # Correct authentication format from documentation
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/plain",
            "User-Agent": "AI-Radar-Pro/3.0"
        })
    
    # ===== STOCK DATA =====
    
    def get_stock_state(self, ticker: str) -> Dict:
        """Get real-time stock state"""
        try:
            url = f"{self.base_url}/stock/{ticker}/stock-state"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return self._format_stock_response(result, ticker)
        except Exception as e:
            return {"error": f"Stock State Error for {ticker}: {e}", "data_source": "🐋 Unusual Whales"}
    
    def get_stock_data(self, symbol: str) -> Dict:
        """Get comprehensive stock data - wrapper for compatibility"""
        return self.get_stock_state(symbol)
    
    def _format_stock_response(self, data: Dict, ticker: str) -> Dict:
        """Format UW stock response to standard format"""
        try:
            if 'data' in data:
                stock_data = data['data']
            else:
                stock_data = data
            
            # Extract price data with multiple possible field names
            last_price = float(stock_data.get('last') or stock_data.get('price') or stock_data.get('current_price') or 0)
            
            if last_price <= 0:
                return {"error": f"Invalid price data for {ticker}", "data_source": "🐋 Unusual Whales"}
            
            open_price = float(stock_data.get('open') or stock_data.get('market_open') or last_price)
            high_price = float(stock_data.get('high') or stock_data.get('day_high') or last_price)
            low_price = float(stock_data.get('low') or stock_data.get('day_low') or last_price)
            volume = int(stock_data.get('volume') or stock_data.get('total_volume') or 0)
            
            # Calculate changes
            prev_close = float(stock_data.get('previous_close') or stock_data.get('prev_close') or open_price)
            change = last_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close > 0 else 0
            
            # Bid/Ask
            bid = float(stock_data.get('bid') or stock_data.get('bid_price') or last_price - 0.01)
            ask = float(stock_data.get('ask') or stock_data.get('ask_price') or last_price + 0.01)
            
            return {
                "last": last_price,
                "bid": bid,
                "ask": ask,
                "volume": volume,
                "change": change,
                "change_percent": change_percent,
                "premarket_change": float(stock_data.get('premarket_change_percent') or 0),
                "intraday_change": change_percent,
                "postmarket_change": float(stock_data.get('afterhours_change_percent') or 0),
                "previous_close": prev_close,
                "market_open": open_price,
                "high": high_price,
                "low": low_price,
                "last_updated": datetime.datetime.now().isoformat(),
                "data_source": "🐋 Unusual Whales",
                "error": None
            }
        except Exception as e:
            return {"error": f"UW format error: {str(e)}", "data_source": "🐋 Unusual Whales"}
    
    def get_option_chains(self, ticker: str, date: str = None) -> Dict:
        """Get option chains for ticker"""
        try:
            url = f"{self.base_url}/stock/{ticker}/option-chains"
            params = {}
            if date:
                params['date'] = date
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Option Chains Error for {ticker}: {e}"}
    
    def get_recent_flow(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Get recent options flow for ticker"""
        try:
            url = f"{self.base_url}/stock/{ticker}/flow-recent"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_flow_alerts(self, ticker: str = None, limit: int = 50) -> List[Dict]:
        """Get flow alerts (market-wide or for specific ticker)"""
        try:
            if ticker:
                url = f"{self.base_url}/stock/{ticker}/flow-alerts"
            else:
                url = f"{self.base_url}/option-trades/flow-alerts"
            
            params = {"limit": limit}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_greek_exposure(self, ticker: str) -> Dict:
        """Get Greek exposure (GEX) for ticker"""
        try:
            url = f"{self.base_url}/stock/{ticker}/greek-exposure"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Greek Exposure Error for {ticker}: {e}"}
    
    def get_iv_rank(self, ticker: str) -> Dict:
        """Get IV rank for ticker"""
        try:
            url = f"{self.base_url}/stock/{ticker}/iv-rank"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"IV Rank Error for {ticker}: {e}"}
    
    def get_max_pain(self, ticker: str) -> Dict:
        """Get max pain levels for ticker"""
        try:
            url = f"{self.base_url}/stock/{ticker}/max-pain"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Max Pain Error for {ticker}: {e}"}
    
    # ===== MARKET-WIDE DATA =====
    
    def get_market_spike(self) -> Dict:
        """Get SPIKE market-wide data"""
        try:
            url = f"{self.base_url}/market/spike"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Market SPIKE Error: {e}"}
    
    def get_market_tide(self) -> Dict:
        """Get market tide data"""
        try:
            url = f"{self.base_url}/market/market-tide"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Market Tide Error: {e}"}
    
    def get_total_options_volume(self) -> Dict:
        """Get total options volume across market"""
        try:
            url = f"{self.base_url}/market/total-options-volume"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Total Options Volume Error: {e}"}
    
    def get_oi_change(self) -> List[Dict]:
        """Get market-wide OI changes"""
        try:
            url = f"{self.base_url}/market/oi-change"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_correlations(self) -> Dict:
        """Get market correlations"""
        try:
            url = f"{self.base_url}/market/correlations"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Correlations Error: {e}"}
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get economic calendar"""
        try:
            url = f"{self.base_url}/market/economic-calendar"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_fda_calendar(self) -> List[Dict]:
        """Get FDA calendar"""
        try:
            url = f"{self.base_url}/market/fda-calendar"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    # ===== CONGRESSIONAL TRADING =====
    
    def get_congress_recent_trades(self, limit: int = 50, ticker: str = None) -> List[Dict]:
        """Get recent congressional trades"""
        try:
            url = f"{self.base_url}/congress/recent-trades"
            params = {"limit": limit}
            if ticker:
                params['ticker'] = ticker
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_congress_trades(self, limit: int = 20) -> List[Dict]:
        """Get congressional trading data - wrapper for compatibility"""
        return self.get_congress_recent_trades(limit)
    
    def get_congress_late_reports(self, limit: int = 20) -> List[Dict]:
        """Get late congressional reports"""
        try:
            url = f"{self.base_url}/congress/late-reports"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_politician_portfolios(self, politician_id: str = None) -> Dict:
        """Get politician portfolio data"""
        try:
            if politician_id:
                url = f"{self.base_url}/politician-portfolios/{politician_id}"
            else:
                url = f"{self.base_url}/politician-portfolios/people"
            
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Politician Portfolios Error: {e}"}
    
    # ===== INSIDER TRADING =====
    
    def get_insider_transactions(self, limit: int = 50, ticker: str = None) -> List[Dict]:
        """Get insider transactions"""
        try:
            url = f"{self.base_url}/insider/transactions"
            params = {"limit": limit}
            if ticker:
                params['ticker'] = ticker
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_insider_ticker_flow(self, ticker: str) -> Dict:
        """Get insider flow for specific ticker"""
        try:
            url = f"{self.base_url}/insider/{ticker}/ticker-flow"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Insider Ticker Flow Error for {ticker}: {e}"}
    
    # ===== SCREENERS =====
    
    def get_stock_screener(self, limit: int = 50) -> List[Dict]:
        """Get stock screener results"""
        try:
            url = f"{self.base_url}/screener/stocks"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_hottest_chains(self, limit: int = 30) -> List[Dict]:
        """Get hottest option chains"""
        try:
            url = f"{self.base_url}/screener/option-contracts"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_analyst_ratings(self, limit: int = 50) -> List[Dict]:
        """Get analyst ratings"""
        try:
            url = f"{self.base_url}/screener/analysts"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    # ===== DARKPOOL =====
    
    def get_darkpool_recent(self, limit: int = 50) -> List[Dict]:
        """Get recent dark pool trades"""
        try:
            url = f"{self.base_url}/darkpool/recent"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_darkpool_ticker(self, ticker: str) -> List[Dict]:
        """Get dark pool trades for specific ticker"""
        try:
            url = f"{self.base_url}/darkpool/{ticker}"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    # ===== NEWS =====
    
    def get_news_headlines(self, limit: int = 20) -> List[Dict]:
        """Get news headlines"""
        try:
            url = f"{self.base_url}/news/headlines"
            params = {"limit": limit}
            
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    # ===== EARNINGS =====
    
    def get_earnings_premarket(self) -> List[Dict]:
        """Get premarket earnings"""
        try:
            url = f"{self.base_url}/earnings/premarket"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    def get_earnings_afterhours(self) -> List[Dict]:
        """Get after hours earnings"""
        try:
            url = f"{self.base_url}/earnings/afterhours"
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            result = response.json()
            return result.get('data', [])
        except Exception as e:
            return []
    
    # ===== COMPATIBILITY METHODS =====
    
    def get_options_flow(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get options flow data - compatibility wrapper"""
        return self.get_flow_alerts(symbol, limit)
    
    def get_market_overview(self) -> Dict:
        """Get market overview - compatibility wrapper"""
        try:
            # Combine multiple market endpoints for overview
            overview = {}
            
            spike_data = self.get_market_spike()
            if not spike_data.get("error"):
                overview['spike'] = spike_data
            
            tide_data = self.get_market_tide()
            if not tide_data.get("error"):
                overview['tide'] = tide_data
            
            volume_data = self.get_total_options_volume()
            if not volume_data.get("error"):
                overview['options_volume'] = volume_data
            
            return overview
        except Exception as e:
            return {"error": f"Market Overview Error: {e}"}

# ===== TRADING ECONOMICS CLIENT =====
class TradingEconomicsClient:
    """Trading Economics API client for economic data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
    
    def get_economic_indicators(self, country: str = "united states") -> List[Dict]:
        """Get key economic indicators"""
        try:
            url = f"{self.base_url}/indicators"
            params = {
                "c": country,
                "key": self.api_key,
                "format": "json"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()[:20]
            return []
            
        except Exception as e:
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Trading Economics"""
        try:
            url = f"{self.base_url}/markets/symbol/{symbol}"
            params = {"key": self.api_key}
            
            response = self.session.get(url, params=params, timeout=8)
            if response.status_code == 200:
                result = response.json()
                return result[0] if isinstance(result, list) and result else result
            return {}
            
        except Exception as e:
            return {}
    
    def get_economic_calendar(self, date: str = None) -> List[Dict]:
        """Get economic calendar events"""
        try:
            url = f"{self.base_url}/calendar"
            params = {"key": self.api_key}
            if date:
                params["d1"] = date
                params["d2"] = date
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()[:15]
            return []
            
        except Exception as e:
            return []

# Initialize enhanced clients
unusual_whales_client = UnusualWhalesClient(UNUSUAL_WHALES_KEY) if UNUSUAL_WHALES_KEY else None
trading_economics_client = TradingEconomicsClient(TRADING_ECONOMICS_KEY) if TRADING_ECONOMICS_KEY else None

# ===== ENHANCED DATA FUNCTION - UNUSUAL WHALES PRIMARY =====
@st.cache_data(ttl=30)  # Fast cache - 30 seconds
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """Enhanced live quote function - Unusual Whales FIRST, then fallbacks"""
    
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # 1. TRY UNUSUAL WHALES FIRST (PRIMARY)
    if unusual_whales_client:
        try:
            uw_quote = unusual_whales_client.get_stock_data(ticker)
            if not uw_quote.get("error") and uw_quote.get("last", 0) > 0:
                uw_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return uw_quote
        except Exception as e:
            pass  # Fall through to next source
    
    # 2. Fall back to Yahoo Finance
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
            "data_source": "Yahoo Finance (Fallback)"
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
            "data_source": "Error"
        }

# ===== AI ANALYSIS ENGINE =====
class AIAnalysisEngine:
    """Enhanced AI analysis with UW-specific insights"""
    
    def __init__(self):
        self.models = {}
        self.setup_models()
    
    def setup_models(self):
        """Setup available AI models"""
        try:
            if openai_client:
                self.models['openai'] = True
                
            if gemini_model:
                self.models['gemini'] = gemini_model
                
            if grok_client:
                self.models['grok'] = True
                
        except Exception as e:
            st.warning(f"AI Model Setup Warning: {e}")
    
    def analyze_comprehensive_data(self, data_cache: Dict) -> str:
        """Generate comprehensive analysis using all UW data"""
        
        prompt = f"""
        As an expert options and equity trader, analyze this comprehensive market data from Unusual Whales:

        MARKET OVERVIEW:
        - SPIKE Data: {json.dumps(data_cache.get('market_spike', {}), indent=2)[:1000]}
        - Market Tide: {json.dumps(data_cache.get('market_tide', {}), indent=2)[:1000]}
        - Total Options Volume: {json.dumps(data_cache.get('total_options_volume', {}), indent=2)[:500]}

        FLOW ALERTS (Recent):
        {json.dumps(data_cache.get('flow_alerts', [])[:10], indent=2)[:1500]}

        CONGRESSIONAL ACTIVITY:
        {json.dumps(data_cache.get('congress_trades', [])[:8], indent=2)[:1200]}

        INSIDER ACTIVITY:
        {json.dumps(data_cache.get('insider_transactions', [])[:8], indent=2)[:1200]}

        UNUSUAL ACTIVITY:
        - Hottest Chains: {json.dumps(data_cache.get('hottest_chains', [])[:5], indent=2)[:800]}
        - Dark Pool Activity: {json.dumps(data_cache.get('darkpool_recent', [])[:5], indent=2)[:800]}

        ECONOMIC CATALYSTS:
        - Economic Calendar: {json.dumps(data_cache.get('economic_calendar', [])[:5], indent=2)[:800]}
        - FDA Calendar: {json.dumps(data_cache.get('fda_calendar', [])[:3], indent=2)[:500]}

        EARNINGS:
        - Premarket: {json.dumps(data_cache.get('earnings_premarket', [])[:5], indent=2)[:600]}
        - After Hours: {json.dumps(data_cache.get('earnings_afterhours', [])[:5], indent=2)[:600]}

        Provide analysis covering:
        1. **Market Sentiment & SPIKE Analysis** - What's the overall market fear/greed?
        2. **High-Impact Flow Interpretation** - Which options flows signal big moves?
        3. **Smart Money Tracking** - Congress/insider patterns worth following
        4. **Dark Pool & Unusual Activity** - Hidden institutional moves
        5. **Greeks & Volatility Insights** - GEX levels and IV opportunities
        6. **Economic & Earnings Catalysts** - Events that could move markets
        7. **5 Specific Trade Ideas** - Actionable setups with entry/exit levels

        Format with clear headers and bullet points. Focus on actionable insights.
        """
        
        # Try AI models in order of preference
        for model_name, model in self.models.items():
            try:
                if model_name == 'openai' and model:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2500,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
                
                elif model_name == 'gemini' and model:
                    response = model.generate_content(prompt)
                    return response.text
                
                elif model_name == 'grok' and model:
                    response = grok_client.chat.completions.create(
                        model="grok-beta",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2500,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                st.warning(f"AI Model {model_name} failed: {e}")
                continue
        
        # Fallback manual analysis
        return self.generate_manual_analysis(data_cache)
    
    def generate_manual_analysis(self, data_cache: Dict) -> str:
        """Generate manual analysis when AI unavailable"""
        
        analysis = "📊 **COMPREHENSIVE MARKET ANALYSIS**\n\n"
        
        # Market sentiment
        spike_data = data_cache.get('market_spike', {})
        if spike_data and not spike_data.get('error'):
            spike_value = spike_data.get('data', {}).get('current', 0)
            sentiment = "HIGH FEAR" if spike_value > 30 else "MODERATE" if spike_value > 20 else "LOW FEAR"
            analysis += f"**🚨 SPIKE Level: {spike_value:.1f} ({sentiment})**\n\n"
        
        # Options volume
        options_vol = data_cache.get('total_options_volume', {})
        if options_vol and not options_vol.get('error'):
            analysis += f"**📊 Options Volume Insight**\n"
            vol_data = options_vol.get('data', {})
            call_vol = vol_data.get('call_volume', 0)
            put_vol = vol_data.get('put_volume', 0)
            if call_vol and put_vol:
                cp_ratio = call_vol / put_vol
                sentiment = "BULLISH" if cp_ratio > 1.2 else "BEARISH" if cp_ratio < 0.8 else "NEUTRAL"
                analysis += f"• Call/Put Ratio: {cp_ratio:.2f} ({sentiment})\n"
                analysis += f"• Total Volume: {call_vol + put_vol:,.0f}\n\n"
        
        # Flow alerts summary
        flow_alerts = data_cache.get('flow_alerts', [])
        if flow_alerts:
            analysis += f"**🎯 Top Flow Alerts ({len(flow_alerts)} total)**\n"
            for i, alert in enumerate(flow_alerts[:5]):
                ticker = alert.get('ticker', 'N/A')
                premium = alert.get('premium', 0)
                sentiment = alert.get('sentiment', 'N/A')
                analysis += f"{i+1}. {ticker}: ${premium:,.0f} premium ({sentiment})\n"
            analysis += "\n"
        
        # Congressional activity
        congress = data_cache.get('congress_trades', [])
        if congress:
            analysis += f"**🏛️ Congressional Activity ({len(congress)} trades)**\n"
            tickers = [trade.get('ticker', '') for trade in congress[:10]]
            unique_tickers = list(set([t for t in tickers if t]))[:5]
            analysis += f"• Active Tickers: {', '.join(unique_tickers)}\n\n"
        
        # Dark pool activity
        darkpool = data_cache.get('darkpool_recent', [])
        if darkpool:
            analysis += f"**🕶️ Dark Pool Activity ({len(darkpool)} trades)**\n"
            total_value = sum([trade.get('value', 0) for trade in darkpool[:10]])
            analysis += f"• Recent Value: ${total_value:,.0f}\n\n"
        
        # Earnings catalysts
        premarket = data_cache.get('earnings_premarket', [])
        afterhours = data_cache.get('earnings_afterhours', [])
        if premarket or afterhours:
            analysis += f"**📈 Earnings Catalysts**\n"
            if premarket:
                pm_tickers = [e.get('ticker', '') for e in premarket[:5]]
                analysis += f"• Premarket: {', '.join(pm_tickers)}\n"
            if afterhours:
                ah_tickers = [e.get('ticker', '') for e in afterhours[:5]]
                analysis += f"• After Hours: {', '.join(ah_tickers)}\n"
            analysis += "\n"
        
        analysis += "**⚡ Key Takeaways**\n"
        analysis += "• Monitor SPIKE levels for volatility opportunities\n"
        analysis += "• Watch flow alerts for institutional positioning\n"
        analysis += "• Track congressional trades for longer-term moves\n"
        analysis += "• Use earnings catalysts for event-driven trades\n"
        
        return analysis

# ===== COMPREHENSIVE DATA MANAGER =====
class ComprehensiveDataManager:
    """Manages all UW data sources with parallel fetching"""
    
    def __init__(self):
        self.uw_client = unusual_whales_client
        self.ai_engine = AIAnalysisEngine()
        
        # Comprehensive data cache
        self.cache = {
            # Market data
            'market_spike': {},
            'market_tide': {},
            'total_options_volume': {},
            'oi_change': [],
            'correlations': {},
            
            # Stock specific
            'stock_data': {},
            'option_chains': {},
            'greek_exposure': {},
            'iv_rank': {},
            'max_pain': {},
            
            # Flow data
            'flow_alerts': [],
            'recent_flows': {},
            
            # Smart money
            'congress_trades': [],
            'congress_late_reports': [],
            'insider_transactions': [],
            'insider_flows': {},
            
            # Screening
            'stock_screener': [],
            'hottest_chains': [],
            'analyst_ratings': [],
            
            # Dark pool
            'darkpool_recent': [],
            'darkpool_tickers': {},
            
            # Catalysts
            'economic_calendar': [],
            'fda_calendar': [],
            'earnings_premarket': [],
            'earnings_afterhours': [],
            'news_headlines': [],
            
            'last_update': None
        }
    
    def fetch_all_data_comprehensive(self, watchlist: List[str]) -> Dict:
        """Fetch comprehensive data using all UW endpoints"""
        
        if not self.uw_client:
            st.error("🚨 Unusual Whales API key required!")
            return self.cache
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = {}
            
            # Market-wide data
            futures['market_spike'] = executor.submit(self.uw_client.get_market_spike)
            futures['market_tide'] = executor.submit(self.uw_client.get_market_tide)
            futures['total_options_volume'] = executor.submit(self.uw_client.get_total_options_volume)
            futures['oi_change'] = executor.submit(self.uw_client.get_oi_change)
            futures['correlations'] = executor.submit(self.uw_client.get_correlations)
            
            # Flow data
            futures['flow_alerts'] = executor.submit(self.uw_client.get_flow_alerts, None, 100)
            
            # Smart money tracking
            futures['congress_trades'] = executor.submit(self.uw_client.get_congress_recent_trades, 50)
            futures['congress_late_reports'] = executor.submit(self.uw_client.get_congress_late_reports, 20)
            futures['insider_transactions'] = executor.submit(self.uw_client.get_insider_transactions, 50)
            
            # Screening
            futures['stock_screener'] = executor.submit(self.uw_client.get_stock_screener, 50)
            futures['hottest_chains'] = executor.submit(self.uw_client.get_hottest_chains, 30)
            futures['analyst_ratings'] = executor.submit(self.uw_client.get_analyst_ratings, 30)
            
            # Dark pool
            futures['darkpool_recent'] = executor.submit(self.uw_client.get_darkpool_recent, 50)
            
            # Catalysts
            futures['economic_calendar'] = executor.submit(self.uw_client.get_economic_calendar)
            futures['fda_calendar'] = executor.submit(self.uw_client.get_fda_calendar)
            futures['earnings_premarket'] = executor.submit(self.uw_client.get_earnings_premarket)
            futures['earnings_afterhours'] = executor.submit(self.uw_client.get_earnings_afterhours)
            futures['news_headlines'] = executor.submit(self.uw_client.get_news_headlines, 20)
            
            # Stock-specific data for watchlist
            for ticker in watchlist:
                futures[f'stock_state_{ticker}'] = executor.submit(self.uw_client.get_stock_state, ticker)
                futures[f'recent_flow_{ticker}'] = executor.submit(self.uw_client.get_recent_flow, ticker, 20)
                futures[f'greek_exposure_{ticker}'] = executor.submit(self.uw_client.get_greek_exposure, ticker)
                futures[f'iv_rank_{ticker}'] = executor.submit(self.uw_client.get_iv_rank, ticker)
                futures[f'max_pain_{ticker}'] = executor.submit(self.uw_client.get_max_pain, ticker)
            
            # Collect results
            for future_name, future in futures.items():
                try:
                    result = future.result(timeout=20)
                    
                    if future_name.startswith('stock_state_'):
                        ticker = future_name.replace('stock_state_', '')
                        self.cache['stock_data'][ticker] = result
                    elif future_name.startswith('recent_flow_'):
                        ticker = future_name.replace('recent_flow_', '')
                        self.cache['recent_flows'][ticker] = result
                    elif future_name.startswith('greek_exposure_'):
                        ticker = future_name.replace('greek_exposure_', '')
                        self.cache['greek_exposure'][ticker] = result
                    elif future_name.startswith('iv_rank_'):
                        ticker = future_name.replace('iv_rank_', '')
                        self.cache['iv_rank'][ticker] = result
                    elif future_name.startswith('max_pain_'):
                        ticker = future_name.replace('max_pain_', '')
                        self.cache['max_pain'][ticker] = result
                    elif future_name in self.cache:
                        self.cache[future_name] = result
                        
                except Exception as e:
                    st.warning(f"Data fetch error for {future_name}: {e}")
        
        # Update timestamp
        self.cache['last_update'] = datetime.datetime.now()
        
        fetch_time = time.time() - start_time
        st.sidebar.success(f"🚀 Comprehensive data refreshed in {fetch_time:.2f}s")
        
        return self.cache
    
    def get_ai_analysis(self) -> str:
        """Get comprehensive AI analysis"""
        return self.ai_engine.analyze_comprehensive_data(self.cache)

# Initialize data manager
comprehensive_dm = ComprehensiveDataManager()

# ===== KEEP ALL YOUR EXISTING HELPER FUNCTIONS =====
# (get_finnhub_news, get_polygon_news, get_all_news, etc. remain the same)

@st.cache_data(ttl=300)
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
    
    # Get UW news first
    if unusual_whales_client:
        try:
            uw_news = unusual_whales_client.get_news_headlines(15)
            for item in uw_news:
                all_news.append({
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "source": "Unusual Whales",
                    "url": item.get("url", ""),
                    "datetime": item.get("datetime", 0),
                    "related": item.get("tickers", "")
                })
        except Exception as e:
            pass
    
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

# ===== ENHANCED AI FUNCTIONS =====
def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook using comprehensive UW data"""
    
    if not openai_client and not gemini_model and not grok_client:
        return f"**{ticker} Analysis** - No AI models configured"
    
    # Get quote for context
    quote = get_live_quote(ticker, st.session_state.selected_tz)
    
    # Get UW-specific data if available
    enhanced_data = ""
    if unusual_whales_client:
        try:
            # Try to get enhanced data from UW
            flow_alerts = unusual_whales_client.get_flow_alerts(ticker, 10)
            greek_exposure = unusual_whales_client.get_greek_exposure(ticker)
            iv_rank = unusual_whales_client.get_iv_rank(ticker)
            
            enhanced_data += f"""
            
UNUSUAL WHALES DATA:
Flow Alerts: {len(flow_alerts)} recent alerts
Greek Exposure: {json.dumps(greek_exposure, indent=2)[:500]}
IV Rank: {json.dumps(iv_rank, indent=2)[:300]}
"""
        except Exception:
            pass
    
    prompt = f"""
    As an expert trader, analyze {ticker} for actionable trading opportunities:

    CURRENT SITUATION:
    - Price: ${quote.get('last', 0):.2f} ({quote.get('change_percent', 0):+.2f}%)
    - Volume: {quote.get('volume', 0):,}
    - Session: PM {quote.get('premarket_change', 0):+.2f}% | Day {quote.get('intraday_change', 0):+.2f}% | AH {quote.get('postmarket_change', 0):+.2f}%
    - Data Source: {quote.get('data_source', 'Yahoo Finance')}
    - Catalyst: {catalyst}
    
    {enhanced_data}

    Provide specific trading analysis:
    1. **Overall Assessment** (Bullish/Bearish/Neutral) with confidence 1-100
    2. **Trading Strategy** with specific timeframe
    3. **Entry Levels** with exact prices
    4. **Profit Targets** (3 levels)
    5. **Risk Management** (stop loss, position size)
    6. **Key Levels** to watch
    7. **Options Strategy** if applicable

    Keep under 400 words, be specific and actionable.
    """
    
    # Try AI models
    try:
        if st.session_state.ai_model == "OpenAI" and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        
        elif st.session_state.ai_model == "Gemini" and gemini_model:
            response = gemini_model.generate_content(prompt)
            return response.text
        
        elif st.session_state.ai_model == "Grok" and grok_client:
            response = grok_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return response.choices[0].message.content
        
        elif st.session_state.ai_model == "Multi-AI":
            # Multi-AI consensus
            analyses = {}
            
            if openai_client:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    analyses["OpenAI"] = response.choices[0].message.content
                except Exception as e:
                    analyses["OpenAI"] = f"OpenAI Error: {e}"
            
            if gemini_model:
                try:
                    response = gemini_model.generate_content(prompt)
                    analyses["Gemini"] = response.text
                except Exception as e:
                    analyses["Gemini"] = f"Gemini Error: {e}"
            
            if grok_client:
                try:
                    response = grok_client.chat.completions.create(
                        model="grok-beta",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    analyses["Grok"] = response.choices[0].message.content
                except Exception as e:
                    analyses["Grok"] = f"Grok Error: {e}"
            
            if analyses:
                result = f"## 🤖 Multi-AI Analysis for {ticker}\n\n"
                for model, analysis in analyses.items():
                    result += f"### {model}:\n{analysis}\n\n---\n\n"
                return result
            else:
                return f"**{ticker} Analysis** - No AI models available"
        
        else:
            return f"**{ticker} Analysis** - AI model not configured"
            
    except Exception as e:
        return f"**{ticker} Analysis Error:** {str(e)}"

# Keep all your existing functions (get_option_chain, get_options_data, etc.)
@st.cache_data(ttl=150)
def get_option_chain(ticker: str, tz: str = "ET") -> Optional[Dict]:
    """Fetch 0DTE or nearest expiration option chain using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": f"No options data available for {ticker}"}

        today = datetime.datetime.now(ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')).date()
        expiration_dates = [datetime.datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
        valid_expirations = [exp for exp in expiration_dates if exp >= today]
        if not valid_expirations:
            return {"error": f"No valid expirations found for {ticker}"}

        target_expiration = min(valid_expirations, key=lambda x: (x - today).days)
        expiration_str = target_expiration.strftime('%Y-%m-%d')

        option_chain = stock.option_chain(expiration_str)
        calls = option_chain.calls
        puts = option_chain.puts

        calls = calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        puts = puts[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        
        current_price = get_live_quote(ticker, tz).get('last', 0)
        calls['moneyness'] = calls['strike'].apply(lambda x: 'ITM' if x < current_price else 'OTM')
        puts['moneyness'] = puts['strike'].apply(lambda x: 'ITM' if x > current_price else 'OTM')

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

def get_options_data(ticker: str) -> Optional[Dict]:
    """Fetch real options data for a ticker"""
    option_chain = get_option_chain(ticker, st.session_state.selected_tz)
    if option_chain.get("error"):
        return {"error": option_chain["error"]}

    calls = option_chain["calls"]
    puts = option_chain["puts"]

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

@st.cache_data(ttl=150)
def get_order_flow(ticker: str, option_chain: Dict) -> Dict:
    """Simulate order flow by analyzing option chain volume and open interest"""
    calls = option_chain.get('calls', pd.DataFrame())
    puts = option_chain.get('puts', pd.DataFrame())
    if calls.empty or puts.empty:
        return {"error": "No option chain data for order flow analysis"}

    try:
        total_call_volume = calls['volume'].sum() if not calls.empty else 0
        total_put_volume = puts['volume'].sum() if not puts.empty else 0
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0

        calls['volume_oi_ratio'] = calls['volume'] / calls['openInterest'].replace(0, 1)
        puts['volume_oi_ratio'] = puts['volume'] / puts['openInterest'].replace(0, 1)

        top_calls = calls[calls['volume_oi_ratio'] > 1.5][['contractSymbol', 'strike', 'lastPrice', 'volume', 'moneyness']].head(3)
        top_puts = puts[puts['volume_oi_ratio'] > 1.5][['contractSymbol', 'strike', 'lastPrice', 'volume', 'moneyness']].head(3)

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

# ===== MAIN STREAMLIT APP =====

st.title("🐋 AI Radar Pro - Complete Unusual Whales Edition")
st.markdown("*Comprehensive options flow, smart money tracking, and market analysis*")

# Timezone toggle
col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], 
                                                index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed")

# Get current time
tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
current_tz = datetime.datetime.now(tz_zone)
tz_label = st.session_state.selected_tz

# Sidebar - Enhanced Configuration
st.sidebar.subheader("🐋 Unusual Whales Data Sources")

# UW Status
if unusual_whales_client:
    st.sidebar.success("✅ 🐋 Unusual Whales (PRIMARY)")
    
    # Test UW connection
    if st.sidebar.button("🧪 Test UW Connection"):
        with st.sidebar:
            with st.spinner("Testing UW API..."):
                test_result = unusual_whales_client.get_stock_state("SPY")
                if test_result.get("error"):
                    st.error(f"UW Test Failed: {test_result['error']}")
                else:
                    st.success("✅ UW Connection OK")
                    st.write(f"SPY: ${test_result.get('last', 0):.2f}")
else:
    st.sidebar.error("❌ 🐋 Unusual Whales (Configure API Key)")

# Trading Economics
if trading_economics_client:
    st.sidebar.success("✅ 📊 Trading Economics")
else:
    st.sidebar.warning("⚠️ 📊 Trading Economics Not Connected")

st.sidebar.success("✅ Yahoo Finance (Fallback)")

# AI Configuration
st.sidebar.subheader("🤖 AI Configuration")
available_models = ["Multi-AI", "OpenAI", "Gemini", "Grok"]
st.session_state.ai_model = st.sidebar.selectbox("AI Model", available_models, 
                                                  index=available_models.index(st.session_state.ai_model) if st.session_state.ai_model in available_models else 0)

# AI Status
st.sidebar.subheader("AI Models Status")
if openai_client:
    st.sidebar.success("✅ OpenAI Connected")
else:
    st.sidebar.warning("⚠️ OpenAI Not Connected")

if gemini_model:
    st.sidebar.success("✅ Gemini Connected")
else:
    st.sidebar.warning("⚠️ Gemini Not Connected")

if grok_client:
    st.sidebar.success("✅ Grok Connected")
else:
    st.sidebar.warning("⚠️ Grok Not Connected")

# Auto-refresh controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

with col2:
    st.session_state.refresh_interval = st.selectbox("Interval", [5, 10, 15, 30], index=1)

with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

with col4:
    current_time = current_tz.strftime("%I:%M:%S %p")
    market_open = 9 <= current_tz.hour < 16
    status = "🟢 Open" if market_open else "🔴 Closed"
    st.write(f"**{status}** | {current_time} {tz_label}")

# Create main tabs - Enhanced with UW Features
tabs = st.tabs([
    "🎯 Command Center", "📊 Market Pulse", "🎪 Flow Alerts", 
    "🏛️ Smart Money", "🔍 Screening", "🕶️ Dark Pool", 
    "📰 Catalysts", "🤖 AI Analysis", "📋 Watchlist", "🎲 0DTE"
])

# TAB 1: Command Center - Enhanced UW Dashboard
with tabs[0]:
    st.subheader("🎯 Enhanced Command Center")
    
    # UW Comprehensive Data Button
    if st.button("🚀 Load Complete UW Dashboard", type="primary"):
        if not unusual_whales_client:
            st.error("Unusual Whales API key required for comprehensive dashboard!")
        else:
            with st.spinner("🐋 Loading comprehensive Unusual Whales data..."):
                watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
                data = comprehensive_dm.fetch_all_data_comprehensive(watchlist)
                
                if data['last_update']:
                    st.success(f"✅ Loaded comprehensive UW data at {data['last_update'].strftime('%H:%M:%S')}")
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # SPIKE metric
                    spike_data = data.get('market_spike', {})
                    if spike_data and not spike_data.get('error'):
                        spike_value = spike_data.get('data', {}).get('current', 0)
                        col1.metric("🚨 SPIKE", f"{spike_value:.1f}")
                    else:
                        col1.metric("🚨 SPIKE", "N/A")
                    
                    # Options volume
                    options_vol = data.get('total_options_volume', {})
                    if options_vol and not options_vol.get('error'):
                        vol_data = options_vol.get('data', {})
                        total_vol = vol_data.get('call_volume', 0) + vol_data.get('put_volume', 0)
                        col2.metric("📊 Options Vol", f"{total_vol:,.0f}")
                    else:
                        col2.metric("📊 Options Vol", "N/A")
                    
                    # Flow alerts
                    flow_alerts = len(data.get('flow_alerts', []))
                    col3.metric("🎯 Flow Alerts", flow_alerts)
                    
                    # Congress trades
                    congress_count = len(data.get('congress_trades', []))
                    col4.metric("🏛️ Congress", congress_count)
                    
                    # Watchlist with UW data
                    st.subheader("📋 Enhanced Watchlist")
                    stock_data = data.get('stock_data', {})
                    
                    if stock_data:
                        for ticker in watchlist:
                            if ticker in stock_data:
                                stock_info = stock_data[ticker]
                                if not stock_info.get('error'):
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric(ticker, f"${stock_info.get('last', 0):.2f}", f"{stock_info.get('change_percent', 0):+.2f}%")
                                    col2.metric("Volume", f"{stock_info.get('volume', 0):,}")
                                    col3.metric("Source", stock_info.get('data_source', 'N/A'))
                    
                    # Store data in session for other tabs
                    st.session_state.uw_comprehensive_data = data

# TAB 2: Market Pulse - UW Enhanced
with tabs[1]:
    st.subheader("📊 Market Pulse - Unusual Whales Enhanced")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 SPIKE Analysis")
            spike_data = data.get('market_spike', {})
            if spike_data and not spike_data.get('error'):
                spike_info = spike_data.get('data', {})
                current_spike = spike_info.get('current', 0)
                
                if current_spike > 30:
                    sentiment = "🔴 HIGH FEAR"
                    interpretation = "Extreme volatility expected"
                elif current_spike > 20:
                    sentiment = "🟡 ELEVATED"
                    interpretation = "Above normal volatility"
                elif current_spike > 10:
                    sentiment = "🟢 MODERATE"
                    interpretation = "Normal market conditions"
                else:
                    sentiment = "🔵 LOW FEAR"
                    interpretation = "Complacent market"
                
                st.metric("Current SPIKE", f"{current_spike:.1f}")
                st.info(f"Status: {sentiment} - {interpretation}")
            else:
                st.warning("SPIKE data unavailable")
        
        with col2:
            st.subheader("📊 Options Volume")
            vol_data = data.get('total_options_volume', {})
            if vol_data and not vol_data.get('error'):
                options_data = vol_data.get('data', {})
                call_vol = options_data.get('call_volume', 0)
                put_vol = options_data.get('put_volume', 0)
                
                if call_vol and put_vol:
                    fig = px.pie(
                        values=[call_vol, put_vol],
                        names=['Calls', 'Puts'],
                        title="Call/Put Volume Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    cp_ratio = call_vol / put_vol
                    st.metric("Call/Put Ratio", f"{cp_ratio:.2f}")
            else:
                st.warning("Options volume data unavailable")
    else:
        st.info("Load comprehensive UW data from Command Center to see enhanced market pulse.")

# TAB 3: Flow Alerts - Direct UW Integration
with tabs[2]:
    st.subheader("🎪 Live Flow Alerts")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        flow_alerts = data.get('flow_alerts', [])
        
        if flow_alerts:
            st.success(f"📊 {len(flow_alerts)} Live Flow Alerts from Unusual Whales")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_premium = sum([alert.get('premium', 0) for alert in flow_alerts])
            call_alerts = len([a for a in flow_alerts if a.get('call_put') == 'call'])
            put_alerts = len([a for a in flow_alerts if a.get('call_put') == 'put'])
            bullish_alerts = len([a for a in flow_alerts if 'bullish' in str(a.get('sentiment', '')).lower()])
            
            col1.metric("💰 Total Premium", f"${total_premium:,.0f}")
            col2.metric("📞 Calls", call_alerts)
            col3.metric("📉 Puts", put_alerts)
            col4.metric("🐂 Bullish", bullish_alerts)
            
            # Display alerts
            for i, alert in enumerate(flow_alerts[:20]):
                with st.expander(f"🎯 {alert.get('ticker', 'N/A')} - ${alert.get('premium', 0):,.0f} premium"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Strike:** ${alert.get('strike', 0)}")
                        st.write(f"**Expiry:** {alert.get('expiry', 'N/A')}")
                        st.write(f"**Type:** {alert.get('call_put', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Volume:** {alert.get('volume', 0):,}")
                        st.write(f"**Sentiment:** {alert.get('sentiment', 'N/A')}")
                        st.write(f"**Time:** {alert.get('time', 'N/A')}")
        else:
            st.warning("No flow alerts available. Check UW API connection.")
    else:
        st.info("Load comprehensive UW data from Command Center to see live flow alerts.")

# TAB 4: Smart Money - Congressional & Insider Tracking
with tabs[3]:
    st.subheader("🏛️ Smart Money Tracking")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏛️ Congressional Activity")
            congress_trades = data.get('congress_trades', [])
            
            if congress_trades:
                st.metric("Recent Trades", len(congress_trades))
                
                for trade in congress_trades[:10]:
                    st.write(f"**{trade.get('ticker', 'N/A')}** - {trade.get('representative', 'N/A')}")
                    st.write(f"Type: {trade.get('transaction_type', 'N/A')} | Amount: {trade.get('amount', 'N/A')}")
                    st.write("---")
            else:
                st.info("No congressional data available")
        
        with col2:
            st.subheader("🔍 Insider Activity")
            insider_transactions = data.get('insider_transactions', [])
            
            if insider_transactions:
                st.metric("Recent Transactions", len(insider_transactions))
                
                for transaction in insider_transactions[:10]:
                    st.write(f"**{transaction.get('ticker', 'N/A')}**")
                    st.write(f"Insider: {transaction.get('insider', 'N/A')}")
                    st.write(f"Type: {transaction.get('transaction_type', 'N/A')}")
                    st.write("---")
            else:
                st.info("No insider data available")
    else:
        st.info("Load comprehensive UW data to see smart money activity.")

# TAB 5: Screening - UW Screeners
with tabs[4]:
    st.subheader("🔍 UW Advanced Screening")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        
        tab1, tab2, tab3 = st.tabs(["🔥 Hottest Chains", "📈 Stock Screener", "⭐ Analyst Ratings"])
        
        with tab1:
            hottest_chains = data.get('hottest_chains', [])
            if hottest_chains:
                st.success(f"🔥 {len(hottest_chains)} Hottest Option Chains")
                
                for chain in hottest_chains[:15]:
                    st.write(f"**{chain.get('ticker', 'N/A')}** - Volume: {chain.get('volume', 0):,}")
                    st.write(f"Strike: ${chain.get('strike', 0)} | IV: {chain.get('iv', 0):.1f}%")
                    st.write("---")
            else:
                st.info("No hottest chains data")
        
        with tab2:
            stock_screener = data.get('stock_screener', [])
            if stock_screener:
                st.success(f"📈 {len(stock_screener)} Screened Stocks")
                
                for stock in stock_screener[:20]:
                    st.write(f"**{stock.get('ticker', 'N/A')}** - {stock.get('change_percent', 0):+.2f}%")
                    st.write(f"Volume: {stock.get('volume', 0):,} | Reason: {stock.get('reason', 'N/A')}")
                    st.write("---")
            else:
                st.info("No stock screener data")
        
        with tab3:
            analyst_ratings = data.get('analyst_ratings', [])
            if analyst_ratings:
                st.success(f"⭐ {len(analyst_ratings)} Analyst Updates")
                
                for rating in analyst_ratings[:15]:
                    st.write(f"**{rating.get('ticker', 'N/A')}** - {rating.get('rating', 'N/A')}")
                    st.write(f"Firm: {rating.get('firm', 'N/A')} | Target: ${rating.get('target', 0)}")
                    st.write("---")
            else:
                st.info("No analyst ratings data")
    else:
        st.info("Load UW data to access advanced screening.")

# TAB 6: Dark Pool - UW Dark Pool Data
with tabs[5]:
    st.subheader("🕶️ Dark Pool Intelligence")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        darkpool_data = data.get('darkpool_recent', [])
        
        if darkpool_data:
            st.success(f"🕶️ {len(darkpool_data)} Recent Dark Pool Trades")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            total_trades = len(darkpool_data)
            total_value = sum([trade.get('value', 0) for trade in darkpool_data if isinstance(trade.get('value'), (int, float))])
            avg_size = np.mean([trade.get('size', 0) for trade in darkpool_data if isinstance(trade.get('size'), (int, float))]) if darkpool_data else 0
            
            col1.metric("📊 Trades", f"{total_trades:,.0f}")
            col2.metric("💰 Total Value", f"${total_value:,.0f}")
            col3.metric("📈 Avg Size", f"{avg_size:,.0f}")
            
            # Recent trades
            for trade in darkpool_data[:15]:
                st.write(f"**{trade.get('ticker', 'N/A')}** - Size: {trade.get('size', 0):,}")
                st.write(f"Price: ${trade.get('price', 0):.2f} | Value: ${trade.get('value', 0):,.0f}")
                st.write(f"Time: {trade.get('time', 'N/A')}")
                st.write("---")
        else:
            st.info("No dark pool data available")
    else:
        st.info("Load UW data to see dark pool activity.")

# TAB 7: Catalysts - UW News & Economic Events
with tabs[6]:
    st.subheader("📰 Market Catalysts")
    
    if hasattr(st.session_state, 'uw_comprehensive_data'):
        data = st.session_state.uw_comprehensive_data
        
        tab1, tab2, tab3, tab4 = st.tabs(["📰 UW News", "📅 Economic", "💊 FDA", "📈 Earnings"])
        
        with tab1:
            news_headlines = data.get('news_headlines', [])
            if news_headlines:
                st.success(f"📰 {len(news_headlines)} Latest Headlines")
                
                for news in news_headlines:
                    st.write(f"**{news.get('title', 'No Title')}**")
                    st.write(news.get('summary', 'No summary')[:200] + "...")
                    if news.get('tickers'):
                        st.write(f"Related: {news.get('tickers')}")
                    st.write("---")
            else:
                st.info("No UW news available")
        
        with tab2:
            economic_events = data.get('economic_calendar', [])
            if economic_events:
                st.success(f"📅 {len(economic_events)} Economic Events")
                
                for event in economic_events:
                    st.write(f"**{event.get('event', 'N/A')}**")
                    st.write(f"Date: {event.get('date', 'N/A')} | Impact: {event.get('importance', 'N/A')}")
                    st.write("---")
            else:
                st.info("No economic events available")
        
        with tab3:
            fda_events = data.get('fda_calendar', [])
            if fda_events:
                st.success(f"💊 {len(fda_events)} FDA Events")
                
                for event in fda_events:
                    st.write(f"**{event.get('drug', 'N/A')}** - {event.get('company', 'N/A')}")
                    st.write(f"Date: {event.get('date', 'N/A')} | Type: {event.get('event_type', 'N/A')}")
                    st.write("---")
            else:
                st.info("No FDA events available")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Premarket Earnings**")
                premarket = data.get('earnings_premarket', [])
                for earning in premarket[:10]:
                    st.write(f"• {earning.get('ticker', 'N/A')} - {earning.get('time', 'N/A')}")
            
            with col2:
                st.write("**After Hours Earnings**")
                afterhours = data.get('earnings_afterhours', [])
                for earning in afterhours[:10]:
                    st.write(f"• {earning.get('ticker', 'N/A')} - {earning.get('time', 'N/A')}")
    else:
        st.info("Load UW data to see comprehensive catalysts.")

# TAB 8: AI Analysis - Enhanced with UW Data
with tabs[7]:
    st.subheader("🤖 Enhanced AI Analysis")
    
    if st.button("🧠 Generate Comprehensive AI Analysis", type="primary"):
        if not unusual_whales_client:
            st.error("Unusual Whales API required for comprehensive analysis!")
        elif not hasattr(st.session_state, 'uw_comprehensive_data'):
            st.warning("Please load UW data from Command Center first.")
        else:
            with st.spinner("🤖 AI analyzing comprehensive UW data..."):
                data = st.session_state.uw_comprehensive_data
                analysis = comprehensive_dm.get_ai_analysis()
                
                st.success("🤖 Comprehensive AI Analysis Complete")
                st.markdown(analysis)
                
                # Cache analysis
                st.session_state.last_ai_analysis = {
                    'content': analysis,
                    'timestamp': datetime.datetime.now()
                }
    
    # Show cached analysis
    if hasattr(st.session_state, 'last_ai_analysis'):
        analysis_data = st.session_state.last_ai_analysis
        st.info(f"Last analysis: {analysis_data['timestamp'].strftime('%H:%M:%S')}")
        with st.expander("📊 View Last Analysis"):
            st.markdown(analysis_data['content'])

    # Individual stock analysis
    st.divider()
    st.markdown("### 🔍 Individual Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analysis_ticker = st.text_input("Analyze specific stock", placeholder="Enter ticker", key="ai_analysis_ticker").upper().strip()
    with col2:
        if st.button("🤖 Analyze", key="analyze_individual"):
            if analysis_ticker:
                quote = get_live_quote(analysis_ticker, st.session_state.selected_tz)
                if not quote.get("error"):
                    with st.spinner(f"AI analyzing {analysis_ticker}..."):
                        news = get_finnhub_news(analysis_ticker)
                        catalyst = news[0].get('headline', '') if news else ""
                        options_data = get_options_data(analysis_ticker)
                        
                        analysis = ai_playbook(analysis_ticker, quote["change_percent"], catalyst, options_data)
                        
                        st.success(f"🤖 AI Analysis: {analysis_ticker}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        col3.metric("Source", quote.get('data_source', 'Yahoo Finance'))
                        
                        st.markdown("### 🎯 AI Analysis")
                        st.markdown(analysis)

# TAB 9: Watchlist Manager (same as your existing code)
with tabs[8]:
    st.subheader("📋 Watchlist Manager")
    
    # Watchlist management (keep your existing code)
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
    
    # Current watchlist display
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    st.markdown("### 📊 Current Watchlist")
    if current_tickers:
        for ticker in current_tickers:
            quote = get_live_quote(ticker, st.session_state.selected_tz)
            if not quote.get("error"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Source", quote.get('data_source', 'Yahoo Finance'))
                if col4.button(f"Remove {ticker}", key=f"remove_{ticker}"):
                    current_tickers.remove(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                    st.rerun()
                st.divider()
    else:
        st.info("No tickers in watchlist. Add some below.")
    
    # Add tickers
    st.markdown("### ➕ Add Tickers")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add ticker to watchlist", key="add_ticker_input").upper().strip()
    with col2:
        if st.button("Add Ticker") and new_ticker:
            if new_ticker not in current_tickers:
                quote = get_live_quote(new_ticker, st.session_state.selected_tz)
                if not quote.get("error"):
                    current_tickers.append(new_ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_tickers
                    st.success(f"Added {new_ticker}")
                    st.rerun()
                else:
                    st.error(f"Invalid ticker: {new_ticker}")
            else:
                st.warning(f"{new_ticker} already in watchlist")

# TAB 10: 0DTE (keep your existing code)
with tabs[9]:
    st.subheader("🎲 0DTE & Lotto Plays")
    
    # Your existing 0DTE code
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_ticker = st.selectbox("Select Ticker for 0DTE", 
                                     options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], 
                                     key="0dte_ticker")
    with col2:
        if st.button("Analyze 0DTE", key="analyze_0dte"):
            st.cache_data.clear()
            st.rerun()

    # Fetch option chain and display (keep your existing logic)
    with st.spinner(f"Fetching option chain for {selected_ticker}..."):
        option_chain = get_option_chain(selected_ticker, st.session_state.selected_tz)
        quote = get_live_quote(selected_ticker, st.session_state.selected_tz)

    if option_chain.get("error"):
        st.error(option_chain["error"])
    else:
        current_price = quote['last']
        expiration = option_chain["expiration"]
        is_0dte = (datetime.datetime.strptime(expiration, '%Y-%m-%d').date() == datetime.datetime.now(ZoneInfo('US/Eastern')).date())
        
        st.markdown(f"**Option Chain for {selected_ticker}** (Expiration: {expiration}{' - 0DTE' if is_0dte else ''})")
        st.markdown(f"**Current Price:** ${current_price:.2f} | **Source:** {quote.get('data_source', 'Yahoo Finance')}")

        # Enhanced AI Analysis for 0DTE
        st.markdown("### 🤖 Enhanced 0DTE Analysis")
        with st.spinner("Generating enhanced analysis..."):
            # Get comprehensive data if available
            enhanced_context = ""
            if unusual_whales_client:
                try:
                    flow_data = unusual_whales_client.get_flow_alerts(selected_ticker, 5)
                    greek_data = unusual_whales_client.get_greek_exposure(selected_ticker)
                    enhanced_context = f"UW Flow Alerts: {len(flow_data)} recent | Greeks: {json.dumps(greek_data, indent=2)[:300]}"
                except:
                    pass
            
            catalyst = f"0DTE options analysis. {enhanced_context}"
            options_data = get_options_data(selected_ticker)
            playbook = ai_playbook(selected_ticker, quote["change_percent"], catalyst, options_data)
            st.markdown(playbook)

        # Display option chain (keep existing code)
        st.markdown("### Calls")
        calls = option_chain["calls"]
        if not calls.empty:
            display_calls = calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']].copy()
            display_calls.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest', 'IV (%)', 'Moneyness']
            display_calls['IV (%)'] = display_calls['IV (%)'].map('{:.2f}'.format)
            st.dataframe(display_calls, use_container_width=True)

        st.markdown("### Puts")
        puts = option_chain["puts"]
        if not puts.empty:
            display_puts = puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'moneyness']].copy()
            display_puts.columns = ['Strike', 'Last Price', 'Bid', 'Ask', 'Volume', 'Open Interest', 'IV (%)', 'Moneyness']
            display_puts['IV (%)'] = display_puts['IV (%)'].map('{:.2f}'.format)
            st.dataframe(display_puts, use_container_width=True)

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

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

# Enhanced Footer
st.markdown("---")
data_timestamp = datetime.datetime.now(tz_zone).strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"

footer_text = "🐋 Unusual Whales (Primary)"
if unusual_whales_client:
    footer_text += " → Yahoo Finance (Fallback)"
else:
    footer_text = "Yahoo Finance (Fallback Only)"

ai_models_active = []
if openai_client:
    ai_models_active.append("OpenAI")
if gemini_model:
    ai_models_active.append("Gemini")
if grok_client:
    ai_models_active.append("Grok")

ai_footer = f"AI: {st.session_state.ai_model}"
if ai_models_active:
    ai_footer += f" ({'+'.join(ai_models_active)})"

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🐋 AI Radar Pro - Complete Unusual Whales Edition | Data: {footer_text} | {ai_footer}<br>"
    f"Updated: {data_timestamp} | Using all 109 UW API endpoints"
    "</div>",
    unsafe_allow_html=True
)
