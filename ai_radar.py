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

# ===== IMPORT YOUR UNUSUAL WHALES MODULE =====
# Your UW API Functions (with typo fix)
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# === 1. STOCK DATA ===
def get_stock_state(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/stock-state"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# === 2. OPTION CHAIN ===
def get_option_chains(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/option-chains"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# === 3. UNUSUAL OPTION FLOW ===
def get_recent_flow(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-recent"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_flow_by_strike(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-per-strike"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_flow_by_expiry(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-per-expiry"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_historic_trades(ticker: str, limit=100, direction="all") -> dict:
    url = f"https://api.unusualwhales.com/api/historic_chains/{ticker}"
    params = {
        "limit": limit,
        "direction": direction
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# === 4. EARNINGS ===
def get_earnings_premarket() -> dict:
    url = "https://api.unusualwhales.com/api/earnings/premarket"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_earnings_afterhours() -> dict:
    url = "https://api.unusualwhales.com/api/earnings/afterhours"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_earnings_for_ticker(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/earnings/{ticker}"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# === 5. FLOW ALERTS (Fixed URL) ===
def get_flow_alerts(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-alerts"  # Fixed typo
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ===== END UW MODULE =====

# API Keys
try:
    TRADING_ECONOMICS_KEY = st.secrets.get("TRADING_ECONOMICS_KEY", "")
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROK_API_KEY = st.secrets.get("GROK_API_KEY", "")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    TWELVEDATA_KEY = st.secrets.get("TWELVEDATA_API_KEY", "")
except Exception as e:
    st.error(f"Error loading API keys: {e}")
    TRADING_ECONOMICS_KEY = ""

# ===== TRADING ECONOMICS API INTEGRATION =====
class TradingEconomicsClient:
    """Trading Economics API client for economic data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Radar-Pro/2.0"
        })
    
    def get_economic_calendar(self, days: int = 7) -> List[Dict]:
        """Get economic calendar events"""
        try:
            url = f"{self.base_url}/calendar"
            params = {
                "c": self.api_key,
                "d1": datetime.date.today().strftime("%Y-%m-%d"),
                "d2": (datetime.date.today() + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []
    
    def get_economic_indicators(self, country: str = "united states") -> List[Dict]:
        """Get key economic indicators"""
        try:
            url = f"{self.base_url}/indicators"
            params = {
                "c": self.api_key,
                "country": country
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Trading Economics"""
        try:
            url = f"{self.base_url}/markets/symbol/{symbol}"
            params = {"c": self.api_key}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"TE market data error: {str(e)}"}

# Initialize TE client
te_client = TradingEconomicsClient(TRADING_ECONOMICS_KEY) if TRADING_ECONOMICS_KEY else None

# Enhanced UW data transformation
def transform_uw_stock_data(uw_data: dict, ticker: str, tz: str = "ET") -> Dict:
    """Transform UW stock state data to our standard format"""
    try:
        tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
        tz_label = "ET" if tz == "ET" else "CT"
        
        # Extract data from UW response
        price = float(uw_data.get('price', uw_data.get('last', 0)))
        change = float(uw_data.get('change', 0))
        change_percent = float(uw_data.get('change_percent', 0))
        volume = int(uw_data.get('volume', 0))
        
        return {
            "last": price,
            "bid": float(uw_data.get('bid', price - 0.01)),
            "ask": float(uw_data.get('ask', price + 0.01)),
            "volume": volume,
            "change": change,
            "change_percent": change_percent,
            "premarket_change": float(uw_data.get('premarket_change_percent', 0)),
            "intraday_change": change_percent,  # UW main change is intraday
            "postmarket_change": float(uw_data.get('afterhours_change_percent', 0)),
            "previous_close": float(uw_data.get('previous_close', price - change)),
            "market_open": float(uw_data.get('open', 0)),
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "data_source": "Unusual Whales",
            "error": None,
            "raw_uw_data": uw_data
        }
    except Exception as e:
        return {"error": f"UW data transformation error: {str(e)}", "data_source": "Unusual Whales"}

# Fallback function for when UW fails
def get_stock_data_fallback(ticker: str, tz: str = "ET") -> Dict:
    """Fallback to yfinance when UW fails"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_1d = stock.history(period="1d", interval="1m", prepost=True)
        
        if hist_1d.empty:
            hist_1d = stock.history(period="1d", prepost=True)
        
        current_price = float(info.get('currentPrice', info.get('regularMarketPrice', hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0)))
        previous_close = info.get('previousClose', 0)
        
        tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
        tz_label = "ET" if tz == "ET" else "CT"
        
        return {
            "last": current_price,
            "bid": float(info.get('bid', current_price - 0.01)),
            "ask": float(info.get('ask', current_price + 0.01)),
            "volume": int(info.get('volume', hist_1d['Volume'].iloc[-1] if not hist_1d.empty else 0)),
            "change": current_price - previous_close if previous_close else 0,
            "change_percent": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            "premarket_change": 0,  # Simplified for fallback
            "intraday_change": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            "postmarket_change": 0,
            "previous_close": previous_close,
            "market_open": float(info.get('regularMarketOpen', 0)),
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "data_source": "Yahoo Finance (Fallback)",
            "error": None
        }
    except Exception as e:
        return {"error": f"Fallback error: {str(e)}", "data_source": "Fallback Failed"}

# Enhanced primary data function using your UW module
@st.cache_data(ttl=15)  # FASTER REFRESH: 15 seconds
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """Get live stock quote - UW PRIMARY using your module, then fallbacks"""
    
    # TRY YOUR UW MODULE FIRST (PRIMARY)
    try:
        uw_data = get_stock_state(ticker)
        if not uw_data.get("error") and uw_data:
            # Transform UW data to our format
            transformed_data = transform_uw_stock_data(uw_data, ticker, tz)
            if not transformed_data.get("error") and transformed_data.get("last", 0) > 0:
                return transformed_data
        else:
            st.warning(f"UW API returned error for {ticker}: {uw_data.get('error', 'Unknown error')} — using fallback")
    except Exception as e:
        st.warning(f"UW API error for {ticker}: {str(e)} — using fallback")
    
    # FALLBACK to yfinance if UW fails
    return get_stock_data_fallback(ticker, tz)

# Enhanced options data using your UW module
@st.cache_data(ttl=30)  # Faster refresh for options
def get_options_data_enhanced(ticker: str) -> Dict:
    """Enhanced options data using your UW module"""
    
    try:
        # Use your option chains function
        uw_options = get_option_chains(ticker)
        if not uw_options.get("error"):
            return {
                "data_source": "Unusual Whales",
                "options_data": uw_options,
                "error": None
            }
        else:
            st.warning(f"UW options error for {ticker}: {uw_options.get('error')}")
    except Exception as e:
        st.warning(f"UW options API error for {ticker}: {e}")
    
    # Fallback to yfinance options
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": f"No options data available for {ticker}"}
        
        # Get nearest expiration
        today = datetime.datetime.now().date()
        expiration_dates = [datetime.datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]
        valid_expirations = [exp for exp in expiration_dates if exp >= today]
        
        if not valid_expirations:
            return {"error": f"No valid expirations found for {ticker}"}
        
        target_expiration = min(valid_expirations, key=lambda x: (x - today).days)
        expiration_str = target_expiration.strftime('%Y-%m-%d')
        
        option_chain = stock.option_chain(expiration_str)
        
        return {
            "data_source": "Yahoo Finance (Fallback)",
            "calls": option_chain.calls,
            "puts": option_chain.puts,
            "expiration": expiration_str,
            "error": None
        }
    except Exception as e:
        return {"error": f"Options data error: {str(e)}"}

# Enhanced flow analysis using your UW module
def get_comprehensive_flow_analysis(ticker: str) -> Dict:
    """Comprehensive flow analysis using your UW module functions"""
    try:
        flow_data = {
            "recent_flow": get_recent_flow(ticker),
            "flow_by_strike": get_flow_by_strike(ticker),
            "flow_by_expiry": get_flow_by_expiry(ticker),
            "historic_trades": get_historic_trades(ticker, limit=50),
            "flow_alerts": get_flow_alerts(ticker),
            "data_source": "Unusual Whales"
        }
        
        # Count successful data retrievals
        successful_calls = sum(1 for key, value in flow_data.items() 
                              if key != "data_source" and not value.get("error"))
        
        flow_data["successful_endpoints"] = successful_calls
        flow_data["total_endpoints"] = 5
        
        return flow_data
    except Exception as e:
        return {"error": f"Flow analysis error: {str(e)}"}

# Enhanced earnings data using your UW module
@st.cache_data(ttl=600)  # Cache earnings for 10 minutes
def get_enhanced_earnings_data() -> Dict:
    """Get comprehensive earnings data using your UW module"""
    try:
        return {
            "premarket_earnings": get_earnings_premarket(),
            "afterhours_earnings": get_earnings_afterhours(),
            "data_source": "Unusual Whales"
        }
    except Exception as e:
        return {"error": f"Earnings data error: {str(e)}"}

# Core tickers and ETFs
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

ETF_TICKERS = [
    "SPY", "QQQ", "XLF", "XLE", "XLK", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC",
    "SPX", "NDX", "IWM", "IWF", "HOOY", "MSTY", "NVDY", "CONY"
]

# Initialize session state
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"Default": ["AAPL", "NVDA", "TSLA", "SPY", "AMD", "MSFT"]}
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 15  # Faster default refresh
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"

# Initialize AI clients
try:
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
    st.error(f"Error loading AI clients: {e}")

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
                    "content": "You are an expert trading analyst with access to real-time Unusual Whales data. Provide concise, actionable trading analysis with specific entry/exit levels and risk management."
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

grok_enhanced = GrokClient(GROK_API_KEY) if GROK_API_KEY else None

# Enhanced Technical Analysis with UW data
@st.cache_data(ttl=180)  # Cache for 3 minutes
def get_comprehensive_technical_analysis(ticker: str) -> Dict:
    """Enhanced technical analysis using UW and fallback data"""
    try:
        # Try to get enhanced data from UW first
        try:
            uw_data = get_stock_state(ticker)
            if not uw_data.get("error"):
                # Transform and use UW data for basic analysis
                transformed = transform_uw_stock_data(uw_data, ticker)
                if not transformed.get("error"):
                    current_price = transformed.get("last", 0)
                    change_pct = transformed.get("change_percent", 0)
                    volume = transformed.get("volume", 0)
                    
                    # Use yfinance for historical technical analysis
                    stock = yf.Ticker(ticker)
                    hist_3mo = stock.history(period="3mo")
                    
                    if not hist_3mo.empty:
                        indicators = calculate_indicators_with_uw(hist_3mo, transformed)
                        indicators["data_source"] = "UW + yfinance"
                        return indicators
        except Exception as e:
            st.warning(f"UW technical analysis error: {e}")
        
        # Fallback to standard yfinance analysis
        stock = yf.Ticker(ticker)
        hist_3mo = stock.history(period="3mo")
        
        if hist_3mo.empty:
            return {"error": "No historical data available"}
        
        indicators = calculate_standard_indicators(hist_3mo)
        indicators["data_source"] = "yfinance"
        return indicators
        
    except Exception as e:
        return {"error": f"Technical analysis error: {str(e)}"}

def calculate_indicators_with_uw(hist_df: pd.DataFrame, uw_data: Dict) -> Dict:
    """Calculate indicators enhanced with UW real-time data"""
    try:
        current_price = uw_data.get("last", hist_df['Close'].iloc[-1])
        
        # RSI calculation
        delta = hist_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Moving averages
        sma_20 = hist_df['Close'].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else hist_df['Close'].mean()
        sma_50 = hist_df['Close'].rolling(50).mean().iloc[-1] if len(hist_df) >= 50 else None
        
        # Bollinger Bands
        bb_period = 20
        if len(hist_df) >= bb_period:
            sma = hist_df['Close'].rolling(bb_period).mean()
            std = hist_df['Close'].rolling(bb_period).std()
            bb_upper = (sma + (std * 2)).iloc[-1]
            bb_lower = (sma - (std * 2)).iloc[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        else:
            bb_upper = bb_lower = bb_position = None
        
        # Trend analysis with UW data
        if sma_50 and current_price > sma_20 > sma_50:
            trend = "Strong Bullish"
        elif current_price > sma_20:
            trend = "Bullish"
        elif sma_50 and current_price < sma_20 < sma_50:
            trend = "Strong Bearish"
        elif current_price < sma_20:
            trend = "Bearish"
        else:
            trend = "Sideways"
        
        return {
            "current_price": current_price,
            "rsi": rsi,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_position": bb_position,
            "trend_analysis": trend,
            "volume_ratio": uw_data.get("volume", 0) / hist_df['Volume'].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else 1,
            "uw_enhanced": True,
            "change_percent": uw_data.get("change_percent", 0)
        }
    except Exception as e:
        return {"error": f"UW-enhanced indicator calculation error: {str(e)}"}

def calculate_standard_indicators(df: pd.DataFrame) -> Dict:
    """Standard indicator calculation for fallback"""
    try:
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Moving averages
        sma_20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].mean()
        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        
        current_price = df['Close'].iloc[-1]
        
        # Trend
        if sma_50 and current_price > sma_20 > sma_50:
            trend = "Strong Bullish"
        elif current_price > sma_20:
            trend = "Bullish"
        else:
            trend = "Bearish"
        
        return {
            "current_price": current_price,
            "rsi": rsi,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "trend_analysis": trend,
            "uw_enhanced": False
        }
    except Exception as e:
        return {"error": f"Standard indicator calculation error: {str(e)}"}

# Multi-AI Analysis System
class MultiAIAnalyzer:
    """Enhanced multi-AI analysis system with UW data integration"""
    
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
                temperature=0.3,
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

# Initialize Multi-AI Analyzer
multi_ai = MultiAIAnalyzer()

# Enhanced AI analysis with UW data using your module
def ai_playbook_enhanced(ticker: str, change: float, catalyst: str = "", uw_data: Optional[Dict] = None, flow_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook using your UW module data"""
    
    # Construct enhanced prompt with UW data
    if uw_data and not uw_data.get("error"):
        data_context = f"""
**UNUSUAL WHALES DATA for {ticker}:**
- Current Price: ${uw_data.get('last', 0):.2f}
- Change: {uw_data.get('change_percent', 0):+.2f}%
- Premarket: {uw_data.get('premarket_change', 0):+.2f}%
- Intraday: {uw_data.get('intraday_change', 0):+.2f}%
- After Hours: {uw_data.get('postmarket_change', 0):+.2f}%
- Volume: {uw_data.get('volume', 0):,}
- Bid/Ask: ${uw_data.get('bid', 0):.2f}/${uw_data.get('ask', 0):.2f}
- Source: {uw_data.get('data_source', 'UW')}
"""
    else:
        data_context = f"**DATA for {ticker}:** Change: {change:+.2f}%"
    
    # Add flow data context if available
    flow_context = ""
    if flow_data and not flow_data.get("error"):
        successful_endpoints = flow_data.get("successful_endpoints", 0)
        total_endpoints = flow_data.get("total_endpoints", 5)
        flow_context = f"""
**UW FLOW ANALYSIS:**
- Data endpoints active: {successful_endpoints}/{total_endpoints}
- Recent flow data available: {'✅' if not flow_data.get('recent_flow', {}).get('error') else '❌'}
- Strike-level flow: {'✅' if not flow_data.get('flow_by_strike', {}).get('error') else '❌'}
- Historic trades: {'✅' if not flow_data.get('historic_trades', {}).get('error') else '❌'}
- Flow alerts: {'✅' if not flow_data.get('flow_alerts', {}).get('error') else '❌'}
"""
    
    # Get technical analysis
    technical = get_comprehensive_technical_analysis(ticker)
    tech_context = ""
    if not technical.get("error"):
        tech_context = f"""
**TECHNICAL ANALYSIS:**
- RSI: {technical.get('rsi', 0):.1f}
- Trend: {technical.get('trend_analysis', 'Unknown')}
- SMA20: ${technical.get('sma_20', 0):.2f}
- Data Source: {technical.get('data_source', 'Standard')}
"""
        if technical.get('uw_enhanced'):
            tech_context += "- Enhanced with UW real-time data ✅"
    
    prompt = f"""
{data_context}

{flow_context}

{tech_context}

**Catalyst:** {catalyst if catalyst else "Market movement analysis"}

Based on this UNUSUAL WHALES enhanced data using comprehensive flow analysis, provide:

1. **Overall Assessment** (Bullish/Bearish/Neutral) with confidence (1-100)
2. **Trading Strategy** (Scalp/Day Trade/Swing/Position/Avoid)
3. **Entry Strategy**: Specific price levels and conditions
4. **Profit Targets**: 2-3 realistic levels with reasoning
5. **Risk Management**: Stop loss and position sizing
6. **Technical Outlook**: Key support/resistance levels
7. **Options Strategy**: Specific plays based on UW flow data
8. **Time Horizon**: Recommended holding period
9. **UW Flow Insights**: What the options flow suggests
10. **Risk Factors**: What could invalidate this setup

Keep analysis under 400 words but be specific and actionable.
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
            result = f"## 🐋 UW Flow-Enhanced Multi-AI Analysis for {ticker}\n\n"
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            return result
        else:
            return f"**{ticker} Analysis** - No AI models available"
    elif st.session_state.ai_model == "OpenAI":
        return multi_ai.analyze_with_openai(prompt)
    elif st.session_state.ai_model == "Gemini":
        return multi_ai.analyze_with_gemini(prompt)
    elif st.session_state.ai_model == "Grok":
        return multi_ai.analyze_with_grok(prompt)
    
    return "No AI model selected"

# Enhanced market data with TE integration
@st.cache_data(ttl=300)  # 5 minute cache for economic data
def get_economic_context() -> Dict:
    """Get economic context from Trading Economics"""
    if not te_client:
        return {"error": "Trading Economics not configured"}
    
    try:
        calendar = te_client.get_economic_calendar(7)  # Next 7 days
        indicators = te_client.get_economic_indicators()
        
        return {
            "calendar": calendar[:10],  # Top 10 events
            "indicators": indicators[:5],  # Top 5 indicators
            "data_source": "Trading Economics",
            "error": None
        }
    except Exception as e:
        return {"error": f"TE error: {str(e)}"}

# Enhanced news with catalyst analysis
@st.cache_data(ttl=300)
def get_enhanced_news() -> List[Dict]:
    """Get enhanced news"""
    all_news = []
    
    # Get standard news
    if FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                news_data = response.json()[:10]
                for item in news_data:
                    all_news.append({
                        "title": item.get("headline", ""),
                        "summary": item.get("summary", ""),
                        "source": "Finnhub",
                        "url": item.get("url", ""),
                        "datetime": item.get("datetime", 0)
                    })
        except Exception as e:
            st.warning(f"News API error: {e}")
    
    return all_news

# Main app
st.title("🔥 AI Radar Pro — UW Flow Enhanced")

# Timezone toggle
col_tz, _ = st.columns([1, 10])
with col_tz:
    st.session_state.selected_tz = st.selectbox("TZ:", ["ET", "CT"], 
                                                index=0 if st.session_state.selected_tz == "ET" else 1, 
                                                label_visibility="collapsed")

# Enhanced sidebar with UW and TE status
st.sidebar.subheader("🐋 UW MODULE STATUS")

# Test your UW module
if st.sidebar.button("🧪 Test UW Module"):
    test_ticker = "AAPL"
    
    # Test stock state
    st.sidebar.write("**Testing stock_state:**")
    stock_test = get_stock_state(test_ticker)
    if stock_test.get("error"):
        st.sidebar.error(f"❌ Stock State: {stock_test['error']}")
    else:
        st.sidebar.success(f"✅ Stock State: ${stock_test.get('price', 0):.2f}")
    
    # Test options
    st.sidebar.write("**Testing option_chains:**")
    options_test = get_option_chains(test_ticker)
    if options_test.get("error"):
        st.sidebar.error(f"❌ Options: {options_test['error']}")
    else:
        st.sidebar.success("✅ Options: Data retrieved")
    
    # Test flow
    st.sidebar.write("**Testing recent_flow:**")
    flow_test = get_recent_flow(test_ticker)
    if flow_test.get("error"):
        st.sidebar.error(f"❌ Flow: {flow_test['error']}")
    else:
        st.sidebar.success("✅ Flow: Data retrieved")

# UW Functions Status
st.sidebar.write("**Available UW Functions:**")
st.sidebar.write("✅ get_stock_state")
st.sidebar.write("✅ get_option_chains")
st.sidebar.write("✅ get_recent_flow")
st.sidebar.write("✅ get_flow_by_strike")
st.sidebar.write("✅ get_flow_by_expiry")
st.sidebar.write("✅ get_historic_trades")
st.sidebar.write("✅ get_earnings_premarket")
st.sidebar.write("✅ get_earnings_afterhours")
st.sidebar.write("✅ get_earnings_for_ticker")
st.sidebar.write("✅ get_flow_alerts (URL fixed)")

# TE Status
if te_client:
    st.sidebar.success("✅ Trading Economics Connected")
else:
    st.sidebar.warning("⚠️ Trading Economics Not Connected")

# AI Models
st.sidebar.subheader("🤖 AI MODELS")
available_models = ["Multi-AI"] + multi_ai.get_available_models()
st.session_state.ai_model = st.sidebar.selectbox("AI Model", available_models, 
                                                  index=available_models.index(st.session_state.ai_model) if st.session_state.ai_model in available_models else 0)

# Enhanced refresh controls
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
    tz_zone = ZoneInfo('US/Eastern') if st.session_state.selected_tz == "ET" else ZoneInfo('US/Central')
    current_tz = datetime.datetime.now(tz_zone)
    current_time = current_tz.strftime("%I:%M:%S %p")
    market_open = 9 <= current_tz.hour < 16
    status = "🟢 Open" if market_open else "🔴 Closed"
    st.write(f"**{status}** | {current_time} {st.session_state.selected_tz}")

# Create tabs
tabs = st.tabs([
    "📊 Live Quotes (UW)", 
    "📋 Watchlist Manager", 
    "🐋 UW Flow Scanner", 
    "📈 Market Analysis", 
    "🤖 AI Playbooks",
    "🎯 UW Earnings",
    "📊 Economic Calendar",
    "🎲 0DTE & Options"
])

# TAB 1: Enhanced Live Quotes with your UW module
with tabs[0]:
    st.subheader("📊 UW Module-Enhanced Real-Time Quotes")
    
    # Show data source priority
    st.info("🐋 **Primary Source:** Your UW Module | **Fallback:** Yahoo Finance")
    
    # Search any ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        search_ticker = st.text_input("🔍 Search Any Stock", placeholder="Enter ticker", key="search_quotes_uw").upper().strip()
    with col2:
        search_quotes = st.button("Get UW Quote", key="search_quotes_uw_btn")
    
    if search_quotes and search_ticker:
        with st.spinner(f"Getting UW-enhanced quote for {search_ticker}..."):
            quote = get_live_quote(search_ticker, st.session_state.selected_tz)
            
            if not quote.get("error"):
                st.success(f"✅ {search_ticker} Quote - Source: {quote.get('data_source', 'Unknown')}")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(search_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.metric("Volume", f"{quote['volume']:,}")
                col4.metric("Source", quote.get('data_source', 'Unknown'))
                
                # Enhanced session data
                st.markdown("#### 🕐 Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
                # Show raw UW data if available
                if quote.get('raw_uw_data') and st.checkbox(f"Show Raw UW Data for {search_ticker}"):
                    st.json(quote['raw_uw_data'])
                
                # UW-Enhanced Analysis
                if st.button(f"🐋 UW Enhanced Analysis", key=f"uw_analysis_{search_ticker}"):
                    with st.spinner("Running UW flow-enhanced analysis..."):
                        flow_data = get_comprehensive_flow_analysis(search_ticker)
                        analysis = ai_playbook_enhanced(search_ticker, quote['change_percent'], "User-requested analysis", quote, flow_data)
                        st.markdown("### 🐋 UW Flow-Enhanced AI Analysis")
                        st.markdown(analysis)
                
                st.divider()
            else:
                st.error(f"Error getting quote for {search_ticker}: {quote['error']}")
    
    # Watchlist with UW data
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    st.markdown("### 📋 Your Watchlist (UW-Enhanced)")
    
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab.")
    else:
        for ticker in tickers:
            quote = get_live_quote(ticker, st.session_state.selected_tz)
            
            if quote.get("error"):
                st.error(f"{ticker}: {quote['error']}")
                continue
            
            with st.container():
                # Main row
                col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                
                # Price with enhanced change tracking
                price_change_color = "🟢" if quote['change_percent'] > 0 else "🔴" if quote['change_percent'] < 0 else "⚪"
                col1.metric(f"{price_change_color} {ticker}", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                
                col2.write("**Bid/Ask**")
                col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                
                col3.write("**Volume**")
                col3.write(f"{quote['volume']:,}")
                col3.caption(f"Source: {quote.get('data_source', 'Unknown')}")
                
                # Enhanced session breakdown
                sess_col1, sess_col2, sess_col3, sess_col4 = st.columns([2, 2, 2, 4])
                sess_col1.caption(f"**PM:** {quote['premarket_change']:+.2f}%")
                sess_col2.caption(f"**Day:** {quote['intraday_change']:+.2f}%")
                sess_col3.caption(f"**AH:** {quote['postmarket_change']:+.2f}%")
                
                # UW-enhanced analysis button for significant movers
                if abs(quote['change_percent']) >= 2.0:
                    if col4.button(f"🐋 UW Analysis", key=f"uw_watchlist_{ticker}"):
                        with st.spinner(f"UW flow-enhanced analysis for {ticker}..."):
                            flow_data = get_comprehensive_flow_analysis(ticker)
                            analysis = ai_playbook_enhanced(ticker, quote['change_percent'], "Significant movement detected", quote, flow_data)
                            st.success(f"🐋 UW Flow-Enhanced Analysis: {ticker}")
                            st.markdown(analysis)
                
                st.divider()

# TAB 2: Watchlist Manager (simplified)
with tabs[1]:
    st.subheader("📋 Watchlist Manager")
    
    # Add ticker
    col1, col2 = st.columns([3, 1])
    with col1:
        add_ticker = st.text_input("Add ticker", placeholder="Enter ticker", key="add_ticker_input").upper().strip()
    with col2:
        if st.button("Add", key="add_ticker_btn") and add_ticker:
            current_list = st.session_state.watchlists[st.session_state.active_watchlist]
            if add_ticker not in current_list:
                current_list.append(add_ticker)
                st.success(f"✅ Added {add_ticker}")
                st.rerun()
    
    # Current watchlist
    current_tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    if current_tickers:
        st.write("**Current Watchlist:**")
        for ticker in current_tickers:
            col1, col2 = st.columns([3, 1])
            col1.write(ticker)
            if col2.button("Remove", key=f"remove_{ticker}"):
                current_tickers.remove(ticker)
                st.rerun()

# TAB 3: UW Flow Scanner using your module
with tabs[2]:
    st.subheader("🐋 UW Flow Scanner (Your Module)")
    
    st.success("✅ Using your Unusual Whales module functions")
    
    # Stock-specific comprehensive flow analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        flow_ticker = st.text_input("Analyze comprehensive flow for stock", placeholder="Enter ticker", key="flow_ticker").upper().strip()
    with col2:
        analyze_flow = st.button("Analyze Flow", key="analyze_flow_btn")
    
    if analyze_flow and flow_ticker:
        with st.spinner(f"Getting comprehensive UW flow for {flow_ticker}..."):
            quote = get_live_quote(flow_ticker, st.session_state.selected_tz)
            flow_data = get_comprehensive_flow_analysis(flow_ticker)
            
            if not quote.get("error"):
                st.success(f"🐋 {flow_ticker} Comprehensive Flow Analysis")
                
                # Current price context
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("UW Endpoints", f"{flow_data.get('successful_endpoints', 0)}/{flow_data.get('total_endpoints', 5)}")
                
                # Individual flow data sections
                col1, col2 = st.columns(2)
                
                with col1:
                    # Recent Flow
                    st.markdown("#### 🌊 Recent Flow")
                    recent_flow = flow_data.get("recent_flow", {})
                    if recent_flow.get("error"):
                        st.error(f"Error: {recent_flow['error']}")
                    else:
                        st.success("✅ Recent flow data retrieved")
                        with st.expander("View Recent Flow Data"):
                            st.json(recent_flow)
                    
                    # Flow by Strike
                    st.markdown("#### 🎯 Flow by Strike")
                    strike_flow = flow_data.get("flow_by_strike", {})
                    if strike_flow.get("error"):
                        st.error(f"Error: {strike_flow['error']}")
                    else:
                        st.success("✅ Strike-level flow data retrieved")
                        with st.expander("View Strike Flow Data"):
                            st.json(strike_flow)
                
                with col2:
                    # Flow by Expiry
                    st.markdown("#### 📅 Flow by Expiry")
                    expiry_flow = flow_data.get("flow_by_expiry", {})
                    if expiry_flow.get("error"):
                        st.error(f"Error: {expiry_flow['error']}")
                    else:
                        st.success("✅ Expiry-level flow data retrieved")
                        with st.expander("View Expiry Flow Data"):
                            st.json(expiry_flow)
                    
                    # Flow Alerts
                    st.markdown("#### 🚨 Flow Alerts")
                    flow_alerts = flow_data.get("flow_alerts", {})
                    if flow_alerts.get("error"):
                        st.error(f"Error: {flow_alerts['error']}")
                    else:
                        st.success("✅ Flow alerts retrieved")
                        with st.expander("View Flow Alerts"):
                            st.json(flow_alerts)
                
                # Historic Trades
                st.markdown("#### 📈 Historic Trades")
                historic_trades = flow_data.get("historic_trades", {})
                if historic_trades.get("error"):
                    st.error(f"Error: {historic_trades['error']}")
                else:
                    st.success("✅ Historic trades data retrieved")
                    chains = historic_trades.get("chains", [])
                    if chains:
                        st.info(f"Found {len(chains)} historic trades")
                        with st.expander("View Historic Trades"):
                            st.json(chains[:10])  # Show first 10
                
                # UW Flow-Enhanced AI Analysis
                if st.button(f"🤖 AI Flow Analysis", key=f"ai_flow_{flow_ticker}"):
                    analysis = ai_playbook_enhanced(flow_ticker, quote['change_percent'], "Comprehensive flow analysis", quote, flow_data)
                    st.markdown("### 🤖 AI Flow Analysis")
                    st.markdown(analysis)
            else:
                st.error(f"Error getting data for {flow_ticker}: {quote['error']}")

# TAB 4: Market Analysis with TE
with tabs[3]:
    st.subheader("📈 Enhanced Market Analysis")
    
    # UW-enhanced market analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        analysis_ticker = st.text_input("Analyze specific stock", placeholder="Enter ticker", key="market_analysis").upper().strip()
    with col2:
        analyze_market = st.button("Analyze Stock", key="analyze_market_btn")
    
    if analyze_market and analysis_ticker:
        with st.spinner(f"UW-enhanced analysis for {analysis_ticker}..."):
            quote = get_live_quote(analysis_ticker, st.session_state.selected_tz)
            
            if not quote.get("error"):
                flow_data = get_comprehensive_flow_analysis(analysis_ticker)
                catalyst = f"Market analysis request. UW endpoints: {flow_data.get('successful_endpoints', 0)}/{flow_data.get('total_endpoints', 5)}"
                analysis = ai_playbook_enhanced(analysis_ticker, quote['change_percent'], catalyst, quote, flow_data)
                
                st.success(f"🐋 UW Flow-Enhanced Analysis: {analysis_ticker}")
                
                # Price metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Source", quote.get('data_source', 'Unknown'))
                col4.metric("UW Endpoints", f"{flow_data.get('successful_endpoints', 0)}/{flow_data.get('total_endpoints', 5)}")
                
                st.markdown("### 🤖 AI Analysis")
                st.markdown(analysis)
            else:
                st.error(f"Error analyzing {analysis_ticker}: {quote['error']}")

# TAB 5: Enhanced AI Playbooks
with tabs[4]:
    st.subheader("🤖 UW Flow-Enhanced AI Playbooks")
    
    st.info(f"🤖 AI Mode: **{st.session_state.ai_model}** | 🐋 Enhanced with your UW module")
    
    # UW flow-enhanced search
    col1, col2 = st.columns([3, 1])
    with col1:
        playbook_ticker = st.text_input("Generate UW flow-enhanced playbook", placeholder="Enter ticker", key="playbook_uw").upper().strip()
    with col2:
        generate_playbook = st.button("Generate Playbook", key="generate_playbook_btn")
    
    if generate_playbook and playbook_ticker:
        with st.spinner(f"Generating UW flow-enhanced playbook for {playbook_ticker}..."):
            quote = get_live_quote(playbook_ticker, st.session_state.selected_tz)
            
            if not quote.get("error"):
                flow_data = get_comprehensive_flow_analysis(playbook_ticker)
                analysis = ai_playbook_enhanced(playbook_ticker, quote['change_percent'], "Comprehensive playbook request", quote, flow_data)
                
                st.success(f"🐋 UW Flow-Enhanced Playbook: {playbook_ticker}")
                
                # Enhanced metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Data Source", quote.get('data_source', 'Unknown'))
                col4.metric("UW Endpoints", f"{flow_data.get('successful_endpoints', 0)}/{flow_data.get('total_endpoints', 5)}")
                
                st.markdown("### 🤖 UW Flow-Enhanced AI Playbook")
                st.markdown(analysis)
                
                # Show flow data availability
                if flow_data.get('successful_endpoints', 0) > 0:
                    st.success(f"✅ Analysis enhanced with {flow_data['successful_endpoints']} UW flow endpoints")
                else:
                    st.info("ℹ️ Analysis using standard data (UW flow endpoints unavailable)")
            else:
                st.error(f"Error generating playbook for {playbook_ticker}: {quote['error']}")

# TAB 6: UW Earnings using your module
with tabs[5]:
    st.subheader("🎯 UW Earnings Data")
    
    st.success("✅ Using your UW earnings functions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📅 Premarket Earnings", type="primary"):
            with st.spinner("Getting premarket earnings from UW..."):
                premarket = get_earnings_premarket()
                if premarket.get("error"):
                    st.error(f"Error: {premarket['error']}")
                else:
                    st.success("✅ Premarket earnings retrieved")
                    st.json(premarket)
    
    with col2:
        if st.button("🌙 After Hours Earnings", type="primary"):
            with st.spinner("Getting after hours earnings from UW..."):
                afterhours = get_earnings_afterhours()
                if afterhours.get("error"):
                    st.error(f"Error: {afterhours['error']}")
                else:
                    st.success("✅ After hours earnings retrieved")
                    st.json(afterhours)
    
    with col3:
        earnings_ticker = st.text_input("Get earnings for ticker", key="earnings_ticker").upper().strip()
        if st.button("Get Earnings", key="get_earnings_btn") and earnings_ticker:
            with st.spinner(f"Getting earnings for {earnings_ticker}..."):
                earnings = get_earnings_for_ticker(earnings_ticker)
                if earnings.get("error"):
                    st.error(f"Error: {earnings['error']}")
                else:
                    st.success(f"✅ Earnings for {earnings_ticker} retrieved")
                    st.json(earnings)

# TAB 7: Economic Calendar
with tabs[6]:
    st.subheader("📊 Economic Calendar & Market Context")
    
    if te_client:
        if st.button("📅 Get Economic Calendar", type="primary"):
            with st.spinner("Getting economic calendar from Trading Economics..."):
                econ_data = get_economic_context()
                
                if not econ_data.get("error"):
                    st.success("📅 Economic Calendar Retrieved")
                    
                    if econ_data["calendar"]:
                        for event in econ_data["calendar"]:
                            importance = event.get('Importance', 'Unknown')
                            importance_emoji = "🔴" if importance == "High" else "🟡" if importance == "Medium" else "🟢"
                            
                            with st.expander(f"{importance_emoji} {event.get('Event', 'Unknown')} - {event.get('Country', 'Unknown')}"):
                                st.json(event)
                else:
                    st.error(f"Trading Economics error: {econ_data['error']}")
    else:
        st.warning("⚠️ Trading Economics not configured")

# TAB 8: Enhanced Options using your UW module
with tabs[7]:
    st.subheader("🎲 UW Module-Enhanced Options & 0DTE")
    
    st.success("✅ UW options data available via your module")
    
    # Ticker selection
    col1, col2 = st.columns([3, 1])
    with col1:
        options_ticker = st.selectbox("Select ticker for options analysis", 
                                    options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], 
                                    key="options_ticker_uw")
    with col2:
        analyze_options = st.button("Analyze Options", key="analyze_options_uw")
    
    if analyze_options:
        with st.spinner(f"Getting UW options data for {options_ticker}..."):
            quote = get_live_quote(options_ticker, st.session_state.selected_tz)
            options_data = get_options_data_enhanced(options_ticker)
            flow_data = get_comprehensive_flow_analysis(options_ticker)
            
            if not quote.get("error"):
                st.success(f"🐋 Options Analysis: {options_ticker} - Source: {quote.get('data_source', 'Unknown')}")
                
                # Price context
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Options Source", options_data.get('data_source', 'Unknown'))
                col4.metric("Flow Endpoints", f"{flow_data.get('successful_endpoints', 0)}/{flow_data.get('total_endpoints', 5)}")
                
                # UW options chain
                if options_data.get('options_data'):
                    st.markdown("### 📊 UW Options Chain")
                    with st.expander("UW Options Data"):
                        st.json(options_data['options_data'])
                
                # Flow analysis summary
                if flow_data.get('successful_endpoints', 0) > 0:
                    st.markdown("### 🌊 Flow Analysis Summary")
                    st.info(f"🐋 {flow_data['successful_endpoints']} UW flow endpoints active")
                    
                    if not flow_data.get('recent_flow', {}).get('error'):
                        st.success("✅ Recent flow data available")
                    if not flow_data.get('flow_alerts', {}).get('error'):
                        st.success("✅ Flow alerts available")
                
                # UW flow-enhanced AI analysis
                if st.button(f"🤖 UW Options AI Analysis", key=f"options_ai_{options_ticker}"):
                    options_context = f"Comprehensive options analysis. UW flow endpoints: {flow_data.get('successful_endpoints', 0)}"
                    analysis = ai_playbook_enhanced(options_ticker, quote['change_percent'], options_context, quote, flow_data)
                    st.markdown("### 🤖 UW Flow-Enhanced Options Analysis")
                    st.markdown(analysis)
            else:
                st.error(f"Error getting options data for {options_ticker}: {quote['error']}")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🔥 AI Radar Pro Enhanced | 🐋 Your UW Module Integrated | 📊 Trading Economics"
    f" | AI: {st.session_state.ai_model} | Refresh: {st.session_state.refresh_interval}s"
    "</div>",
    unsafe_allow_html=True
)
