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
    st.session_state.refresh_interval = 30
if "selected_tz" not in st.session_state:
    st.session_state.selected_tz = "ET"  # Default to ET
if "etf_list" not in st.session_state:
    st.session_state.etf_list = list(ETF_TICKERS)
if "data_source" not in st.session_state:
    st.session_state.data_source = "Unusual Whales"  # Primary data source now UW
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"  # Default to multi-AI
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()
if "live_data_cache" not in st.session_state:
    st.session_state.live_data_cache = {}

# API Keys
try:
    UNUSUAL_WHALES_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROK_API_KEY = st.secrets.get("GROK_API_KEY", "")
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

# =============================================================================
# UNUSUAL WHALES API CLIENT - PRIMARY DATA SOURCE
# =============================================================================

class UnusualWhalesClient:
    """Enhanced Unusual Whales API client for comprehensive market data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/plain"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return {"data": response.json(), "error": None}
        except requests.exceptions.RequestException as e:
            return {"data": None, "error": f"UW API Error: {str(e)}"}
        except Exception as e:
            return {"data": None, "error": f"UW Processing Error: {str(e)}"}
    
    # =================================================================
    # STOCK DATA METHODS
    # =================================================================
    
    def get_stock_state(self, ticker: str) -> Dict:
        """Get real-time stock state using UW stock-state endpoint"""
        endpoint = f"/api/stock/{ticker}/stock-state"
        
        result = self._make_request(endpoint)
        if result["error"]:
            return {"error": result["error"]}
        
        try:
            response_data = result["data"]
            if response_data and "data" in response_data:
                # Extract the actual data from the nested structure
                data = response_data["data"]
                
                # Parse UW stock state response - all prices come as strings
                close = float(data.get("close", "0"))
                open_price = float(data.get("open", str(close)))
                high = float(data.get("high", str(close)))
                low = float(data.get("low", str(close)))
                volume = int(data.get("volume", 0))
                total_volume = int(data.get("total_volume", volume))
                prev_close = float(data.get("prev_close", str(close)))
                market_time = data.get("market_time", "market")
                tape_time = data.get("tape_time", "")
                
                # Calculate changes
                change_dollar = close - prev_close
                change_percent = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                # Estimate bid/ask spread based on daily range
                daily_range = high - low
                spread_estimate = max(0.01, daily_range * 0.1)  # At least 1 cent, or 10% of daily range
                bid_estimate = close - (spread_estimate / 2)
                ask_estimate = close + (spread_estimate / 2)
                
                return {
                    "last": close,
                    "bid": bid_estimate,
                    "ask": ask_estimate,
                    "volume": volume,
                    "total_volume": total_volume,
                    "change": change_dollar,
                    "change_percent": change_percent,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "previous_close": prev_close,
                    "market_time": market_time,
                    "tape_time": tape_time,
                    "data_source": "Unusual Whales",
                    "error": None
                }
            else:
                return {"error": "No stock state data found for ticker"}
        except Exception as e:
            return {"error": f"Error parsing stock state data: {str(e)}"}
    
    def get_flow_alerts(self, ticker: str = None) -> Dict:
        """Get options flow alerts"""
        endpoint = "/api/option-trades/flow-alerts"
        params = {
            "all_opening": "true",
            "is_floor": "true", 
            "is_sweep": "true",
            "is_call": "true",
            "is_put": "true",
            "is_ask_side": "true",
            "is_bid_side": "true",
            "is_otm": "true"
        }
        if ticker:
            # Use stock-specific flow alerts
            endpoint = f"/api/stock/{ticker}/flow-alerts"
            params = {"is_ask_side": "true", "is_bid_side": "true"}
        
        return self._make_request(endpoint, params)
    
    def get_options_volume(self, ticker: str, limit: int = 1) -> Dict:
        """Get options volume data for ticker"""
        endpoint = f"/api/stock/{ticker}/options-volume"
        params = {"limit": limit}
        return self._make_request(endpoint, params)
    
    def get_hottest_chains(self, date: str = None, limit: int = 50) -> Dict:
        """Get hottest option chains"""
        endpoint = "/api/screener/option-contracts"
        params = {
            "limit": limit,
            "is_otm": "false"  # Include ITM contracts too
        }
        if date:
            params["date"] = date
        return self._make_request(endpoint, params)
    
    def get_stock_flow_recent(self, ticker: str) -> Dict:
        """Get recent options flow for stock"""
        endpoint = f"/api/stock/{ticker}/flow-recent"
        return self._make_request(endpoint)
    
    def get_greek_exposure(self, ticker: str, date: str = None) -> Dict:
        """Get Greek exposure for stock"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = f"/api/stock/{ticker}/greek-exposure"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_greek_exposure_by_expiry(self, ticker: str, date: str = None) -> Dict:
        """Get Greek exposure by expiry"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = f"/api/stock/{ticker}/greek-exposure/expiry"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_flow_per_strike(self, ticker: str, date: str = None) -> Dict:
        """Get flow per strike"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = f"/api/stock/{ticker}/flow-per-strike"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_atm_chains(self, ticker: str, expirations: List[str] = None) -> Dict:
        """Get at-the-money option chains with enhanced error handling"""
        endpoint = f"/api/stock/{ticker.upper()}/atm-chains"
        
        print(f'Fetching ATM chains from:', endpoint)
        
        # If no expirations provided, get current and next Friday
        if not expirations:
            today = datetime.date.today()
            # Find next Friday (most common options expiration)
            days_ahead = 4 - today.weekday()  # Friday is weekday 4
            if days_ahead <= 0:  # Target next Friday
                days_ahead += 7
            next_friday = today + datetime.timedelta(days_ahead)
            
            # Also include the Friday after that
            following_friday = next_friday + datetime.timedelta(7)
            
            expirations = [
                next_friday.strftime("%Y-%m-%d"),
                following_friday.strftime("%Y-%m-%d")
            ]
        
        params = {
            "expirations[]": expirations
        }
        
        result = self._make_request(endpoint, params)
        
        if result["error"]:
            print(f"ATM chains error for {ticker}: {result['error']}")
            return {"error": result["error"]}
        
        # Enhanced data processing
        try:
            response_data = result["data"]
            print(f"ATM chains raw response for {ticker}:", response_data)
            
            if response_data and "data" in response_data:
                chains_data = response_data["data"]
                if isinstance(chains_data, list) and len(chains_data) > 0:
                    print(f"Successfully got {len(chains_data)} ATM chains for {ticker}")
                    return {"data": chains_data, "error": None}
                else:
                    print(f"Empty ATM chains array for {ticker}")
                    return {"data": [], "error": "No ATM chains data available"}
            else:
                print(f"No ATM chains data in response for {ticker}")
                return {"data": [], "error": "No ATM chains data found"}
                    
        except Exception as e:
            error_msg = f"Error processing ATM chains for {ticker}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    # =================================================================
    # MARKET DATA METHODS
    # =================================================================
    
    def get_market_tide(self, date: str = None) -> Dict:
        """Get market tide data"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/market/market-tide"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_oi_change(self, date: str = None) -> Dict:
        """Get open interest changes"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/market/oi-change"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_spike_data(self, date: str = None) -> Dict:
        """Get spike detection data"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/market/spike"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_top_net_impact(self, date: str = None) -> Dict:
        """Get top net impact trades"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/market/top-net-impact"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_sector_etfs(self) -> Dict:
        """Get sector ETF data"""
        endpoint = "/api/market/sector-etfs"
        return self._make_request(endpoint)
    
    def get_insider_trades(self) -> Dict:
        """Get insider buy/sells"""
        endpoint = "/api/market/insider-buy-sells"
        return self._make_request(endpoint)
    
    def get_economic_calendar(self) -> Dict:
        """Get economic calendar"""
        endpoint = "/api/market/economic-calendar"
        return self._make_request(endpoint)
    
    def get_fda_calendar(self, ticker: str = None) -> Dict:
        """Get FDA calendar"""
        endpoint = "/api/market/fda-calendar"
        params = {"ticker": ticker} if ticker else {}
        return self._make_request(endpoint, params)
    
    # =================================================================
    # OPTIONS SPECIFIC METHODS
    # =================================================================
    
    def get_net_flow_by_expiry(self, date: str = None) -> Dict:
        """Get net flow by expiration"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/net-flow/expiry"
        params = {"date": date}
        return self._make_request(endpoint, params)
    
    def get_otm_contracts(self, date: str = None) -> Dict:
        """Get out-of-the-money contracts"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/screener/option-contracts"
        params = {"is_otm": "true", "date": date}
        return self._make_request(endpoint, params)
    
    def get_total_options_volume(self) -> Dict:
        """Get total options volume"""
        endpoint = "/api/market/total-options-volume"
        return self._make_request(endpoint)
    
    # =================================================================
    # INSTITUTIONAL DATA
    # =================================================================
    
    def get_institution_details(self, name: str) -> Dict:
        """Get institution details"""
        endpoint = "/api/institutions"
        params = {"name": name}
        return self._make_request(endpoint, params)
    
    def get_latest_filings(self, name: str, date: str = None) -> Dict:
        """Get latest institutional filings"""
        if not date:
            date = datetime.date.today().isoformat()
        endpoint = "/api/institutions/latest_filings"
        params = {"name": name, "date": date}
        return self._make_request(endpoint, params)
    
    # =================================================================
    # NEWS AND HEADLINES
    # =================================================================
    
    def get_news_headlines(self) -> Dict:
        """Get news headlines"""
        endpoint = "/api/news/headlines"
        return self._make_request(endpoint)
    
    # =================================================================
    # COMPREHENSIVE STOCK ANALYSIS WITH ENHANCED FLOW DATA
    # =================================================================
    
    def get_comprehensive_flow_data(self, ticker: str) -> Dict:
        """Get comprehensive flow data including alerts, volume, and hottest chains"""
        today = datetime.date.today().isoformat()
        
        # Gather all flow-related data
        data = {
            "flow_alerts": self.get_flow_alerts(ticker),
            "options_volume": self.get_options_volume(ticker),
            "flow_recent": self.get_stock_flow_recent(ticker),
            "flow_per_strike": self.get_flow_per_strike(ticker, today),
            "greek_exposure": self.get_greek_exposure(ticker, today),
            "greek_by_expiry": self.get_greek_exposure_by_expiry(ticker, today),
            "atm_chains": self.get_atm_chains(ticker),
            "timestamp": datetime.datetime.now().isoformat(),
            "data_source": "Unusual Whales"
        }
        
        return data
    
    def get_comprehensive_stock_data(self, ticker: str) -> Dict:
        """Get comprehensive stock analysis data"""
        today = datetime.date.today().isoformat()
        
        # Gather all relevant data
        data = {
            "quote": self.get_stock_state(ticker),
            "flow_alerts": self.get_flow_alerts(ticker),
            "options_volume": self.get_options_volume(ticker),
            "flow_recent": self.get_stock_flow_recent(ticker),
            "greek_exposure": self.get_greek_exposure(ticker, today),
            "greek_by_expiry": self.get_greek_exposure_by_expiry(ticker, today),
            "flow_per_strike": self.get_flow_per_strike(ticker, today),
            "atm_chains": self.get_atm_chains(ticker),
            "timestamp": datetime.datetime.now().isoformat(),
            "data_source": "Unusual Whales"
        }
        
        return data

# Initialize UW client
uw_client = UnusualWhalesClient(UNUSUAL_WHALES_KEY) if UNUSUAL_WHALES_KEY else None

def debug_atm_chains(ticker: str):
    """Debug function for testing ATM chains"""
    if not uw_client:
        print("UW client not available")
        return None
        
    print(f"Testing ATM chains for {ticker}...")
    try:
        atm_result = uw_client.get_atm_chains(ticker)
        print(f"ATM chains result: {atm_result}")
        return atm_result
    except Exception as e:
        print(f"Debug error: {e}")
        return None

# =================================================================
# ENHANCED FLOW ANALYSIS FUNCTIONS
# =================================================================

def analyze_flow_alerts(flow_alerts_data: Dict, ticker: str) -> Dict:
    """Analyze flow alerts data from UW"""
    if flow_alerts_data.get("error"):
        return {"error": flow_alerts_data["error"]}
    
    try:
        alerts = flow_alerts_data.get("data", [])
        if not alerts:
            return {"summary": "No flow alerts found", "alerts": []}
        
        # Process alerts data
        processed_alerts = []
        call_alerts = []
        put_alerts = []
        
        total_premium = 0
        bullish_flow = 0
        bearish_flow = 0
        
        for alert in alerts:
            if isinstance(alert, dict):
                alert_type = alert.get("type", "").lower()
                premium = float(alert.get("total_premium", 0)) if alert.get("total_premium") else 0
                volume = int(alert.get("volume", 0)) if alert.get("volume") else 0
                
                processed_alert = {
                    "type": alert_type,
                    "strike": alert.get("strike", 0),
                    "premium": premium,
                    "volume": volume,
                    "ticker": alert.get("ticker", ticker),
                    "expiry": alert.get("expiry", ""),
                    "price": alert.get("price", 0),
                    "underlying_price": alert.get("underlying_price", 0),
                    "time": alert.get("created_at", ""),
                    "is_sweep": alert.get("is_sweep", False),
                    "is_opening": alert.get("all_opening", False)
                }
                
                processed_alerts.append(processed_alert)
                total_premium += premium
                
                if alert_type == "call":
                    call_alerts.append(processed_alert)
                    bullish_flow += premium
                elif alert_type == "put":
                    put_alerts.append(processed_alert)
                    bearish_flow += premium
        
        # Calculate summary metrics
        total_alerts = len(processed_alerts)
        call_count = len(call_alerts)
        put_count = len(put_alerts)
        
        flow_sentiment = "Neutral"
        if bullish_flow > bearish_flow * 1.2:
            flow_sentiment = "Bullish"
        elif bearish_flow > bullish_flow * 1.2:
            flow_sentiment = "Bearish"
        
        return {
            "summary": {
                "total_alerts": total_alerts,
                "call_alerts": call_count,
                "put_alerts": put_count,
                "total_premium": total_premium,
                "bullish_flow": bullish_flow,
                "bearish_flow": bearish_flow,
                "flow_sentiment": flow_sentiment
            },
            "alerts": processed_alerts,
            "call_alerts": call_alerts,
            "put_alerts": put_alerts,
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Error analyzing flow alerts: {str(e)}"}

def analyze_options_volume(volume_data: Dict, ticker: str) -> Dict:
    """Analyze options volume data from UW"""
    if volume_data.get("error"):
        return {"error": volume_data["error"]}
    
    try:
        data = volume_data.get("data", [])
        if not data:
            return {"summary": "No volume data found", "volume_data": []}
        
        # Process volume data
        processed_data = []
        total_call_volume = 0
        total_put_volume = 0
        total_call_premium = 0
        total_put_premium = 0
        
        for item in data:
            if isinstance(item, dict):
                avg_30_day_call_volume = float(item.get("avg_30_day_call_volume", 0)) if item.get("avg_30_day_call_volume") else 0
                avg_30_day_put_volume = float(item.get("avg_30_day_put_volume", 0)) if item.get("avg_30_day_put_volume") else 0
                call_volume = int(item.get("call_volume", 0)) if item.get("call_volume") else 0
                put_volume = int(item.get("put_volume", 0)) if item.get("put_volume") else 0
                
                processed_item = {
                    "date": item.get("date", ""),
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "avg_30_day_call_volume": avg_30_day_call_volume,
                    "avg_30_day_put_volume": avg_30_day_put_volume,
                    "call_volume_ratio": call_volume / max(avg_30_day_call_volume, 1),
                    "put_volume_ratio": put_volume / max(avg_30_day_put_volume, 1),
                    "bearish_premium": float(item.get("bearish_premium", 0)) if item.get("bearish_premium") else 0,
                    "bullish_premium": float(item.get("bullish_premium", 0)) if item.get("bullish_premium") else 0,
                    "call_premium": float(item.get("call_premium", 0)) if item.get("call_premium") else 0,
                    "put_premium": float(item.get("put_premium", 0)) if item.get("put_premium") else 0,
                    "net_call_premium": float(item.get("net_call_premium", 0)) if item.get("net_call_premium") else 0,
                    "net_put_premium": float(item.get("net_put_premium", 0)) if item.get("net_put_premium") else 0
                }
                
                processed_data.append(processed_item)
                total_call_volume += call_volume
                total_put_volume += put_volume
                total_call_premium += processed_item["call_premium"]
                total_put_premium += processed_item["put_premium"]
        
        # Calculate summary
        put_call_ratio = total_put_volume / max(total_call_volume, 1)
        premium_ratio = total_put_premium / max(total_call_premium, 1)
        
        return {
            "summary": {
                "total_call_volume": total_call_volume,
                "total_put_volume": total_put_volume,
                "put_call_ratio": put_call_ratio,
                "total_call_premium": total_call_premium,
                "total_put_premium": total_put_premium,
                "premium_ratio": premium_ratio
            },
            "volume_data": processed_data,
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Error analyzing options volume: {str(e)}"}

def get_hottest_chains_analysis() -> Dict:
    """Get and analyze hottest option chains"""
    if not uw_client:
        return {"error": "UW client not available"}
    
    try:
        chains_data = uw_client.get_hottest_chains()
        if chains_data.get("error"):
            return chains_data
        
        data = chains_data.get("data", [])
        if not data:
            return {"summary": "No hottest chains data found", "chains": []}
        
        processed_chains = []
        total_volume = 0
        total_premium = 0
        
        for chain in data:
            if isinstance(chain, dict):
                volume = int(chain.get("volume", 0)) if chain.get("volume") else 0
                premium = float(chain.get("total_premium", 0)) if chain.get("total_premium") else 0
                
                processed_chain = {
                    "ticker": chain.get("ticker", ""),
                    "strike": float(chain.get("strike", 0)) if chain.get("strike") else 0,
                    "type": chain.get("type", ""),
                    "volume": volume,
                    "premium": premium,
                    "expiry": chain.get("expiry", ""),
                    "underlying_price": float(chain.get("underlying_price", 0)) if chain.get("underlying_price") else 0,
                    "price": float(chain.get("price", 0)) if chain.get("price") else 0,
                    "iv": float(chain.get("iv", 0)) if chain.get("iv") else 0
                }
                
                processed_chains.append(processed_chain)
                total_volume += volume
                total_premium += premium
        
        # Sort by volume
        processed_chains.sort(key=lambda x: x["volume"], reverse=True)
        
        return {
            "summary": {
                "total_chains": len(processed_chains),
                "total_volume": total_volume,
                "total_premium": total_premium
            },
            "chains": processed_chains,
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Error analyzing hottest chains: {str(e)}"}

# =================================================================
# AI ANALYSIS WITH ENHANCED FLOW DATA
# =================================================================

def generate_flow_analysis_prompt(ticker: str, flow_data: Dict, volume_data: Dict, hottest_chains: Dict) -> str:
    """Generate comprehensive flow analysis prompt for AI"""
    
    prompt = f"""
    COMPREHENSIVE OPTIONS FLOW ANALYSIS FOR {ticker}
    
    === FLOW ALERTS ANALYSIS ===
    """
    
    if not flow_data.get("error"):
        summary = flow_data.get("summary", {})
        prompt += f"""
        - Total Flow Alerts: {summary.get('total_alerts', 0)}
        - Call Alerts: {summary.get('call_alerts', 0)}
        - Put Alerts: {summary.get('put_alerts', 0)}
        - Total Premium: ${summary.get('total_premium', 0):,.2f}
        - Bullish Flow: ${summary.get('bullish_flow', 0):,.2f}
        - Bearish Flow: ${summary.get('bearish_flow', 0):,.2f}
        - Flow Sentiment: {summary.get('flow_sentiment', 'Neutral')}
        
        Top Flow Alerts:
        """
        
        # Add top alerts
        alerts = flow_data.get("alerts", [])
        for alert in alerts[:5]:
            prompt += f"""
        - {alert['type'].upper()}: Strike ${alert['strike']}, Premium ${alert['premium']:,.2f}, Volume {alert['volume']}
        """
    else:
        prompt += f"Flow Alerts Error: {flow_data['error']}\n"
    
    prompt += f"""
    
    === OPTIONS VOLUME ANALYSIS ===
    """
    
    if not volume_data.get("error"):
        vol_summary = volume_data.get("summary", {})
        prompt += f"""
        - Total Call Volume: {vol_summary.get('total_call_volume', 0):,}
        - Total Put Volume: {vol_summary.get('total_put_volume', 0):,}
        - Put/Call Ratio: {vol_summary.get('put_call_ratio', 0):.2f}
        - Call Premium: ${vol_summary.get('total_call_premium', 0):,.2f}
        - Put Premium: ${vol_summary.get('total_put_premium', 0):,.2f}
        - Premium Ratio: {vol_summary.get('premium_ratio', 0):.2f}
        """
    else:
        prompt += f"Volume Data Error: {volume_data['error']}\n"
    
    prompt += f"""
    
    === HOTTEST CHAINS ANALYSIS ===
    """
    
    if not hottest_chains.get("error"):
        chains_summary = hottest_chains.get("summary", {})
        prompt += f"""
        - Total Hottest Chains: {chains_summary.get('total_chains', 0)}
        - Combined Volume: {chains_summary.get('total_volume', 0):,}
        - Combined Premium: ${chains_summary.get('total_premium', 0):,.2f}
        
        Top Hottest Chains:
        """
        
        chains = hottest_chains.get("chains", [])
        for chain in chains[:5]:
            prompt += f"""
        - {chain['ticker']} {chain['type'].upper()}: Strike ${chain['strike']}, Volume {chain['volume']}, Premium ${chain['premium']:,.2f}
        """
    else:
        prompt += f"Hottest Chains Error: {hottest_chains['error']}\n"
    
    prompt += f"""
    
    ANALYSIS REQUEST:
    Based on this comprehensive options flow data, provide:
    
    1. **Flow Sentiment Analysis**: Overall bullish/bearish bias from the data
    2. **Key Flow Patterns**: Notable unusual activity or patterns
    3. **Volume Analysis**: How current volume compares to averages
    4. **Premium Flow**: Where the big money is going
    5. **Trading Opportunities**: Specific strikes and strategies
    6. **Risk Assessment**: Key levels and timing considerations
    7. **Institutional Activity**: Signs of smart money positioning
    
    Keep analysis under 400 words but be specific about actionable insights.
    Focus on the most significant flow patterns and their trading implications.
    """
    
    return prompt

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
twelvedata_client = TwelveDataClient(TWELVEDATA_KEY) if TWELVEDATA_KEY else None

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

# =============================================================================
# ENHANCED OPTIONS ANALYSIS WITH UW DATA
# =============================================================================

def get_enhanced_options_analysis(ticker: str) -> Dict:
    """Comprehensive options analysis using UW data and yfinance fallback"""
    try:
        if not uw_client:
            return get_advanced_options_analysis_yf(ticker)
        
        # Get comprehensive UW options data
        uw_data = uw_client.get_comprehensive_stock_data(ticker)
        
        # Extract and analyze UW data
        analysis = {
            "uw_flow_alerts": uw_data.get("flow_alerts", {}),
            "uw_options_volume": uw_data.get("options_volume", {}),
            "uw_greek_exposure": uw_data.get("greek_exposure", {}),
            "uw_greek_by_expiry": uw_data.get("greek_by_expiry", {}),
            "uw_flow_per_strike": uw_data.get("flow_per_strike", {}),
            "uw_atm_chains": uw_data.get("atm_chains", {}),
            "enhanced_metrics": analyze_uw_options_data(uw_data),
            "data_source": "Unusual Whales",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Enhanced options analysis error: {str(e)}"}

def analyze_uw_options_data(uw_data: Dict) -> Dict:
    """Analyze UW options data to extract key metrics"""
    try:
        metrics = {}
        
        # Flow alerts analysis
        flow_alerts = uw_data.get("flow_alerts", {})
        if flow_alerts.get("data"):
            alerts = flow_alerts["data"]
            metrics["total_flow_alerts"] = len(alerts) if isinstance(alerts, list) else 0
            
            if isinstance(alerts, list) and len(alerts) > 0:
                call_alerts = [a for a in alerts if a.get("type", "").lower() == "call"]
                put_alerts = [a for a in alerts if a.get("type", "").lower() == "put"]
                metrics["call_flow_alerts"] = len(call_alerts)
                metrics["put_flow_alerts"] = len(put_alerts)
                metrics["flow_sentiment"] = "Bullish" if len(call_alerts) > len(put_alerts) else "Bearish" if len(put_alerts) > len(call_alerts) else "Neutral"
        
        # Options volume analysis
        options_volume = uw_data.get("options_volume", {})
        if options_volume.get("data"):
            volume_data = options_volume["data"]
            if isinstance(volume_data, list) and len(volume_data) > 0:
                latest_volume = volume_data[0] if volume_data else {}
                metrics["call_volume"] = latest_volume.get("call_volume", 0)
                metrics["put_volume"] = latest_volume.get("put_volume", 0)
                metrics["call_premium"] = latest_volume.get("call_premium", 0)
                metrics["put_premium"] = latest_volume.get("put_premium", 0)
                metrics["volume_put_call_ratio"] = latest_volume.get("put_volume", 0) / max(latest_volume.get("call_volume", 1), 1)
        
        # Greek exposure analysis
        greek_exposure = uw_data.get("greek_exposure", {})
        if greek_exposure.get("data"):
            greek_data = greek_exposure["data"]
            if isinstance(greek_data, dict):
                metrics["total_delta"] = greek_data.get("total_delta", 0)
                metrics["total_gamma"] = greek_data.get("total_gamma", 0)
                metrics["total_theta"] = greek_data.get("total_theta", 0)
                metrics["total_vega"] = greek_data.get("total_vega", 0)
        
        # Enhanced ATM chains analysis
        atm_chains = uw_data.get("atm_chains", {})
        if atm_chains.get("data") and not atm_chains.get("error"):
            chains_data = atm_chains["data"]
            if isinstance(chains_data, list) and len(chains_data) > 0:
                total_call_volume = 0
                total_put_volume = 0
                total_call_oi = 0
                total_put_oi = 0
                
                for chain in chains_data:
                    try:
                        volume = int(chain.get("volume", 0)) if chain.get("volume") else 0
                        oi = int(chain.get("open_interest", 0)) if chain.get("open_interest") else 0
                        
                        is_call = False
                        if "option_symbol" in chain:
                            option_symbol = chain["option_symbol"]
                            is_call = "C" in option_symbol and "P" not in option_symbol
                        elif "type" in chain:
                            is_call = chain["type"].lower() == "call"
                        elif "call_put" in chain:
                            is_call = chain["call_put"].lower() == "call"
                        
                        if is_call:
                            total_call_volume += volume
                            total_call_oi += oi
                        else:
                            total_put_volume += volume
                            total_put_oi += oi
                            
                    except Exception as e:
                        print(f"Error processing ATM chain item: {e}")
                        continue
                
                metrics["atm_call_volume"] = total_call_volume
                metrics["atm_put_volume"] = total_put_volume
                metrics["atm_call_oi"] = total_call_oi
                metrics["atm_put_oi"] = total_put_oi
                
                if total_call_volume > 0:
                    metrics["atm_put_call_ratio"] = total_put_volume / total_call_volume
                else:
                    metrics["atm_put_call_ratio"] = 0
                    
                print(f"ATM analysis complete: Calls={total_call_volume}, Puts={total_put_volume}")
            else:
                print("ATM chains data is empty or invalid format")
                metrics["atm_chains_status"] = "No data"
        else:
            # Log the ATM chains error for debugging  
            if atm_chains.get("error"):
                metrics["atm_chains_error"] = atm_chains["error"]
                print(f"ATM chains error: {atm_chains['error']}")
            else:
                print("No ATM chains data available")
                metrics["atm_chains_status"] = "No data available"
        
        return metrics
        
    except Exception as e:
        return {"error": f"UW options analysis error: {str(e)}"}

def get_advanced_options_analysis_yf(ticker: str) -> Dict:
    """Fallback yfinance options analysis"""
    try:
        option_chain = get_option_chain(ticker, st.session_state.selected_tz)
        if option_chain.get("error"):
            return {"error": option_chain["error"]}
        
        calls = option_chain["calls"]
        puts = option_chain["puts"]
        current_price = option_chain["current_price"]
        
        # Advanced analysis
        analysis = {
            "basic_metrics": calculate_basic_options_metrics(calls, puts),
            "flow_analysis": analyze_options_flow(calls, puts, current_price),
            "unusual_activity": detect_unusual_activity(calls, puts),
            "gamma_levels": calculate_gamma_levels(calls, puts, current_price),
            "sentiment_indicators": calculate_options_sentiment(calls, puts),
            "expiration": option_chain["expiration"],
            "current_price": current_price,
            "data_source": "Yahoo Finance"
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

# =============================================================================
# LIVE DATA UPDATE FUNCTIONS
# =============================================================================

def get_live_quote_with_cache(ticker: str, tz: str = "ET") -> Dict:
    """Get live quote with intelligent caching for speed"""
    current_time = time.time()
    cache_key = f"{ticker}_{tz}"
    
    # Check if we have recent cached data (within 30 seconds)
    if (cache_key in st.session_state.live_data_cache and 
        current_time - st.session_state.live_data_cache[cache_key].get('timestamp', 0) < 30):
        return st.session_state.live_data_cache[cache_key]['data']
    
    # Fetch new data
    quote = get_live_quote(ticker, tz)
    
    # Cache the result
    st.session_state.live_data_cache[cache_key] = {
        'data': quote,
        'timestamp': current_time
    }
    
    return quote

# =============================================================================
# PRIMARY DATA FUNCTION - UW FIRST, FALLBACK TO OTHERS
# =============================================================================

def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """
    Enhanced live quote using UW first, then fallback hierarchy
    """
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # Try Unusual Whales first (primary data source using stock-state endpoint)
    if uw_client:
        try:
            uw_quote = uw_client.get_stock_state(ticker)
            if not uw_quote.get("error") and uw_quote.get("last", 0) > 0:
                # Enhance UW stock-state data with session tracking
                enhanced_quote = enhance_uw_stock_state_with_sessions(uw_quote, ticker, tz_zone, tz_label)
                return enhanced_quote
        except Exception as e:
            print(f"UW stock-state error for {ticker}: {str(e)}")
    
    # Try Twelve Data
    if twelvedata_client:
        try:
            twelve_quote = twelvedata_client.get_quote(ticker)
            if not twelve_quote.get("error") and twelve_quote.get("last", 0) > 0:
                twelve_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return twelve_quote
        except Exception as e:
            print(f"Twelve Data error for {ticker}: {str(e)}")
    
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

def enhance_uw_stock_state_with_sessions(uw_stock_state: Dict, ticker: str, tz_zone, tz_label: str) -> Dict:
    """Enhance UW stock-state data with session tracking"""
    try:
        # Extract UW stock-state data
        current_price = uw_stock_state.get("last", 0)
        open_price = uw_stock_state.get("open", current_price)
        previous_close = uw_stock_state.get("previous_close", open_price)
        market_time = uw_stock_state.get("market_time", "market")
        
        # Calculate session changes based on UW data and market_time
        premarket_change = 0
        intraday_change = 0
        postmarket_change = 0
        
        if market_time == "premarket":
            # Currently in premarket
            premarket_change = ((current_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
            intraday_change = 0
            postmarket_change = 0
        elif market_time == "market":
            # Currently in market hours
            if open_price != previous_close:
                premarket_change = ((open_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
            intraday_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
            postmarket_change = 0
        elif market_time == "postmarket":
            # Currently in after hours
            if open_price != previous_close:
                premarket_change = ((open_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
            # Assume market close was same as open for simplification (could enhance with more data)
            intraday_change = ((open_price - open_price) / open_price) * 100 if open_price > 0 else 0
            postmarket_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
        
        # Enhance UW stock-state with session data
        enhanced_quote = {
            "last": current_price,
            "bid": uw_stock_state.get("bid", current_price - 0.01),
            "ask": uw_stock_state.get("ask", current_price + 0.01),
            "volume": uw_stock_state.get("volume", 0),
            "total_volume": uw_stock_state.get("total_volume", uw_stock_state.get("volume", 0)),
            "change": uw_stock_state.get("change", 0),
            "change_percent": uw_stock_state.get("change_percent", 0),
            "premarket_change": premarket_change,
            "intraday_change": intraday_change,
            "postmarket_change": postmarket_change,
            "previous_close": previous_close,
            "market_open": open_price,
            "open": open_price,
            "high": uw_stock_state.get("high", current_price),
            "low": uw_stock_state.get("low", current_price),
            "market_time": market_time,
            "tape_time": uw_stock_state.get("tape_time", ""),
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None,
            "data_source": "Unusual Whales"
        }
        
        return enhanced_quote
        
    except Exception as e:
        # Return basic UW quote if enhancement fails
        uw_stock_state["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
        uw_stock_state["data_source"] = "Unusual Whales"
        uw_stock_state.setdefault("premarket_change", 0)
        uw_stock_state.setdefault("intraday_change", 0)
        uw_stock_state.setdefault("postmarket_change", 0)
        return uw_stock_state

# =============================================================================
# NEWS AND MARKET DATA WITH UW INTEGRATION
# =============================================================================

@st.cache_data(ttl=600)
def get_uw_news() -> List[Dict]:
    """Get news from Unusual Whales"""
    if not uw_client:
        return []
    
    try:
        news_result = uw_client.get_news_headlines()
        if news_result.get("error"):
            return []
        
        news_data = news_result.get("data", [])
        formatted_news = []
        
        for item in news_data:
            formatted_news.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": "Unusual Whales",
                "url": item.get("url", ""),
                "datetime": item.get("published_at", ""),
                "related": ",".join(item.get("symbols", [])) if item.get("symbols") else "",
                "provider": "Unusual Whales API"
            })
        
        return formatted_news[:15]  # Return top 15
        
    except Exception as e:
        print(f"UW news error: {e}")
        return []

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

def get_all_news() -> List[Dict]:
    """Enhanced news with UW integration"""
    all_news = []
    
    # UW news first (primary source)
    uw_news = get_uw_news()
    all_news.extend(uw_news)
    
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
    
    # Get UW news if available
    if uw_client:
        try:
            uw_news = get_uw_news()
            # Filter for ticker-specific news
            for item in uw_news:
                if ticker.upper() in item.get("related", "").upper():
                    all_news.append(item)
        except Exception:
            pass
    
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
    uw_general = get_uw_news()
    finnhub_general = get_finnhub_news()  # General news
    yahoo_general = get_yfinance_news()
    
    # Process UW news
    for item in uw_general:
        catalyst_analysis = analyze_catalyst_impact(
            item.get("title", ""), 
            item.get("summary", "")
        )
        item["catalyst_analysis"] = catalyst_analysis
        all_news.append(item)
    
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

# New function to fetch option chain data
@st.cache_data(ttl=300)  # Cache for 5 minutes
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
        current_price = get_live_quote_with_cache(ticker, tz).get('last', 0)
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
@st.cache_data(ttl=300)
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

def get_earnings_calendar() -> List[Dict]:
    """Enhanced earnings calendar using UW data"""
    if uw_client:
        try:
            # Try to get economic calendar from UW (may include earnings)
            calendar_result = uw_client.get_economic_calendar()
            if not calendar_result.get("error") and calendar_result.get("data"):
                calendar_data = calendar_result["data"]
                earnings_events = []
                
                # Filter for earnings-related events
                for event in calendar_data:
                    if "earnings" in event.get("title", "").lower() or "eps" in event.get("title", "").lower():
                        earnings_events.append({
                            "ticker": event.get("symbol", ""),
                            "date": event.get("date", ""),
                            "time": event.get("time", ""),
                            "estimate": event.get("estimate", ""),
                            "source": "Unusual Whales"
                        })
                
                if earnings_events:
                    return earnings_events
        except Exception as e:
            print(f"UW earnings calendar error: {e}")
    
    # Fallback to placeholder data
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    return [
        {"ticker": "MSFT", "date": today, "time": "After Hours", "estimate": "$2.50", "source": "Simulated"},
        {"ticker": "NVDA", "date": today, "time": "Before Market", "estimate": "$1.20", "source": "Simulated"},
        {"ticker": "TSLA", "date": today, "time": "After Hours", "estimate": "$0.75", "source": "Simulated"},
    ]

# Enhanced AI analysis functions
def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook using comprehensive technical, fundamental, and options analysis"""
    
    # Get comprehensive analysis data
    with st.spinner(f"Gathering comprehensive data for {ticker}..."):
        quote = get_live_quote_with_cache(ticker, st.session_state.selected_tz)
        technical_analysis = get_comprehensive_technical_analysis(ticker)
        fundamental_analysis = get_fundamental_analysis(ticker)
        
        # Use UW options analysis if available, fallback to yfinance
        if uw_client:
            options_analysis = get_enhanced_options_analysis(ticker)
        else:
            options_analysis = get_advanced_options_analysis_yf(ticker)
        
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

# Enhanced AI Analysis Prompt Construction
def construct_comprehensive_analysis_prompt(ticker: str, quote: Dict, technical: Dict, fundamental: Dict, options: Dict, news_context: str = "") -> str:
    """Construct comprehensive analysis prompt with all data"""
    
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
        # Enhanced technical from Twelve Data
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
    
    # Options summary (enhanced for UW data)
    options_summary = "Options Analysis:\n"
    if options.get("error"):
        options_summary += f"Options Error: {options['error']}\n"
    else:
        # Check if this is UW data or yfinance data
        if options.get("data_source") == "Unusual Whales":
            enhanced = options.get('enhanced_metrics', {})
            if enhanced:
                options_summary += f"- Flow Alerts: {enhanced.get('total_flow_alerts', 'N/A')}\n"
                options_summary += f"- Flow Sentiment: {enhanced.get('flow_sentiment', 'Neutral')}\n"
                
                # Safe handling of ATM P/C Ratio
                atm_pc_ratio = enhanced.get('atm_put_call_ratio', None)
                if atm_pc_ratio is not None and isinstance(atm_pc_ratio, (int, float)):
                    options_summary += f"- ATM P/C Ratio: {atm_pc_ratio:.2f}\n"
                else:
                    options_summary += f"- ATM P/C Ratio: N/A\n"
                
                options_summary += f"- Total Delta: {enhanced.get('total_delta', 'N/A')}\n"
                options_summary += f"- Total Gamma: {enhanced.get('total_gamma', 'N/A')}\n"
                options_summary += f"- Data Source: Unusual Whales (Premium)\n"
            else:
                options_summary += f"- UW Data: No enhanced metrics available\n"
                options_summary += f"- Data Source: Unusual Whales (Limited)\n"
        else:
            # Standard yfinance data
            basic = options.get('basic_metrics', {})
            flow = options.get('flow_analysis', {})
            unusual = options.get('unusual_activity', {})
            if basic:
                pc_ratio = basic.get('put_call_volume_ratio', 0)
                if isinstance(pc_ratio, (int, float)):
                    options_summary += f"- Put/Call Ratio: {pc_ratio:.2f}\n"
                else:
                    options_summary += f"- Put/Call Ratio: N/A\n"
                
                avg_iv = basic.get('avg_call_iv', 0)
                if isinstance(avg_iv, (int, float)):
                    options_summary += f"- Average IV: {avg_iv:.1f}%\n"
                else:
                    options_summary += f"- Average IV: N/A\n"
                
                iv_skew = basic.get('iv_skew', 0)
                if isinstance(iv_skew, (int, float)):
                    options_summary += f"- IV Skew: {iv_skew:.1f}%\n"
                else:
                    options_summary += f"- IV Skew: N/A\n"
                
                options_summary += f"- Flow Sentiment: {flow.get('flow_sentiment', 'Neutral')}\n"
                options_summary += f"- Unusual Activity: {unusual.get('total_unusual_contracts', 0)} contracts\n"
                
                total_volume = basic.get('total_call_volume', 0) + basic.get('total_put_volume', 0)
                options_summary += f"- Total Volume: {total_volume:,}\n"
            else:
                options_summary += f"- Options data: Limited metrics available\n"
    
    # Session data
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
Comprehensive Trading Analysis for {ticker}:

{session_summary}

{tech_summary}

{fund_summary}

{options_summary}
{news_section}

Based on this comprehensive analysis, provide:

1. **Overall Assessment** (Bullish/Bearish/Neutral) with confidence rating (1-100)
2. **Trading Strategy** (Scalp/Day Trade/Swing/Position/Avoid) with specific timeframe
3. **Entry Strategy**: Exact price levels and conditions
4. **Profit Targets**: 3 realistic target levels with rationale
5. **Risk Management**: Stop loss levels and position sizing guidance
6. **Technical Outlook**: Key levels to watch and breakout scenarios
7. **Fundamental Justification**: How fundamentals support the technical setup
8. **Options Strategy**: Specific options plays if applicable
9. **Catalyst Watch**: Events or levels that could trigger major moves
10. **Risk Factors**: What could invalidate this analysis

Keep analysis under 400 words but be specific and actionable.
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
    
    def multi_ai_consensus(self, ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> Dict[str, str]:
        """Get consensus analysis from all available AI models with enhanced prompts"""
        
        # Use the enhanced comprehensive prompt
        prompt = construct_comprehensive_analysis_prompt(ticker, {"last": 0, "change_percent": change}, {}, {}, options_data or {}, catalyst)
        
        analyses = {}
        
        # Get analysis from each available model
        if self.openai_client:
            analyses["OpenAI"] = self.analyze_with_openai(prompt)
        
        if self.gemini_model:
            analyses["Gemini"] = self.analyze_with_gemini(prompt)
        
        if self.grok_client:
            analyses["Grok"] = self.analyze_with_grok(prompt)
        
        return analyses
    
    def synthesize_consensus(self, analyses: Dict[str, str], ticker: str) -> str:
        """Synthesize multiple AI analyses into a consensus view"""
        if not analyses:
            return "No AI models available for analysis."
        
        # Create synthesis prompt
        analysis_text = "\n\n".join([f"**{model} Analysis:**\n{analysis}" for model, analysis in analyses.items()])
        
        synthesis_prompt = f"""
        Based on the following AI analyses for {ticker}, provide a synthesized consensus view:
        
        {analysis_text}
        
        Synthesize into:
        1. **Consensus Sentiment** and average confidence
        2. **Agreed Trading Strategy**
        3. **Consensus Price Levels** (entry, targets, stops)
        4. **Risk Assessment**
        5. **Key Points of Agreement/Disagreement**
        
        Prioritize areas where models agree and note any significant disagreements.
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

# Enhanced auto-generation with comprehensive analysis
def ai_auto_generate_plays_enhanced(tz: str):
    """Enhanced auto-generation with comprehensive analysis using UW data"""
    plays = []
    
    try:
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        
        # Scan for significant movers with enhanced criteria
        candidates = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(get_live_quote_with_cache, ticker, tz): ticker for ticker in scan_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    quote = future.result()
                    if not quote["error"]:
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
            
            # Get comprehensive analysis data
            technical_analysis = get_comprehensive_technical_analysis(ticker)
            fundamental_analysis = get_fundamental_analysis(ticker)
            
            # Use UW options analysis if available
            if uw_client:
                options_analysis = get_enhanced_options_analysis(ticker)
            else:
                options_analysis = get_advanced_options_analysis_yf(ticker)
            
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
                "options_summary": generate_enhanced_options_summary(options_analysis),
                "significance_score": candidate["significance"]
            }
            plays.append(play)
        
        return plays
    except Exception as e:
        st.error(f"Error generating enhanced auto plays: {str(e)}")
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

def generate_enhanced_options_summary(options: Dict) -> str:
    """Generate enhanced options summary with UW support"""
    if options.get("error"):
        return f"Options Error: {options['error']}"
    
    if options.get("data_source") == "Unusual Whales":
        # UW enhanced options metrics
        enhanced = options.get('enhanced_metrics', {})
        flow_alerts = enhanced.get('total_flow_alerts', 'N/A')
        sentiment = enhanced.get('flow_sentiment', 'Neutral')
        pc_ratio = enhanced.get('atm_put_call_ratio', 'N/A')
        return f"UW Flow Alerts: {flow_alerts}, Sentiment: {sentiment}, ATM P/C: {pc_ratio}"
    else:
        # Standard yfinance summary
        basic = options.get('basic_metrics', {})
        flow = options.get('flow_analysis', {})
        pc_ratio = basic.get('put_call_volume_ratio', 0)
        pc_ratio_str = f"{pc_ratio:.2f}" if pc_ratio is not None else "N/A"
        sentiment = flow.get('flow_sentiment', 'Neutral')
        return f"P/C Ratio: {pc_ratio_str}, Flow: {sentiment}"

# Enhanced market analysis
def ai_market_analysis_enhanced(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis with comprehensive data"""
    
    # Gather market-wide technical data
    market_technical = {}
    key_indices = ["SPY", "QQQ", "IWM"]
    
    for index in key_indices:
        try:
            quote = get_live_quote_with_cache(index)
            technical = get_comprehensive_technical_analysis(index)
            market_technical[index] = {
                "price": quote['last'],
                "change": quote['change_percent'],
                "technical": technical
            }
        except:
            continue
    
    # Analyze sector rotation using UW or fallback
    sector_data = analyze_sector_rotation()
    
    # Add UW market insights if available
    uw_insights = ""
    if uw_client:
        try:
            market_tide = uw_client.get_market_tide()
            if not market_tide.get("error"):
                uw_insights = "\nUnusual Whales Market Insights: Market tide data analyzed\n"
            
            oi_changes = uw_client.get_oi_change()
            if not oi_changes.get("error"):
                uw_insights += "Open Interest changes detected\n"
                
            spike_data = uw_client.get_spike_data()
            if not spike_data.get("error"):
                uw_insights += "Spike activity monitored\n"
        except Exception:
            pass
    
    # Construct enhanced market analysis prompt
    market_context = f"""
Market Technical Overview:
{format_market_technical(market_technical)}

Sector Analysis:
{sector_data}

{uw_insights}

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
            quote = get_live_quote_with_cache(etf)
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

# Function to get important economic events using AI and UW
def get_important_events() -> List[Dict]:
    # Try UW economic calendar first
    if uw_client:
        try:
            calendar_result = uw_client.get_economic_calendar()
            if not calendar_result.get("error") and calendar_result.get("data"):
                events = []
                for event in calendar_result["data"]:
                    events.append({
                        "event": event.get("title", ""),
                        "date": event.get("date", ""),
                        "time": event.get("time", ""),
                        "impact": event.get("impact", "Medium")
                    })
                return events[:10]  # Return top 10
        except Exception:
            pass
    
    # Fallback to AI generation
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

# =============================================================================
# HELPER FUNCTIONS FOR TIMEFRAME OPTIONS ANALYSIS
# =============================================================================

@st.cache_data(ttl=300)
def get_options_by_timeframe(ticker: str, timeframe: str, tz: str = "ET") -> Dict:
    """Get options filtered by timeframe (0DTE, Swing, LEAPS)"""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"error": f"No options data available for {ticker}"}

        today = datetime.datetime.now(ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')).date()
        expiration_dates = []
        
        for exp in expirations:
            exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            
            if timeframe == "0DTE" and days_to_exp == 0:
                expiration_dates.append(exp)
            elif timeframe == "Swing" and 2 <= days_to_exp <= 89:
                expiration_dates.append(exp)
            elif timeframe == "LEAPS" and days_to_exp >= 90:
                expiration_dates.append(exp)
        
        if not expiration_dates:
            return {"error": f"No {timeframe} options available for {ticker}"}

        # Get closest expiration in timeframe
        target_expiration = min(expiration_dates, key=lambda x: abs((datetime.datetime.strptime(x, '%Y-%m-%d').date() - today).days))
        
        # Fetch option chain
        option_chain = stock.option_chain(target_expiration)
        calls = option_chain.calls
        puts = option_chain.puts

        # Clean and format data
        calls = calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        puts = puts[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        
        # Determine moneyness
        current_price = get_live_quote_with_cache(ticker, tz).get('last', 0)
        calls['moneyness'] = calls['strike'].apply(lambda x: 'ITM' if x < current_price else 'OTM')
        puts['moneyness'] = puts['strike'].apply(lambda x: 'ITM' if x > current_price else 'OTM')

        # Convert IV to percentage
        calls['impliedVolatility'] = calls['impliedVolatility'] * 100
        puts['impliedVolatility'] = puts['impliedVolatility'] * 100

        return {
            "calls": calls,
            "puts": puts,
            "expiration": target_expiration,
            "current_price": current_price,
            "days_to_expiration": (datetime.datetime.strptime(target_expiration, '%Y-%m-%d').date() - today).days,
            "timeframe": timeframe,
            "all_expirations": expiration_dates,
            "error": None
        }
    except Exception as e:
        return {"error": f"Error fetching {timeframe} options for {ticker}: {str(e)}"}

def analyze_timeframe_options_with_flow(ticker: str, option_data: Dict, flow_data: Dict, volume_data: Dict, hottest_chains: Dict, timeframe: str) -> str:
    """Generate timeframe-specific AI analysis with enhanced flow data"""
    
    if option_data.get("error"):
        return f"Unable to analyze {timeframe} options: {option_data['error']}"
    
    calls = option_data.get("calls", pd.DataFrame())
    puts = option_data.get("puts", pd.DataFrame())
    days_to_exp = option_data.get("days_to_expiration", 0)
    current_price = option_data.get("current_price", 0)
    
    # Calculate key metrics
    total_call_volume = calls['volume'].sum() if not calls.empty else 0
    total_put_volume = puts['volume'].sum() if not puts.empty else 0
    avg_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2 if not calls.empty and not puts.empty else 0
    
    # Generate enhanced flow analysis prompt
    flow_prompt = generate_flow_analysis_prompt(ticker, flow_data, volume_data, hottest_chains)
    
    # Create timeframe-specific prompt with flow integration
    prompt = f"""
    ENHANCED {timeframe} OPTIONS FLOW ANALYSIS FOR {ticker}:
    
    === BASIC OPTIONS DATA ===
    Current Price: ${current_price:.2f}
    Days to Expiration: {days_to_exp}
    Timeframe Category: {timeframe}
    
    Options Metrics:
    - Total Call Volume: {total_call_volume:,}
    - Total Put Volume: {total_put_volume:,}
    - Put/Call Volume Ratio: {total_put_volume/max(total_call_volume, 1):.2f}
    - Average IV: {avg_iv:.1f}%
    
    Top 5 Call Strikes by Volume:
    {calls.nlargest(5, 'volume')[['strike', 'lastPrice', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False) if not calls.empty else 'No call data'}
    
    Top 5 Put Strikes by Volume:
    {puts.nlargest(5, 'volume')[['strike', 'lastPrice', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False) if not puts.empty else 'No put data'}
    
    {flow_prompt}
    
    === ANALYSIS REQUEST ===
    Provide comprehensive {timeframe}-specific analysis covering:
    
    1. **Flow-Based Strategy**: How the unusual flow data impacts {timeframe} positioning
    2. **Optimal Entry Timing**: When to enter {timeframe} positions based on flow patterns
    3. **Key Strike Selection**: Which strikes show the most institutional interest
    4. **Time Decay Considerations**: How flow patterns affect theta decay for {timeframe}
    5. **Volume vs Open Interest**: What the flow tells us about new vs existing positions
    6. **IV and Volatility Outlook**: How flow impacts implied volatility expectations
    7. **Risk Management**: Position sizing and stops based on flow sentiment
    8. **Catalyst Timing**: How to time {timeframe} trades around expected catalysts
    
    Tailor advice specifically for {timeframe} characteristics with flow-based insights.
    Keep analysis under 400 words but be actionable and flow-focused.
    """
    
    # Use selected AI model for enhanced analysis
    if st.session_state.ai_model == "Multi-AI":
        analyses = multi_ai.multi_ai_consensus_enhanced(prompt)
        if analyses:
            result = f"## 🤖 Enhanced Multi-AI {timeframe} Flow Analysis\n\n"
            for model, analysis in analyses.items():
                result += f"### {model}:\n{analysis}\n\n---\n\n"
            return result
        else:
            return f"No AI models available for {timeframe} analysis."
    elif st.session_state.ai_model == "OpenAI" and openai_client:
        return multi_ai.analyze_with_openai(prompt)
    elif st.session_state.ai_model == "Gemini" and gemini_model:
        return multi_ai.analyze_with_gemini(prompt)
    elif st.session_state.ai_model == "Grok" and grok_enhanced:
        return multi_ai.analyze_with_grok(prompt)
    else:
        return f"No AI model configured for {timeframe} analysis."

def analyze_timeframe_options(ticker: str, option_data: Dict, uw_data: Dict, timeframe: str) -> str:
    """Generate timeframe-specific AI analysis (backward compatibility)"""
    
    if option_data.get("error"):
        return f"Unable to analyze {timeframe} options: {option_data['error']}"
    
    calls = option_data.get("calls", pd.DataFrame())
    puts = option_data.get("puts", pd.DataFrame())
    days_to_exp = option_data.get("days_to_expiration", 0)
    current_price = option_data.get("current_price", 0)
    
    # Calculate key metrics
    total_call_volume = calls['volume'].sum() if not calls.empty else 0
    total_put_volume = puts['volume'].sum() if not puts.empty else 0
    avg_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2 if not calls.empty and not puts.empty else 0
    
    # Get UW enhanced metrics if available
    uw_metrics = ""
    if uw_data and not uw_data.get("error"):
        enhanced = uw_data.get('enhanced_metrics', {})
        uw_metrics = f"""
        🔥 Unusual Whales Flow Data:
        - Flow Alerts: {enhanced.get('total_flow_alerts', 'N/A')}
        - Flow Sentiment: {enhanced.get('flow_sentiment', 'Neutral')}
        - ATM P/C Ratio: {enhanced.get('atm_put_call_ratio', 'N/A')}
        - Total Delta: {enhanced.get('total_delta', 'N/A')}
        - Total Gamma: {enhanced.get('total_gamma', 'N/A')}
        """
    
    # Create timeframe-specific prompt
    prompt = f"""
    {timeframe} Options Analysis for {ticker}:
    
    Current Price: ${current_price:.2f}
    Days to Expiration: {days_to_exp}
    Timeframe Category: {timeframe}
    
    Options Metrics:
    - Total Call Volume: {total_call_volume:,}
    - Total Put Volume: {total_put_volume:,}
    - Put/Call Volume Ratio: {total_put_volume/max(total_call_volume, 1):.2f}
    - Average IV: {avg_iv:.1f}%
    
    {uw_metrics}
    
    Top 5 Call Strikes by Volume:
    {calls.nlargest(5, 'volume')[['strike', 'lastPrice', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False) if not calls.empty else 'No call data'}
    
    Top 5 Put Strikes by Volume:
    {puts.nlargest(5, 'volume')[['strike', 'lastPrice', 'volume', 'impliedVolatility', 'moneyness']].to_string(index=False) if not puts.empty else 'No put data'}
    
    Provide {timeframe}-specific analysis covering:
    1. Optimal strategy for this timeframe ({days_to_exp} days)
    2. Key levels and price targets
    3. Time decay considerations
    4. IV and volatility outlook
    5. Risk management for this timeframe
    6. Entry/exit timing strategies
    
    Tailor advice specifically for {timeframe} characteristics.
    Keep analysis under 300 words but be actionable.
    """
    
    # Use selected AI model
    if st.session_state.ai_model == "Multi-AI":
        analyses = multi_ai.multi_ai_consensus_enhanced(prompt)
        if analyses:
            result = f"## 🤖 Multi-AI {timeframe} Analysis\n\n"
            for model, analysis in analyses.items():
                result += f"### {model}:\n{analysis}\n\n---\n\n"
            return result
        else:
            return f"No AI models available for {timeframe} analysis."
    elif st.session_state.ai_model == "OpenAI" and openai_client:
        return multi_ai.analyze_with_openai(prompt)
    elif st.session_state.ai_model == "Gemini" and gemini_model:
        return multi_ai.analyze_with_gemini(prompt)
    elif st.session_state.ai_model == "Grok" and grok_enhanced:
        return multi_ai.analyze_with_grok(prompt)
    else:
        return f"No AI model configured for {timeframe} analysis."

# =============================================================================
# LIVE UPDATE SYSTEM
# =============================================================================

def update_live_data():
    """Update live data in background without full refresh"""
    current_time = time.time()
    if current_time - st.session_state.last_update_time < 10:  # Throttle updates
        return
    
    # Update watchlist data
    watchlist_tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(get_live_quote_with_cache, ticker, st.session_state.selected_tz): ticker 
                  for ticker in watchlist_tickers[:10]}  # Limit to 10 for speed
        
        updated_quotes = {}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                quote = future.result()
                updated_quotes[ticker] = quote
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
    
    # Update session state with new data
    st.session_state.last_update_time = current_time
    return updated_quotes

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Main app
st.title("🔥 AI Radar Pro — Live Trading Assistant with Unusual Whales")

# Auto-refresh mechanism
if st.session_state.auto_refresh:
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = 0
    
    current_time = time.time()
    if current_time - st.session_state.last_auto_refresh > st.session_state.refresh_interval:
        st.session_state.last_auto_refresh = current_time
        st.rerun()

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

# Data Source Configuration
st.sidebar.subheader("📊 Data Configuration")
available_sources = ["Unusual Whales"]
if twelvedata_client:
    available_sources.append("Twelve Data")
available_sources.append("Yahoo Finance")

st.session_state.data_source = st.sidebar.selectbox("Primary Data Source", available_sources, 
                                                     index=available_sources.index(st.session_state.data_source) if st.session_state.data_source in available_sources else 0)

# Data source status
st.sidebar.subheader("Data Sources Status")

if uw_client:
    st.sidebar.success("🔥 Unusual Whales Connected (PRIMARY)")
else:
    st.sidebar.error("❌ Unusual Whales Not Connected")

if twelvedata_client:
    st.sidebar.success("✅ Twelve Data Connected")
else:
    st.sidebar.warning("⚠️ Twelve Data Not Connected")

st.sidebar.success("✅ Yahoo Finance Connected (Fallback)")

if FINNHUB_KEY:
    st.sidebar.success("✅ Finnhub API Connected")
else:
    st.sidebar.warning("⚠️ Finnhub API Not Found")

# Debug toggle and API test
debug_mode = st.sidebar.checkbox("🛠 Debug Mode", help="Show API response details")
st.session_state.debug_mode = debug_mode

if debug_mode:
    st.sidebar.subheader("🔬 UW Enhanced Data Debug")
    debug_ticker = st.sidebar.selectbox("Debug Ticker", CORE_TICKERS[:10])
    
    if st.sidebar.button("🧪 Test UW Integration"):
        with st.sidebar:
            st.write("**Testing UW Functions:**")
            
            if uw_client:
                # Test UW quote
                uw_quote = uw_client.get_stock_state(debug_ticker)
                st.write(f"UW Quote: {'✅' if not uw_quote.get('error') else '❌'}")
                
                # Test UW flow
                uw_flow = uw_client.get_flow_alerts(debug_ticker)
                st.write(f"UW Flow: {'✅' if not uw_flow.get('error') else '❌'}")
                
                # Test UW Greeks
                uw_greeks = uw_client.get_greek_exposure(debug_ticker)
                st.write(f"UW Greeks: {'✅' if not uw_greeks.get('error') else '❌'}")
                
                # Test ATM Chains
                uw_atm = uw_client.get_atm_chains(debug_ticker)
                st.write(f"UW ATM Chains: {'✅' if not uw_atm.get('error') else '❌'}")
                
                if st.checkbox("Show UW Raw Data"):
                    st.json({"quote": uw_quote, "flow": uw_flow, "greeks": uw_greeks, "atm": uw_atm})
            else:
                st.error("UW Client not initialized")
            
            # Test enhanced analysis
            enhanced_opts = get_enhanced_options_analysis(debug_ticker)
            st.write(f"Enhanced Options: {'✅' if not enhanced_opts.get('error') else '❌'}")

# Auto-refresh controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)

with col2:
    st.session_state.refresh_interval = st.selectbox("Interval", [10, 30, 60], index=1)

with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.session_state.live_data_cache.clear()
        st.rerun()

with col4:
    current_time = current_tz.strftime("%I:%M:%S %p")
    market_open = 9 <= current_tz.hour < 16
    status = "🟢 Open" if market_open else "🔴 Closed"
    st.write(f"**{status}** | {current_time} {tz_label}")

# Create tabs
tabs = st.tabs([
    "📊 Live Quotes", 
    "📋 Watchlist Manager", 
    "🔥 Catalyst Scanner", 
    "📈 Market Analysis", 
    "🤖 AI Playbooks", 
    "🌐 Sector/ETF Tracking", 
    "🎯 Options Flow", 
    "💰 Lottos", 
    "🗓️ Earnings Plays", 
    "📰 Important News",
    "🦅 Twitter/X Market Sentiment & Rumors"
])

# Global timestamp
data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
data_sources = []
if uw_client:
    data_sources.append("Unusual Whales")
if twelvedata_client:
    data_sources.append("Twelve Data")
data_sources.append("Yahoo Finance")
data_source_info = " + ".join(data_sources)

# AI model info
ai_info = f"AI: {st.session_state.ai_model}"
if st.session_state.ai_model == "Multi-AI":
    active_models = multi_ai.get_available_models()
    ai_info += f" ({'+'.join(active_models)})" if active_models else " (None Available)"

# TAB 1: Live Quotes (Optimized)
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist & Market Movers")
    
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
        with st.spinner(f"Getting quote for {search_ticker}..."):
            quote = get_live_quote_with_cache(search_ticker, tz_label)
            if not quote["error"]:
                st.success(f"Quote for {search_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                col1.metric(search_ticker, f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.metric("Volume", f"{quote['volume']:,}")
                
                # Show UW-specific data if available
                if quote.get('data_source') == 'Unusual Whales':
                    col4.metric("Market Time", quote.get('market_time', 'Unknown'))
                
                # Extended UW data display
                if quote.get('data_source') == 'Unusual Whales':
                    st.markdown("#### 🔥 Unusual Whales Extended Data")
                    uw_col1, uw_col2, uw_col3, uw_col4, uw_col5 = st.columns(5)
                    uw_col1.metric("Open", f"${quote.get('open', 0):.2f}")
                    uw_col2.metric("High", f"${quote.get('high', 0):.2f}")
                    uw_col3.metric("Low", f"${quote.get('low', 0):.2f}")
                    uw_col4.metric("Total Volume", f"{quote.get('total_volume', 0):,}")
                    uw_col5.metric("Prev Close", f"${quote.get('previous_close', 0):.2f}")
                    
                    # Show tape time if available
                    tape_time = quote.get('tape_time', '')
                    if tape_time:
                        st.caption(f"**Tape Time:** {tape_time}")
                
                # Session breakdown
                st.markdown("#### Session Performance")
                sess_col1, sess_col2, sess_col3 = st.columns(3)
                sess_col1.metric("Premarket", f"{quote['premarket_change']:+.2f}%")
                sess_col2.metric("Intraday", f"{quote['intraday_change']:+.2f}%")
                sess_col3.metric("After Hours", f"{quote['postmarket_change']:+.2f}%")
                
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
    
    # Watchlist display with live updates
    tickers = st.session_state.watchlists[st.session_state.active_watchlist]
    st.markdown("### Your Watchlist")
    if not tickers:
        st.warning("No symbols in watchlist. Add some in the Watchlist Manager tab or check Market Movers below.")
    else:
        # Create placeholder for live updates
        watchlist_placeholder = st.empty()
        
        with watchlist_placeholder.container():
            for ticker in tickers:
                quote = get_live_quote_with_cache(ticker, tz_label)
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
                    
                    # Show UW-specific data if available
                    if quote.get('data_source') == 'Unusual Whales':
                        col4.write("**🔥 UW Data**")
                        col4.write(f"Market Time: {quote.get('market_time', 'Unknown')}")
                        col4.write(f"Total Vol: {quote.get('total_volume', 0):,}")
                        col4.write(f"OHLC: {quote.get('open', 0):.2f}/{quote.get('high', 0):.2f}/{quote.get('low', 0):.2f}/{quote['last']:.2f}")
                        tape_time = quote.get('tape_time', '')
                        if tape_time:
                            col4.caption(f"Tape: {tape_time[-8:]}")  # Show just the time part
                    
                    if abs(quote['change_percent']) >= 2.0:
                        if col4.button(f"🎯 AI Analysis", key=f"quotes_ai_{ticker}"):
                            with st.spinner(f"Analyzing {ticker}..."):
                                if uw_client:
                                    options_data = get_enhanced_options_analysis(ticker)
                                else:
                                    options_data = get_options_data(ticker)
                                analysis = ai_playbook(ticker, quote['change_percent'], "", options_data)
                                st.success(f"🤖 {ticker} Analysis")
                                st.markdown(analysis)
                    
                    # Session data
                    sess_col1, sess_col2, sess_col3, sess_col4 = st.columns([2, 2, 2, 4])
                    sess_col1.caption(f"**PM:** {quote['premarket_change']:+.2f}%")
                    sess_col2.caption(f"**Day:** {quote['intraday_change']:+.2f}%")
                    sess_col3.caption(f"**AH:** {quote['postmarket_change']:+.2f}%")
                    
                    # Show extended UW data in session row for UW sources
                    if quote.get('data_source') == 'Unusual Whales':
                        sess_col4.caption(f"🔥 Prev Close: ${quote.get('previous_close', 0):.2f}")
                    
                    st.divider()

    # Top Market Movers (Optimized)
    st.markdown("### 🌟 Top Market Movers")
    st.caption("Stocks with significant intraday movement from CORE_TICKERS")
    
    # Create movers with concurrent processing
    movers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(get_live_quote_with_cache, ticker, tz_label): ticker for ticker in CORE_TICKERS[:20]}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                quote = future.result()
                if not quote["error"]:
                    mover_data = {
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"],
                        "volume": quote["volume"],
                        "data_source": quote.get("data_source", "Yahoo Finance")
                    }
                    
                    # Add UW-specific fields if available
                    if quote.get('data_source') == 'Unusual Whales':
                        mover_data.update({
                            "open": quote.get("open", 0),
                            "high": quote.get("high", 0),
                            "low": quote.get("low", 0),
                            "total_volume": quote.get("total_volume", 0),
                            "market_time": quote.get("market_time", "Unknown"),
                            "tape_time": quote.get("tape_time", ""),
                            "previous_close": quote.get("previous_close", 0)
                        })
                    
                    movers.append(mover_data)
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')
    
    movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    top_movers = movers[:10]  # Show top 10 movers

    for mover in top_movers:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            direction = "🚀" if mover["change_pct"] > 0 else "📉"
            col1.metric(f"{direction} {mover['ticker']}", f"${mover['price']:.2f}", f"{mover['change_pct']:+.2f}%")
            
            # Show enhanced data if from UW
            if mover.get('data_source') == 'Unusual Whales':
                col2.write("**🔥 UW OHLC**")
                col2.write(f"O: ${mover.get('open', 0):.2f}")
                col2.write(f"H: ${mover.get('high', 0):.2f}")
                col2.write(f"L: ${mover.get('low', 0):.2f}")
                
                col3.write("**Volume/Total**")
                col3.write(f"{mover['volume']:,}")
                col3.write(f"Total: {mover.get('total_volume', 0):,}")
                col3.caption(f"Market: {mover.get('market_time', 'Unknown')}")
            else:
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
            quote = get_live_quote_with_cache(search_add_ticker, tz_label)
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

# TAB 3: Enhanced Catalyst Scanner
with tabs[2]:
    st.subheader("🔥 Enhanced Real-Time Catalyst Scanner")
    st.caption("Comprehensive news analysis from Unusual Whales, Finnhub, and Yahoo Finance")
    
    # Show data sources status
    sources_status = []
    if uw_client:
        sources_status.append("🔥 Unusual Whales")
    else:
        sources_status.append("❌ Unusual Whales")
    if FINNHUB_KEY:
        sources_status.append("✅ Finnhub")
    else:
        sources_status.append("❌ Finnhub")
    sources_status.append("✅ Yahoo Finance")
    
    st.info(f"**News Sources:** {' | '.join(sources_status)}")
    
    # Search specific stock catalysts
    col1, col2 = st.columns([3, 1])
    with col1:
        search_catalyst_ticker = st.text_input("🔍 Search catalysts for stock", placeholder="Enter ticker", key="search_catalyst").upper().strip()
    with col2:
        search_catalyst = st.button("🔍 Analyze Catalysts", key="search_catalyst_btn")
    
    if search_catalyst and search_catalyst_ticker:
        with st.spinner(f"Searching all news sources for {search_catalyst_ticker} catalysts..."):
            # Get comprehensive catalyst analysis
            catalyst_data = get_stock_specific_catalysts(search_catalyst_ticker)
            quote = get_live_quote_with_cache(search_catalyst_ticker, tz_label)
            
            if not quote["error"]:
                st.success(f"Catalyst Analysis for {search_catalyst_ticker} - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
                # Price and volume info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                col2.metric("Volume", f"{quote['volume']:,}")
                col3.metric("Total Catalysts", catalyst_data["catalyst_summary"]["total_catalysts"])
                col4.metric("Highest Impact", f"{catalyst_data['catalyst_summary']['highest_impact']}")
                
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
            quote = get_live_quote_with_cache(search_analysis_ticker, tz_label)
            if not quote["error"]:
                news = get_finnhub_news(search_analysis_ticker)
                catalyst = news[0].get('headline', '') if news else "Recent market movement"
                
                # Get enhanced options data for analysis
                if uw_client:
                    options_data = get_enhanced_options_analysis(search_analysis_ticker)
                else:
                    options_data = get_options_data(search_analysis_ticker)
                
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
                
                # Show enhanced options data if available
                if options_data and not options_data.get("error"):
                    st.markdown("#### Options Metrics")
                    if options_data.get("data_source") == "Unusual Whales":
                        st.markdown("**🔥 Unusual Whales Options Data**")
                        enhanced = options_data.get('enhanced_metrics', {})
                        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                        opt_col1.metric("Flow Alerts", enhanced.get('total_flow_alerts', 'N/A'))
                        opt_col2.metric("Flow Sentiment", enhanced.get('flow_sentiment', 'Neutral'))
                        opt_col3.metric("ATM P/C Ratio", f"{enhanced.get('atm_put_call_ratio', 0):.2f}")
                        opt_col4.metric("Delta/Gamma", f"{enhanced.get('total_delta', 'N/A')}/{enhanced.get('total_gamma', 'N/A')}")
                    else:
                        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                        opt_col1.metric("IV", f"{options_data.get('iv', 0):.1f}%")
                        opt_col2.metric("Put/Call", f"{options_data.get('put_call_ratio', 0):.2f}")
                        opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,}")
                        opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,}")
                        st.caption("Note: Options data from Yahoo Finance (fallback)")
                
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
                quote = get_live_quote_with_cache(ticker, tz_label)
                if not quote["error"]:
                    movers.append({
                        "ticker": ticker,
                        "change_pct": quote["change_percent"],
                        "price": quote["last"],
                        "data_source": quote.get("data_source", "Yahoo Finance")
                    })
            
            analysis = ai_market_analysis(news_items, movers)
            
            st.success("🤖 AI Market Analysis Complete")
            st.markdown(analysis)
            
            with st.expander("📊 Supporting Data"):
                st.write("**Top Market Movers:**")
                for mover in sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)[:5]:
                    st.write(f"• {mover['ticker']}: {mover['change_pct']:+.2f}% | Source: {mover.get('data_source', 'Yahoo Finance')}")
                
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
                            
                            # Display AI play analysis
                            st.markdown("**AI Trading Play:**")
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
        quote = get_live_quote_with_cache(search_playbook_ticker, tz_label)
        
        if not quote["error"]:
            with st.spinner(f"AI generating playbook for {search_playbook_ticker}..."):
                news = get_finnhub_news(search_playbook_ticker)
                catalyst = news[0].get('headline', '') if news else ""
                
                # Get enhanced options data for playbook
                if uw_client:
                    options_data = get_enhanced_options_analysis(search_playbook_ticker)
                else:
                    options_data = get_options_data(search_playbook_ticker)
                
                playbook = ai_playbook(search_playbook_ticker, quote["change_percent"], catalyst, options_data)
                
                st.success(f"✅ {search_playbook_ticker} Trading Playbook - Updated: {quote['last_updated']} | Source: {quote.get('data_source', 'Yahoo Finance')}")
                
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
                
                # Show enhanced options data if available
                if options_data and not options_data.get("error"):
                    st.markdown("#### Enhanced Options Analysis")
                    if options_data.get("data_source") == "Unusual Whales":
                        st.markdown("**🔥 Unusual Whales Premium Options Data**")
                        enhanced = options_data.get('enhanced_metrics', {})
                        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                        opt_col1.metric("Flow Alerts", enhanced.get('total_flow_alerts', 'N/A'))
                        opt_col2.metric("Flow Sentiment", enhanced.get('flow_sentiment', 'Neutral'))
                        opt_col3.metric("ATM P/C Ratio", f"{enhanced.get('atm_put_call_ratio', 0):.2f}")
                        opt_col4.metric("Greeks", f"Δ:{enhanced.get('total_delta', 'N/A')} Γ:{enhanced.get('total_gamma', 'N/A')}")
                    else:
                        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                        opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                        opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                        opt_col3.metric("Call OI", f"{options_data.get('top_call_oi', 0):,} @ ${options_data.get('top_call_oi_strike', 0)}")
                        opt_col4.metric("Put OI", f"{options_data.get('top_put_oi', 0):,} @ ${options_data.get('top_put_oi_strike', 0)}")
                        st.caption("Note: Using Yahoo Finance options data (fallback)")
                
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
            quote = get_live_quote_with_cache(selected_ticker, tz_label)
            
            if not quote["error"]:
                with st.spinner(f"AI analyzing {selected_ticker}..."):
                    # Get enhanced options data for analysis
                    if uw_client:
                        options_data = get_enhanced_options_analysis(selected_ticker)
                    else:
                        options_data = get_options_data(selected_ticker)
                    
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
                quote = get_live_quote_with_cache(etf_search_ticker)
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
        quote = get_live_quote_with_cache(ticker, tz_label)
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
            
            if col4.button(f"Add {ticker} to Watchlist", key=f"sector_etf_add_{ticker}"):
                current_list = st.session_state.watchlists[st.session_state.active_watchlist]
                if ticker not in current_list:
                    current_list.append(ticker)
                    st.session_state.watchlists[st.session_state.active_watchlist] = current_list
                    st.success(f"Added {ticker} to watchlist!")
                    st.rerun()

            st.divider()

# TAB 7: Enhanced Options Flow with UW Integration (Optimized)
with tabs[6]:
    st.subheader("🎯 Enhanced Options Flow Analysis")
    st.markdown("**Advanced options flow analysis with Unusual Whales integration and timeframe-specific strategies.**")

    # Ticker selection
    col1, col2 = st.columns([3, 1])
    with col1:
        flow_ticker = st.selectbox("Select Ticker for Options Flow", options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], key="flow_ticker")
    with col2:
        if st.button("Refresh All Data", key="refresh_flow_data"):
            st.cache_data.clear()
            st.session_state.live_data_cache.clear()
            st.rerun()

    # Get base data
    quote = get_live_quote_with_cache(flow_ticker, st.session_state.selected_tz)
    
    if not quote.get("error"):
        # Basic quote info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
        col2.metric("Volume", f"{quote['volume']:,}")
        col3.metric("Data Source", quote.get('data_source', 'Unknown'))
        col4.metric("Last Updated", quote['last_updated'][-8:])

        # Enhanced UW Flow Analysis Section
        if uw_client:
            st.markdown("### 🔥 Unusual Whales Flow Intelligence")
            
            with st.spinner(f"Fetching comprehensive flow data from Unusual Whales for {flow_ticker}..."):
                # Get comprehensive flow data with caching
                flow_alerts_data = uw_client.get_flow_alerts(flow_ticker)
                options_volume_data = uw_client.get_options_volume(flow_ticker)
                hottest_chains_data = get_hottest_chains_analysis()
                
                # Analyze the data
                flow_analysis = analyze_flow_alerts(flow_alerts_data, flow_ticker)
                volume_analysis = analyze_options_volume(options_volume_data, flow_ticker)
                
                # Display UW Flow Alerts
                st.markdown("#### 🔥 Flow Alerts")
                if not flow_analysis.get("error"):
                    summary = flow_analysis.get("summary", {})
                    
                    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
                    alert_col1.metric("Total Alerts", summary.get("total_alerts", 0))
                    alert_col2.metric("Call Alerts", summary.get("call_alerts", 0))
                    alert_col3.metric("Put Alerts", summary.get("put_alerts", 0))
                    alert_col4.metric("Flow Sentiment", summary.get("flow_sentiment", "Neutral"))
                    
                    # Premium metrics
                    prem_col1, prem_col2, prem_col3 = st.columns(3)
                    prem_col1.metric("Total Premium", f"${summary.get('total_premium', 0):,.0f}")
                    prem_col2.metric("Bullish Flow", f"${summary.get('bullish_flow', 0):,.0f}")
                    prem_col3.metric("Bearish Flow", f"${summary.get('bearish_flow', 0):,.0f}")
                    
                    # Display top alerts
                    if flow_analysis.get("alerts"):
                        with st.expander("📋 Recent Flow Alerts"):
                            alerts_df = pd.DataFrame(flow_analysis["alerts"])
                            if not alerts_df.empty:
                                # Sort by premium
                                alerts_df = alerts_df.sort_values('premium', ascending=False)
                                st.dataframe(alerts_df.head(10), use_container_width=True)
                else:
                    st.info(f"Flow Alerts: {flow_analysis.get('error', 'No data available')}")
        else:
            st.error("🔥 Unusual Whales API required for premium options flow analysis")
            st.info("Configure your Unusual Whales API key to access enhanced flow data")
            flow_analysis = {"error": "UW not available"}
            volume_analysis = {"error": "UW not available"}
            hottest_chains_data = {"error": "UW not available"}

# Continue with remaining tabs 8-11 implementation...

# TAB 8: Enhanced Lottos
with tabs[7]:
    st.subheader("💰 Enhanced Lotto Plays with Flow Intelligence")
    st.markdown("**High-risk, high-reward options under $1.00 with Unusual Whales flow analysis for better edge detection.**")

    # Ticker selection
    col1, col2 = st.columns([3, 1])
    with col1:
        lotto_ticker = st.selectbox("Select Ticker for Lotto Analysis", options=CORE_TICKERS + st.session_state.watchlists[st.session_state.active_watchlist], key="lotto_ticker")
    with col2:
        if st.button("Find Enhanced Lottos", key="find_enhanced_lottos"):
            st.cache_data.clear()
            st.rerun()

    # Basic lotto implementation
    option_chain = get_option_chain(lotto_ticker, st.session_state.selected_tz)
    if not option_chain.get("error"):
        calls = option_chain["calls"]
        puts = option_chain["puts"]
        
        # Find lotto opportunities
        lotto_calls = calls[calls['lastPrice'] <= 1.0].copy() if not calls.empty else pd.DataFrame()
        lotto_puts = puts[puts['lastPrice'] <= 1.0].copy() if not puts.empty else pd.DataFrame()
        
        st.success(f"Found {len(lotto_calls)} call lottos and {len(lotto_puts)} put lottos under $1.00")

# TAB 9: Earnings Plays
with tabs[8]:
    st.subheader("🗓️ Earnings Plays with UW Integration")
    
    st.write("Track upcoming earnings reports and get AI analysis for potential earnings plays using Unusual Whales and other data sources.")
    
    # Try to get UW economic calendar first
    if uw_client:
        st.info("🔥 Using Unusual Whales economic calendar for enhanced earnings detection")
    else:
        st.info("Using simulated earnings data. For live earnings calendar with UW integration, configure your Unusual Whales API key.")
    
    if st.button("📊 Get Enhanced Earnings Plays", type="primary"):
        with st.spinner("AI analyzing earnings reports with enhanced data..."):
            earnings_today = get_earnings_calendar()
            
            if not earnings_today:
                st.info("No earnings reports found for today.")
            else:
                st.markdown("### Today's Earnings Reports")
                for report in earnings_today:
                    ticker = report["ticker"]
                    time_str = report["time"]
                    source = report.get("source", "Unknown")
                    
                    st.markdown(f"**{ticker}** - Earnings **{time_str}** | Source: {source}")
                    
                    # Get live quote and enhanced options data for earnings analysis
                    quote = get_live_quote_with_cache(ticker)
                    
                    # Use UW options analysis if available
                    if uw_client:
                        options_analysis = get_enhanced_options_analysis(ticker)
                        options_source = "Unusual Whales"
                    else:
                        options_analysis = get_advanced_options_analysis_yf(ticker)
                        options_source = "Yahoo Finance"
                    
                    if not quote.get("error"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Current Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        col3.metric("Data Source", quote.get('data_source', 'Unknown'))
                        col4.metric("Options Source", options_source)
                    
                    # Enhanced AI earnings analysis
                    if not options_analysis.get("error"):
                        ai_analysis = ai_playbook(ticker, quote.get("change_percent", 0), f"Earnings {time_str} - Enhanced Analysis", options_analysis)
                    else:
                        ai_analysis = f"""
                        **Enhanced AI Analysis for {ticker} Earnings:**
                        - **Date:** {report["date"]}
                        - **Time:** {time_str}
                        - **Current Price:** ${quote.get('last', 0):.2f}
                        - **Daily Change:** {quote.get('change_percent', 0):+.2f}%
                        - **Volume:** {quote.get('volume', 0):,}
                        - **Data Source:** {quote.get('data_source', 'Unknown')}
                        - **Options Source:** {options_source}
                        
                        **Enhanced Analysis Notes:**
                        Monitor for post-earnings volatility and unusual options activity.
                        {"Enhanced UW flow data provides superior institutional insight." if options_source == "Unusual Whales" else "Consider upgrading to UW for premium options flow insights."}
                        """
                    
                    with st.expander(f"🔮 Enhanced AI Analysis for {ticker}"):
                        st.markdown(ai_analysis)
                    st.divider()

# TAB 10: Important News & Economic Calendar
with tabs[9]:
    st.subheader("📰 Important News & Economic Calendar")

    if st.button("📊 Get This Week's Events", type="primary"):
        with st.spinner("Fetching important events from UW and AI sources..."):
            important_events = get_important_events()

            if not important_events:
                st.info("No major economic events scheduled for this week.")
            else:
                st.markdown("### Major Market-Moving Events")
                
                # Check if events came from UW
                if uw_client:
                    try:
                        calendar_result = uw_client.get_economic_calendar()
                        if not calendar_result.get("error"):
                            st.success("🔥 Events sourced from Unusual Whales economic calendar")
                        else:
                            st.info("Events generated by AI (UW calendar unavailable)")
                    except:
                        st.info("Events generated by AI")
                else:
                    st.info("Events generated by AI (UW not configured)")

                for event in sorted(important_events, key=lambda x: x['date']):
                    st.markdown(f"**{event['event']}**")
                    st.write(f"**Date:** {event['date']}")
                    st.write(f"**Time:** {event['time']}")
                    st.write(f"**Impact:** {event['impact']}")
                    st.divider()

# TAB 11: Twitter/X Market Sentiment & Rumors
with tabs[10]:
    st.subheader("🦅 Twitter/X Market Sentiment & Rumors")

    # Important disclaimer
    st.warning("⚠️ **Risk Disclaimer:** Social media content includes unverified rumors and speculation. "
               "Always verify information through official sources before making trading decisions. "
               "Grok analysis may include both verified news and unconfirmed rumors - trade responsibly.")

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
                    st.markdown("### 🦅 Twitter/X Market Analysis")
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
                    quote = get_live_quote_with_cache(social_ticker, tz_label)

                    col1, col2, col3 = st.columns(3)
                    if not quote.get("error"):
                        col1.metric(f"{social_ticker} Price", f"${quote['last']:.2f}", f"{quote['change_percent']:+.2f}%")
                        col2.metric("Volume", f"{quote['volume']:,}")
                        col3.metric("Data Source", quote.get('data_source', 'Yahoo Finance'))

                    # Get Twitter sentiment
                    sentiment_analysis = grok_enhanced.get_twitter_market_sentiment(social_ticker)
                    st.markdown(f"### 🦅 Twitter/X Sentiment for {social_ticker}")
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

# ===== FOOTER (only once, outside all tabs) =====
st.markdown("---")
footer_sources = []
if uw_client:
    footer_sources.append("🔥 Unusual Whales")
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
    f"🔥 AI Radar Pro with Unusual Whales Integration | Data: {footer_text} | {ai_footer}"
    "</div>",
    unsafe_allow_html=True
)
