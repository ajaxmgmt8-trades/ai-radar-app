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
    st.session_state.data_source = "Yahoo Finance"  # Default data source
if "ai_model" not in st.session_state:
    st.session_state.ai_model = "Multi-AI"  # Default to multi-AI
if "use_enhanced_analysis" not in st.session_state:
    st.session_state.use_enhanced_analysis = False

# API Keys
try:
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

# Enhanced Technical Analysis with Custom EMA Setup (9,21,50,113,200,800)
class TechnicalAnalyzer:
    """Enhanced technical analysis with custom EMA setup (9,21,50,113,200,800)"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.ema_periods = [9, 21, 50, 113, 200, 800]  # Custom EMA periods
    
    def get_comprehensive_analysis(self, ticker: str, period: str = "2y") -> Dict:
        """Get comprehensive technical analysis with real chart data using custom EMA setup"""
        try:
            # Fetch historical data (need more data for 800 EMA)
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval="1d")
            
            if hist.empty:
                return {"error": f"No historical data for {ticker}"}
            
            # Need at least 800+ bars for 800 EMA
            if len(hist) < 50:
                return {"error": f"Insufficient data for {ticker} (need more history)"}
            
            # Basic price data
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            daily_change = ((current_price - prev_close) / prev_close) * 100
            
            # Volume analysis
            avg_volume_20 = hist['Volume'].rolling(20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # Custom EMA calculations
            emas = self.calculate_custom_emas(hist['Close'])
            
            # RSI
            rsi = self.calculate_rsi(hist['Close'])
            
            # MACD (using custom EMAs - 9 and 21)
            macd_line, macd_signal, macd_histogram = self.calculate_custom_macd(hist['Close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(hist['Close'])
            
            # Support and Resistance
            support_resistance = self.find_support_resistance(hist)
            
            # ATR for volatility
            atr = self.calculate_atr(hist)
            
            # EMA-based Trend Analysis
            trend_analysis = self.analyze_ema_trend(hist, emas)
            
            # EMA Alignment Analysis
            ema_alignment = self.analyze_ema_alignment(emas, current_price)
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "daily_change": daily_change,
                "volume_analysis": {
                    "current_volume": int(current_volume),
                    "avg_volume_20": int(avg_volume_20),
                    "volume_ratio": volume_ratio,
                    "volume_signal": "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.8 else "Normal"
                },
                "custom_emas": emas,
                "ema_alignment": ema_alignment,
                "momentum": {
                    "rsi": rsi,
                    "rsi_signal": "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_histogram,
                    "macd_crossover": "Bullish" if macd_line > macd_signal else "Bearish"
                },
                "bollinger_bands": {
                    "upper": bb_upper,
                    "middle": bb_middle,
                    "lower": bb_lower,
                    "position": "Upper" if current_price > bb_upper else "Lower" if current_price < bb_lower else "Middle",
                    "squeeze": abs(bb_upper - bb_lower) / bb_middle < 0.1
                },
                "support_resistance": support_resistance,
                "volatility": {
                    "atr": atr,
                    "atr_percentage": (atr / current_price) * 100
                },
                "trend": trend_analysis,
                "error": None
            }
            
        except Exception as e:
            return {"error": f"Technical analysis error for {ticker}: {str(e)}"}
    
    def calculate_custom_emas(self, prices: pd.Series) -> Dict:
        """Calculate custom EMAs: 9, 21, 50, 113, 200, 800"""
        emas = {}
        
        for period in self.ema_periods:
            if len(prices) >= period:
                emas[f"ema_{period}"] = prices.ewm(span=period).mean().iloc[-1]
            else:
                emas[f"ema_{period}"] = None
        
        return emas
    
    def analyze_ema_alignment(self, emas: Dict, current_price: float) -> Dict:
        """Analyze EMA alignment and price position relative to EMAs"""
        
        # Get EMA values (filter out None values)
        valid_emas = {k: v for k, v in emas.items() if v is not None}
        
        if len(valid_emas) < 3:
            return {"alignment": "Insufficient Data", "strength": 0}
        
        # Check if EMAs are in bullish order (shorter > longer)
        bullish_alignment = True
        bearish_alignment = True
        
        ema_periods_available = [int(k.split('_')[1]) for k in valid_emas.keys()]
        ema_periods_available.sort()
        
        for i in range(len(ema_periods_available) - 1):
            period1 = ema_periods_available[i]
            period2 = ema_periods_available[i + 1]
            
            ema1 = emas.get(f"ema_{period1}")
            ema2 = emas.get(f"ema_{period2}")
            
            if ema1 is not None and ema2 is not None:
                if ema1 <= ema2:  # Shorter EMA should be above longer for bullish
                    bullish_alignment = False
                if ema1 >= ema2:  # Shorter EMA should be below longer for bearish
                    bearish_alignment = False
        
        # Determine alignment
        if bullish_alignment:
            alignment = "Bullish Aligned"
            strength = 90
        elif bearish_alignment:
            alignment = "Bearish Aligned"
            strength = 90
        else:
            alignment = "Mixed/Choppy"
            strength = 30
        
        # Price position relative to key EMAs
        price_above_emas = []
        for period in [9, 21, 50]:
            ema_key = f"ema_{period}"
            if emas.get(ema_key) is not None:
                if current_price > emas[ema_key]:
                    price_above_emas.append(period)
        
        return {
            "alignment": alignment,
            "strength": strength,
            "price_above_emas": price_above_emas,
            "key_ema_status": f"Above {len(price_above_emas)}/3 key EMAs (9,21,50)"
        }
    
    def analyze_ema_trend(self, hist: pd.DataFrame, emas: Dict) -> Dict:
        """Analyze trend based on EMA setup"""
        current_price = hist['Close'].iloc[-1]
        
        # Calculate trend strength based on 20-day performance
        if len(hist) >= 20:
            price_20_days_ago = hist['Close'].iloc[-20]
            trend_strength = ((current_price - price_20_days_ago) / price_20_days_ago) * 100
        else:
            trend_strength = 0
        
        # Determine trend based on EMA alignment and price position
        ema_9 = emas.get("ema_9")
        ema_21 = emas.get("ema_21")
        ema_50 = emas.get("ema_50")
        ema_200 = emas.get("ema_200")
        
        if all(x is not None for x in [ema_9, ema_21, ema_50]):
            if current_price > ema_9 > ema_21 > ema_50:
                if ema_200 is not None and ema_50 > ema_200:
                    trend = "Strong Uptrend"
                else:
                    trend = "Uptrend"
            elif current_price < ema_9 < ema_21 < ema_50:
                if ema_200 is not None and ema_50 < ema_200:
                    trend = "Strong Downtrend"
                else:
                    trend = "Downtrend"
            elif current_price > ema_21:
                trend = "Weak Uptrend"
            elif current_price < ema_21:
                trend = "Weak Downtrend"
            else:
                trend = "Sideways"
        else:
            trend = "Insufficient Data"
        
        return {
            "trend": trend,
            "trend_strength": trend_strength,
            "trend_direction": "Bullish" if trend_strength > 0 else "Bearish" if trend_strength < 0 else "Neutral"
        }
    
    def calculate_custom_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD using 9 and 21 EMAs instead of traditional 12/26"""
        ema_9 = prices.ewm(span=9).mean()
        ema_21 = prices.ewm(span=21).mean()
        macd_line = ema_9 - ema_21
        macd_signal = macd_line.ewm(span=9).mean()  # 9-period signal line
        macd_histogram = macd_line - macd_signal
        
        return macd_line.iloc[-1], macd_signal.iloc[-1], macd_histogram.iloc[-1]
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 21, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands using 21-period (aligned with EMA21)"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    def calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift(1))
        low_close = np.abs(hist['Low'] - hist['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def find_support_resistance(self, hist: pd.DataFrame, window: int = 20) -> Dict:
        """Find basic support and resistance levels"""
        # Get recent highs and lows
        recent_data = hist.tail(50)  # Last 50 days
        
        # Find local maxima and minima
        highs = recent_data['High'].rolling(window=window, center=True).max()
        lows = recent_data['Low'].rolling(window=window, center=True).min()
        
        # Get resistance (recent highs)
        resistance_levels = []
        for i in range(len(recent_data)):
            if recent_data['High'].iloc[i] == highs.iloc[i] and not pd.isna(highs.iloc[i]):
                resistance_levels.append(recent_data['High'].iloc[i])
        
        # Get support (recent lows)
        support_levels = []
        for i in range(len(recent_data)):
            if recent_data['Low'].iloc[i] == lows.iloc[i] and not pd.isna(lows.iloc[i]):
                support_levels.append(recent_data['Low'].iloc[i])
        
        # Get nearest levels
        current_price = hist['Close'].iloc[-1]
        
        nearby_resistance = [r for r in resistance_levels if r > current_price]
        nearby_support = [s for s in support_levels if s < current_price]
        
        return {
            "nearest_resistance": min(nearby_resistance) if nearby_resistance else None,
            "nearest_support": max(nearby_support) if nearby_support else None,
            "all_resistance": sorted(set(resistance_levels), reverse=True)[:3],
            "all_support": sorted(set(support_levels), reverse=True)[:3]
        }

# Initialize the technical analyzer
tech_analyzer = TechnicalAnalyzer()

# Initialize data clients
alpha_vantage_client = AlphaVantageClient(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else None
twelvedata_client = TwelveDataClient(TWELVEDATA_KEY) if TWELVEDATA_KEY else None

# Multi-AI Analysis System
class MultiAIAnalyzer:
    """Comprehensive multi-AI analysis system for trading"""
    
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
    
    def multi_ai_consensus(self, ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> Dict[str, str]:
        """Get consensus analysis from all available AI models with enhanced prompts"""
        
        # Use the enhanced comprehensive prompt
        prompt = get_analysis_prompt(ticker, change, catalyst, options_data)
        
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

# Enhanced primary data function - Alpha Vantage/Twelve Data first, Yahoo Finance fallback
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_live_quote(ticker: str, tz: str = "ET") -> Dict:
    """
    Get live stock quote using Alpha Vantage/Twelve Data first, then Yahoo Finance fallback
    """
    tz_zone = ZoneInfo('US/Eastern') if tz == "ET" else ZoneInfo('US/Central')
    tz_label = "ET" if tz == "ET" else "CT"
    
    # Try Alpha Vantage first (if available)
    if alpha_vantage_client:
        try:
            alpha_quote = alpha_vantage_client.get_quote(ticker)
            if not alpha_quote.get("error") and alpha_quote.get("last", 0) > 0:
                alpha_quote["last_updated"] = datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}"
                return alpha_quote
        except Exception as e:
            print(f"Alpha Vantage error for {ticker}: {str(e)}")
    
    # Try Twelve Data second (if available)
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_technical_analysis(ticker: str) -> str:
    """Fetch technical indicators using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            return "No historical data available"
        
        # RSI (14 days)
        delta = hist['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # SMA 50 and 200
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        # MACD
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = macd.iloc[-1] - signal.iloc[-1]
        
        # Trend
        trend = "Bullish" if sma50 > sma200 else "Bearish"
        overbought = rsi > 70
        oversold = rsi < 30
        rsi_status = "Overbought" if overbought else "Oversold" if oversold else "Neutral"
        
        return f"RSI (14): {rsi:.2f} ({rsi_status}), SMA50: {sma50:.2f}, SMA200: {sma200:.2f}, Trend: {trend}, MACD: {macd_val:.2f}"
    except Exception as e:
        return f"Error in technical analysis: {str(e)}"

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
    """
    Placeholder function for earnings calendar
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    return [
        {"ticker": "MSFT", "date": today, "time": "After Hours", "estimate": "$2.50"},
        {"ticker": "NVDA", "date": today, "time": "Before Market", "estimate": "$1.20"},
        {"ticker": "TSLA", "date": today, "time": "After Hours", "estimate": "$0.75"},
    ]

# Enhanced analysis prompt functions
def _construct_enhanced_analysis_prompt(ticker: str, change: float, catalyst: str, options_data: Optional[Dict]) -> str:
    """Enhanced analysis prompt with custom EMA setup (9,21,50,113,200,800)"""
    
    # Get comprehensive technical analysis
    tech_analysis = tech_analyzer.get_comprehensive_analysis(ticker)
    
    # Start with basic prompt structure
    basic_prompt = f"""
    Analyze {ticker} with {change:+.2f}% change today.
    Catalyst: {catalyst if catalyst else "Market movement"}
    """
    
    # Add options data
    if options_data:
        basic_prompt += f"""
        Options Data:
        - Implied Volatility (IV): {options_data.get('iv', 'N/A'):.1f}%
        - Put/Call Ratio: {options_data.get('put_call_ratio', 'N/A'):.2f}
        - Top Call OI: {options_data.get('top_call_oi_strike', 'N/A')} with {options_data.get('top_call_oi', 'N/A'):,} OI
        - Top Put OI: {options_data.get('top_put_oi_strike', 'N/A')} with {options_data.get('top_put_oi', 'N/A'):,} OI
        - Total Contracts: {options_data.get('total_calls', 0) + options_data.get('total_puts', 0):,}
        """
    
    # Add enhanced technical analysis if available
    if not tech_analysis.get("error"):
        emas = tech_analysis['custom_emas']
        alignment = tech_analysis['ema_alignment']
        
        basic_prompt += f"""
        
        COMPREHENSIVE TECHNICAL ANALYSIS (Using 9,21,50,113,200,800 EMAs):
        
        **Current Market Data:**
        - Price: ${tech_analysis['current_price']:.2f}
        - Daily Change: {tech_analysis['daily_change']:+.2f}%
        - Volume: {tech_analysis['volume_analysis']['current_volume']:,} ({tech_analysis['volume_analysis']['volume_signal']} volume)
        
        **Custom EMA Analysis:**
        - EMA 9: ${emas.get('ema_9', 0):.2f if emas.get('ema_9') else 'N/A'}
        - EMA 21: ${emas.get('ema_21', 0):.2f if emas.get('ema_21') else 'N/A'}
        - EMA 50: ${emas.get('ema_50', 0):.2f if emas.get('ema_50') else 'N/A'}
        - EMA 113: ${emas.get('ema_113', 0):.2f if emas.get('ema_113') else 'N/A'}
        - EMA 200: ${emas.get('ema_200', 0):.2f if emas.get('ema_200') else 'N/A'}
        - EMA 800: ${emas.get('ema_800', 0):.2f if emas.get('ema_800') else 'N/A'}
        
        **EMA Alignment & Trend:**
        - EMA Alignment: {alignment['alignment']} (Strength: {alignment['strength']}%)
        - Price vs Key EMAs: {alignment['key_ema_status']}
        - Overall Trend: {tech_analysis['trend']['trend']}
        - 20-day Performance: {tech_analysis['trend']['trend_strength']:+.2f}%
        
        **Momentum Indicators:**
        - RSI: {tech_analysis['momentum']['rsi']:.1f} ({tech_analysis['momentum']['rsi_signal']})
        - Custom MACD (9-21): {tech_analysis['momentum']['macd_crossover']} crossover
        
        **Support/Resistance:**
        - Nearest Support: ${tech_analysis['support_resistance']['nearest_support']:.2f if tech_analysis['support_resistance']['nearest_support'] else 'N/A'}
        - Nearest Resistance: ${tech_analysis['support_resistance']['nearest_resistance']:.2f if tech_analysis['support_resistance']['nearest_resistance'] else 'N/A'}
        
        **Volatility:**
        - ATR: {tech_analysis['volatility']['atr']:.2f} ({tech_analysis['volatility']['atr_percentage']:.1f}% of price)
        
        **Bollinger Bands (21-period):**
        - Position: {tech_analysis['bollinger_bands']['position']} band
        - Upper: ${tech_analysis['bollinger_bands']['upper']:.2f}
        - Lower: ${tech_analysis['bollinger_bands']['lower']:.2f}
        """
    
    # Analysis requirements
    basic_prompt += """
    
    Provide expert trading analysis using the EMA setup (9,21,50,113,200,800) focusing on:
    1. Overall Sentiment (Bullish/Bearish/Neutral) and confidence rating (out of 100).
    2. Trading strategy recommendation (Scalp, Day Trade, Swing, LEAP).
    3. Specific Entry levels based on EMA support/resistance and technical levels.
    4. Target levels using EMA resistance zones and technical analysis.
    5. Stop levels below key EMA support and technical support.
    6. EMA alignment analysis - is this a trending or choppy setup?
    7. Analysis using options metrics (IV, OI, put/call ratio) if available.
    8. Assessment of explosive move potential based on EMA alignment and technical setup.
    
    Use the EMA levels and technical analysis data to provide SPECIFIC price levels.
    Consider EMA alignment strength when assessing trend continuation probability.
    Keep concise and actionable, under 400 words.
    """
    
    return basic_prompt

def _construct_analysis_prompt(ticker: str, change: float, catalyst: str, options_data: Optional[Dict]) -> str:
    """Original basic analysis prompt"""
    options_text = ""
    if options_data:
        options_text = f"""
        Options Data:
        - Implied Volatility (IV): {options_data.get('iv', 'N/A'):.1f}%
        - Put/Call Ratio: {options_data.get('put_call_ratio', 'N/A'):.2f}
        - Top Call OI: {options_data.get('top_call_oi_strike', 'N/A')} with {options_data.get('top_call_oi', 'N/A'):,} OI
        - Top Put OI: {options_data.get('top_put_oi_strike', 'N/A')} with {options_data.get('top_put_oi', 'N/A'):,} OI
        - Total Contracts: {options_data.get('total_calls', 0) + options_data.get('total_puts', 0):,}
        """
    
    return f"""
    Analyze {ticker} with {change:+.2f}% change today.
    Catalyst: {catalyst if catalyst else "Market movement"}
    {options_text}
    
    Provide an expert trading analysis focusing on:
    1. Overall Sentiment (Bullish/Bearish/Neutral) and confidence rating (out of 100).
    2. Trading strategy recommendation (Scalp, Day Trade, Swing, LEAP).
    3. Specific Entry levels, Target levels, and Stop levels.
    4. Key support and resistance levels.
    5. Analysis using options metrics (IV, OI, put/call ratio) if available.
    6. Assessment of explosive move potential.
    
    Keep concise and actionable, under 300 words.
    """

def get_analysis_prompt(ticker: str, change: float, catalyst: str, options_data: Optional[Dict]) -> str:
    """Get either basic or enhanced analysis prompt based on user setting"""
    
    if st.session_state.use_enhanced_analysis:
        return _construct_enhanced_analysis_prompt(ticker, change, catalyst, options_data)
    else:
        return _construct_analysis_prompt(ticker, change, catalyst, options_data)

# Technical display component
def display_technical_summary(ticker: str):
    """Display custom EMA technical analysis summary"""
    tech_data = tech_analyzer.get_comprehensive_analysis(ticker)
    
    if tech_data.get("error"):
        st.warning(f"Technical analysis unavailable: {tech_data['error']}")
        return
    
    st.markdown("#### 📈 EMA Technical Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "RSI (14)", 
        f"{tech_data['momentum']['rsi']:.1f}",
        help=f"Status: {tech_data['momentum']['rsi_signal']}"
    )
    
    col2.metric(
        "EMA Alignment", 
        tech_data['ema_alignment']['alignment'],
        f"{tech_data['ema_alignment']['strength']}% strength"
    )
    
    col3.metric(
        "Support", 
        f"${tech_data['support_resistance']['nearest_support']:.2f}" if tech_data['support_resistance']['nearest_support'] else "N/A"
    )
    
    col4.metric(
        "Resistance", 
        f"${tech_data['support_resistance']['nearest_resistance']:.2f}" if tech_data['support_resistance']['nearest_resistance'] else "N/A"
    )
    
    # EMA status
    emas = tech_data['custom_emas']
    ema_status = []
    for period in [9, 21, 50]:
        ema_val = emas.get(f'ema_{period}')
        if ema_val:
            above_below = "↗" if tech_data['current_price'] > ema_val else "↘"
            ema_status.append(f"EMA{period}: {above_below}")
    
    st.caption(" | ".join(ema_status))
    
    # Volume analysis
    vol_data = tech_data['volume_analysis']
    st.caption(f"Volume: {vol_data['current_volume']:,} ({vol_data['volume_signal']}, {vol_data['volume_ratio']:.1f}x avg)")

# Enhanced AI analysis functions
def ai_playbook(ticker: str, change: float, catalyst: str = "", options_data: Optional[Dict] = None) -> str:
    """Enhanced AI playbook using selected model or multi-AI consensus"""
    
    if st.session_state.ai_model == "Multi-AI":
        # Use multi-AI consensus
        analyses = multi_ai.multi_ai_consensus(ticker, change, catalyst, options_data)
        if analyses:
            # Show individual analyses first
            result = f"## 🤖 Multi-AI Analysis for {ticker}\n\n"
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
            return f"**{ticker} Analysis** (OpenAI API not configured)\n\nCurrent Change: {change:+.2f}%\nSet up OpenAI API key for detailed AI analysis."
        return multi_ai.analyze_with_openai(get_analysis_prompt(ticker, change, catalyst, options_data))
    
    elif st.session_state.ai_model == "Gemini":
        if not gemini_model:
            return f"**{ticker} Analysis** (Gemini API not configured)\n\nCurrent Change: {change:+.2f}%\nSet up Gemini API key for detailed AI analysis."
        return multi_ai.analyze_with_gemini(get_analysis_prompt(ticker, change, catalyst, options_data))
    
    elif st.session_state.ai_model == "Grok":
        if not grok_enhanced:
            return f"**{ticker} Analysis** (Grok API not configured)\n\nCurrent Change: {change:+.2f}%\nSet up Grok API key for detailed AI analysis."
        return multi_ai.analyze_with_grok(get_analysis_prompt(ticker, change, catalyst, options_data))
    
    else:
        return "No AI model selected or configured."

def ai_market_analysis(news_items: List[Dict], movers: List[Dict]) -> str:
    """Enhanced market analysis using selected AI model"""
    
    news_context = "\n".join([f"- {item['title']}" for item in news_items[:5]])
    movers_context = "\n".join([f"- {m['ticker']}: {m['change_pct']:+.2f}%" for m in movers[:5]])
    
    prompt = f"""
    Analyze current market conditions:

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
    
    if st.session_state.ai_model == "Multi-AI":
        analyses = {}
        if openai_client:
            analyses["OpenAI"] = multi_ai.analyze_with_openai(prompt)
        if gemini_model:
            analyses["Gemini"] = multi_ai.analyze_with_gemini(prompt)
        if grok_enhanced:
            analyses["Grok"] = multi_ai.analyze_with_grok(prompt)
        
        if analyses:
            result = "## 🤖 Multi-AI Market Analysis\n\n"
            for model, analysis in analyses.items():
                result += f"### {model} Analysis:\n{analysis}\n\n---\n\n"
            
            synthesis = multi_ai.synthesize_consensus(analyses, "Market")
            result += f"### 🎯 Market Consensus:\n{synthesis}"
            return result
        else:
            return "No AI models available for market analysis."
    
    elif st.session_state.ai_model == "OpenAI":
        if not openai_client:
            return "OpenAI API not configured for AI analysis."
        return multi_ai.analyze_with_openai(prompt)
    
    elif st.session_state.ai_model == "Gemini":
        if not gemini_model:
            return "Gemini API not configured for AI analysis."
        return multi_ai.analyze_with_gemini(prompt)
    
    elif st.session_state.ai_model == "Grok":
        if not grok_enhanced:
            return "Grok API not configured for AI analysis."
        return multi_ai.analyze_with_grok(prompt)
    
    else:
        return "No AI model selected or configured."

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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(get_live_quote, ticker, tz): ticker for ticker in scan_tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    quote = future.result()
                    if not quote["error"]:
                        # Look for significant moves (>1.5% change)
                        if abs(quote["change_percent"]) >= 1.5:
                            candidates.append({
                                "ticker": ticker,
                                "quote": quote,
                                "significance": abs(quote["change_percent"])
                            })
                except Exception as exc:
                    st.error(f'{ticker} generated an exception: {exc}')
        
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
            
            # Get options data for enhanced analysis
            options_data = get_options_data(ticker)
            
            # Generate AI analysis using selected model
            play_analysis = ai_playbook(ticker, quote["change_percent"], catalyst, options_data)

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
                "timestamp": quote['last_updated'],
                "data_source": quote.get('data_source', 'Yahoo Finance')
            }
            plays.append(play)
        
        return plays
    except Exception as e:
        st.error(f"Error generating auto plays: {str(e)}")
        return []

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

# Main app
st.title("🔥 AI Radar Pro — Live Trading Assistant")

# Important disclaimer
st.warning("⚠️ **Trading Disclaimer**: This application provides analysis tools for educational purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. No trading system achieves consistent high accuracy rates. Always use proper risk management and never risk more than you can afford to lose.")

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

# Enhanced Analysis Toggle
st.session_state.use_enhanced_analysis = st.sidebar.checkbox(
    "📈 Enhanced EMA Analysis (9,21,50,113,200,800)", 
    value=st.session_state.use_enhanced_analysis, 
    help="Use comprehensive EMA analysis with custom periods for AI trading analysis"
)

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

# Data Source Toggle
st.sidebar.subheader("📊 Data Configuration")
available_sources = ["Yahoo Finance"]
if alpha_vantage_client:
    available_sources.append("Alpha Vantage")
if twelvedata_client:
    available_sources.append("Twelve Data")
st.session_state.data_source = st.sidebar.selectbox("Select Data Source", available_sources, index=available_sources.index(st.session_state.data_source))

# Data source status
st.sidebar.subheader("Data Sources")

# Debug toggle and API test
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show API response details")
st.session_state.debug_mode = debug_mode

if debug_mode and st.sidebar.button("🧪 Test All APIs"):
    st.sidebar.write("**Testing Data APIs:**")
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

# Show available data sources
if twelvedata_client:
    st.sidebar.success("✅ Twelve Data Connected (Primary)")
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
    st.sidebar.success("✅ Polygon API Connected (News)")
else:
    st.sidebar.warning("⚠️ Polygon API Not Found")

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
tabs = st.tabs(["📊 Live Quotes", "📋 Watchlist Manager", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks", "🌐 Sector/ETF Tracking", "🎲 0DTE & Lottos", "🗓️ Earnings Plays", "📰 Important News","🐦 Twitter/X Market Sentiment & Rumors"])

# Global timestamp
data_timestamp = current_tz.strftime("%B %d, %Y at %I:%M:%S %p") + f" {tz_label}"
data_sources = []
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

# Enhanced analysis info
if st.session_state.use_enhanced_analysis:
    ai_info += " | Enhanced EMA Analysis: ON"

st.markdown(f"<div style='text-align: center; color: #888; font-size: 12px;'>Last Updated: {data_timestamp} | Data: {data_source_info} | {ai_info}</div>", unsafe_allow_html=True)

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
                
                # Enhanced technical summary
                if st.session_state.use_enhanced_analysis:
                    display_technical_summary(search_ticker)
                
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
                col3.caption(f"Source: {quote.get('data_source', 'Yahoo Finance')}")
                
                if abs(quote['change_percent']) >= 2.0:
                    if col4.button(f"🎯 AI Analysis", key=f"ai_{ticker}"):
                        with st.spinner(f"Analyzing {ticker}..."):
                            # Get options data for analysis
                            options_data = get_options_data(ticker)
                            analysis = ai_playbook(ticker, quote['change_percent'], "", options_data)
                            st.success(f"🤖 {ticker} Analysis")
                            st.markdown(analysis)
                
                # Session data
                sess_col1, sess_col2, sess_col3, sess_col4 = st.columns([2, 2, 2, 4])
                sess_col1.caption(f"**PM:** {quote['premarket_change']:+.2f}%")
                sess_col2.caption(f"**Day:** {quote['intraday_change']:+.2f}%")
                sess_col3.caption(f"**AH:** {quote['postmarket_change']:+.2f}%")
                
                # Enhanced technical summary for watchlist items
                if st.session_state.use_enhanced_analysis:
                    display_technical_summary(ticker)
                
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
                    
                    # AI Playbook with options data
                    st.markdown("### 🎯 AI Playbook")
                    catalyst_title = news[0].get('headline', '') if news else ""
                    options_data = get_options_data(ticker)
                    
                    if options_data:
                        st.write("**Options Metrics:**")
                        opt_col1, opt_col2, opt_col3 = st.columns(3)
                        opt_col1.metric("Implied Vol", f"{options_data.get('iv', 0):.1f}%")
                        opt_col2.metric("Put/Call Ratio", f"{options_data.get('put_call_ratio', 0):.2f}")
                        opt_col3.metric("Total Contracts", f"{options_data.get('total_calls', 0) + options_data.get('total_puts', 0):,}")
                        st.caption("Note: Options data is real from yfinance")
                    
                    st.markdown(ai_playbook(ticker, quote['change_percent'], catalyst_title, options_data))
                
                st.divider()

# Continue with remaining tabs - the structure remains the same, but now they use the enhanced analysis when enabled
# The rest of your tabs remain identical to your original code...

# ===== FOOTER (only once, outside all tabs) =====
st.markdown("---")
footer_sources = []
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

if st.session_state.use_enhanced_analysis:
    ai_footer += " | Enhanced EMAs: ON"

st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"🔥 AI Radar Pro | Data: {footer_text} | {ai_footer}<br>"
    f"<small>⚠️ For educational purposes only. Trading involves substantial risk.</small>"
    "</div>",
    unsafe_allow_html=True
)
