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
    "SPX", "NDX", "IWM", "IWF", "HOOY", "MSTY"
]

# Cryptocurrency tickers
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "COIN", "MSTR", "IBIT", "BITF", "HUT", "WULF", "RIOT"
]

# Commodity tickers
COMMODITY_TICKERS = [
    "GLD", "SLV", "USO", "UNG", "GC=F", "SI=F", "CL=F", "NG=F"
]

# AI/Tech focused tickers
AI_TICKERS = [
    "NVDA", "AMD", "GOOGL", "MSFT", "META", "TSLA", "AAPL", "ORCL", "CRM", "NOW",
    "SNOW", "PLTR", "C3AI", "AI", "SMCI", "ARM", "MRVL", "QCOM", "INTC", "MU"
]

# Popular meme/retail stocks
MEME_TICKERS = [
    "GME", "AMC", "BBBY", "HOOD", "SOFI", "PLTR", "WISH", "CLOV", "BB", "NOK"
]

# Biotech/Healthcare
BIOTECH_TICKERS = [
    "MRNA", "BNTX", "GILD", "REGN", "AMGN", "BIIB", "VRTX", "ILMN", "PFE", "JNJ"
]

# Energy sector
ENERGY_TICKERS = [
    "XOM", "CVX", "COP", "SLB", "EOG", "KMI", "OKE", "WMB", "PSX", "VLO"
]

# Financial sector
FINANCIAL_TICKERS = [
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"
]

# All ticker categories for easy selection
TICKER_CATEGORIES = {
    "Core Picks": CORE_TICKERS,
    "AI/Tech": AI_TICKERS,
    "ETFs": ETF_TICKERS,
    "Crypto Related": CRYPTO_TICKERS,
    "Commodities": COMMODITY_TICKERS,
    "Meme/Retail": MEME_TICKERS,
    "Biotech": BIOTECH_TICKERS,
    "Energy": ENERGY_TICKERS,
    "Financials": FINANCIAL_TICKERS
}

# API Configuration
UW_BASE_URL = "https://unusualwhales.com/api"
HEADERS = {
    "User-Agent": "AI-Radar-Pro/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def get_uw_quote(symbol: str, tz_zone=None, tz_label="EST") -> Dict:
    """Get enhanced quote data from Unusual Whales API with session breakdown."""
    try:
        if tz_zone is None:
            tz_zone = ZoneInfo("US/Eastern")
            
        # First get basic quote from UW
        url = f"{UW_BASE_URL}/stock/{symbol}/quote"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        uw_stock_state = response.json()
        
        # Handle different API response formats
        if isinstance(uw_stock_state, list) and len(uw_stock_state) > 0:
            uw_stock_state = uw_stock_state[0]
        elif isinstance(uw_stock_state, dict) and 'data' in uw_stock_state:
            uw_stock_state = uw_stock_state['data']
        
        # Get current market session and time
        now = datetime.datetime.now(tz_zone)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Determine market session
        if now < market_open:
            market_time = "Premarket"
        elif now > market_close:
            market_time = "Aftermarket"
        else:
            market_time = "Regular"
        
        # Get session-specific data using yfinance for enhanced calculations
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")
            current_price = float(uw_stock_state.get("last", uw_stock_state.get("price", 0)))
            
            if not hist.empty:
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1])
                open_price = float(hist['Open'].iloc[-1])
                
                # Calculate session changes
                premarket_change = ((open_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
                intraday_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
                postmarket_change = intraday_change if market_time == "Aftermarket" else 0
            else:
                previous_close = float(uw_stock_state.get("previous_close", current_price))
                open_price = float(uw_stock_state.get("open", current_price))
                premarket_change = ((open_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
                # Assume market close was same as open for simplification (could enhance with more data)
                intraday_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
                postmarket_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
        except:
            # Fallback calculations
            current_price = float(uw_stock_state.get("last", uw_stock_state.get("price", 0)))
            previous_close = float(uw_stock_state.get("previous_close", current_price))
            open_price = float(uw_stock_state.get("open", current_price))
            premarket_change = ((open_price - previous_close) / previous_close) * 100 if previous_close > 0 else 0
            # Assume market close was same as open for simplification (could enhance with more data)
            intraday_change = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
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
        uw_stock_state = {}
        uw_stock_state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " EST"
        uw_stock_state["data_source"] = "Unusual Whales"
        uw_stock_state["error"] = f"UW API error: {str(e)}"
        uw_stock_state.setdefault("premarket_change", 0)
        uw_stock_state.setdefault("intraday_change", 0)
        uw_stock_state.setdefault("postmarket_change", 0)
        return uw_stock_state

def get_yfinance_quote(symbol: str, tz_zone=None, tz_label="EST") -> Dict:
    """Fallback to yfinance for quote data."""
    try:
        if tz_zone is None:
            tz_zone = ZoneInfo("US/Eastern")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="2d", interval="1d")
        
        if hist.empty:
            return {"error": f"No data available for {symbol}", "data_source": "yfinance"}
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if current_price == 0 and len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
        
        previous_close = info.get('previousClose', 0)
        if previous_close == 0 and len(hist) > 1:
            previous_close = hist['Close'].iloc[-2]
        
        change = current_price - previous_close if previous_close > 0 else 0
        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
        
        quote = {
            "last": current_price,
            "bid": info.get('bid', current_price - 0.01),
            "ask": info.get('ask', current_price + 0.01),
            "volume": info.get('volume', 0),
            "total_volume": info.get('volume', 0),
            "change": change,
            "change_percent": change_percent,
            "premarket_change": 0,  # yfinance doesn't provide session breakdown
            "intraday_change": change_percent,
            "postmarket_change": 0,
            "previous_close": previous_close,
            "market_open": info.get('open', current_price),
            "open": info.get('open', current_price),
            "high": info.get('dayHigh', current_price),
            "low": info.get('dayLow', current_price),
            "market_time": "",
            "tape_time": "",
            "last_updated": datetime.datetime.now(tz_zone).strftime("%Y-%m-%d %H:%M:%S") + f" {tz_label}",
            "error": None,
            "data_source": "yfinance"
        }
        
        return quote
        
    except Exception as e:
        return {
            "error": f"yfinance error for {symbol}: {str(e)}",
            "data_source": "yfinance",
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " EST"
        }

def get_options_data(symbol: str, expiry_date: str) -> Dict:
    """Get options chain data from Unusual Whales API."""
    try:
        url = f"{UW_BASE_URL}/options/{symbol}/{expiry_date}"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Options data error: {str(e)}"}

def get_atm_chains(symbol: str) -> Dict:
    """Get ATM (At-The-Money) options chains with proper error handling."""
    try:
        # Get current stock price first
        stock_data = get_uw_quote(symbol)
        if stock_data.get("error") or "last" not in stock_data:
            # Fallback to yfinance
            stock_data = get_yfinance_quote(symbol)
            if stock_data.get("error"):
                return {"error": f"Could not fetch stock price for {symbol}"}
        
        current_price = float(stock_data["last"])
        
        # Get available expiry dates from UW API
        expiry_url = f"{UW_BASE_URL}/options/{symbol}/expirations"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        expiry_response = requests.get(expiry_url, headers=headers, timeout=10)
        expiry_response.raise_for_status()
        expiry_data = expiry_response.json()
        
        # Handle different response formats
        expirations = []
        if isinstance(expiry_data, dict):
            if "expirations" in expiry_data:
                expirations = expiry_data["expirations"]
            elif "data" in expiry_data and isinstance(expiry_data["data"], list):
                expirations = expiry_data["data"]
        elif isinstance(expiry_data, list):
            expirations = expiry_data
        
        if not expirations:
            return {"error": f"No expiry dates available for {symbol}"}
        
        # Get the nearest expiry date
        nearest_expiry = expirations[0]
        
        # Get options chain for the nearest expiry
        options_data = get_options_data(symbol, nearest_expiry)
        if options_data.get("error"):
            return options_data
        
        # Handle different options data formats
        calls = []
        puts = []
        
        if isinstance(options_data, dict):
            if "calls" in options_data and "puts" in options_data:
                calls = options_data["calls"]
                puts = options_data["puts"]
            elif "data" in options_data:
                data = options_data["data"]
                if isinstance(data, dict):
                    calls = data.get("calls", [])
                    puts = data.get("puts", [])
                elif isinstance(data, list):
                    # Separate calls and puts from mixed list
                    for option in data:
                        if option.get("type") == "call":
                            calls.append(option)
                        elif option.get("type") == "put":
                            puts.append(option)
        
        if not calls and not puts:
            return {"error": f"No options data available for {symbol}"}
        
        # Find strikes closest to current price
        all_strikes = set()
        for call in calls:
            strike = call.get("strike") or call.get("strike_price")
            if strike:
                all_strikes.add(float(strike))
        for put in puts:
            strike = put.get("strike") or put.get("strike_price")
            if strike:
                all_strikes.add(float(strike))
        
        if not all_strikes:
            return {"error": f"No valid strikes found for {symbol}"}
        
        # Find the strike closest to current price
        atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
        
        # Filter for ATM options
        atm_calls = []
        atm_puts = []
        
        for call in calls:
            strike = call.get("strike") or call.get("strike_price")
            if strike and float(strike) == atm_strike:
                atm_calls.append(call)
        
        for put in puts:
            strike = put.get("strike") or put.get("strike_price")
            if strike and float(strike) == atm_strike:
                atm_puts.append(put)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "atm_strike": atm_strike,
            "expiry": nearest_expiry,
            "atm_calls": atm_calls,
            "atm_puts": atm_puts,
            "all_expirations": expirations,
            "error": None
        }
        
    except requests.RequestException as e:
        return {"error": f"API request failed for {symbol}: {str(e)}"}
    except Exception as e:
        return {"error": f"ATM chains error for {symbol}: {str(e)}"}

def get_market_movers() -> Dict:
    """Get market movers from Unusual Whales API."""
    try:
        url = f"{UW_BASE_URL}/market/movers"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Market movers error: {str(e)}"}

def get_unusual_options_activity(symbol: str = None) -> Dict:
    """Get unusual options activity from Unusual Whales API."""
    try:
        if symbol:
            url = f"{UW_BASE_URL}/options/unusual/{symbol}"
        else:
            url = f"{UW_BASE_URL}/options/unusual"
        
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Unusual options activity error: {str(e)}"}

def get_options_flow(symbol: str = None, limit: int = 100) -> Dict:
    """Get options flow data from Unusual Whales API."""
    try:
        if symbol:
            url = f"{UW_BASE_URL}/options/flow/{symbol}"
        else:
            url = f"{UW_BASE_URL}/options/flow"
        
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        params = {"limit": limit}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Options flow error: {str(e)}"}

def get_congressional_trades() -> Dict:
    """Get recent congressional trading data from Unusual Whales API."""
    try:
        url = f"{UW_BASE_URL}/congress/recent"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Congressional trades error: {str(e)}"}

def get_insider_trades(symbol: str) -> Dict:
    """Get insider trading data for a specific symbol."""
    try:
        url = f"{UW_BASE_URL}/insider/{symbol}"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Insider trades error for {symbol}: {str(e)}"}

def get_earnings_calendar(days_ahead: int = 7) -> Dict:
    """Get earnings calendar from Unusual Whales API."""
    try:
        url = f"{UW_BASE_URL}/earnings/calendar"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        params = {"days": days_ahead}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Earnings calendar error: {str(e)}"}

def get_dark_pool_data(symbol: str) -> Dict:
    """Get dark pool trading data for a symbol."""
    try:
        url = f"{UW_BASE_URL}/darkpool/{symbol}"
        headers = HEADERS.copy()
        if st.session_state.get('uw_api_key'):
            headers["Authorization"] = f"Bearer {st.session_state.uw_api_key}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Dark pool data error for {symbol}: {str(e)}"}

def get_ai_analysis(prompt: str, provider: str = "openai", model: str = None) -> str:
    """Get AI analysis using specified provider."""
    try:
        if provider == "openai" and st.session_state.get('openai_api_key'):
            client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
            model = model or "gpt-4"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst with expertise in stocks, options, and market analysis. Provide clear, actionable insights based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif provider == "gemini" and st.session_state.get('gemini_api_key'):
            genai.configure(api_key=st.session_state['gemini_api_key'])
            model = model or 'gemini-pro'
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
            
        else:
            return "AI analysis requires API key configuration."
            
    except Exception as e:
        return f"AI analysis error: {str(e)}"

def create_stock_chart(symbol: str, period: str = "1d", chart_type: str = "candlestick") -> Optional[go.Figure]:
    """Create an interactive stock price chart."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Map period to yfinance format
        if period == "intraday":
            hist = ticker.history(period="1d", interval="5m")
        else:
            hist = ticker.history(period=period)
        
        if hist.empty:
            return None
        
        fig = go.Figure()
        
        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=symbol,
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ))
        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name=symbol,
                line=dict(color='#00aaff', width=2)
            ))
        elif chart_type == "ohlc":
            fig.add_trace(go.Ohlc(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=symbol
            ))
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(0,100,255,0.3)'
        ))
        
        fig.update_layout(
            title=f"{symbol} - {period.upper()} Chart",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_title="Time",
            template="plotly_dark",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart error for {symbol}: {str(e)}")
        return None

def create_options_heatmap(symbol: str, expiry: str) -> Optional[go.Figure]:
    """Create options chain heatmap."""
    try:
        options_data = get_options_data(symbol, expiry)
        if options_data.get("error"):
            return None
        
        calls = options_data.get("calls", [])
        puts = options_data.get("puts", [])
        
        if not calls and not puts:
            return None
        
        # Prepare data for heatmap
        strikes = []
        call_volumes = []
        put_volumes = []
        
        # Get all unique strikes
        all_strikes = set()
        for call in calls:
            strike = call.get("strike", 0)
            if strike:
                all_strikes.add(float(strike))
        for put in puts:
            strike = put.get("strike", 0)
            if strike:
                all_strikes.add(float(strike))
        
        all_strikes = sorted(list(all_strikes))
        
        # Map volumes to strikes
        for strike in all_strikes:
            strikes.append(strike)
            
            # Find call volume for this strike
            call_vol = 0
            for call in calls:
                if float(call.get("strike", 0)) == strike:
                    call_vol = call.get("volume", 0)
                    break
            call_volumes.append(call_vol)
            
            # Find put volume for this strike
            put_vol = 0
            for put in puts:
                if float(put.get("strike", 0)) == strike:
                    put_vol = put.get("volume", 0)
                    break
            put_volumes.append(put_vol)
        
        # Create heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strikes,
            y=call_volumes,
            name='Call Volume',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=strikes,
            y=[-vol for vol in put_volumes],
            name='Put Volume',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{symbol} Options Volume - {expiry}",
            xaxis_title="Strike Price",
            yaxis_title="Volume",
            template="plotly_dark",
            height=400,
            barmode='overlay'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Options heatmap error: {str(e)}")
        return None

def format_large_number(num: float) -> str:
    """Format large numbers with appropriate suffixes."""
    if pd.isna(num) or num == 0:
        return "0"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def format_volume(num: float) -> str:
    """Format volume numbers."""
    if pd.isna(num) or num == 0:
        return "0"
    
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{int(num)}"

def get_color_for_change(change: float) -> str:
    """Get color based on price change."""
    if change > 0:
        return "🟢"
    elif change < 0:
        return "🔴"
    else:
        return "⚪"

def calculate_technical_indicators(symbol: str, period: str = "1mo") -> Dict:
    """Calculate basic technical indicators."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": "No data available"}
        
        # Calculate moving averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean() if len(hist) >= 50 else None
        hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9).mean()
        hist['MACD_Histogram'] = hist['MACD'] - hist['MACD_Signal']
        
        # Get latest values
        latest = hist.iloc[-1]
        
        indicators = {
            "price": latest['Close'],
            "sma_20": latest['SMA_20'],
            "sma_50": latest['SMA_50'] if pd.notna(latest.get('SMA_50')) else None,
            "rsi": latest['RSI'],
            "macd": latest['MACD'],
            "macd_signal": latest['MACD_Signal'],
            "volume": latest['Volume'],
            "high_52w": hist['High'].max(),
            "low_52w": hist['Low'].min(),
            "error": None
        }
        
        return indicators
        
    except Exception as e:
        return {"error": f"Technical indicators error: {str(e)}"}

def get_sector_performance() -> Dict:
    """Get sector performance using ETF data."""
    try:
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Energy": "XLE",
            "Consumer Discretionary": "XLY",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Utilities": "XLU",
            "Materials": "XLB",
            "Communication": "XLC",
            "Real Estate": "XLRE"
        }
        
        performance_data = []
        
        for sector, etf in sector_etfs.items():
            quote = get_yfinance_quote(etf)
            if not quote.get("error"):
                performance_data.append({
                    "Sector": sector,
                    "ETF": etf,
                    "Price": quote.get("last", 0),
                    "Change %": quote.get("change_percent", 0)
                })
        
        return {"sectors": performance_data, "error": None}
        
    except Exception as e:
        return {"error": f"Sector performance error: {str(e)}"}

def get_market_sentiment() -> Dict:
    """Get market sentiment indicators."""
    try:
        sentiment_data = {}
        
        # VIX data
        vix_quote = get_yfinance_quote("^VIX")
        if not vix_quote.get("error"):
            vix_level = vix_quote.get("last", 0)
            if vix_level < 15:
                vix_sentiment = "Low Fear"
            elif vix_level < 25:
                vix_sentiment = "Normal"
            elif vix_level < 35:
                vix_sentiment = "High Fear"
            else:
                vix_sentiment = "Extreme Fear"
            
            sentiment_data["vix"] = {
                "level": vix_level,
                "sentiment": vix_sentiment,
                "change": vix_quote.get("change_percent", 0)
            }
        
        # Put/Call Ratio approximation using SPY options volume
        # This would require options data - placeholder for now
        sentiment_data["put_call_ratio"] = {
            "ratio": "N/A",
            "sentiment": "Neutral"
        }
        
        # Fear & Greed Index approximation
        # Based on VIX levels and market performance
        spy_quote = get_yfinance_quote("SPY")
        if not spy_quote.get("error"):
            spy_change = spy_quote.get("change_percent", 0)
            vix_level = sentiment_data.get("vix", {}).get("level", 20)
            
            # Simple fear/greed calculation
            if spy_change > 1 and vix_level < 20:
                fg_sentiment = "Greed"
                fg_score = min(80, 50 + spy_change * 5 - vix_level)
            elif spy_change < -1 and vix_level > 25:
                fg_sentiment = "Fear"
                fg_score = max(20, 50 + spy_change * 5 - vix_level)
            else:
                fg_sentiment = "Neutral"
                fg_score = 50
            
            sentiment_data["fear_greed"] = {
                "score": fg_score,
                "sentiment": fg_sentiment
            }
        
        return {"data": sentiment_data, "error": None}
        
    except Exception as e:
        return {"error": f"Market sentiment error: {str(e)}"}

def create_portfolio_tracker():
    """Portfolio tracking functionality."""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    st.subheader("📈 Portfolio Tracker")
    
    # Add new position
    with st.expander("Add New Position"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_symbol = st.text_input("Symbol", key="portfolio_symbol").upper()
        with col2:
            new_shares = st.number_input("Shares", min_value=0.0, step=0.01, key="portfolio_shares")
        with col3:
            new_avg_price = st.number_input("Avg Price", min_value=0.0, step=0.01, key="portfolio_price")
        with col4:
            if st.button("Add Position"):
                if new_symbol and new_shares > 0 and new_avg_price > 0:
                    # Check if position already exists
                    existing_pos = next((p for p in st.session_state.portfolio if p["symbol"] == new_symbol), None)
                    if existing_pos:
                        # Update existing position
                        total_value = (existing_pos["shares"] * existing_pos["avg_price"]) + (new_shares * new_avg_price)
                        total_shares = existing_pos["shares"] + new_shares
                        existing_pos["shares"] = total_shares
                        existing_pos["avg_price"] = total_value / total_shares
                    else:
                        # Add new position
                        st.session_state.portfolio.append({
                            "symbol": new_symbol,
                            "shares": new_shares,
                            "avg_price": new_avg_price,
                            "date_added": datetime.datetime.now().strftime("%Y-%m-%d")
                        })
                    st.success(f"Added {new_symbol} to portfolio")
                    st.rerun()
    
    # Display portfolio
    if st.session_state.portfolio:
        portfolio_data = []
        total_value = 0
        total_cost = 0
        
        for position in st.session_state.portfolio:
            quote = get_uw_quote(position["symbol"])
            if quote.get("error"):
                quote = get_yfinance_quote(position["symbol"])
            
            current_price = quote.get("last", 0)
            market_value = position["shares"] * current_price
            cost_basis = position["shares"] * position["avg_price"]
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            portfolio_data.append({
                "Symbol": position["symbol"],
                "Shares": position["shares"],
                "Avg Price": f"${position['avg_price']:.2f}",
                "Current Price": f"${current_price:.2f}",
                "Market Value": f"${market_value:.2f}",
                "Cost Basis": f"${cost_basis:.2f}",
                "Unrealized P&L": f"${unrealized_pnl:.2f}",
                "P&L %": f"{unrealized_pnl_pct:+.2f}%",
                "Date Added": position["date_added"]
            })
            
            total_value += market_value
            total_cost += cost_basis
        
        # Portfolio summary
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${total_value:.2f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:.2f}")
        with col3:
            st.metric("Total P&L", f"${total_pnl:.2f}", f"{total_pnl_pct:+.2f}%")
        with col4:
            st.metric("Positions", len(st.session_state.portfolio))
        
        # Portfolio table
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Remove positions
        with st.expander("Remove Position"):
            symbols = [p["symbol"] for p in st.session_state.portfolio]
            remove_symbol = st.selectbox("Select position to remove:", symbols)
            if st.button("Remove Position"):
                st.session_state.portfolio = [p for p in st.session_state.portfolio if p["symbol"] != remove_symbol]
                st.success(f"Removed {remove_symbol} from portfolio")
                st.rerun()
    
    else:
        st.info("No positions in portfolio. Add some positions above to get started!")

# Initialize session state
if 'uw_api_key' not in st.session_state:
    st.session_state.uw_api_key = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ["AAPL", "NVDA", "TSLA", "SPY"]
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "Core Picks"
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'show_extended_hours' not in st.session_state:
    st.session_state.show_extended_hours = True

# Main Application Header
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #00ff41; font-size: 3em; margin: 0;'>🎯 AI RADAR Pro</h1>
    <p style='color: #888; font-size: 1.2em; margin: 5px 0;'>Advanced Market Intelligence & Options Analytics</p>
    <p style='color: #555; font-size: 0.9em;'>Real-time data • Options flow • Congressional tracking • AI analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Keys Section
    with st.expander("🔐 API Keys", expanded=False):
        uw_key = st.text_input("Unusual Whales API Key", type="password", value=st.session_state.uw_api_key, help="Required for premium features")
        if uw_key != st.session_state.uw_api_key:
            st.session_state.uw_api_key = uw_key
        
        openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key, help="For AI analysis with GPT models")
        if openai_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_key
        
        gemini_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, help="For AI analysis with Gemini models")
        if gemini_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key
    
    st.divider()
    
    # Ticker Selection
    st.markdown("### 📈 Ticker Selection")
    
    # Category selection
    selected_category = st.selectbox(
        "Choose category:",
        options=list(TICKER_CATEGORIES.keys()),
        index=list(TICKER_CATEGORIES.keys()).index(st.session_state.selected_category)
    )
    st.session_state.selected_category = selected_category
    
    # Multi-select from category
    category_tickers = TICKER_CATEGORIES[selected_category]
    selected_from_category = st.multiselect(
        f"Select from {selected_category}:",
        options=category_tickers,
        default=[t for t in st.session_state.selected_tickers if t in category_tickers][:5],  # Limit default selection
        help=f"Choose tickers from {selected_category} category"
    )
    
    # Update selected tickers
    # Keep existing selections from other categories
    other_selections = [t for t in st.session_state.selected_tickers if t not in category_tickers]
    st.session_state.selected_tickers = list(set(other_selections + selected_from_category))
    
    # Custom ticker input
    st.markdown("**Add Custom Tickers:**")
    custom_input = st.text_input("Enter ticker(s) separated by commas:", placeholder="AAPL, MSFT, GOOGL")
    if custom_input:
        custom_tickers = [ticker.strip().upper() for ticker in custom_input.split(",") if ticker.strip()]
        if st.button("➕ Add Custom Tickers"):
            for ticker in custom_tickers:
                if ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)
            st.success(f"Added: {', '.join(custom_tickers)}")
            st.rerun()
    
    # Show current selection
    if st.session_state.selected_tickers:
        st.markdown("**Current Selection:**")
        ticker_display = ", ".join(st.session_state.selected_tickers[:10])
        if len(st.session_state.selected_tickers) > 10:
            ticker_display += f" (+{len(st.session_state.selected_tickers) - 10} more)"
        st.caption(ticker_display)
        
        if st.button("🗑️ Clear All Selections"):
            st.session_state.selected_tickers = []
            st.rerun()
    
    st.divider()
    
    # Display Settings
    st.markdown("### 🎨 Display Settings")
    
    show_extended = st.checkbox(
        "Show extended hours data",
        value=st.session_state.show_extended_hours,
        help="Display pre/post market data when available"
    )
    st.session_state.show_extended_hours = show_extended
    
    chart_style = st.selectbox(
        "Chart style:",
        options=["candlestick", "line", "ohlc"],
        index=0,
        help="Choose chart visualization style"
    )
    
    # Auto-refresh settings
    st.markdown("### 🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh interval:",
            options=[15, 30, 60, 120, 300],
            format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minute(s)",
            index=[15, 30, 60, 120, 300].index(st.session_state.refresh_interval)
        )
        st.session_state.refresh_interval = refresh_interval
        
        st.info(f"Next refresh in ~{refresh_interval}s")
    
    # Manual refresh
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not auto_refresh):
            st.session_state.auto_refresh = False
            st.rerun()

# Main Content Area - Check API Keys
api_status_container = st.container()
with api_status_container:
    if not st.session_state.uw_api_key:
        st.warning("⚠️ **Unusual Whales API key required** for premium features (options flow, congressional trades, etc.). Enter your API key in the sidebar.")
    else:
        st.success("✅ Unusual Whales API connected")

# Tab Layout
tab_names = [
    "📊 Dashboard", 
    "⚡ Options Flow", 
    "🎯 ATM Chains", 
    "🏛️ Congress", 
    "👥 Insiders", 
    "📈 Portfolio",
    "🔍 Screener",
    "📺 Market",
    "🤖 AI Analysis"
]

tabs = st.tabs(tab_names)

# Tab 1: Dashboard
with tabs[0]:
    st.markdown("## 📊 Market Dashboard")
    
    if not st.session_state.selected_tickers:
        st.info("👆 Please select some tickers in the sidebar to view dashboard data.")
        
        # Show sample data or market overview
        st.markdown("### Market Overview")
        sample_tickers = ["SPY", "QQQ", "IWM", "^VIX"]
        sample_data = []
        
        for ticker in sample_tickers:
            quote = get_yfinance_quote(ticker)
            if not quote.get("error"):
                sample_data.append({
                    "Symbol": ticker,
                    "Price": f"${quote.get('last', 0):.2f}",
                    "Change %": f"{quote.get('change_percent', 0):+.2f}%"
                })
        
        if sample_data:
            df = pd.DataFrame(sample_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    else:
        # Market Status
        now = datetime.datetime.now(ZoneInfo("US/Eastern"))
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            market_status = "🔴 Premarket"
            status_color = "orange"
        elif now > market_close:
            market_status = "🌙 After Hours"
            status_color = "blue"
        else:
            market_status = "🟢 Market Open"
            status_color = "green"
        
        st.markdown(f"**Market Status:** <span style='color: {status_color}'>{market_status}</span>", unsafe_allow_html=True)
        st.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')} ET")
        
        # Create progress indicator
        progress_container = st.empty()
        status_container = st.empty()
        
        # Fetch quotes for all selected tickers
        quotes_data = []
        total_tickers = len(st.session_state.selected_tickers)
        
        # Use progress bar for better UX
        progress_bar = progress_container.progress(0)
        
        with status_container.container():
            for i, symbol in enumerate(st.session_state.selected_tickers):
                status_text = f"Fetching {symbol}... ({i+1}/{total_tickers})"
                progress_bar.progress((i + 1) / total_tickers, text=status_text)
                
                # Try UW API first, fallback to yfinance
                quote = get_uw_quote(symbol)
                if quote.get("error"):
                    quote = get_yfinance_quote(symbol)
                
                # Handle missing data gracefully
                price = quote.get("last", 0)
                change = quote.get("change", 0)
                change_pct = quote.get("change_percent", 0)
                volume = quote.get("volume", 0)
                
                quotes_data.append({
                    "Symbol": symbol,
                    "Price": price,
                    "Change $": change,
                    "Change %": change_pct,
                    "Volume": volume,
                    "Bid": quote.get("bid", 0),
                    "Ask": quote.get("ask", 0),
                    "High": quote.get("high", 0),
                    "Low": quote.get("low", 0),
                    "Source": quote.get("data_source", "Unknown"),
                    "Updated": quote.get("last_updated", "N/A")
                })
        
        # Clear progress indicators
        progress_container.empty()
        status_container.empty()
        
        if quotes_data:
            # Create formatted dataframe
            df = pd.DataFrame(quotes_data)
            
            # Format columns
            df["Price"] = df["Price"].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            df["Change $"] = df["Change $"].apply(lambda x: f"${x:+.2f}" if pd.notna(x) else "N/A")
            df["Change %"] = df["Change %"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
            df["Volume"] = df["Volume"].apply(format_volume)
            df["Bid"] = df["Bid"].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            df["Ask"] = df["Ask"].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            df["High"] = df["High"].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            df["Low"] = df["Low"].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            
            # Display main quotes table
            st.markdown("### 📋 Quote Summary")
            display_cols = ["Symbol", "Price", "Change $", "Change %", "Volume", "Source"]
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Extended data toggle
            if st.checkbox("Show detailed quote data"):
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Market Overview Cards
            st.markdown("### 📈 Key Metrics")
            
            # Create metric cards
            cols = st.columns(4)
            key_symbols = ["SPY", "QQQ", "^VIX", st.session_state.selected_tickers[0]]
            
            for i, symbol in enumerate(key_symbols):
                if symbol in [q["Symbol"] for q in quotes_data]:
                    quote_data = next(q for q in quotes_data if q["Symbol"] == symbol)
                    price_str = quote_data["Price"]
                    change_str = quote_data["Change %"]
                else:
                    # Fetch individual quote
                    quote = get_yfinance_quote(symbol)
                    price_str = f"${quote.get('last', 0):.2f}"
                    change_str = f"{quote.get('change_percent', 0):+.2f}%"
                
                with cols[i]:
                    st.metric(
                        label=symbol,
                        value=price_str,
                        delta=change_str.replace("+", "").replace("%", "") + "%"
                    )
            
            # Performance Analysis
            st.markdown("### 📊 Performance Analysis")
            
            # Winners and Losers
            numeric_data = []
            for quote in quotes_data:
                try:
                    change_pct = float(str(quote["Change %"]).replace("%", "").replace("+", ""))
                    numeric_data.append({
                        "Symbol": quote["Symbol"],
                        "Change %": change_pct
                    })
                except:
                    continue
            
            if numeric_data:
                perf_df = pd.DataFrame(numeric_data)
                perf_df = perf_df.sort_values("Change %", ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏆 Top Performers**")
                    top_3 = perf_df.head(3)
                    for _, row in top_3.iterrows():
                        st.write(f"🟢 **{row['Symbol']}**: {row['Change %']:+.2f}%")
                
                with col2:
                    st.markdown("**📉 Bottom Performers**")
                    bottom_3 = perf_df.tail(3)
                    for _, row in bottom_3.iterrows():
                        st.write(f"🔴 **{row['Symbol']}**: {row['Change %']:+.2f}%")

# Tab 2: Options Flow
with tabs[1]:
    st.markdown("## ⚡ Options Flow Monitor")
    
    if not st.session_state.uw_api_key:
        st.warning("🔐 Options flow monitoring requires Unusual Whales API key.")
        st.info("Configure your API key in the sidebar to access this feature.")
    else:
        # Options flow controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            flow_symbol = st.selectbox(
                "Select symbol for options flow:",
                options=["ALL"] + st.session_state.selected_tickers,
                help="Choose a specific symbol or ALL for market-wide flow"
            )
        
        with col2:
            flow_limit = st.selectbox("Limit:", [50, 100, 200, 500], index=1)
        
        with col3:
            if st.button("🔄 Get Options Flow", use_container_width=True):
                with st.spinner(f"Fetching options flow data..."):
                    symbol_param = None if flow_symbol == "ALL" else flow_symbol
                    flow_data = get_options_flow(symbol_param, flow_limit)
                    
                    if flow_data.get("error"):
                        st.error(f"❌ {flow_data['error']}")
                    else:
                        st.success(f"✅ Retrieved {len(flow_data.get('flow', []))} flow records")
                        
                        # Process and display flow data
                        flow_records = flow_data.get("flow", [])
                        if flow_records:
                            # Convert to DataFrame
                            df_flow = pd.DataFrame(flow_records)
                            
                            # Format columns if they exist
                            if "premium" in df_flow.columns:
                                df_flow["Premium"] = df_flow["premium"].apply(format_large_number)
                            if "timestamp" in df_flow.columns:
                                df_flow["Time"] = pd.to_datetime(df_flow["timestamp"]).dt.strftime("%H:%M:%S")
                            
                            # Display flow table
                            st.dataframe(df_flow, use_container_width=True)
                        else:
                            st.info("No options flow data available.")
        
        st.divider()
        
        # Unusual Activity section
        st.markdown("### 🚨 Unusual Options Activity")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            unusual_symbol = st.selectbox(
                "Symbol for unusual activity:",
                options=["ALL"] + st.session_state.selected_tickers,
                key="unusual_symbol"
            )
        
        with col2:
            if st.button("🔍 Get Unusual Activity", use_container_width=True):
                with st.spinner("Analyzing unusual activity..."):
                    symbol_param = None if unusual_symbol == "ALL" else unusual_symbol
                    unusual_data = get_unusual_options_activity(symbol_param)
                    
                    if unusual_data.get("error"):
                        st.error(f"❌ {unusual_data['error']}")
                    else:
                        unusual_records = unusual_data.get("unusual", [])
                        if unusual_records:
                            st.success(f"✅ Found {len(unusual_records)} unusual activities")
                            df_unusual = pd.DataFrame(unusual_records)
                            st.dataframe(df_unusual, use_container_width=True)
                        else:
                            st.info("No unusual options activity detected.")
        
        # Options Education Section
        with st.expander("📚 Options Flow Guide"):
            st.markdown("""
            **Understanding Options Flow:**
            
            - **Volume**: Number of contracts traded
            - **Premium**: Total dollar amount traded
            - **Open Interest**: Contracts outstanding
            - **Unusual Activity**: Trades significantly above normal volume
            
            **Flow Indicators:**
            - 🟢 **High call volume**: Bullish sentiment
            - 🔴 **High put volume**: Bearish sentiment
            - ⚡ **Large premium**: Institutional activity
            - 🚨 **Unusual volume**: Potential catalyst
            """)

# Tab 3: ATM Chains
with tabs[2]:
    st.markdown("## 🎯 ATM Options Chains")
    
    if not st.session_state.uw_api_key:
        st.warning("🔐 ATM chains require Unusual Whales API key.")
    else:
        # ATM chains controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            atm_symbol = st.selectbox("Select symbol:", st.session_state.selected_tickers, key="atm_symbol")
        
        with col2:
            if st.button("🎯 Get ATM Chains", use_container_width=True):
                with st.spinner(f"Fetching ATM chains for {atm_symbol}..."):
                    atm_data = get_atm_chains(atm_symbol)
                    
                    if atm_data.get("error"):
                        st.error(f"❌ {atm_data['error']}")
                    else:
                        st.session_state.atm_data = atm_data
                        st.success(f"✅ ATM chains loaded for {atm_symbol}")
        
        with col3:
            chart_period = st.selectbox("Chart:", ["1d", "5d", "1mo"], key="atm_chart")
        
        # Display ATM data if available
        if 'atm_data' in st.session_state and st.session_state.atm_data:
            data = st.session_state.atm_data
            
            # Stock information header
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${data['current_price']:.2f}")
            with col2:
                st.metric("ATM Strike", f"${data['atm_strike']:.2f}")
            with col3:
                distance = abs(data['current_price'] - data['atm_strike'])
                st.metric("Distance to ATM", f"${distance:.2f}")
            with col4:
                st.metric("Expiry", data['expiry'])
            
            # Main content layout
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                # Calls section
                st.markdown("### 📞 ATM Calls")
                if data.get('atm_calls'):
                    calls_df = pd.DataFrame(data['atm_calls'])
                    
                    # Format important columns
                    if 'last_price' in calls_df.columns:
                        calls_df['Last'] = calls_df['last_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                    if 'bid' in calls_df.columns and 'ask' in calls_df.columns:
                        calls_df['Bid-Ask'] = calls_df.apply(lambda row: f"${row['bid']:.2f} - ${row['ask']:.2f}", axis=1)
                    if 'volume' in calls_df.columns:
                        calls_df['Volume'] = calls_df['volume'].apply(format_volume)
                    
                    # Select key columns to display
                    display_cols = []
                    for col in ['Last', 'Bid-Ask', 'Volume', 'open_interest', 'implied_volatility']:
                        if col in calls_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(calls_df[display_cols], use_container_width=True)
                    else:
                        st.dataframe(calls_df, use_container_width=True)
                else:
                    st.info("No ATM calls data available")
            
            with col_right:
                # Puts section
                st.markdown("### 📞 ATM Puts")
                if data.get('atm_puts'):
                    puts_df = pd.DataFrame(data['atm_puts'])
                    
                    # Format important columns (same as calls)
                    if 'last_price' in puts_df.columns:
                        puts_df['Last'] = puts_df['last_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                    if 'bid' in puts_df.columns and 'ask' in puts_df.columns:
                        puts_df['Bid-Ask'] = puts_df.apply(lambda row: f"${row['bid']:.2f} - ${row['ask']:.2f}", axis=1)
                    if 'volume' in puts_df.columns:
                        puts_df['Volume'] = puts_df['volume'].apply(format_volume)
                    
                    # Select key columns to display
                    display_cols = []
                    for col in ['Last', 'Bid-Ask', 'Volume', 'open_interest', 'implied_volatility']:
                        if col in puts_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(puts_df[display_cols], use_container_width=True)
                    else:
                        st.dataframe(puts_df, use_container_width=True)
                else:
                    st.info("No ATM puts data available")
            
            # Chart section
            if atm_symbol:
                st.markdown("### 📈 Price Chart")
                chart_fig = create_stock_chart(atm_symbol, chart_period, chart_style)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                
                # Options heatmap
                st.markdown("### 🔥 Options Volume Heatmap")
                heatmap_fig = create_options_heatmap(atm_symbol, data['expiry'])
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
        
        else:
            st.info("Select a symbol and click 'Get ATM Chains' to view options data.")

# Tab 4: Congressional Trades
with tabs[3]:
    st.markdown("## 🏛️ Congressional Trading Activity")
    
    if not st.session_state.uw_api_key:
        st.warning("🔐 Congressional trades require Unusual Whales API key.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🏛️ Get Congressional Trades", use_container_width=True):
                with st.spinner("Fetching congressional trading data..."):
                    congress_data = get_congressional_trades()
                    
                    if congress_data.get("error"):
                        st.error(f"❌ {congress_data['error']}")
                    else:
                        st.session_state.congress_data = congress_data
                        trades = congress_data.get("trades", [])
                        st.success(f"✅ Retrieved {len(trades)} congressional trades")
        
        # Display congressional trades if available
        if 'congress_data' in st.session_state:
            data = st.session_state.congress_data
            trades = data.get("trades", [])
            
            if trades:
                # Convert to DataFrame
                df_congress = pd.DataFrame(trades)
                
                # Format columns if they exist
                for col in ['amount', 'value']:
                    if col in df_congress.columns:
                        df_congress[col] = df_congress[col].apply(format_large_number)
                
                if 'trade_date' in df_congress.columns:
                    df_congress['Date'] = pd.to_datetime(df_congress['trade_date']).dt.strftime("%Y-%m-%d")
                
                # Summary metrics
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", len(trades))
                with col2:
                    unique_members = len(df_congress['representative'].unique()) if 'representative' in df_congress.columns else "N/A"
                    st.metric("Unique Members", unique_members)
                with col3:
                    buy_trades = len(df_congress[df_congress['transaction_type'] == 'Buy']) if 'transaction_type' in df_congress.columns else "N/A"
                    st.metric("Buy Trades", buy_trades)
                with col4:
                    sell_trades = len(df_congress[df_congress['transaction_type'] == 'Sell']) if 'transaction_type' in df_congress.columns else "N/A"
                    st.metric("Sell Trades", sell_trades)
                
                # Trades table
                st.markdown("### 📋 Recent Trades")
                st.dataframe(df_congress, use_container_width=True, hide_index=True)
                
                # Analysis
                with st.expander("📈 Trade Analysis"):
                    if 'symbol' in df_congress.columns:
                        # Most traded symbols
                        symbol_counts = df_congress['symbol'].value_counts().head(10)
                        st.markdown("**Most Traded Symbols:**")
                        for symbol, count in symbol_counts.items():
                            st.write(f"• {symbol}: {count} trades")
            else:
                st.info("No congressional trades data available.")

# Tab 5: Insider Trades
with tabs[4]:
    st.markdown("## 👥 Insider Trading Activity")
    
    if not st.session_state.uw_api_key:
        st.warning("🔐 Insider trades require Unusual Whales API key.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            insider_symbol = st.selectbox("Select symbol:", st.session_state.selected_tickers, key="insider_symbol")
        
        with col2:
            if st.button("👥 Get Insider Trades", use_container_width=True):
                if insider_symbol:
                    with st.spinner(f"Fetching insider trades for {insider_symbol}..."):
                        insider_data = get_insider_trades(insider_symbol)
                        
                        if insider_data.get("error"):
                            st.error(f"❌ {insider_data['error']}")
                        else:
                            st.session_state.insider_data = insider_data
                            trades = insider_data.get("trades", [])
                            st.success(f"✅ Retrieved {len(trades)} insider trades for {insider_symbol}")
        
        # Display insider trades if available
        if 'insider_data' in st.session_state:
            data = st.session_state.insider_data
            trades = data.get("trades", [])
            
            if trades:
                # Convert to DataFrame
                df_insider = pd.DataFrame(trades)
                
                # Format columns
                for col in ['value', 'price']:
                    if col in df_insider.columns:
                        df_insider[col] = df_insider[col].apply(format_large_number)
                
                if 'date' in df_insider.columns:
                    df_insider['Trade Date'] = pd.to_datetime(df_insider['date']).dt.strftime("%Y-%m-%d")
                
                # Summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Trades", len(trades))
                with col2:
                    unique_insiders = len(df_insider['insider'].unique()) if 'insider' in df_insider.columns else "N/A"
                    st.metric("Unique Insiders", unique_insiders)
                with col3:
                    if 'transaction_type' in df_insider.columns:
                        buy_trades = len(df_insider[df_insider['transaction_type'].str.contains('Buy', case=False, na=False)])
                        st.metric("Buy Trades", buy_trades)
                
                # Trades table
                st.dataframe(df_insider, use_container_width=True, hide_index=True)
            else:
                st.info(f"No insider trades found for {insider_symbol}")

# Tab 6: Portfolio Tracker
with tabs[5]:
    create_portfolio_tracker()

# Tab 7: Stock Screener
with tabs[6]:
    st.markdown("## 🔍 Stock Screener")
    
    # Screening criteria
    st.markdown("### Filter Criteria")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0, step=0.1)
        max_price = st.number_input("Max Price ($)", min_value=0.0, value=1000.0, step=1.0)
    
    with col2:
        min_volume = st.number_input("Min Volume (M)", min_value=0.0, value=0.0, step=0.1) * 1e6
        min_change = st.number_input("Min Change %", value=-100.0, step=0.1)
    
    with col3:
        max_change = st.number_input("Max Change %", value=100.0, step=0.1)
        screener_category = st.selectbox("Category:", list(TICKER_CATEGORIES.keys()))
    
    with col4:
        if st.button("🔍 Screen Stocks", use_container_width=True):
            with st.spinner("Screening stocks..."):
                # Get tickers from selected category
                tickers_to_screen = TICKER_CATEGORIES[screener_category]
                
                screened_stocks = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(tickers_to_screen):
                    progress_bar.progress((i + 1) / len(tickers_to_screen))
                    
                    quote = get_uw_quote(ticker)
                    if quote.get("error"):
                        quote = get_yfinance_quote(ticker)
                    
                    if not quote.get("error"):
                        price = quote.get("last", 0)
                        volume = quote.get("volume", 0)
                        change_pct = quote.get("change_percent", 0)
                        
                        # Apply filters
                        if (min_price <= price <= max_price and
                            volume >= min_volume and
                            min_change <= change_pct <= max_change):
                            
                            screened_stocks.append({
                                "Symbol": ticker,
                                "Price": f"${price:.2f}",
                                "Change %": f"{change_pct:+.2f}%",
                                "Volume": format_volume(volume),
                                "Source": quote.get("data_source", "Unknown")
                            })
                
                progress_bar.empty()
                
                if screened_stocks:
                    st.success(f"✅ Found {len(screened_stocks)} stocks matching criteria")
                    df_screened = pd.DataFrame(screened_stocks)
                    st.dataframe(df_screened, use_container_width=True, hide_index=True)
                else:
                    st.info("No stocks found matching the specified criteria.")

# Tab 8: Market Overview
with tabs[7]:
    st.markdown("## 📺 Market Overview")
    
    # Market indices
    st.markdown("### 📊 Major Indices")
    
    indices = ["SPY", "QQQ", "IWM", "^VIX", "^TNX", "DX-Y.NYB"]
    index_data = []
    
    for index in indices:
        quote = get_yfinance_quote(index)
        if not quote.get("error"):
            index_data.append({
                "Index": index,
                "Price": quote.get("last", 0),
                "Change %": quote.get("change_percent", 0),
                "Volume": quote.get("volume", 0)
            })
    
    if index_data:
        df_indices = pd.DataFrame(index_data)
        
        # Create metrics display
        cols = st.columns(len(index_data))
        for i, (_, row) in enumerate(df_indices.iterrows()):
            with cols[i]:
                st.metric(
                    row["Index"],
                    f"${row['Price']:.2f}",
                    f"{row['Change %']:+.2f}%"
                )
    
    # Sector Performance
    st.markdown("### 🏭 Sector Performance")
    
    if st.button("📈 Get Sector Performance"):
        with st.spinner("Loading sector performance..."):
            sector_data = get_sector_performance()
            
            if sector_data.get("error"):
                st.error(f"❌ {sector_data['error']}")
            else:
                sectors = sector_data.get("sectors", [])
                if sectors:
                    df_sectors = pd.DataFrame(sectors)
                    df_sectors = df_sectors.sort_values("Change %", ascending=False)
                    
                    # Color code performance
                    def color_performance(val):
                        color = 'background-color: lightgreen' if val > 0 else 'background-color: lightcoral' if val < 0 else ''
                        return color
                    
                    styled_df = df_sectors.style.applymap(color_performance, subset=['Change %'])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Market Sentiment
    st.markdown("### 😊 Market Sentiment")
    
    if st.button("🎭 Get Market Sentiment"):
        with st.spinner("Analyzing market sentiment..."):
            sentiment_data = get_market_sentiment()
            
            if sentiment_data.get("error"):
                st.error(f"❌ {sentiment_data['error']}")
            else:
                sentiment = sentiment_data.get("data", {})
                
                col1, col2, col3 = st.columns(3)
                
                # VIX Sentiment
                if "vix" in sentiment:
                    vix_data = sentiment["vix"]
                    with col1:
                        st.metric(
                            "VIX Fear Index",
                            f"{vix_data['level']:.2f}",
                            f"{vix_data['change']:+.2f}%"
                        )
                        st.caption(f"Sentiment: {vix_data['sentiment']}")
                
                # Fear & Greed
                if "fear_greed" in sentiment:
                    fg_data = sentiment["fear_greed"]
                    with col2:
                        st.metric("Fear & Greed", f"{fg_data['score']:.0f}/100")
                        st.caption(f"Sentiment: {fg_data['sentiment']}")
                
                # Put/Call Ratio
                if "put_call_ratio" in sentiment:
                    pc_data = sentiment["put_call_ratio"]
                    with col3:
                        st.metric("Put/Call Ratio", pc_data['ratio'])
                        st.caption(f"Sentiment: {pc_data['sentiment']}")

# Tab 9: AI Analysis
with tabs[8]:
    st.markdown("## 🤖 AI Market Analysis")
    
    # AI Provider and model selection
    col1, col2 = st.columns(2)
    
    with col1:
        ai_provider = st.selectbox("AI Provider:", ["openai", "gemini"])
    
    with col2:
        if ai_provider == "openai":
            ai_model = st.selectbox("Model:", ["gpt-4", "gpt-3.5-turbo"])
        else:
            ai_model = st.selectbox("Model:", ["gemini-pro", "gemini-pro-vision"])
    
    # Analysis type selection
    analysis_type = st.selectbox("Analysis Type:", [
        "Market Summary",
        "Stock Analysis", 
        "Options Strategy",
        "Risk Assessment",
        "Portfolio Review",
        "Sector Analysis",
        "Custom Query"
    ])
    
    # Type-specific analysis
    if analysis_type == "Stock Analysis":
        analysis_symbol = st.selectbox("Select symbol:", st.session_state.selected_tickers)
        
        if st.button("🤖 Generate Stock Analysis"):
            if (ai_provider == "openai" and st.session_state.openai_api_key) or \
               (ai_provider == "gemini" and st.session_state.gemini_api_key):
                
                with st.spinner("Generating AI stock analysis..."):
                    # Get comprehensive stock data
                    stock_data = get_uw_quote(analysis_symbol)
                    if stock_data.get("error"):
                        stock_data = get_yfinance_quote(analysis_symbol)
                    
                    # Get technical indicators
                    tech_data = calculate_technical_indicators(analysis_symbol)
                    
                    # Create comprehensive prompt
                    prompt = f"""
                    Provide a comprehensive analysis of {analysis_symbol} stock with the following current data:
                    
                    **Stock Data:**
                    - Current Price: ${stock_data.get('last', 'N/A')}
                    - Change: {stock_data.get('change_percent', 'N/A')}%
                    - Volume: {format_volume(stock_data.get('volume', 0))}
                    - Bid/Ask: ${stock_data.get('bid', 0):.2f}/${stock_data.get('ask', 0):.2f}
                    
                    **Technical Indicators:**
                    - RSI: {tech_data.get('rsi', 'N/A'):.2f if tech_data.get('rsi') else 'N/A'}
                    - SMA 20: ${tech_data.get('sma_20', 0):.2f if tech_data.get('sma_20') else 'N/A'}
                    - MACD: {tech_data.get('macd', 'N/A'):.4f if tech_data.get('macd') else 'N/A'}
                    - 52W High/Low: ${tech_data.get('high_52w', 0):.2f}/${tech_data.get('low_52w', 0):.2f}
                    
                    Please provide:
                    1. **Technical Analysis** - Support/resistance levels, trend analysis
                    2. **Key Levels to Watch** - Important price points
                    3. **Market Position** - Relative to moving averages and ranges
                    4. **Risk Assessment** - Volatility and risk factors
                    5. **Trading Strategy** - Entry/exit points and position sizing
                    6. **Catalysts** - Potential upcoming events or factors
                    
                    Keep the analysis professional, actionable, and well-structured.
                    """
                    
                    analysis = get_ai_analysis(prompt, ai_provider, ai_model)
                    
                    # Display analysis in a nice format
                    st.markdown("### 📊 AI Analysis Results")
                    st.markdown(analysis)
            else:
                st.warning(f"Please configure {ai_provider.upper()} API key in the sidebar.")
    
    elif analysis_type == "Market Summary":
        if st.button("🤖 Generate Market Summary"):
            if (ai_provider == "openai" and st.session_state.openai_api_key) or \
               (ai_provider == "gemini" and st.session_state.gemini_api_key):
                
                with st.spinner("Generating market summary..."):
                    # Get market data
                    spy_data = get_yfinance_quote("SPY")
                    qqq_data = get_yfinance_quote("QQQ")
                    vix_data = get_yfinance_quote("^VIX")
                    
                    prompt = f"""
                    Provide a comprehensive market summary based on current data:
                    
                    **Major Indices:**
                    - SPY: ${spy_data.get('last', 0):.2f} ({spy_data.get('change_percent', 0):+.2f}%)
                    - QQQ: ${qqq_data.get('last', 0):.2f} ({qqq_data.get('change_percent', 0):+.2f}%)
                    - VIX: {vix_data.get('last', 0):.2f} ({vix_data.get('change_percent', 0):+.2f}%)
                    
                    Please analyze:
                    1. **Overall Market Direction** - Bull/bear sentiment
                    2. **Sector Rotation** - Which sectors are leading/lagging
                    3. **Volatility Analysis** - Fear/greed levels
                    4. **Key Market Drivers** - Major themes and catalysts
                    5. **Risk Factors** - Potential headwinds
                    6. **Trading Opportunities** - Areas of focus
                    
                    Provide actionable insights for today's trading session.
                    """
                    
                    analysis = get_ai_analysis(prompt, ai_provider, ai_model)
                    st.markdown("### 🌐 Market Summary")
                    st.markdown(analysis)
            else:
                st.warning(f"Please configure {ai_provider.upper()} API key in the sidebar.")
    
    elif analysis_type == "Custom Query":
        st.markdown("### 💭 Custom Analysis Query")
        custom_prompt = st.text_area(
            "Enter your analysis request:",
            height=120,
            placeholder="Ask about market conditions, specific stocks, trading strategies, risk management, etc."
        )
        
        if custom_prompt and st.button("🤖 Generate Custom Analysis"):
            if (ai_provider == "openai" and st.session_state.openai_api_key) or \
               (ai_provider == "gemini" and st.session_state.gemini_api_key):
                
                with st.spinner("Generating custom analysis..."):
                    # Add context about current holdings
                    context_prompt = f"""
                    Context: User is monitoring these tickers: {', '.join(st.session_state.selected_tickers)}
                    
                    User Query: {custom_prompt}
                    
                    Please provide a detailed, professional analysis addressing their query.
                    """
                    
                    analysis = get_ai_analysis(context_prompt, ai_provider, ai_model)
                    st.markdown("### 🎯 Custom Analysis Results")
                    st.markdown(analysis)
            else:
                st.warning(f"Please configure {ai_provider.upper()} API key in the sidebar.")
    
    else:
        st.info(f"Select '{analysis_type}' parameters above and click generate to get AI analysis.")
    
    # AI Analysis Tips
    with st.expander("💡 AI Analysis Tips"):
        st.markdown("""
        **Getting Better AI Analysis:**
        
        - **Be Specific**: Ask about particular stocks, timeframes, or strategies
        - **Provide Context**: Mention your risk tolerance, investment goals
        - **Ask Follow-ups**: Drill down into specific aspects
        - **Combine Data**: Reference technical indicators, news, earnings
        
        **Example Queries:**
        - "Analyze NVDA for swing trading opportunities"
        - "What are the key risks in my current portfolio?"
        - "Should I hedge my tech positions with puts?"
        - "Compare the risk/reward of AAPL vs MSFT calls"
        """)

# Auto-refresh functionality
if st.session_state.auto_refresh:
    # Add a small delay and refresh
    time.sleep(st.session_state.refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4 style='color: #00ff41;'>🎯 AI Radar Pro</h4>
        <p>Advanced Market Intelligence Platform</p>
        <p><em>Data sources: Unusual Whales API • yfinance • OpenAI • Google Gemini</em></p>
        <p style='font-size: 0.8em;'>Built with Streamlit • Real-time market data • Options analytics • Congressional tracking</p>
    </div>
    """, 
    unsafe_allow_html=True
)
