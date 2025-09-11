# Debug Public.com API Integration - FIXED VERSION

import requests
import datetime
from typing import Dict, List
import streamlit as st

# First, let's create a debug function to test the actual Public.com API
def debug_public_api():
    """Debug function to test Public.com API endpoints"""
    
    # NOTE: Replace this with your actual SECRET KEY from Public.com settings
    secret_key = "YOUR_SECRET_KEY_HERE"  # Get this from Account Settings > Security > API
    
    # Step 1: Get access token using official flow
    auth_url = "https://api.public.com/userapiauthservice/personal/access-tokens"
    auth_payload = {
        "validityInMinutes": 123,
        "secret": secret_key
    }
    
    try:
        auth_response = requests.post(auth_url, json=auth_payload, timeout=10)
        if auth_response.status_code != 200:
            print(f"Authentication failed: {auth_response.status_code}")
            print(f"Response: {auth_response.text}")
            return None, None
        
        access_token = auth_response.json().get("accessToken")
        if not access_token:
            print("No access token received")
            return None, None
        
        print(f"✅ Authentication successful! Token: {access_token[:20]}...")
        
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None, None
    
    # Step 2: Test endpoints with access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Updated endpoints based on official API
    test_endpoints = [
        "https://api.public.com/userapigateway/trading/account",
        "https://api.public.com/userapigateway/trading/quotes/AAPL",
        "https://api.public.com/userapigateway/trading/quote/AAPL", 
        "https://api.public.com/userapigateway/market/quotes/AAPL",
        "https://api.public.com/userapigateway/quotes/AAPL",
        "https://api.public.com/userapigateway/trading/instruments/AAPL/quote",
        "https://api.public.com/userapigateway/trading/market-data/AAPL",
        "https://api.public.com/userapigateway/trading/stocks/AAPL"
    ]
    
    print("Testing Public.com API endpoints...")
    
    for endpoint in test_endpoints:
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            print(f"\n{endpoint}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"SUCCESS! Data: {response.json()}")
                return endpoint, response.json()
            else:
                print(f"Response: {response.text[:200]}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return None, None

# Updated Public.com client with proper error handling and debugging
class PublicDataClient:
    """Enhanced Public.com client with debugging"""
    
    def __init__(self, secret_key: str):  # Changed from api_token to secret_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_expires_at = None
        self.base_url = "https://api.public.com"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Radar-Pro/1.0"
        }
        self.working_endpoints = {}
        
        # Authenticate immediately
        self._authenticate()
    
    def _authenticate(self):
        """Get access token using official Public.com flow"""
        auth_url = f"{self.base_url}/userapiauthservice/personal/access-tokens"
        payload = {
            "validityInMinutes": 123,
            "secret": self.secret_key
        }
        
        try:
            response = requests.post(auth_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("accessToken")
                if self.access_token:
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    self.token_expires_at = datetime.datetime.now() + datetime.timedelta(minutes=120)
                    return True
        except Exception as e:
            print(f"Authentication error: {str(e)}")
        
        return False
    
    def _ensure_authenticated(self):
        """Make sure we have a valid access token"""
        if not self.access_token or (self.token_expires_at and datetime.datetime.now() >= self.token_expires_at):
            return self._authenticate()
        return True
    
    def test_connection(self) -> Dict:
        """Test connection and find working endpoints"""
        if not self._ensure_authenticated():
            return {"status": "failed", "error": "Authentication failed"}
        
        try:
            # Test the official account endpoint
            test_url = f"{self.base_url}/userapigateway/trading/account"
            response = requests.get(test_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return {"status": "connected", "endpoint": test_url, "response": response.status_code}
            else:
                return {"status": "failed", "error": f"Status {response.status_code}"}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    def get_quote(self, symbol: str) -> Dict:
        """Get quote with multiple endpoint attempts"""
        
        if not self._ensure_authenticated():
            return {"error": "Authentication failed"}
        
        # Try different quote endpoint patterns
        quote_endpoints = [
            f"/userapigateway/trading/quotes/{symbol}",
            f"/userapigateway/trading/quote/{symbol}",
            f"/userapigateway/market/quotes/{symbol}",
            f"/userapigateway/quotes/{symbol}",
            f"/userapigateway/trading/instruments/{symbol}/quote",
            f"/userapigateway/trading/market-data/{symbol}",
            f"/userapigateway/trading/stocks/{symbol}"
        ]
        
        for endpoint in quote_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    # Cache this working endpoint
                    self.working_endpoints['quote'] = endpoint
                    
                    # Map the response data (structure will depend on actual API)
                    return self._map_quote_response(data, symbol)
                    
                elif response.status_code == 401:
                    # Try re-authenticating once
                    if self._authenticate():
                        response = requests.get(url, headers=self.headers, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            return self._map_quote_response(data, symbol)
                    return {"error": "Authentication failed - check secret key"}
                elif response.status_code == 403:
                    return {"error": "Access forbidden - check API permissions"}
                    
            except Exception as e:
                continue
        
        return {"error": "No working quote endpoints found"}
    
    def _map_quote_response(self, data: Dict, symbol: str) -> Dict:
        """Map Public.com response to our standard format"""
        
        try:
            # Different possible structures to try
            quote_mappings = [
                # Structure 1: Direct fields
                {
                    "price": ["price", "last_price", "last", "current_price"],
                    "bid": ["bid", "bid_price"],
                    "ask": ["ask", "ask_price"], 
                    "volume": ["volume", "day_volume", "total_volume"],
                    "change": ["change", "net_change", "day_change"],
                    "change_percent": ["change_percent", "percent_change", "day_change_percent"]
                },
                # Structure 2: Nested in quote object
                {
                    "price": ["quote.price", "quote.last_price", "quote.last"],
                    "bid": ["quote.bid", "quote.bid_price"],
                    "ask": ["quote.ask", "quote.ask_price"],
                    "volume": ["quote.volume", "quote.day_volume"]
                },
                # Structure 3: Market data nested
                {
                    "price": ["market_data.price", "market_data.last_price"],
                    "bid": ["market_data.bid"],
                    "ask": ["market_data.ask"],
                    "volume": ["market_data.volume"]
                }
            ]
            
            # Try to extract data using different mappings
            result = {
                "last": 0,
                "bid": 0,
                "ask": 0, 
                "volume": 0,
                "change": 0,
                "change_percent": 0,
                "premarket_change": 0,
                "intraday_change": 0,
                "postmarket_change": 0,
                "previous_close": 0,
                "market_open": 0,
                "last_updated": datetime.datetime.now().isoformat(),
                "data_source": "Public.com",
                "error": None,
                "raw_data": data  # Include raw data for debugging
            }
            
            # Helper function to get nested value
            def get_nested_value(obj, path):
                keys = path.split('.')
                for key in keys:
                    if isinstance(obj, dict) and key in obj:
                        obj = obj[key]
                    else:
                        return None
                return obj
            
            # Try each mapping structure
            for mapping in quote_mappings:
                for field, possible_paths in mapping.items():
                    for path in possible_paths:
                        value = get_nested_value(data, path)
                        if value is not None:
                            if field == "price":
                                result["last"] = float(value)
                            elif field in result:
                                result[field] = float(value)
                            break
                    if result.get("last", 0) > 0:  # If we found price, this mapping might work
                        break
                if result.get("last", 0) > 0:
                    break
            
            return result
            
        except Exception as e:
            return {"error": f"Error mapping response: {str(e)}", "raw_data": data}
    
    def get_options_data(self, symbol: str) -> Dict:
        """Get options data from Public.com"""
        
        if not self._ensure_authenticated():
            return {"error": "Authentication failed"}
        
        options_endpoints = [
            f"/userapigateway/trading/options/{symbol}",
            f"/userapigateway/options/{symbol}",
            f"/userapigateway/trading/instruments/{symbol}/options"
        ]
        
        for endpoint in options_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
        
        return {"error": "Options data not available"}
    
    def get_earnings_data(self, symbol: str) -> Dict:
        """Get earnings data from Public.com"""
        
        if not self._ensure_authenticated():
            return {"error": "Authentication failed"}
        
        earnings_endpoints = [
            f"/userapigateway/trading/earnings/{symbol}",
            f"/userapigateway/earnings/{symbol}",
            f"/userapigateway/trading/fundamentals/{symbol}/earnings"
        ]
        
        for endpoint in earnings_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
        
        return {"error": "Earnings data not available"}
    
    def get_news(self, symbol: str = None) -> List[Dict]:
        """Get news from Public.com"""
        
        if not self._ensure_authenticated():
            return []
        
        if symbol:
            news_endpoints = [
                f"/userapigateway/trading/news/{symbol}",
                f"/userapigateway/news/{symbol}"
            ]
        else:
            news_endpoints = [
                "/userapigateway/trading/news",
                "/userapigateway/news"
            ]
        
        for endpoint in news_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
        
        return []

# Add this debug function to your Streamlit app
def debug_public_integration():
    """Add this to your Streamlit sidebar for debugging"""
    
    st.sidebar.subheader("🔧 Public.com Debug")
    
    if st.sidebar.button("Test API Connection"):
        if st.session_state.get('public_client'):
            with st.spinner("Testing Public.com API..."):
                # Test connection
                result = st.session_state.public_client.test_connection()
                st.sidebar.write(f"Connection test: {result}")
                
                # Test quote endpoint
                quote_result = st.session_state.public_client.get_quote("AAPL")
                st.sidebar.write("AAPL Quote test:")
                st.sidebar.json(quote_result)
        else:
            st.sidebar.error("No Public.com client initialized")
    
    if st.sidebar.button("Debug Raw API Response"):
        if st.session_state.get('public_client'):
            # Show raw API response for debugging
            quote = st.session_state.public_client.get_quote("AAPL")
            if "raw_data" in quote:
                st.sidebar.write("Raw API Response:")
                st.sidebar.json(quote["raw_data"])

# Enhanced get_live_quote function with better Public.com integration
@st.cache_data(ttl=60)
def get_live_quote_enhanced(ticker: str, tz: str = "ET") -> Dict:
    """Enhanced quote function with better Public.com integration"""
    
    # Try Public.com first if available
    if st.session_state.get('public_client'):
        try:
            public_quote = st.session_state.public_client.get_quote(ticker)
            
            # Check if we got valid data
            if not public_quote.get("error") and public_quote.get("last", 0) > 0:
                return public_quote
            else:
                # Log the error for debugging
                if public_quote.get("error"):
                    st.sidebar.warning(f"Public.com error: {public_quote['error']}")
                
        except Exception as e:
            st.sidebar.warning(f"Public.com exception: {str(e)}")
    
    # Fall back to Yahoo Finance with enhanced error handling
    return get_live_quote_yahoo_fallback(ticker, tz)

def get_live_quote_yahoo_fallback(ticker: str, tz: str = "ET") -> Dict:
    """Yahoo Finance fallback with the original logic"""
    # [Keep all your original Yahoo Finance code here]
    # This ensures we always have data even if Public.com fails
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")
        
        if not hist.empty:
            last_price = hist['Close'].iloc[-1]
            return {
                "last": float(last_price),
                "bid": float(info.get('bid', 0)),
                "ask": float(info.get('ask', 0)),
                "volume": int(info.get('volume', 0)),
                "change": float(info.get('regularMarketChange', 0)),
                "change_percent": float(info.get('regularMarketChangePercent', 0)),
                "previous_close": float(info.get('regularMarketPreviousClose', 0)),
                "data_source": "Yahoo Finance (fallback)",
                "last_updated": datetime.datetime.now().isoformat(),
                "error": None
            }
    except Exception as e:
        return {
            "error": f"All data sources failed: {str(e)}",
            "data_source": "Error",
            "last_updated": datetime.datetime.now().isoformat()
        }

# Setup function for your Streamlit app
def setup_public_client(secret_key: str):
    """Setup Public.com client in your Streamlit app"""
    try:
        client = PublicDataClient(secret_key)
        test_result = client.test_connection()
        
        if test_result.get("status") == "connected":
            st.session_state.public_client = client
            st.success("✅ Public.com connected successfully!")
            return True
        else:
            st.error(f"❌ Connection failed: {test_result.get('error')}")
            return False
    except Exception as e:
        st.error(f"❌ Setup failed: {str(e)}")
        return False

"""
INTEGRATION INSTRUCTIONS:

1. Get your SECRET KEY from Public.com:
   - Go to Account Settings > Security > API
   - Generate a SECRET KEY (not the same as API key)

2. In your Streamlit app, add this to your sidebar:
   
   secret_key = st.sidebar.text_input("Public.com Secret Key", type="password")
   if secret_key and st.sidebar.button("Connect Public.com"):
       setup_public_client(secret_key)

3. Add the debug interface:
   debug_public_integration()

4. Replace your quote function:
   quote_data = get_live_quote_enhanced(ticker_symbol)

MAIN CHANGES FROM YOUR ORIGINAL:
- Changed from api_token to secret_key 
- Added authentication step (secret_key -> access_token)
- Updated endpoints to use /userapigateway/trading/...
- Added token refresh logic
- Fixed headers to use Bearer access_token instead of secret_key

Your existing class structure and methods are preserved!
"""
