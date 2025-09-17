import streamlit as st
import requests
from datetime import date

# Load the API key
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# Define both stock and chain endpoints
endpoints = {
    # 📈 Stock-Level
    "📊 Stock State": "/api/stock/{ticker}/stock-state",
    "📈 Options Volume": "/api/stock/{ticker}/options-volume",
    "📉 Volatility (Realized)": "/api/stock/{ticker}/volatility/realized",
    "📊 Volatility Stats": "/api/stock/{ticker}/volatility/stats",
    "📉 Volatility Term Structure": "/api/stock/{ticker}/volatility/term-structure",
    "📊 Spot GEX (1min)": "/api/stock/{ticker}/spot-exposures",
    "🎯 Spot GEX (By Strike)": "/api/stock/{ticker}/spot-exposures/strike",
    "📆 Spot GEX (By Expiry & Strike)": "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "📈 Price Levels (Lit/Off)": "/api/stock/{ticker}/stock-volume-price-levels",

    # 🔗 Chain-Level
    "📉 Intraday Stats": "/api/stock/{ticker}/intraday/stats",
    "🕰️ Historic Option Flow": "/api/historic_chains/{ticker}?date={date}",
    "📅 Chains (Today)": "/api/stock/{ticker}/chains/today",
    "📅 Chains (Date)": "/api/stock/{ticker}/chains/date/{date}",
    "📅 Chains (Expirations)": "/api/stock/{ticker}/chains/expirations",
    "🎯 Chains (Strikes)": "/api/stock/{ticker}/chains/strikes"
}

# Streamlit UI
st.title("🧪 Test Unusual Whales API Endpoints")

endpoint_key = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Ticker Symbol", value="AAPL")
selected_date = st.date_input("Select Date", value=date.today())

if st.button("Run Connection Test"):
    path_template = endpoints[endpoint_key]
    
    # Replace {ticker}
    final_path = path_template.replace("{ticker}", ticker.upper())
    
    # Replace {date} if needed
    if "{date}" in final_path:
        final_path = final_path.replace("{date}", selected_date.isoformat())

    url = f"https://api.unusualwhales.com{final_path}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Connected successfully!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")

