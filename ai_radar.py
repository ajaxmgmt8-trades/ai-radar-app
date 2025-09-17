import streamlit as st
import requests
from datetime import date

# Load API key from Streamlit secrets
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# All valid endpoints from document.yaml
endpoints = {
    # 📈 Stock-Level Endpoints
    "📊 Stock State": "/api/stock/{ticker}/stock-state",
    "📈 Options Volume": "/api/stock/{ticker}/options-volume",
    "📉 Volatility (Realized)": "/api/stock/{ticker}/volatility/realized",
    "📊 Volatility Stats": "/api/stock/{ticker}/volatility/stats",
    "📉 Volatility Term Structure": "/api/stock/{ticker}/volatility/term-structure",
    "📊 Spot GEX (1min)": "/api/stock/{ticker}/spot-exposures",
    "🎯 Spot GEX (By Strike)": "/api/stock/{ticker}/spot-exposures/strike",
    "📆 Spot GEX (By Expiry & Strike)": "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "📈 Price Levels (Lit/Off)": "/api/stock/{ticker}/stock-volume-price-levels",

    # 🔗 Chain-Level Endpoints
    "🕰️ Intraday Stats": "/api/stock/{ticker}/intraday/stats",
    "🕰️ Historic Option Flow": "/api/historic_chains/{ticker}?date={date}",
    "📅 Chains (Today)": "/api/option_chains/{ticker}/chains/today",
    "📅 Chains (Date)": "/api/option_chains/{ticker}/chains/date/{date}",
    "📆 Chains (Expirations)": "/api/option_chains/{ticker}/chains/expirations",
    "🎯 Chains (Strikes by Expiration)": "/api/option_chains/{ticker}/chains/strikes/{expiration}"
}

# Streamlit UI
st.title("🧪 Test Unusual Whales API (All Endpoints)")
endpoint_key = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Ticker Symbol", value="AAPL")
selected_date = st.date_input("Select Date (if required)", value=date.today())
selected_expiration = st.text_input("Expiration (YYYY-MM-DD) for strikes", value="2025-09-20")

# When user clicks button
if st.button("Run Connection Test"):
    path = endpoints[endpoint_key]
    path = path.replace("{ticker}", ticker.upper())
    path = path.replace("{date}", selected_date.isoformat()) if "{date}" in path else path
    path = path.replace("{expiration}", selected_expiration) if "{expiration}" in path else path

    url = f"https://api.unusualwhales.com{path}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Connected successfully!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
