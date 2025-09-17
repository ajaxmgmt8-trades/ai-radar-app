import streamlit as st
import requests

# Load the Unusual Whales API key from secrets
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# Define the correct endpoints from the YAML (stock-level)
endpoints = {
    "Stock State": "/api/stock/{ticker}/stock-state",
    "Options Volume": "/api/stock/{ticker}/options-volume",
    "Volatility (Realized)": "/api/stock/{ticker}/volatility/realized",
    "Volatility Stats": "/api/stock/{ticker}/volatility/stats",
    "Volatility Term Structure": "/api/stock/{ticker}/volatility/term-structure",
    "Spot GEX (1min)": "/api/stock/{ticker}/spot-exposures",
    "Spot GEX (By Strike)": "/api/stock/{ticker}/spot-exposures/strike",
    "Spot GEX (By Expiry & Strike)": "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "Price Levels (Lit/Off)": "/api/stock/{ticker}/stock-volume-price-levels"
}

# UI
st.title("🧪 Test Unusual Whales API Endpoints")
st.markdown("**Choose an endpoint and enter a ticker to test live Unusual Whales data.**")

selected_endpoint = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Enter Ticker Symbol", value="AAPL")

if st.button("Run Connection Test"):
    endpoint_path = endpoints[selected_endpoint].replace("{ticker}", ticker.upper())
    url = f"https://api.unusualwhales.com{endpoint_path}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("Connected successfully!")
        st.json(response.json())
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP error: {http_err}")
    except Exception as err:
        st.error(f"❌ Other error: {err}")

