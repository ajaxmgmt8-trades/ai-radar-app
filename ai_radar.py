
import streamlit as st
import requests
from datetime import date

# Load API key securely
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# Endpoint map (including corrected path)
endpoints = {
    "📊 Stock State": "/api/stock/{ticker}/stock-state",
    "📈 Options Volume": "/api/stock/{ticker}/options-volume",
    "📉 Volatility (Realized)": "/api/stock/{ticker}/volatility/realized",
    "📊 Volatility Stats": "/api/stock/{ticker}/volatility/stats",
    "📉 Volatility Term Structure": "/api/stock/{ticker}/volatility/term-structure",
    "📊 Spot GEX (1min)": "/api/stock/{ticker}/spot-exposures",
    "🎯 Spot GEX (By Strike)": "/api/stock/{ticker}/spot-exposures/strike",
    "📆 Spot GEX (By Expiry & Strike)": "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "📈 Price Levels (Lit/Off)": "/api/stock/{ticker}/stock-volume-price-levels",

    "🕰️ Intraday Stats": "/api/stock/{ticker}/intraday/stats",
    "🕰️ Historic Option Flow": "/api/option-contract/{id}/historic",
    "📅 Chains (Today)": "/api/option-chains/{ticker}/chains/today",
    "📅 Chains (Date)": "/api/option_chains/by-date/{ticker}/{date}",
    "📆 Chains (Expirations)": "/api/option_chains/{ticker}/expirations",
    "🎯 Chains (Strikes by Expiration)": "/api/option_chains/{ticker}/chains/strikes/{expiration}"
}

# UI
st.title("🧪 Test Unusual Whales API Endpoints")
endpoint_key = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Ticker Symbol", value="AAPL")
selected_date = st.date_input("Select Date (if needed)", value=date.today())
selected_expiration = st.text_input("Expiration Date (YYYY-MM-DD)", value="2025-10-18")

if st.button("Run Test"):
    path_template = endpoints[endpoint_key]
    final_path = path_template.replace("{ticker}", ticker.upper())
    if "{date}" in final_path:
        final_path = final_path.replace("{date}", selected_date.isoformat())
    if "{expiration}" in final_path:
        final_path = final_path.replace("{expiration}", selected_expiration)

    url = f"https://api.unusualwhales.com{final_path}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Connected!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

