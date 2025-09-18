import streamlit as st
import requests
from datetime import date

# Load API key securely
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json, text/plain"
}

# Supported endpoints from documentation
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
    "📅 Option Chains by Date": "/api/stock/{ticker}/option-chains?date={date}",
    "📜 Historic Chains by Date": "/api/historic_chains/{ticker}?date={date}"
}

# ──────────────────────────────────────────────────────────────
st.title("🧪 Test Unusual Whales API Endpoints")

endpoint_key = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Ticker Symbol", value="AAPL")
selected_date = st.date_input("Select Date (if needed)", value=date.today())

if st.button("Run Test"):
    path_template = endpoints[endpoint_key]
    path = path_template.replace("{ticker}", ticker.upper())
    if "{date}" in path:
        path = path.replace("{date}", selected_date.isoformat())

    # Handle query parameters if needed
    if "?" in path:
        base_path, query = path.split("?")
        query_params = dict(param.split("=") for param in query.split("&"))
        query_params["date"] = selected_date.isoformat()
        url = f"https://api.unusualwhales.com{base_path}"
        params = query_params
    else:
        url = f"https://api.unusualwhales.com{path}"
        params = {}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Connected!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("📜 Option Contract History Test")

contract_id = st.text_input("Enter Option Contract ID (e.g. TSLA230526P00167500)")
limit = st.number_input("Limit (optional)", min_value=1, value=1, step=1)

if st.button("Run Option Contract History"):
    url = f"https://api.unusualwhales.com/api/option-contract/{contract_id}/historic"
    params = {"limit": int(limit)} if limit else {}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Data retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
