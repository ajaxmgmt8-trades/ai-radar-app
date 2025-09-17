import streamlit as st
import requests

# 🔐 Load API Key
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]

# 🧠 App Title
st.title("🧪 Test Unusual Whales API Endpoints")

# 🔒 Headers
headers = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

# 📊 Define endpoints to test
endpoints = {
    "📈 Stock State": "https://api.unusualwhales.com/api/stock/AAPL/stock-state",
    "📊 Intraday Stats": "https://api.unusualwhales.com/api/stock/AAPL/intraday-stats",
    "🕰️ Historic Option Flow": "https://api.unusualwhales.com/api/historic_chains/AAPL?date=2025-09-13",
    "🧮 Chains (Today)": "https://api.unusualwhales.com/api/stock/AAPL/chains/today",
    "📅 Chains (Date)": "https://api.unusualwhales.com/api/stock/AAPL/chains/date/2025-09-13",
    "📆 Chains (Expirations)": "https://api.unusualwhales.com/api/stock/AAPL/chains/expirations",
    "🎯 Chains (Strikes)": "https://api.unusualwhales.com/api/stock/AAPL/chains/strikes?expiration=2025-09-20"
}

# 🧭 Endpoint selector
selected_name = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
selected_url = endpoints[selected_name]

# 🔁 API request function
def test_unusual_whales_endpoint(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ▶️ Test trigger
if st.button("Run Connection Test"):
    result = test_unusual_whales_endpoint(selected_url)
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        st.success("✅ Connected successfully!")
        st.json(result)
