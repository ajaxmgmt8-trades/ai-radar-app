import streamlit as st
import requests

# Load the trial API key from secrets
UW_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")

def get_supported_tickers():
    """
    Fetch supported tickers (trial-accessible endpoint)
    """
    url = "https://api.unusualwhales.com/api/supported_tickers/"
    headers = {
        "Authorization": f"Bearer {UW_KEY}",
        "accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        st.write(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# Streamlit UI
st.title("✅ Unusual Whales Trial API - Supported Tickers")

if st.button("Fetch Tickers"):
    with st.spinner("Loading tickers from Unusual Whales Trial API..."):
        result = get_supported_tickers()
        if isinstance(result, dict) and result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"Loaded {len(result)} tickers")
            st.dataframe(result)
