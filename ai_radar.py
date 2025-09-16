import streamlit as st
import requests

UW_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")

def get_option_chains(ticker: str):
    """
    Fetch option chains for a given ticker.
    """
    url = f"https://api.unusualwhales.com/api/stock/{ticker.upper()}/option-chains"
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

# UI
st.title("📊 Unusual Whales - Option Chains")

ticker = st.text_input("Enter stock ticker", value="AAPL")

if st.button("Fetch Option Chains"):
    with st.spinner(f"Fetching option chains for {ticker}..."):
        data = get_option_chains(ticker)

        if isinstance(data, dict) and data.get("error"):
            st.error(data["error"])
        elif not data:
            st.warning("No data returned.")
        else:
            st.success(f"Loaded {len(data)} contracts")
            st.dataframe(data)
