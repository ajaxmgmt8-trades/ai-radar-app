import requests
import streamlit as st

UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]

def test_unusual_whales_connection():
    url = "https://api.unusualwhales.com/api/stock/AAPL/stock-state"
    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
        "accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# In Streamlit UI
st.subheader("🔌 Test Unusual Whales API")
if st.button("Run Connection Test"):
    result = test_unusual_whales_connection()
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Connected successfully!")
        st.json(result)
