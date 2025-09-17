import requests
import streamlit as st

UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]

def test_endpoint():
    url = "https://api.unusualwhales.com/api/...your-endpoint..."
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

# Streamlit UI
st.subheader("🔌 Test [Descriptive Endpoint Name]")
if st.button("Run Test"):
    result = test_endpoint()
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Connected successfully!")
        st.json(result)
