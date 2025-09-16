import requests
import datetime
import streamlit as st

# Load key
UW_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")

def get_unusual_whales_flow(ticker: str, limit: int = 20):
    """
    Fetch options flow from Unusual Whales
    """
    if not UW_KEY:
        return {"error": "Unusual Whales API key not found in st.secrets"}

    url = f"https://api.unusualwhales.com/api/historic_chains/{ticker}"
    params = {
        "limit": limit,
        "direction": "all",   # "call", "put", or "all"
        "order": "desc"       # most recent first
    }
    headers = {
        "Authorization": f"Bearer {UW_KEY}",
        "accept": "application/json"
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return {"error": f"API Error {r.status_code}: {r.text}"}
        data = r.json()
        return data.get("chains", [])  # UW wraps results under 'chains'
    except Exception as e:
        return {"error": str(e)}

# Example Streamlit UI
st.subheader("🐋 Unusual Whales Options Flow")

ticker = st.text_input("Enter ticker", "AAPL").upper().strip()
if st.button("Fetch Flow"):
    with st.spinner(f"Fetching Unusual Whales flow for {ticker}..."):
        flow = get_unusual_whales_flow(ticker, 10)
        if isinstance(flow, dict) and flow.get("error"):
            st.error(flow["error"])
        elif flow:
            for f in flow:
                st.write(f"**{f.get('symbol')}** | {f.get('type')} | Strike: {f.get('strike')} | Exp: {f.get('expiration')} | Prem: {f.get('premium')}")
        else:
            st.info("No flow data found.")
