import requests
import streamlit as st

# Load Unusual Whales API Key securely from secrets
UW_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "")

def get_unusual_whales_flow(ticker: str, limit: int = 20):
    """
    Fetch options flow data from Unusual Whales API (/v2/historic_chains)
    """
    if not UW_KEY:
        return {"error": "❌ API key not found in st.secrets. Check your secrets configuration."}

    url = f"https://api.unusualwhales.com/v2/historic_chains/{ticker.upper()}"
    params = {
        "limit": limit,
        "direction": "all",
        "order": "desc"
    }
    headers = {
        "Authorization": f"Bearer {UW_KEY}",
        "accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        st.write(f"Status Code: {response.status_code}")

        # Return error if not successful
        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {response.text}"}

        # Parse and return JSON data
        data = response.json()
        return data.get("chains", [])

    except ValueError:
        return {"error": f"Failed to parse JSON. Raw response:\n{response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

# ----- Streamlit UI -----

st.title("🐋 Unusual Whales Options Flow Viewer")

ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL").strip().upper()

if st.button("Fetch Flow"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching options flow for {ticker}..."):
            flow = get_unusual_whales_flow(ticker, limit=10)

            if isinstance(flow, dict) and flow.get("error"):
                st.error(flow["error"])
            elif not flow:
                st.info("No flow data found for this ticker.")
            else:
                st.success(f"Showing top {len(flow)} flow entries for {ticker}")
                for idx, f in enumerate(flow, 1):
                    st.markdown(
                        f"""
                        **{idx}. {f.get('symbol', 'N/A')}**  
                        • Type: `{f.get('type', 'N/A')}`  
                        • Strike: `{f.get('strike', 'N/A')}`  
                        • Expiration: `{f.get('expiration', 'N/A')}`  
                        • Premium: `${f.get('premium', 0):,.2f}`  
                        • Side: `{f.get('side', 'N/A')}`  
                        • Timestamp: `{f.get('timestamp', 'N/A')}`
                        ---
                        """
                    )
