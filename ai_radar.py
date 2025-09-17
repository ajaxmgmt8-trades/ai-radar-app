# ai_radar.py generated with Unusual Whales API support


import requests
import streamlit as st

UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}


# === 1. STOCK DATA ===
def get_stock_state(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/stock-state"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Example:
# stock = get_stock_state("AAPL")
# st.json(stock)


# === 2. OPTION CHAIN ===
def get_option_chains(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/option-chains"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Example:
# chains = get_option_chains("AAPL")
# st.write(chains.get("chains", []))


# === 3. UNUSUAL OPTION FLOW ===
def get_recent_flow(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-recent"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_flow_by_strike(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-per-strike"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_flow_by_expiry(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/stock/{ticker}/flow-per-expiry"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_historic_trades(ticker: str, limit=100, direction="all") -> dict:
    url = f"https://api.unusualwhales.com/api/historic-chains/{ticker}"
    params = {
        "limit": limit,
        "direction": direction
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Example:
# flow = get_recent_flow("AAPL")
# st.dataframe(flow.get("chains", []))


# === 4. EARNINGS ===
def get_earnings_premarket() -> dict:
    url = "https://api.unusualwhales.com/api/earnings/premarket"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_earnings_afterhours() -> dict:
    url = "https://api.unusualwhales.com/api/earnings/afterhours"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_earnings_for_ticker(ticker: str) -> dict:
    url = f"https://api.unusualwhales.com/api/earnings/{ticker}"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Example:
# earnings = get_earnings_for_ticker("MSFT")
# st.json(earnings)


# === 5. FLOW ALERTS ===
def get_flow_alerts(ticker: str) -> dict:
    url = f"https://api/unusualwhales.com/api/stock/{ticker}/flow-alerts"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Example:
# alerts = get_flow_alerts("TSLA")
# st.json(alerts)
