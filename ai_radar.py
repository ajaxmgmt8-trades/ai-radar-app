import requests
import streamlit as st

UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]

def get_option_contract_history(contract_id: str, limit: int = None) -> dict:
    """
    Fetch historical data for a specific option contract by ID.
    
    Parameters:
        contract_id (str): The full ISO-style ID of the contract (e.g. "TSLA230526P00167500").
        limit (int, optional): Number of records to return. If not provided, fetches all.
    
    Returns:
        dict: The JSON response from Unusual Whales API.
    """
    base_url = f"https://api.unusualwhales.com/api/option-contract/{contract_id}/historic"
    
    params = {}
    if limit:
        params["limit"] = limit

    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
        "accept": "application/json"
    }

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
