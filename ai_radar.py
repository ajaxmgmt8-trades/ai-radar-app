import requests

def fetch_uw_chains_by_date(ticker: str, date: str, api_key: str):
    """
    Fetch option chain data for a given ticker and date from Unusual Whales.

    Args:
        ticker (str): The stock ticker (e.g., "AAPL").
        date (str): The date in "YYYY-MM-DD" format.
        api_key (str): Your Unusual Whales API key.

    Returns:
        dict: JSON response from the API.
    """
    url = f"https://api.unusualwhales.com/api/option_chains/by-date/{ticker}/{date}"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

