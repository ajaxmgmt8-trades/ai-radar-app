import requests

POLYGON_KEY = "pk_your_polygon_key_here"  # replace with your actual key

def test_trade():
    url = f"https://api.polygon.io/v2/last/trade/AAPL?apiKey={POLYGON_KEY}"
    r = requests.get(url, timeout=10).json()
    print("Last Trade:", r)

def test_prev_close():
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={POLYGON_KEY}"
    r = requests.get(url, timeout=10).json()
    print("Previous Close:", r)

def test_snapshot():
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={POLYGON_KEY}"
    r = requests.get(url, timeout=10).json()
    print("Snapshot Gainers:", r)

if __name__ == "__main__":
    test_trade()
    test_prev_close()
    test_snapshot()
