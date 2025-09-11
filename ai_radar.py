import requests

POLYGON_KEY = "pk_your_polygon_key_here"
url = f"https://api.polygon.io/v2/last/trade/AAPL?apiKey={POLYGON_KEY}"
r = requests.get(url).json()
print(r)
