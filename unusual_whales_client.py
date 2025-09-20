import requests
from datetime import date

class UnusualWhalesClient:
    def __init__(self, api_key):
        self.base_url = "https://api.unusualwhales.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json, text/plain"
        }
        self.endpoints = {
            "Flow Alerts": "/api/option-trades/flow-alerts?all_opening=true&is_floor=true&is_sweep=true&is_call=true&is_put=true&is_ask_side=true&is_bid_side=true&is_otm=true",
            "Full Tape": "/api/option-trades/full-tape/{date}",
            "Market - Market Tide": "/api/market/market-tide?date={date}",
            "Market - Sector ETFs": "/api/market/sector-etfs",
            "Stock - Flow Recent": "/api/stock/{ticker}/flow-recent",
            "Stock - Greek Flow": "/api/stock/{ticker}/greek-flow?date={date}",
            "Stock - Greeks": "/api/stock/{ticker}/greeks?date={date}",
            "Stock - Flow Per Strike": "/api/stock/{ticker}/flow-per-strike?date={date}",
            "News Headlines": "/api/news/headlines",
            "Market - Economic Calendar": "/api/market/economic-calendar",
            "Market - FDA Calendar": "/api/market/fda-calendar?ticker={ticker}",
            "Stock - ATM Chains": "/api/stock/{ticker}/atm-chains"
        }

    def _get(self, endpoint_name, **kwargs):
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        path = self.endpoints[endpoint_name]
        for key, value in kwargs.items():
            path = path.replace(f"{{{key}}}", value)
        url = f"{self.base_url}{path}"
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_flow_alerts(self):
        return self._get("Flow Alerts")

    def get_full_tape(self, date):
        return self._get("Full Tape", date=date)

    def get_market_tide(self, date):
        return self._get("Market - Market Tide", date=date)

    def get_sector_etfs(self):
        return self._get("Market - Sector ETFs")

    def get_stock_flow_recent(self, ticker):
        return self._get("Stock - Flow Recent", ticker=ticker)

    def get_stock_greek_flow(self, ticker, date):
        return self._get("Stock - Greek Flow", ticker=ticker, date=date)

    def get_stock_greeks(self, ticker, date):
        return self._get("Stock - Greeks", ticker=ticker, date=date)

    def get_flow_per_strike(self, ticker, date):
        return self._get("Stock - Flow Per Strike", ticker=ticker, date=date)

    def get_news_headlines(self):
        return self._get("News Headlines")

    def get_economic_calendar(self):
        return self._get("Market - Economic Calendar")

    def get_fda_calendar(self, ticker):
        return self._get("Market - FDA Calendar", ticker=ticker)

    def get_atm_chains(self, ticker):
        return self._get("Stock - ATM Chains", ticker=ticker)

