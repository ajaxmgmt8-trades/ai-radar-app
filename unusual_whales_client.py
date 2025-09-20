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

    def get_api_stock_ticker_stock-state(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/stock-state"""
        endpoint = f"/api/stock/{ticker}/stock-state".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_options-volume(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/options-volume"""
        endpoint = f"/api/stock/{ticker}/options-volume".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_volatility_realized(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/volatility/realized"""
        endpoint = f"/api/stock/{ticker}/volatility/realized".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_volatility_stats(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/volatility/stats"""
        endpoint = f"/api/stock/{ticker}/volatility/stats".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_volatility_term-structure(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/volatility/term-structure"""
        endpoint = f"/api/stock/{ticker}/volatility/term-structure".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_spot-exposures(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/spot-exposures"""
        endpoint = f"/api/stock/{ticker}/spot-exposures".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_spot-exposures_strike(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/spot-exposures/strike"""
        endpoint = f"/api/stock/{ticker}/spot-exposures/strike".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_spot-exposures_expiry-strike(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/spot-exposures/expiry-strike"""
        endpoint = f"/api/stock/{ticker}/spot-exposures/expiry-strike".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_stock-volume-price-levels(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/stock-volume-price-levels"""
        endpoint = f"/api/stock/{ticker}/stock-volume-price-levels".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_intraday_stats(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/intraday/stats"""
        endpoint = f"/api/stock/{ticker}/intraday/stats".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_option-chains?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/option-chains?date={date}"""
        endpoint = f"/api/stock/{ticker}/option-chains?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_historic_chains_ticker?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/historic_chains/{ticker}?date={date}"""
        endpoint = f"/api/historic_chains/{ticker}?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_option-contract_contract_id_historic(self, ticker):
        """Auto-generated method for endpoint: /api/option-contract/{contract_id}/historic"""
        endpoint = f"/api/option-contract/{contract_id}/historic".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_option-contract_contract_id_intraday?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/option-contract/{contract_id}/intraday?date={date}"""
        endpoint = f"/api/option-contract/{contract_id}/intraday?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_option-contract_contract_id_volume-profile?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/option-contract/{contract_id}/volume-profile?date={date}"""
        endpoint = f"/api/option-contract/{contract_id}/volume-profile?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_earnings_ticker(self, ticker):
        """Auto-generated method for endpoint: /api/earnings/{ticker}"""
        endpoint = f"/api/earnings/{ticker}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_earnings_afterhours?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/earnings/afterhours?date={date}"""
        endpoint = f"/api/earnings/afterhours?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_alerts(self, ticker):
        """Auto-generated method for endpoint: /api/alerts"""
        endpoint = f"/api/alerts".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_alerts_configuration(self, ticker):
        """Auto-generated method for endpoint: /api/alerts/configuration"""
        endpoint = f"/api/alerts/configuration".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_congress_congress-trader?date=date&ticker=ticker(self, ticker):
        """Auto-generated method for endpoint: /api/congress/congress-trader?date={date}&ticker={ticker}"""
        endpoint = f"/api/congress/congress-trader?date={date}&ticker={ticker}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_congress_recent-trades?date=date&ticker=ticker(self, ticker):
        """Auto-generated method for endpoint: /api/congress/recent-trades?date={date}&ticker={ticker}"""
        endpoint = f"/api/congress/recent-trades?date={date}&ticker={ticker}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_darkpool_recent?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/darkpool/recent?date={date}"""
        endpoint = f"/api/darkpool/recent?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_darkpool_ticker?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/darkpool/{ticker}?date={date}"""
        endpoint = f"/api/darkpool/{ticker}?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_etfs_ticker_exposure(self, ticker):
        """Auto-generated method for endpoint: /api/etfs/{ticker}/exposure"""
        endpoint = f"/api/etfs/{ticker}/exposure".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_screener_analysts?ticker=ticker(self, ticker):
        """Auto-generated method for endpoint: /api/screener/analysts?ticker={ticker}"""
        endpoint = f"/api/screener/analysts?ticker={ticker}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_screener_stocks?ticker=ticker&date=date(self, ticker):
        """Auto-generated method for endpoint: /api/screener/stocks?ticker={ticker}&date={date}"""
        endpoint = f"/api/screener/stocks?ticker={ticker}&date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_screener_option-contracts?is_otm=true&date=date(self, ticker):
        """Auto-generated method for endpoint: /api/screener/option-contracts?is_otm=true&date={date}"""
        endpoint = f"/api/screener/option-contracts?is_otm=true&date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_expiry-breakdown?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/expiry-breakdown?date={date}"""
        endpoint = f"/api/stock/{ticker}/expiry-breakdown?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_option-contracts(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/option-contracts"""
        endpoint = f"/api/stock/{ticker}/option-contracts".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_option-trades_flow-alerts?all_opening=true&is_floor=true&is_sweep=true&is_call=true&is_put=true&is_ask_side=true&is_bid_side=true&is_otm=true(self, ticker):
        """Auto-generated method for endpoint: /api/option-trades/flow-alerts?all_opening=true&is_floor=true&is_sweep=true&is_call=true&is_put=true&is_ask_side=true&is_bid_side=true&is_otm=true"""
        endpoint = f"/api/option-trades/flow-alerts?all_opening=true&is_floor=true&is_sweep=true&is_call=true&is_put=true&is_ask_side=true&is_bid_side=true&is_otm=true".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_option-trades_full-tape_date(self, ticker):
        """Auto-generated method for endpoint: /api/option-trades/full-tape/{date}"""
        endpoint = f"/api/option-trades/full-tape/{date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_fda-calendar?ticker=ticker(self, ticker):
        """Auto-generated method for endpoint: /api/market/fda-calendar?ticker={ticker}"""
        endpoint = f"/api/market/fda-calendar?ticker={ticker}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_economic-calendar(self, ticker):
        """Auto-generated method for endpoint: /api/market/economic-calendar"""
        endpoint = f"/api/market/economic-calendar".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_correlations(self, ticker):
        """Auto-generated method for endpoint: /api/market/correlations"""
        endpoint = f"/api/market/correlations".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_insider-buy-sells(self, ticker):
        """Auto-generated method for endpoint: /api/market/insider-buy-sells"""
        endpoint = f"/api/market/insider-buy-sells".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_market-tide?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/market-tide?date={date}"""
        endpoint = f"/api/market/market-tide?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_oi-change?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/oi-change?date={date}"""
        endpoint = f"/api/market/oi-change?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_sector-etfs(self, ticker):
        """Auto-generated method for endpoint: /api/market/sector-etfs"""
        endpoint = f"/api/market/sector-etfs".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_spike?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/spike?date={date}"""
        endpoint = f"/api/market/spike?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_top-net-impact?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/top-net-impact?date={date}"""
        endpoint = f"/api/market/top-net-impact?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_total-options-volume(self, ticker):
        """Auto-generated method for endpoint: /api/market/total-options-volume"""
        endpoint = f"/api/market/total-options-volume".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_sector_sector-tide?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/{sector}/sector-tide?date={date}"""
        endpoint = f"/api/market/{sector}/sector-tide?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_market_ticker_etf-tide?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/market/{ticker}/etf-tide?date={date}"""
        endpoint = f"/api/market/{ticker}/etf-tide?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_net-flow_expiry?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/net-flow/expiry?date={date}"""
        endpoint = f"/api/net-flow/expiry?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_news_headlines(self, ticker):
        """Auto-generated method for endpoint: /api/news/headlines"""
        endpoint = f"/api/news/headlines".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_institutions?name=name(self, ticker):
        """Auto-generated method for endpoint: /api/institutions?name={name}"""
        endpoint = f"/api/institutions?name={name}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_institutions_latest_filings?name=name&date=date(self, ticker):
        """Auto-generated method for endpoint: /api/institutions/latest_filings?name={name}&date={date}"""
        endpoint = f"/api/institutions/latest_filings?name={name}&date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_atm-chains(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/atm-chains"""
        endpoint = f"/api/stock/{ticker}/atm-chains".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_flow-alerts?is_ask_side=true&is_bid_side=true(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/flow-alerts?is_ask_side=true&is_bid_side=true"""
        endpoint = f"/api/stock/{ticker}/flow-alerts?is_ask_side=true&is_bid_side=true".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_flow-per-expiry(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/flow-per-expiry"""
        endpoint = f"/api/stock/{ticker}/flow-per-expiry".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_flow-per-strike?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/flow-per-strike?date={date}"""
        endpoint = f"/api/stock/{ticker}/flow-per-strike?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_flow-per-strike-intraday?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/flow-per-strike-intraday?date={date}"""
        endpoint = f"/api/stock/{ticker}/flow-per-strike-intraday?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_flow-recent(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/flow-recent"""
        endpoint = f"/api/stock/{ticker}/flow-recent".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-exposure?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-exposure?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-exposure?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-exposure_expiry?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-exposure/expiry?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-exposure/expiry?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-exposure_strike?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-exposure/strike?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-exposure/strike?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-exposure_strike-expiry?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-exposure/strike-expiry?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-exposure/strike-expiry?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-flow?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-flow?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-flow?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greek-flow_expiry?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greek-flow/{expiry}?date={date}"""
        endpoint = f"/api/stock/{ticker}/greek-flow/{expiry}?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_ticker_greeks?date=date(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{ticker}/greeks?date={date}"""
        endpoint = f"/api/stock/{ticker}/greeks?date={date}".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_shorts_ticker_data(self, ticker):
        """Auto-generated method for endpoint: /api/shorts/{ticker}/data"""
        endpoint = f"/api/shorts/{ticker}/data".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_shorts_ticker_ftds(self, ticker):
        """Auto-generated method for endpoint: /api/shorts/{ticker}/ftds"""
        endpoint = f"/api/shorts/{ticker}/ftds".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_shorts_ticker_interest-float(self, ticker):
        """Auto-generated method for endpoint: /api/shorts/{ticker}/interest-float"""
        endpoint = f"/api/shorts/{ticker}/interest-float".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_shorts_ticker_volume-and-ratio(self, ticker):
        """Auto-generated method for endpoint: /api/shorts/{ticker}/volume-and-ratio"""
        endpoint = f"/api/shorts/{ticker}/volume-and-ratio".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_shorts_ticker_volumes-by-exchange(self, ticker):
        """Auto-generated method for endpoint: /api/shorts/{ticker}/volumes-by-exchange"""
        endpoint = f"/api/shorts/{ticker}/volumes-by-exchange".replace("{ticker}", ticker)
        return self._make_request(endpoint)

    def get_api_stock_sector_tickers(self, ticker):
        """Auto-generated method for endpoint: /api/stock/{sector}/tickers"""
        endpoint = f"/api/stock/{sector}/tickers".replace("{ticker}", ticker)
        return self._make_request(endpoint)

