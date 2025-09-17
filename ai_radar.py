if __name__ == "__main__":
    data = fetch_uw_chains_by_date(
        ticker="AAPL",
        date="2025-09-19",
        api_key="your_api_key_here"
    )
    print(data)
