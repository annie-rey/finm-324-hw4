import pandas as pd

def load_data(quotes_path, executions_path):
    """Docstring"""
    quotes = pd.read_csv(quotes_path, dtype = {"ticker": "category"})
    quotes["sip_timestamp"] = pd.to_datetime(quotes["sip_timestamp"], unit="ns")

    executions = pd.read_csv(executions_path, dtype={"symbol": "category"})
    executions["execution_time"] = pd.to_datetime(executions["execution_time"])
    executions["order_time"] = pd.to_datetime(executions["order_time"])

    return quotes, executions

def match_categories(quotes, executions):
    """Docstring"""
    #setting the categories to the same values prevents errors with merge_asof down the line
    executions_tickers = set(executions["symbol"].cat.categories)
    quotes_tickers = set(quotes["ticker"].cat.categories)
    all_tickers = executions_tickers.union(quotes_tickers)
    executions.symbol = executions.symbol.cat.set_categories(all_tickers)
    quotes.ticker = quotes.ticker.cat.set_categories(all_tickers)

    return quotes, executions

def merge_data(quotes, executions):
    """Docstring"""
    executions = executions.sort_values(by=["order_time", "symbol"])
    quotes = quotes.sort_values(by=["sip_timestamp", "ticker"])
    
    merged = pd.merge_asof(
        executions,
        quotes,
        left_on='order_time',
        right_on='sip_timestamp',
        left_by='symbol',
        right_by='ticker',
        direction='backward'
    )

    return merged

def clean_merged_data(merged_data):
    """Docstring"""

    merged_data = merged_data.dropna()
    
    #truncate to market hours

    return merged_data

def add_price_improvement(merged_data):
    pass

