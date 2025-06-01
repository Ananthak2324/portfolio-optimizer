import yfinance as yf
import pandas as pd

def get_price_data(tickers, start='2019-01-01', end='2024-12-31'):
    """
    Downloads Close prices for the given tickers between the given date range.
    
    Parameters:
        tickers (list or str): Stock tickers (e.g., ['AAPL', 'MSFT'] or 'AAPL')
        start (str): Start date in format 'YYYY-MM-DD'
        end (str): End date in format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: Close prices with dates as index and tickers as columns
    """
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)

        # Handle MultiIndex if multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            # Single ticker case: wrap it as a DataFrame with ticker name as column
            close_data = data[['Close']]
            if isinstance(tickers, str):
                close_data.columns = [tickers]

        return close_data.dropna()

    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return pd.DataFrame()



