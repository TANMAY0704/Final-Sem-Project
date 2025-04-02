import yfinance as yf
from nsetools import Nse
import pandas as pd

# Initialize NSE object
nse = Nse()

# Fetch all stock codes with names from NSE
def get_stock_codes():
    stock_codes = nse.get_stock_codes()  # Fetch stock codes

    if isinstance(stock_codes, dict):  # Expected old behavior
        stock_codes.pop('SYMBOL', None)  # Remove 'SYMBOL' key safely
        return stock_codes  # Return dictionary {TICKER: NAME}
    
    elif isinstance(stock_codes, list):  # If it's a list, return as a dictionary with empty names
        return {ticker: ticker for ticker in stock_codes}  # {TICKER: TICKER} as default name

    else:
        raise TypeError(f"Unexpected data structure from get_stock_codes(): {type(stock_codes)}")

# Check if ticker has valid data on Yahoo Finance
def is_valid_ticker(ticker):
    try:
        stock_data = yf.download(ticker + ".NS", period="5y", progress=False)
        return not stock_data.empty  # True if data exists, False otherwise
    except Exception as e:
        print(f"⚠️ Error checking {ticker}: {e}")
        return False

# Save valid tickers and names to CSV
def save_valid_tickers():
    stock_codes = get_stock_codes()
    valid_tickers = []

    for ticker, name in stock_codes.items():
        print(f"Checking {ticker}...")
        if is_valid_ticker(ticker):
            valid_tickers.append({"Ticker": ticker + ".NS", "Stock Name": name})
            print(f"✅ {ticker} is valid.")
        else:
            print(f"❌ {ticker} is invalid.")

    # Convert to DataFrame and save
    df_valid_tickers = pd.DataFrame(valid_tickers)
    csv_path = "valid_nse_tickers.csv"
    df_valid_tickers.to_csv(csv_path, index=False)

    print(f"\n📁 Valid tickers saved to `{csv_path}`.")

# Run the script
if __name__ == '__main__':
    save_valid_tickers()
