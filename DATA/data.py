import yfinance as yf
from nsetools import Nse
import os

# Initialize NSE object
nse = Nse()

# Fetch all stock codes from NSE
def get_stock_codes():
    stock_codes = nse.get_stock_codes()  # Fetch stock codes

    if isinstance(stock_codes, dict):  # Old expected behavior
        stock_codes.pop('SYMBOL', None)  # Remove 'SYMBOL' key safely
        return list(stock_codes.keys())
    
    elif isinstance(stock_codes, list):  # Handle unexpected list structure
        return stock_codes  # Simply return the list

    else:
        raise TypeError(f"Unexpected data structure from get_stock_codes(): {type(stock_codes)}")

# Function to download stock data using yfinance
def download_stock_data(stock_codes, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for stock_code in stock_codes:
        try:
            print(f"Downloading data for {stock_code}...")
            stock_data = yf.download(stock_code + '.NS')  # '.NS' suffix for NSE
            stock_data.to_csv(os.path.join(folder_path, f'{stock_code}.csv'))
            print(f"Data for {stock_code} saved.")
        except Exception as e:
            print(f"Could not download data for {stock_code}: {e}")

# Main function
if __name__ == '__main__':
    try:
        stock_codes = get_stock_codes()
        print(f"Found {len(stock_codes)} stocks.")
        
        folder_path = 'nse_stock_data'
        download_stock_data(stock_codes, folder_path)
        print(f"Stock data downloaded and saved to {folder_path}.")
    except Exception as e:
        print(f"Error: {e}")
