import os
import yfinance as yf

# Define top 10 Indian stock indices and their Yahoo Finance symbols
indices = {
    "NIFTY 50": "^NSEI",
    "S&P BSE Sensex": "^BSESN",
    "NIFTY Next 50": "^NSMIDCP",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Midcap 100": "^NSEMDCP50",
    
    "NIFTY 500": "^CRSLDX",
    "NIFTY IT": "^CNXIT",
    "NIFTY FMCG": "^CNXFMCG",
    
}

# Create a folder to store downloaded data
folder_path = "indices"
os.makedirs(folder_path, exist_ok=True)

# Function to download index data with fallback periods
def download_index_data(index_name, symbol):
    fallback_periods = ["max", "10y", "5y", "2y", "1y"]
    for period in fallback_periods:
        try:
            print(f"Trying to download {index_name} ({symbol}) for period: {period}")
            data = yf.download(symbol, period=period)
            if data is not None and not data.empty:
                file_path = os.path.join(folder_path, f"{index_name.replace(' ', '_')}.csv")
                data.to_csv(file_path)
                print(f"Data for {index_name} saved to {file_path}")
                return
        except Exception as e:
            print(f"Failed with period '{period}' for {index_name}: {e}")
    print(f"❌ Could not download data for {index_name} ({symbol}) with any fallback period.")

# Download data for each index
for index_name, symbol in indices.items():
    download_index_data(index_name, symbol)

print("\n✅ All index data download attempts completed.")
