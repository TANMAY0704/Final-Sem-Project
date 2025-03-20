import os
import yfinance as yf

# Define top 10 Indian stock indices and their Yahoo Finance symbols
indices = {
    "NIFTY 50": "^NSEI",
    "S&P BSE Sensex": "^BSESN",
    "NIFTY Next 50": "^NSMIDCP",
    "NIFTY Bank": "^NSEBANK",
    "NIFTY Midcap 100": "^NSEMDCP50",
    "NIFTY Smallcap 100": "^NSESMLCP50",
    "NIFTY 500": "^CRSLDX",
    "NIFTY IT": "^CNXIT",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY Pharma": "^CNXPHARMA"
}

# Create a folder to store downloaded data
folder_path = "indian_stock_indices_data"
os.makedirs(folder_path, exist_ok=True)

# Function to download index data
def download_index_data(index_name, symbol):
    try:
        print(f"Downloading data for {index_name} ({symbol})...")
        data = yf.download(symbol, period="10y")  # Last 10 years of data
        file_path = os.path.join(folder_path, f"{index_name.replace(' ', '_')}.csv")
        data.to_csv(file_path)
        print(f"Data for {index_name} saved to {file_path}")
    except Exception as e:
        print(f"Failed to download data for {index_name}: {e}")

# Download data for each index
for index_name, symbol in indices.items():
    download_index_data(index_name, symbol)

print("\nAll index data downloaded successfully.")
