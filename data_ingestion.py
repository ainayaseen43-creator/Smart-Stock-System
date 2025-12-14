import yfinance as yf

# Select stock ticker
ticker = "AAPL"

# Download historical OHLCV data (last 5 years)
data = yf.download(ticker, start="2019-01-01", end="2024-12-31")

# Print first few rows
print(data.head())

# Save to CSV
data.to_csv("AAPL_data.csv")
print("✅ Saved as AAPL_data.csv")
