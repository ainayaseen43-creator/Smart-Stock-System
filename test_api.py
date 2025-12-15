import requests
import json

ALPHA_VANTAGE_API_KEY = "S7ULZPIQ8OSIWQV8"

def get_stock_data(symbol):
    """Fetch stock data and return as JSON"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}

# Test with multiple symbols - expanded list
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
results = {}

print("Fetching data from Alpha Vantage API...\n")

for symbol in symbols:
    results[symbol] = get_stock_data(symbol)

print(json.dumps(results, indent=2))

with open('api_response.json', 'w') as f:
    json.dump(results, f, indent=2)

