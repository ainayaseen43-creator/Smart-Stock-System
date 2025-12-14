# stock_data.py - COMPLETE REAL-TIME VERSION WITH ALPHA VANTAGE
import yfinance as yf
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
import time
import concurrent.futures
from typing import Dict, List, Any
import requests
import json

print("✅ Loading Real-Time Global Stocks Dashboard with Alpha Vantage")

# ========== CONFIGURATION ==========
ALPHA_VANTAGE_API_KEY = "S7ULZPIQ8OSIWQV8"

class StockDataFetcher:
    """Real-time stock data fetcher using Alpha Vantage as primary source"""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # Cache for 30 seconds
        self.request_count = 0
        self.last_request_time = time.time()
    
    def get_stock_data(self, symbol: str) -> dict:
        """Get real-time stock data with Alpha Vantage as primary source"""
        
        # Check cache first
        current_time = time.time()
        cache_key = f"{symbol}_data"
        if cache_key in self.cache and cache_key in self.cache_time:
            if current_time - self.cache_time[cache_key] < self.cache_duration:
                return self.cache[cache_key]
        
        # Implement rate limiting (5 requests per minute for free tier)
        time_since_last_request = current_time - self.last_request_time
        if self.request_count >= 5 and time_since_last_request < 60:
            sleep_time = 60 - time_since_last_request
            print(f"   ⏳ Rate limit reached. Waiting {sleep_time:.1f}s for {symbol}...")
            time.sleep(sleep_time)
            self.request_count = 0
        
        # Try Alpha Vantage first
        av_data = self._fetch_from_alpha_vantage(symbol)
        
        if av_data['success']:
            self.request_count += 1
            self.last_request_time = time.time()
            self.cache[cache_key] = av_data
            self.cache_time[cache_key] = current_time
            print(f"   ✅ {symbol}: ${av_data['price']:.2f} (Alpha Vantage)")
            return av_data
        
        # Fallback to yfinance
        print(f"   ⚠️  {symbol}: Alpha Vantage failed, trying yfinance...")
        yf_data = self._fetch_from_yfinance(symbol)
        
        if yf_data['success']:
            self.cache[cache_key] = yf_data
            self.cache_time[cache_key] = current_time
            return yf_data
        
        # Return empty data if all sources fail
        return {
            'price': 0.0,
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 0,
            'previous_close': 0.0,
            'success': False,
            'source': 'failed'
        }
    
    def _fetch_from_alpha_vantage(self, symbol: str) -> dict:
        """Fetch real-time data from Alpha Vantage GLOBAL_QUOTE endpoint"""
        try:
            # Use GLOBAL_QUOTE for real-time data
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Debug: Print raw response for first few symbols
            if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                print(f"   📡 Alpha Vantage raw for {symbol}: {json.dumps(data)[:100]}...")
            
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                
                # Extract and parse data safely
                price_str = quote.get("05. price", "0")
                change_str = quote.get("09. change", "0")
                change_pct_str = quote.get("10. change percent", "0%").rstrip('%')
                volume_str = quote.get("06. volume", "0")
                prev_close_str = quote.get("08. previous close", "0")
                
                # Convert to appropriate types
                price = float(price_str) if price_str and price_str != "0" else 0
                change = float(change_str) if change_str else 0
                change_percent = float(change_pct_str) if change_pct_str else 0
                volume = int(float(volume_str)) if volume_str and volume_str != "0" else 0
                prev_close = float(prev_close_str) if prev_close_str and prev_close_str != "0" else price - change
                
                # Validate we have a real price
                if price > 0:
                    return {
                        'price': price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': volume,
                        'previous_close': prev_close,
                        'success': True,
                        'source': 'alpha_vantage'
                    }
            
            # Try alternative endpoint if GLOBAL_QUOTE fails
            return self._fetch_from_alpha_vantage_alternative(symbol)
            
        except Exception as e:
            print(f"   ❌ Alpha Vantage error for {symbol}: {str(e)[:50]}")
            return {'success': False, 'source': 'alpha_vantage'}
    
    def _fetch_from_alpha_vantage_alternative(self, symbol: str) -> dict:
        """Alternative Alpha Vantage endpoint if GLOBAL_QUOTE fails"""
        try:
            # Try TIME_SERIES_INTRADAY for latest price
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Time Series (5min)" in data:
                time_series = data["Time Series (5min)"]
                if time_series:
                    latest_time = max(time_series.keys())
                    latest_data = time_series[latest_time]
                    
                    price = float(latest_data.get("4. close", 0))
                    volume = int(latest_data.get("5. volume", 0))
                    
                    if price > 0:
                        return {
                            'price': price,
                            'change': 0,
                            'change_percent': 0,
                            'volume': volume,
                            'previous_close': price,
                            'success': True,
                            'source': 'alpha_vantage_intraday'
                        }
            
            return {'success': False, 'source': 'alpha_vantage'}
            
        except Exception as e:
            return {'success': False, 'source': 'alpha_vantage'}
    
    def _fetch_from_yfinance(self, symbol: str) -> dict:
        """Fallback to yfinance when Alpha Vantage fails"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get the most recent data possible
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest_price = float(hist['Close'].iloc[-1])
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                
                # Calculate change from previous minute
                if len(hist) >= 2:
                    prev_price = float(hist['Close'].iloc[-2])
                    change = latest_price - prev_price
                    change_percent = (change / prev_price) * 100 if prev_price > 0 else 0
                    prev_close = prev_price
                else:
                    # Get from ticker info if available
                    info = ticker.info
                    prev_close = info.get('previousClose', latest_price)
                    change = latest_price - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                
                return {
                    'price': latest_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume,
                    'previous_close': prev_close,
                    'success': True,
                    'source': 'yfinance'
                }
            
            # Try with longer interval if 1m fails
            hist = ticker.history(period="2d", interval="1d")
            if not hist.empty and len(hist) >= 2:
                latest_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2])
                change = latest_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                
                return {
                    'price': latest_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume,
                    'previous_close': prev_close,
                    'success': True,
                    'source': 'yfinance_daily'
                }
            
            return {'success': False, 'source': 'yfinance'}
            
        except Exception as e:
            print(f"   ❌ yfinance error for {symbol}: {str(e)[:50]}")
            return {'success': False, 'source': 'yfinance'}

# Global fetcher instance
fetcher = StockDataFetcher()

def generate_chart(prices: List[float], change_percent: float, direction: str) -> str:
    """Generate SVG chart based on actual price data"""
    try:
        if prices and len(prices) > 1:
            max_price = max(prices)
            min_price = min(prices)
            price_range = max_price - min_price if max_price != min_price else 1
            
            points = []
            for i, price in enumerate(prices):
                x = i * (120 / max(1, len(prices) - 1))
                y = 40 - ((price - min_price) / price_range * 35)
                points.append(f"{x},{y}")
            
            path_points = "M" + " L".join(points)
            color = "#16a34a" if direction == "up" else "#dc2626"
        else:
            if direction == "up":
                path_points = "M0,30 L20,25 L40,20 L60,25 L80,15 L100,10 L120,5"
                color = "#16a34a"
            else:
                path_points = "M0,5 L20,10 L40,15 L60,10 L80,20 L100,25 L120,30"
                color = "#dc2626"
        
        svg_template = f'''<svg width="120" height="40" xmlns="http://www.w3.org/2000/svg">
            <path d="{path_points}" stroke="{color}" stroke-width="2" fill="none"/>
        </svg>'''
        
        return base64.b64encode(svg_template.encode('utf-8')).decode('utf-8')
    except:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def fetch_stock_data(symbol: str, country: str, currency: str, name: str, sector: str) -> Dict[str, Any]:
    """Fetch real-time data for a single stock"""
    try:
        # Get real-time data from fetcher
        stock_data = fetcher.get_stock_data(symbol)
        
        if not stock_data['success'] or stock_data['price'] <= 0:
            raise ValueError(f"No real-time data available for {symbol}")
        
        price = stock_data['price']
        change_percent = stock_data['change_percent']
        volume = stock_data['volume']
        direction = "up" if change_percent >= 0 else "down"
        
        # Get historical data for chart from yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")
            chart_prices = []
            if not hist.empty and 'Close' in hist.columns:
                chart_prices = hist['Close'].tail(7).tolist()
            
            if len(chart_prices) < 2:
                base_price = price
                chart_prices = []
                for i in range(7):
                    if direction == "up":
                        trend = (i / 6.0) * (abs(change_percent) / 100) * 0.6
                    else:
                        trend = -(i / 6.0) * (abs(change_percent) / 100) * 0.6
                    noise = np.random.uniform(-0.002, 0.002)
                    chart_price = base_price * (1 + trend + noise)
                    chart_prices.append(chart_price)
        except:
            chart_prices = [price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(7)]
        
        return {
            "symbol": symbol,
            "name": name,
            "price": round(price, 2),
            "change_percent": round(change_percent, 2),
            "direction": direction,
            "volume": volume,
            "chart": generate_chart(chart_prices, change_percent, direction),
            "country": country,
            "currency": currency,
            "sector": sector,
            "success": True,
            "source": stock_data.get('source', 'unknown')
        }
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return {
            "symbol": symbol,
            "name": name,
            "price": 0,
            "change_percent": 0,
            "direction": "neutral",
            "volume": 0,
            "chart": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
            "country": country,
            "currency": currency,
            "sector": sector,
            "success": False,
            "source": "error"
        }

def get_live_data():
    """Main function to get live data for all stocks"""
    print(f"\n📊 FETCHING REAL-TIME GLOBAL STOCKS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"🔑 Using Alpha Vantage API key: {ALPHA_VANTAGE_API_KEY[:4]}...{ALPHA_VANTAGE_API_KEY[-4:]}")
    print("="*60)
    
    # Stock definitions - 100+ global stocks
    stocks = [
        # ========== UNITED STATES (30 stocks) ==========
        {"symbol": "AAPL", "country": "US", "currency": "USD", "name": "Apple Inc.", "sector": "Technology"},
        {"symbol": "MSFT", "country": "US", "currency": "USD", "name": "Microsoft Corporation", "sector": "Technology"},
        {"symbol": "GOOGL", "country": "US", "currency": "USD", "name": "Alphabet Inc.", "sector": "Technology"},
        {"symbol": "AMZN", "country": "US", "currency": "USD", "name": "Amazon.com Inc.", "sector": "E-commerce"},
        {"symbol": "TSLA", "country": "US", "currency": "USD", "name": "Tesla Inc.", "sector": "Automotive"},
        {"symbol": "META", "country": "US", "currency": "USD", "name": "Meta Platforms Inc.", "sector": "Technology"},
        {"symbol": "NVDA", "country": "US", "currency": "USD", "name": "NVIDIA Corporation", "sector": "Technology"},
        {"symbol": "JPM", "country": "US", "currency": "USD", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
        {"symbol": "V", "country": "US", "currency": "USD", "name": "Visa Inc.", "sector": "Finance"},
        {"symbol": "JNJ", "country": "US", "currency": "USD", "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"symbol": "WMT", "country": "US", "currency": "USD", "name": "Walmart Inc.", "sector": "Retail"},
        {"symbol": "PG", "country": "US", "currency": "USD", "name": "Procter & Gamble", "sector": "Consumer"},
        {"symbol": "MA", "country": "US", "currency": "USD", "name": "Mastercard Inc.", "sector": "Finance"},
        {"symbol": "DIS", "country": "US", "currency": "USD", "name": "Walt Disney Company", "sector": "Entertainment"},
        {"symbol": "NFLX", "country": "US", "currency": "USD", "name": "Netflix Inc.", "sector": "Entertainment"},
        {"symbol": "ADBE", "country": "US", "currency": "USD", "name": "Adobe Inc.", "sector": "Technology"},
        {"symbol": "PYPL", "country": "US", "currency": "USD", "name": "PayPal Holdings", "sector": "Finance"},
        {"symbol": "INTC", "country": "US", "currency": "USD", "name": "Intel Corporation", "sector": "Technology"},
        {"symbol": "CSCO", "country": "US", "currency": "USD", "name": "Cisco Systems", "sector": "Technology"},
        {"symbol": "PEP", "country": "US", "currency": "USD", "name": "PepsiCo Inc.", "sector": "Consumer"},
        {"symbol": "COST", "country": "US", "currency": "USD", "name": "Costco Wholesale", "sector": "Retail"},
        {"symbol": "MRK", "country": "US", "currency": "USD", "name": "Merck & Co.", "sector": "Healthcare"},
        {"symbol": "ABT", "country": "US", "currency": "USD", "name": "Abbott Laboratories", "sector": "Healthcare"},
        {"symbol": "TMO", "country": "US", "currency": "USD", "name": "Thermo Fisher Scientific", "sector": "Healthcare"},
        {"symbol": "AVGO", "country": "US", "currency": "USD", "name": "Broadcom Inc.", "sector": "Technology"},
        {"symbol": "ACN", "country": "US", "currency": "USD", "name": "Accenture", "sector": "Technology"},
        {"symbol": "CRM", "country": "US", "currency": "USD", "name": "Salesforce", "sector": "Technology"},
        {"symbol": "NKE", "country": "US", "currency": "USD", "name": "Nike Inc.", "sector": "Consumer"},
        {"symbol": "AMD", "country": "US", "currency": "USD", "name": "Advanced Micro Devices", "sector": "Technology"},
        {"symbol": "QCOM", "country": "US", "currency": "USD", "name": "Qualcomm", "sector": "Technology"},
        
        # ========== OTHER COUNTRIES ==========
        # UK
        {"symbol": "HSBC", "country": "UK", "currency": "GBP", "name": "HSBC Holdings", "sector": "Finance"},
        {"symbol": "BP", "country": "UK", "currency": "GBP", "name": "BP PLC", "sector": "Energy"},
        {"symbol": "GSK", "country": "UK", "currency": "GBP", "name": "GSK plc", "sector": "Healthcare"},
        {"symbol": "UL", "country": "UK", "currency": "GBP", "name": "Unilever PLC", "sector": "Consumer"},
        {"symbol": "AZN", "country": "UK", "currency": "GBP", "name": "AstraZeneca PLC", "sector": "Healthcare"},
        {"symbol": "RIO", "country": "UK", "currency": "GBP", "name": "Rio Tinto", "sector": "Mining"},
        {"symbol": "BATS", "country": "UK", "currency": "GBP", "name": "British American Tobacco", "sector": "Consumer"},
        {"symbol": "RDSA", "country": "UK", "currency": "GBP", "name": "Royal Dutch Shell", "sector": "Energy"},
        
        # Canada
        {"symbol": "RY", "country": "Canada", "currency": "CAD", "name": "Royal Bank of Canada", "sector": "Finance"},
        {"symbol": "TD", "country": "Canada", "currency": "CAD", "name": "Toronto-Dominion Bank", "sector": "Finance"},
        {"symbol": "SHOP", "country": "Canada", "currency": "CAD", "name": "Shopify Inc.", "sector": "Technology"},
        {"symbol": "BMO", "country": "Canada", "currency": "CAD", "name": "Bank of Montreal", "sector": "Finance"},
        {"symbol": "CNQ", "country": "Canada", "currency": "CAD", "name": "Canadian Natural Resources", "sector": "Energy"},
        {"symbol": "ENB", "country": "Canada", "currency": "CAD", "name": "Enbridge Inc.", "sector": "Energy"},
        {"symbol": "BCE", "country": "Canada", "currency": "CAD", "name": "BCE Inc.", "sector": "Telecom"},
        
        # Germany
        {"symbol": "SAP", "country": "Germany", "currency": "EUR", "name": "SAP SE", "sector": "Technology"},
        {"symbol": "SIE", "country": "Germany", "currency": "EUR", "name": "Siemens AG", "sector": "Industrial"},
        {"symbol": "DAI", "country": "Germany", "currency": "EUR", "name": "Mercedes-Benz Group", "sector": "Automotive"},
        {"symbol": "BMW", "country": "Germany", "currency": "EUR", "name": "BMW AG", "sector": "Automotive"},
        {"symbol": "BAS", "country": "Germany", "currency": "EUR", "name": "BASF SE", "sector": "Chemical"},
        {"symbol": "ALV", "country": "Germany", "currency": "EUR", "name": "Allianz SE", "sector": "Finance"},
        {"symbol": "DTE", "country": "Germany", "currency": "EUR", "name": "Deutsche Telekom", "sector": "Telecom"},
        
        # France
        {"symbol": "TTE", "country": "France", "currency": "EUR", "name": "TotalEnergies SE", "sector": "Energy"},
        {"symbol": "SAN", "country": "France", "currency": "EUR", "name": "Sanofi", "sector": "Healthcare"},
        {"symbol": "AIR", "country": "France", "currency": "EUR", "name": "Airbus SE", "sector": "Aerospace"},
        {"symbol": "BNP", "country": "France", "currency": "EUR", "name": "BNP Paribas", "sector": "Finance"},
        {"symbol": "MC", "country": "France", "currency": "EUR", "name": "LVMH Moët Hennessy", "sector": "Luxury"},
        {"symbol": "OR", "country": "France", "currency": "EUR", "name": "L'Oréal", "sector": "Consumer"},
        {"symbol": "CAP", "country": "France", "currency": "EUR", "name": "Capgemini", "sector": "Technology"},
        
        # Japan
        {"symbol": "TOYOF", "country": "Japan", "currency": "JPY", "name": "Toyota Motor", "sector": "Automotive"},
        {"symbol": "SONY", "country": "Japan", "currency": "JPY", "name": "Sony Group", "sector": "Entertainment"},
        {"symbol": "NTDOY", "country": "Japan", "currency": "JPY", "name": "Nintendo", "sector": "Gaming"},
        {"symbol": "HMC", "country": "Japan", "currency": "JPY", "name": "Honda Motor", "sector": "Automotive"},
        {"symbol": "MUFG", "country": "Japan", "currency": "JPY", "name": "Mitsubishi UFJ Financial", "sector": "Finance"},
        {"symbol": "NTT", "country": "Japan", "currency": "JPY", "name": "Nippon Telegraph & Telephone", "sector": "Telecom"},
        {"symbol": "HITACHI", "country": "Japan", "currency": "JPY", "name": "Hitachi Ltd", "sector": "Industrial"},
        
        # China
        {"symbol": "BABA", "country": "China", "currency": "CNY", "name": "Alibaba Group", "sector": "E-commerce"},
        {"symbol": "PDD", "country": "China", "currency": "CNY", "name": "Pinduoduo Inc.", "sector": "E-commerce"},
        {"symbol": "TCEHY", "country": "China", "currency": "CNY", "name": "Tencent Holdings", "sector": "Technology"},
        {"symbol": "JD", "country": "China", "currency": "CNY", "name": "JD.com Inc.", "sector": "E-commerce"},
        {"symbol": "NIO", "country": "China", "currency": "CNY", "name": "NIO Inc.", "sector": "Automotive"},
        {"symbol": "BIDU", "country": "China", "currency": "CNY", "name": "Baidu Inc.", "sector": "Technology"},
        {"symbol": "XPEV", "country": "China", "currency": "CNY", "name": "XPeng Inc.", "sector": "Automotive"},
        
        # India
        {"symbol": "INFY", "country": "India", "currency": "INR", "name": "Infosys Ltd", "sector": "Technology"},
        {"symbol": "RELIANCE", "country": "India", "currency": "INR", "name": "Reliance Industries", "sector": "Conglomerate"},
        {"symbol": "TCS", "country": "India", "currency": "INR", "name": "Tata Consultancy Services", "sector": "Technology"},
        {"symbol": "HDB", "country": "India", "currency": "INR", "name": "HDFC Bank", "sector": "Finance"},
        {"symbol": "WIPRO", "country": "India", "currency": "INR", "name": "Wipro Ltd", "sector": "Technology"},
        {"symbol": "SBIN", "country": "India", "currency": "INR", "name": "State Bank of India", "sector": "Finance"},
        {"symbol": "ICICIBANK", "country": "India", "currency": "INR", "name": "ICICI Bank", "sector": "Finance"},
        
        # Australia
        {"symbol": "BHP", "country": "Australia", "currency": "AUD", "name": "BHP Group Ltd", "sector": "Mining"},
        {"symbol": "CBA", "country": "Australia", "currency": "AUD", "name": "Commonwealth Bank", "sector": "Finance"},
        {"symbol": "RIO.AX", "country": "Australia", "currency": "AUD", "name": "Rio Tinto Ltd", "sector": "Mining"},
        {"symbol": "CSL", "country": "Australia", "currency": "AUD", "name": "CSL Limited", "sector": "Healthcare"},
        {"symbol": "WOW", "country": "Australia", "currency": "AUD", "name": "Woolworths Group", "sector": "Retail"},
        
        # Switzerland
        {"symbol": "NVS", "country": "Switzerland", "currency": "CHF", "name": "Novartis AG", "sector": "Healthcare"},
        {"symbol": "UBS", "country": "Switzerland", "currency": "CHF", "name": "UBS Group AG", "sector": "Finance"},
        {"symbol": "ROG", "country": "Switzerland", "currency": "CHF", "name": "Roche Holding AG", "sector": "Healthcare"},
        {"symbol": "NESTLE", "country": "Switzerland", "currency": "CHF", "name": "Nestlé SA", "sector": "Consumer"},
        {"symbol": "ABB", "country": "Switzerland", "currency": "CHF", "name": "ABB Ltd", "sector": "Industrial"},
        
        # Netherlands
        {"symbol": "ASML", "country": "Netherlands", "currency": "EUR", "name": "ASML Holding", "sector": "Technology"},
        {"symbol": "PHG", "country": "Netherlands", "currency": "EUR", "name": "Philips", "sector": "Healthcare"},
        {"symbol": "AD", "country": "Netherlands", "currency": "EUR", "name": "Adyen NV", "sector": "Finance"},
        
        # Singapore
        {"symbol": "DBS", "country": "Singapore", "currency": "SGD", "name": "DBS Group", "sector": "Finance"},
        {"symbol": "SGX", "country": "Singapore", "currency": "SGD", "name": "Singapore Exchange", "sector": "Finance"},
        {"symbol": "SIA", "country": "Singapore", "currency": "SGD", "name": "Singapore Airlines", "sector": "Airlines"}
    ]
    
    stocks_data = []
    total_volume = 0
    positive_stocks = 0
    countries_covered = set()
    sources_used = {}
    
    print(f"📈 Processing {len(stocks)} stocks from 12+ countries...")
    print("Note: Alpha Vantage free tier allows 5 requests/minute")
    print("="*60)
    
    # Use multithreading but respect Alpha Vantage rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data, 
                stock["symbol"], stock["country"], stock["currency"], 
                stock["name"], stock["sector"]): stock for stock in stocks
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_info = future_to_stock[future]
            try:
                result = future.result(timeout=15)
                stocks_data.append(result)
                
                if result["success"]:
                    total_volume += result["volume"]
                    countries_covered.add(result["country"])
                    if result["change_percent"] >= 0:
                        positive_stocks += 1
                    
                    # Track data sources
                    source = result.get("source", "unknown")
                    sources_used[source] = sources_used.get(source, 0) + 1
                
                completed += 1
                
                # Show progress for first few stocks
                if completed <= 10 and result["success"]:
                    print(f"   {result['symbol']}: ${result['price']:.2f} ({result.get('source', '?')})")
                
            except Exception as e:
                print(f"❌ Failed to fetch {stock_info['symbol']}: {e}")
                stocks_data.append({
                    "symbol": stock_info["symbol"],
                    "name": stock_info["name"],
                    "price": 0,
                    "change_percent": 0,
                    "direction": "neutral",
                    "volume": 0,
                    "chart": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                    "country": stock_info["country"],
                    "currency": stock_info["currency"],
                    "sector": stock_info["sector"],
                    "success": False,
                    "source": "failed"
                })
    
    # Calculate market indicators
    successful_stocks = [s for s in stocks_data if s["success"]]
    total_stocks = len(successful_stocks) if successful_stocks else len(stocks_data)
    
    if successful_stocks:
        sentiment = (positive_stocks / total_stocks) * 100
        volatility_indicator = sum(abs(s["change_percent"]) for s in successful_stocks) / total_stocks
    else:
        sentiment = 50.0
        volatility_indicator = 1.5
    
    print(f"\n📊 MARKET SUMMARY:")
    print(f"   Successfully fetched: {len(successful_stocks)}/{len(stocks)} stocks")
    print(f"   Data sources: {sources_used}")
    print(f"   Countries covered: {len(countries_covered)}")
    print(f"   Market sentiment: {sentiment:.1f}% ({positive_stocks}/{total_stocks} bullish)")
    print(f"   Total volume: {total_volume:,}")
    print("="*60)
    print("💡 TIPS:")
    print("   1. Check Alpha Vantage dashboard for API usage")
    print("   2. Some international symbols may need adjustment")
    print("   3. Real-time data is best during market hours")
    print("="*60)
    
    return {
        "market_indicators": {
            "sentiment": round(sentiment, 1),
            "volatility": round(volatility_indicator, 2),
            "total_stocks": total_stocks,
            "total_volume": total_volume,
            "countries_covered": len(countries_covered),
            "countries_list": sorted(list(countries_covered)),
            "market_status": "Open",
            "data_sources": sources_used,
            "last_updated": datetime.now().isoformat()
        },
        "stocks_data": stocks_data
    }

if __name__ == "__main__":
    # Test the updated code
    print("🧪 TESTING UPDATED REAL-TIME SYSTEM")
    print("="*60)
    
    data = get_live_data()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Total stocks processed: {len(data['stocks_data'])}")
    print(f"   Data sources: {data['market_indicators'].get('data_sources', {})}")
    
    print("\n📋 SAMPLE STOCKS (REAL-TIME):")
    sample_count = 0
    for stock in data['stocks_data']:
        if stock["success"] and stock["price"] > 0:
            print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock.get('source', '?')})")
            sample_count += 1
            if sample_count >= 5:
                break
    
    if sample_count == 0:
        print("   ❌ No real-time data received. Check API key and network.")