# Simple app without emoji issues
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import traceback
import sys

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def home():
    try:
        return render_template("dashboard.html")
    except Exception as e:
        print(f"Error loading dashboard: {str(e)}", file=sys.stderr)
        return f"Error loading dashboard: {str(e)}"

@app.route('/get_live_data')
def get_data():
    """Get live stock data"""
    try:
        print("\n" + "="*60)
        print("FETCHING FRESH STOCK DATA")
        print("="*60)
        
        # Import here to avoid emoji issues at startup
        import stock_data
        data = stock_data.get_live_data()
        
        # Verify we have correct prices
        print("\nVERIFYING CORRECT PRICES:")
        for stock in data['stocks_data'][:5]:
            if stock["success"]:
                print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock.get('source', '?')})")
        
        print(f"Market Sentiment: {data['market_indicators']['sentiment']}%")
        print("="*60)
        
        # Create response with NO CACHE headers
        response = jsonify(data)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        print(f"ERROR in get_live_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    from datetime import datetime
    return jsonify({
        "status": "healthy",
        "service": "StockSense AI",
        "version": "3.0",
        "timestamp": datetime.now().isoformat(),
        "message": "Real-time data enabled with correct prices"
    })

if __name__ == "__main__":
    try:
        print("="*60)
        print("STARTING STOCKSENSE SERVER V3.0")
        print("="*60)
        print("FEATURES:")
        print("   - Real-time stock prices via Alpha Vantage")
        print("   - 100+ Global Stocks Dashboard")
        print("   - No Cache - Always Fresh Data")
        print("="*60)
        print("URLs:")
        print("   Dashboard: http://127.0.0.1:5000")
        print("   Live Data: http://127.0.0.1:5000/get_live_data")
        print("="*60)
        
        app.run(debug=True, port=5000, threaded=True, use_reloader=False)
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()
        sys.exit(1)
