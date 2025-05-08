import threading
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import io
import base64
from flask import Flask, render_template, request
import time

app = Flask(__name__)

# List of predefined stock tickers for recommendations
stock_tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'NFLX', 'NVDA', 'META', 'AMD', 'INTC', 'BA', 'JPM', 'XOM', 'V', 'DIS', 'PYPL', 'CSCO', 'IBM', 'GS', 'UBER']

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate SMA Crossover
def calculate_sma(data, short_window=10, long_window=50):
    data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Avoid ambiguous Series comparison
    data['Buy_Signal'] = (data['SMA_Short'] > data['SMA_Long']).astype(int)
    data['Sell_Signal'] = (data['SMA_Short'] < data['SMA_Long']).astype(int)
    
    return data



# Function to fetch data from Yahoo Finance
def fetch_data(ticker):
    data = yf.download(ticker, period='1yr', interval='1d')
    return data

# Function to generate recommendation
def generate_recommendation(data):
    try:
        latest = data.iloc[-1]
        rsi = latest['RSI']
        macd = latest['MACD']
        signal = latest['Signal_Line']
        sma10 = latest['SMA_10']
        sma50 = latest['SMA_50']
    except Exception as e:
        return "Error", f"Error accessing indicators: {e}"

    explanation = []

    if rsi < 30:
        explanation.append("RSI is low (under 30), indicating the stock may be oversold.")
    elif rsi > 70:
        explanation.append("RSI is high (over 70), suggesting the stock might be overbought.")

    if macd > signal:
        explanation.append("MACD is above the signal line — bullish signal.")
    elif macd < signal:
        explanation.append("MACD is below the signal line — bearish signal.")

    if sma10 > sma50:
        explanation.append("Short-term SMA is above long-term SMA — possible upward trend.")
    elif sma10 < sma50:
        explanation.append("Short-term SMA is below long-term SMA — possible downward trend.")

    # Final recommendation logic
    if rsi < 30 and macd > signal and sma10 > sma50:
        reco = "Strong Buy"
    elif rsi > 70 and macd < signal and sma10 < sma50:
        reco = "Strong Sell"
    elif macd > signal:
        reco = "Buy"
    elif macd < signal:
        reco = "Sell"
    else:
        reco = "Hold"

    return reco, " ".join(explanation)


# Function to generate a chart for each stock
def generate_chart(data, ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='blue')
    ax.set_title(f'{ticker} Stock Price & Indicators')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return base64_img



# Periodic task to refresh data
def refresh_data_periodically():
    while True:
        print("Refreshing stock data...")
        time.sleep(1800)



#backtesting page 

# Backtest function
def backtest(data, strategy, buy_threshold=40, sell_threshold=60, initial_cash=1000):
    cash = initial_cash
    shares = 0
    portfolio_values = []
    buy_sell_signals = []

    if strategy == "RSI":
        data = calculate_rsi(data)
        for i in range(1, len(data)):
            close_price = data['Close'].iloc[i]
            rsi = data['RSI'].iloc[i]  # ✅ FIXED: Use iloc[i]

            buy_condition = (rsi < buy_threshold) and (shares == 0)
            sell_condition = (rsi > sell_threshold) and (shares > 0)

            if buy_condition:
                shares = cash // close_price
                cash -= shares * close_price
                buy_sell_signals.append(('Buy', i))
            elif sell_condition:
                cash += shares * close_price
                shares = 0
                buy_sell_signals.append(('Sell', i))

            portfolio_value = cash + (shares * close_price)
            portfolio_values.append(portfolio_value)

    elif strategy == "MACD":
        data = calculate_macd(data)
        for i in range(1, len(data)):
            close_price = data['Close'].iloc[i]
            macd = data['MACD'].iloc[i]  # ✅ FIXED
            signal_line = data['Signal_Line'].iloc[i]

            buy_condition = (macd > signal_line) and (shares == 0)
            sell_condition = (macd < signal_line) and (shares > 0)

            if buy_condition:
                shares = cash // close_price
                cash -= shares * close_price
                buy_sell_signals.append(('Buy', i))
            elif sell_condition:
                cash += shares * close_price
                shares = 0
                buy_sell_signals.append(('Sell', i))

            portfolio_value = cash + (shares * close_price)
            portfolio_values.append(portfolio_value)

    elif strategy == "SMA Crossover":
        data = calculate_sma(data)
        for i in range(1, len(data)):
            close_price = data['Close'].iloc[i]
            sma_short = data['SMA_Short'].iloc[i]  # ✅ FIXED
            sma_long = data['SMA_Long'].iloc[i]

            buy_condition = (sma_short > sma_long) and (shares == 0)
            sell_condition = (sma_short < sma_long) and (shares > 0)

            if buy_condition:
                shares = cash // close_price
                cash -= shares * close_price
                buy_sell_signals.append(('Buy', i))
            elif sell_condition:
                cash += shares * close_price
                shares = 0
                buy_sell_signals.append(('Sell', i))

            portfolio_value = cash + (shares * close_price)
            portfolio_values.append(portfolio_value)

    final_value = cash + (shares * data['Close'].iloc[-1])
    profit = final_value - initial_cash
    
    return {
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "buy_sell_signals": buy_sell_signals,
        "portfolio_values": portfolio_values,
    }

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to fetch data from Yahoo Finance
def fetch_data(ticker):
    data = yf.download(ticker, period='6mo', interval='1d')

    if data.empty or len(data) < 50:
        return None  # prevent indexing errors

    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data = calculate_rsi(data)
    data = calculate_macd(data)

    data.dropna(inplace=True)  # Drop rows with NaN indicators
    return data



# Plotting function
def plot_backtest(data, portfolio_values, buy_sell_signals):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    for signal, index in buy_sell_signals:
        color = 'green' if signal == 'Buy' else 'red'
        ax.scatter(data.index[index], data['Close'].iloc[index], color=color, label=signal, zorder=5)
    
    ax2 = ax.twinx()
    ax2.plot(data.index[:len(portfolio_values)], portfolio_values, label='Portfolio Value', color='orange', linestyle='--')
    
    ax.set_title('Backtest Results')
    ax.set_ylabel('Price')
    ax2.set_ylabel('Portfolio Value')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return base64_img



thread = threading.Thread(target=refresh_data_periodically)
thread.daemon = True
thread.start()


#suggestions page  
# AI-style analysis function
def analyze_stock_with_ai(ticker, data):
    recommendation = generate_recommendation(data)
    latest_rsi = data['RSI'].iloc[-1]
    latest_macd = data['MACD'].iloc[-1]
    signal = data['Signal_Line'].iloc[-1]
    sma10 = data['SMA_10'].iloc[-1]
    sma50 = data['SMA_50'].iloc[-1]
    close_price = data['Close'].iloc[-1]

    explanation = f"{ticker} is currently trading at ${close_price:.2f}. "

    if latest_rsi < 30:
        explanation += f"The RSI is {latest_rsi:.2f}, indicating it may be oversold. "
    elif latest_rsi > 70:
        explanation += f"The RSI is {latest_rsi:.2f}, indicating it may be overbought. "
    else:
        explanation += f"The RSI is {latest_rsi:.2f}, which is in a neutral range. "

    explanation += f"The MACD is {latest_macd:.2f} vs signal line {signal:.2f}. "
    explanation += f"SMA(10) is {sma10:.2f} and SMA(50) is {sma50:.2f}. "

    if recommendation == "Strong Buy":
        explanation += "All indicators align with a strong buying opportunity."
    elif recommendation == "Strong Sell":
        explanation += "All indicators point to a potential strong sell-off."
    else:
        explanation += f"Overall, the stock is rated as '{recommendation}'."

    return {
        "Recommendation": recommendation,
        "Explanation": explanation,
        "RSI": round(latest_rsi, 2),
        "MACD": round(latest_macd, 2),
        "Signal_Line": round(signal, 2),
        "SMA_10": round(sma10, 2),
        "SMA_50": round(sma50, 2),
        "Close": round(close_price, 2),
    }


#routes

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/stock-suggestions", methods=["GET", "POST"])
def stock_suggestions():
    analysis = None
    chart_img = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        question = request.form.get("question", "").strip()

        if ticker:
            data = fetch_data(ticker)
            chart_img = generate_chart(data, ticker)
            analysis = analyze_stock_with_ai(ticker, data)

    return render_template("stock_suggestions.html", analysis=analysis, chart_img=chart_img)


@app.route("/backtesting", methods=["GET", "POST"])
def backtest_page():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
    strategies = ["RSI", "MACD", "SMA Crossover"]
    results = []
    plot_imgs = []

    if request.method == "POST":
        selected_tickers = request.form.getlist("tickers")
        strategy = request.form.get("strategy", "RSI")
        buy_threshold = int(request.form.get("buy_threshold", 40))
        sell_threshold = int(request.form.get("sell_threshold", 60))
        initial_cash = int(request.form.get("initial_cash", 1000))
        
    for ticker in selected_tickers:
     data = fetch_data(ticker)
     if data is None:
            print(f"Skipping {ticker}: Not enough valid data.")
            continue  # skip this ticker

     backtest_results = backtest(data, strategy, buy_threshold, sell_threshold, initial_cash)
     plot_img = plot_backtest(data, backtest_results['portfolio_values'], backtest_results['buy_sell_signals'])
     results.append({
        "ticker": ticker,
        "final_value": backtest_results["final_value"],
        "profit": backtest_results["profit"],
        "plot_img": plot_img
     })

    return render_template("backtesting.html", tickers=tickers, strategies=strategies, results=results)

@app.route("/portfolio")
def portfolio_page():
    portfolio = [
        {"ticker": "AAPL", "shares": 10, "avg_price": 150.00},
        {"ticker": "TSLA", "shares": 5, "avg_price": 700.00},
        {"ticker": "AMZN", "shares": 2, "avg_price": 3300.00}
    ]

    total_value = 0
    total_cost = 0

    for stock in portfolio:
        data = yf.download(stock["ticker"], period="1d")
        latest_price = data["Close"].iloc[-1] if not data.empty else 0
        stock["latest_price"] = round(latest_price, 2)
        stock["current_value"] = round(latest_price * stock["shares"], 2)
        stock["cost_basis"] = round(stock["avg_price"] * stock["shares"], 2)
        stock["profit_loss"] = round(stock["current_value"] - stock["cost_basis"], 2)

        total_value += stock["current_value"]
        total_cost += stock["cost_basis"]

    total_profit_loss = round(total_value - total_cost, 2)

    return render_template("portfolio.html", portfolio=portfolio, total_value=round(total_value, 2), total_profit_loss=total_profit_loss)


if __name__ == "__main__":
    app.run(debug=True)
