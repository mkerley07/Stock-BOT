import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# Function to fetch and process data
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='3mo', interval='1d')
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data = calculate_rsi(data)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to plot data
def plot_data(data, ticker):
    if data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot price and moving averages
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax1.plot(data.index, data['SMA_10'], label='SMA 10', color='orange')
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', color='green')
        ax1.set_title(f'{ticker} Price and Moving Averages')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()

        # Plot RSI
        ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax2.set_title(f'{ticker} RSI')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI Value')
        ax2.legend()
        ax2.grid()

        # Show plot
        plt.tight_layout()
        plt.show()
    else:
        print("No data to plot.")

# Main function
if __name__ == "__main__":
    ticker = input("Enter the ticker symbol (e.g., AAPL, TSLA, BTC-USD): ").upper()
    market_data = fetch_data(ticker)
    
    if market_data is not None:
        print(market_data.tail())  # Show last few rows
        plot_data(market_data, ticker)  # Plot the data
