import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_data(symbol, period='1y'):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError("No data found for this symbol")
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def prepare_data_for_ml(df):
    """Prepare data for ML model."""
    try:
        df = df.copy()
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate BB first since other indicators depend on it
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        df['SO_K'], df['SO_D'] = calculate_stochastic_oscillator(df)
        
        # Average True Range
        df['ATR'] = calculate_atr(df)
        
        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Feature engineering
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Fill NaN values with method='bfill' to ensure no NaN values remain
        df = df.fillna(method='bfill')
        
        # Verify calculated columns
        print("\nVerifying calculated technical indicators:")
        print(f"Shape before dropping NaN: {df.shape}")
        print("\nColumn statistics:")
        print(df[['Close', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD']].describe())
        
        # Verify no NaN values remain
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            print(f"\nWarning: NaN values found in columns: {nan_columns}")
            df = df.fillna(method='ffill')  # Final forward fill if any NaNs remain
        
        return df
        
    except Exception as e:
        print(f"Error in prepare_data_for_ml: {str(e)}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic_oscillator(df, period=14):
    """Calculate Stochastic Oscillator."""
    try:
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        # Calculate %K
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (3-period moving average of %K)
        d = k.rolling(window=3).mean()
        
        return k, d
    except Exception as e:
        print(f"Error in calculate_stochastic_oscillator: {str(e)}")
        return pd.Series(index=df.index), pd.Series(index=df.index)

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    try:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    except Exception as e:
        print(f"Error in calculate_atr: {str(e)}")
        return pd.Series(index=df.index)

def get_company_info(symbol):
    """Get company information."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('forwardPE', 'N/A')
        }
    except:
        return {
            'name': symbol,
            'sector': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A'
        }
