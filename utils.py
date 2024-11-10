import pandas as pd
import numpy as np

def format_large_number(number):
    """Format large numbers for display."""
    if not isinstance(number, (int, float)) or pd.isna(number):
        return 'N/A'
    
    if number >= 1e12:
        return f'${number/1e12:.2f}T'
    elif number >= 1e9:
        return f'${number/1e9:.2f}B'
    elif number >= 1e6:
        return f'${number/1e6:.2f}M'
    else:
        return f'${number:,.2f}'

def calculate_beta(returns, market_returns):
    """Calculate beta coefficient."""
    covariance = np.cov(returns.dropna(), market_returns.dropna())[0][1]
    market_variance = np.var(market_returns.dropna())
    return covariance / market_variance if market_variance != 0 else 0

def calculate_alpha(returns, market_returns):
    """Calculate alpha (Jensen's alpha)."""
    beta = calculate_beta(returns, market_returns)
    return (returns.mean() - 0.02/252) - beta * (market_returns.mean() - 0.02/252)

def calculate_information_ratio(returns, market_returns):
    """Calculate information ratio."""
    active_returns = returns - market_returns
    return active_returns.mean() / active_returns.std() if active_returns.std() != 0 else 0

def calculate_investment_metrics(df, market_data=None):
    """Calculate investment metrics."""
    returns = df['Close'].pct_change()
    monthly_returns = df['Close'].resample('ME').last().pct_change()
    
    metrics = {
        'Monthly Returns': monthly_returns.mean(),
        'Daily Returns': returns.mean(),
        'Volatility': returns.std(),
        'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252),
        'Max Drawdown': ((df['Close'] - df['Close'].expanding().max()) / df['Close'].expanding().max()).min(),
        'Value at Risk (95%)': returns.quantile(0.05),
        'Expected Shortfall': returns[returns <= returns.quantile(0.05)].mean(),
        'Sortino Ratio': (returns.mean() / returns[returns < 0].std()) * np.sqrt(252),
        'Win Rate': len(returns[returns > 0]) / len(returns[returns != 0]),
        'Risk-Adjusted Returns': returns.mean() / returns.std(),
    }
    
    if market_data is not None:
        market_returns = market_data['Close'].pct_change()
        beta = calculate_beta(returns, market_returns)
        alpha = calculate_alpha(returns, market_returns)
        metrics.update({
            'Beta': beta,
            'Alpha': alpha,
            'Information Ratio': calculate_information_ratio(returns, market_returns)
        })
        
    return metrics

def generate_trading_signals(df):
    """Generate basic trading signals."""
    signals = []
    
    # Simple moving average crossover
    if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1] and \
       df['SMA_5'].iloc[-2] <= df['SMA_20'].iloc[-2]:
        signals.append('Buy: Short-term MA crossed above long-term MA')
    
    elif df['SMA_5'].iloc[-1] < df['SMA_20'].iloc[-1] and \
         df['SMA_5'].iloc[-2] >= df['SMA_20'].iloc[-2]:
        signals.append('Sell: Short-term MA crossed below long-term MA')
    
    # RSI signals
    if df['RSI'].iloc[-1] < 30:
        signals.append('Buy: RSI indicates oversold conditions')
    elif df['RSI'].iloc[-1] > 70:
        signals.append('Sell: RSI indicates overbought conditions')
    
    # MACD signals
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and \
       df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
        signals.append('Buy: MACD crossed above signal line')
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and \
         df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
        signals.append('Sell: MACD crossed below signal line')
    
    # Stochastic signals
    if df['SO_K'].iloc[-1] < 20 and df['SO_D'].iloc[-1] < 20:
        signals.append('Buy: Stochastic indicates oversold conditions')
    elif df['SO_K'].iloc[-1] > 80 and df['SO_D'].iloc[-1] > 80:
        signals.append('Sell: Stochastic indicates overbought conditions')
    
    return signals if signals else ['No clear signals at the moment']
