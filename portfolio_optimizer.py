import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, stock_data, risk_free_rate=0.02):
        """Initialize PortfolioOptimizer with stock data and risk-free rate."""
        self.stock_data = stock_data
        self.returns = stock_data['Close'].pct_change().dropna()
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def calculate_metrics(self, weights):
        """Calculate portfolio metrics."""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate * 252) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio(self, objective='sharpe', target_return=None):
        """
        Optimize portfolio based on objective for single-stock scenario.
        
        Args:
            objective (str): 'sharpe' for maximum Sharpe ratio, 'min_risk' for minimum volatility
            target_return (float, optional): Target return for minimum risk portfolio
        """
        num_assets = 1  # Single stock portfolio
        
        def objective_function(weights):
            if objective == 'sharpe':
                return -self.calculate_metrics(weights)['sharpe_ratio']
            elif objective == 'min_risk':
                return self.calculate_metrics(weights)['volatility']
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.calculate_metrics(x)['return'] - target_return
            })
        
        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        initial_weights = np.array([1.0])
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def generate_efficient_frontier(self, num_portfolios=100):
        """Generate efficient frontier points for single stock."""
        returns_range = np.linspace(
            max(0, self.returns.mean() * 252 * 0.5),
            self.returns.mean() * 252 * 1.5,
            num_portfolios
        )
        
        efficient_portfolios = []
        for target_return in returns_range:
            weights = self.optimize_portfolio('min_risk', target_return)
            metrics = self.calculate_metrics(weights)
            efficient_portfolios.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'weights': weights
            })
        
        return pd.DataFrame(efficient_portfolios)
    
    def plot_efficient_frontier(self):
        """Create efficient frontier plot."""
        # Calculate current portfolio metrics
        current_metrics = self.calculate_metrics(np.array([1.0]))
        
        # Generate efficient frontier points
        returns_range = np.linspace(
            max(0, self.returns.mean() * 252 * 0.5),
            self.returns.mean() * 252 * 1.5,
            100
        )
        volatilities = []
        for ret in returns_range:
            vol = np.sqrt(self.returns.var() * 252)  # Simplified for single stock
            volatilities.append(vol)
        
        # Create the plot
        fig = go.Figure()
        
        # Efficient Frontier line
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns_range,
            mode='lines',
            name='Risk-Return Profile',
            line=dict(color='#0031B4', width=2)
        ))
        
        # Current Portfolio point
        fig.add_trace(go.Scatter(
            x=[current_metrics['volatility']],
            y=[current_metrics['return']],
            mode='markers',
            name='Current Position',
            marker=dict(color='#FF4B4B', size=12, symbol='circle')
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Expected Volatility',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def get_optimization_suggestion(self):
        """Generate portfolio optimization suggestions."""
        current_metrics = self.calculate_metrics(np.array([1.0]))
        
        suggestions = []
        
        # Risk assessment
        volatility = current_metrics['volatility']
        risk_level = 'High' if volatility > 0.4 else 'Medium' if volatility > 0.25 else 'Low'
        
        suggestions.append({
            'message': f'Risk Assessment',
            'recommendation': f"""Current volatility ({volatility*100:.1f}%) indicates {risk_level.lower()} risk level. 
            {'Consider hedging strategies to reduce risk exposure.' if risk_level == 'High' else
             'Risk level is balanced.' if risk_level == 'Medium' else
             'Conservative risk profile maintained.'}\n"""
        })
        
        # Return potential
        annual_return = current_metrics['return']
        suggestions.append({
            'message': f'Return Analysis',
            'recommendation': f"""Expected annual return: {annual_return*100:.1f}%
            {' (Strong return potential)' if annual_return > 0.15 else
             ' (Moderate return outlook)' if annual_return > 0.08 else
             ' (Conservative return profile)'}\n"""
        })
        
        # Sharpe ratio analysis
        sharpe = current_metrics['sharpe_ratio']
        suggestions.append({
            'message': f'Risk-Adjusted Performance',
            'recommendation': f"""Sharpe Ratio: {sharpe:.2f}
            {'Excellent risk-adjusted returns' if sharpe > 1.5 else
             'Good risk-return balance' if sharpe > 1.0 else
             'Consider portfolio adjustments to improve risk-adjusted returns'}\n"""
        })
        
        return suggestions