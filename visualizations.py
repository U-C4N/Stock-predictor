import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def create_stock_price_plot(df, future_predictions=None, stock_symbol=""):
    """Create interactive stock price plot with multiple model predictions."""
    try:
        fig = go.Figure()
        
        # Add Bollinger Bands if available
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='Upper BB',
                    line=dict(color='#C8C8C8', dash='dash'),
                    opacity=0.5
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='Lower BB',
                    line=dict(color='#C8C8C8', dash='dash'),
                    fill='tonexty',
                    opacity=0.5
                )
            )
        
        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Actual Prices',
                line=dict(color='#0031B4')
            )
        )
        
        # Add future predictions if available
        if future_predictions is not None:
            if isinstance(future_predictions, dict):
                colors = {
                    'Linear Regression': '#FFA500',
                    'Random Forest': '#FF4B4B',
                    'XGBoost': '#00B4A1',
                    'SVR': '#9B59B6'
                }
                
                future_dates = pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=len(next(iter(future_predictions.values()))),
                    freq='D'
                )
                
                for model_name, predictions in future_predictions.items():
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=predictions,
                            name=f'{model_name} Prediction',
                            line=dict(color=colors.get(model_name, '#000000'))
                        )
                    )
            else:
                future_dates = pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=len(future_predictions),
                    freq='D'
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        name='Predicted Prices',
                        line=dict(color='#FFA500')
                    )
                )
        
        fig.update_layout(
            title=dict(
                text=f'{stock_symbol} Stock Price Prediction',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_stock_price_plot: {str(e)}")
        return go.Figure()

def create_technical_indicators_plot(df):
    """Create technical indicators plot."""
    try:
        # Verify data
        required_columns = ['Close', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD', 'MACD_Signal', 
                          'MACD_Hist', 'SO_K', 'SO_D', 'Volume', 'Volume_MA']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
            subplot_titles=(
                'Price with Bollinger Bands',
                'Stochastic Oscillator (%K/%D)',
                'Relative Strength Index (RSI)',
                'MACD',
                'Volume Analysis'
            )
        )

        # 1. Price and Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name='Price',
                      line=dict(color='#0031B4', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper BB',
                      line=dict(color='gray', dash='dash'), opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower BB',
                      line=dict(color='gray', dash='dash'),
                      fill='tonexty', opacity=0.3),
            row=1, col=1
        )

        # 2. Stochastic Oscillator
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SO_K'], name='%K',
                      line=dict(color='#0031B4')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SO_D'], name='%D',
                      line=dict(color='#FF4B4B')),
            row=2, col=1
        )
        
        # Add reference zones for Stochastic
        for y_val in [20, 80]:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                x1=df.index[-1],
                y0=y_val,
                y1=y_val,
                line=dict(color="gray", dash="dash", width=1),
                opacity=0.5,
                row=2,
                col=1
            )

        # 3. RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='#0031B4')),
            row=3, col=1
        )
        # Add reference lines for RSI
        for y_val in [30, 70]:
            fig.add_shape(
                type="line",
                x0=df.index[0],
                x1=df.index[-1],
                y0=y_val,
                y1=y_val,
                line=dict(color="gray", dash="dash", width=1),
                opacity=0.5,
                row=3,
                col=1
            )

        # 4. MACD
        colors = ['#00B4A1' if val >= 0 else '#FF4B4B' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Histogram',
                  marker_color=colors),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='#0031B4')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                      line=dict(color='#FF4B4B')),
            row=4, col=1
        )

        # 5. Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color='#0031B4', opacity=0.5),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_MA'], name='Volume MA',
                      line=dict(color='#FF4B4B')),
            row=5, col=1
        )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=50, l=50, r=50, b=50)
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="%K/%D", range=[0, 100], row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="MACD", row=4, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Volume", row=5, col=1, gridcolor='lightgray')

        # Update x-axes
        fig.update_xaxes(gridcolor='lightgray', showgrid=True)
        
        return fig
    except Exception as e:
        print(f"Error in create_technical_indicators_plot: {str(e)}")
        return go.Figure()

def create_model_performance_plot(y_test, predictions):
    """Create model performance visualization for multiple models."""
    try:
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_test))),
                y=y_test,
                name='Actual',
                line=dict(color='#0031B4')
            )
        )
        
        # Add predictions for each model
        colors = {
            'Linear Regression': '#FFA500',
            'Random Forest': '#FF4B4B',
            'XGBoost': '#00B4A1',
            'SVR': '#9B59B6'
        }
        
        if isinstance(predictions, dict):
            for model_name, pred in predictions.items():
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(y_test))),
                        y=pred,
                        name=model_name,
                        line=dict(color=colors.get(model_name, '#000000'))
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(y_test))),
                    y=predictions,
                    name='Predicted',
                    line=dict(color='#FF4B4B')
                )
            )
        
        fig.update_layout(
            title=dict(
                text='Model Performance Comparison',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Sample Index',
            yaxis_title='Price',
            template='plotly_white',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_model_performance_plot: {str(e)}")
        return go.Figure()

def create_model_comparison_metrics(metrics):
    """Create model comparison metrics visualization."""
    try:
        models = list(metrics.keys())
        train_rmse = [metrics[model]['train_rmse'] for model in models]
        test_rmse = [metrics[model]['test_rmse'] for model in models]
        r2_scores = [metrics[model]['r2_score'] for model in models]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Training RMSE', 'Testing RMSE', 'R² Score'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors = {
            'Linear Regression': '#FFA500',
            'Random Forest': '#FF4B4B',
            'XGBoost': '#00B4A1',
            'SVR': '#9B59B6'
        }
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=train_rmse,
                name='Training RMSE',
                marker_color=[colors.get(model, '#000000') for model in models]
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=test_rmse,
                name='Testing RMSE',
                marker_color=[colors.get(model, '#000000') for model in models]
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=r2_scores,
                name='R² Score',
                marker_color=[colors.get(model, '#000000') for model in models]
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            title=dict(
                text='Model Comparison Metrics',
                x=0.5,
                xanchor='center'
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_model_comparison_metrics: {str(e)}")
        return go.Figure()
