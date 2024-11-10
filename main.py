import streamlit as st
import pandas as pd
import numpy as np
from data_handler import get_stock_data, prepare_data_for_ml, get_company_info
from model import ModelComparisonPredictor
from portfolio_optimizer import PortfolioOptimizer
from visualizations import (
    create_stock_price_plot,
    create_technical_indicators_plot,
    create_model_performance_plot,
    create_model_comparison_metrics
)
from utils import format_large_number, calculate_investment_metrics, generate_trading_signals
from ai_analyzer import get_stock_recommendation

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Configure Streamlit to handle larger plots
st.markdown("""
    <style>
        .stPlotlyChart {
            width: 100%;
            min-height: 400px;
        }
        .plot-container {
            margin-bottom: 2rem;
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
        .main > div {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        iframe {
            border: none !important;
        }
        .optimization-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="NVDA").upper()
period = st.sidebar.selectbox(
    "Select Time Period",
    options=["6mo", "1y", "2y"],
    index=0
)

# Language selector
language = st.sidebar.selectbox(
    "Select Analysis Language",
    options=["Turkish", "English", "German", "Russian", "French", "Spanish"],
    index=0
)

# Model Selection
selected_model = st.sidebar.selectbox(
    "Select Prediction Model",
    options=["All Models", "Linear Regression", "Random Forest", "XGBoost", "SVR"],
    index=0
)

# Risk Tolerance
risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Conservative", "Moderate", "Aggressive"],
    value="Moderate"
)

# Main content
st.title("Stock Price Prediction Dashboard")

try:
    # Load and prepare data
    df = get_stock_data(stock_symbol, period)
    prepared_df = prepare_data_for_ml(df)
    
    # Company information
    company_info = get_company_info(stock_symbol)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Company", company_info['name'])
    with col2:
        st.metric("Sector", company_info['sector'])
    with col3:
        st.metric("Market Cap", format_large_number(company_info['market_cap']))
    with col4:
        st.metric("P/E Ratio", company_info['pe_ratio'])
    
    # Train models and make predictions
    predictor = ModelComparisonPredictor()
    metrics, (X_test, y_test, test_predictions) = predictor.train(prepared_df)
    
    # Get future predictions
    if selected_model == "All Models":
        future_predictions = predictor.predict_future_all_models(prepared_df)
    else:
        future_predictions = {
            selected_model: predictor.predict_future(prepared_df, selected_model)
        }
    
    # Stock price plot with predictions
    st.subheader("Price History and Prediction")
    price_fig = create_stock_price_plot(df, future_predictions, stock_symbol)
    st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True})
    
    # Portfolio Optimization
    st.subheader("Portfolio Optimization")
    optimizer = PortfolioOptimizer(df)
    
    # Display Efficient Frontier
    ef_fig = optimizer.plot_efficient_frontier()
    st.plotly_chart(ef_fig, use_container_width=True)
    
    # Portfolio Optimization Suggestions
    suggestions = optimizer.get_optimization_suggestion()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Optimization Analysis")
        for suggestion in suggestions:
            with st.container():
                st.markdown(f"""
                <div class="optimization-card">
                    <h4>{suggestion['message']}</h4>
                    <div class="metric-row">
                        {suggestion.get('recommendation', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Risk-Return Profile")
        metrics = optimizer.calculate_metrics(np.array([1.0]))
        st.metric("Expected Annual Return", f"{metrics['return']*100:.1f}%")
        st.metric("Portfolio Volatility", f"{metrics['volatility']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    # Technical indicators
    st.subheader("Technical Indicators")
    with st.container():
        st.markdown("""
            <style>
                [data-testid="stVerticalBlock"] div:has(> [data-testid="stHorizontalBlock"]) {
                    gap: 0.5rem;
                }
            </style>
        """, unsafe_allow_html=True)
        tech_fig = create_technical_indicators_plot(prepared_df)
        st.plotly_chart(
            tech_fig,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
            }
        )
    
    # Model comparison metrics
    st.subheader("Model Comparison Metrics")
    metrics_fig = create_model_comparison_metrics(metrics)
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Model performance visualization
    st.subheader("Model Performance")
    if selected_model == "All Models":
        perf_predictions = test_predictions
    else:
        perf_predictions = {selected_model: test_predictions[selected_model]}
    
    perf_fig = create_model_performance_plot(y_test, perf_predictions)
    st.plotly_chart(perf_fig, use_container_width=True)
    
    # Investment Analysis
    st.subheader("Investment Analysis")
    
    # Calculate investment metrics
    inv_metrics = calculate_investment_metrics(df)
    
    # Display basic metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Monthly Returns", f"{inv_metrics['Monthly Returns']:.2%}")
    with col2:
        st.metric("Daily Returns", f"{inv_metrics['Daily Returns']:.2%}")
    with col3:
        st.metric("Volatility", f"{inv_metrics['Volatility']:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{inv_metrics['Sharpe Ratio']:.2f}")
    with col5:
        st.metric("Max Drawdown", f"{inv_metrics['Max Drawdown']:.2%}")
    
    # Display additional risk metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Value at Risk (95%)", f"{inv_metrics['Value at Risk (95%)']:.2%}")
    with col2:
        st.metric("Expected Shortfall", f"{inv_metrics['Expected Shortfall']:.2%}")
    with col3:
        st.metric("Sortino Ratio", f"{inv_metrics['Sortino Ratio']:.2f}")
    with col4:
        st.metric("Win Rate", f"{inv_metrics['Win Rate']:.2%}")
    with col5:
        st.metric("Risk-Adjusted Returns", f"{inv_metrics['Risk-Adjusted Returns']:.2f}")
    
    # AI Analysis and Recommendation
    st.subheader("AI Analysis and Recommendation")
    
    analysis_data = {
        'model_predictions': future_predictions,
        'technical_indicators': {
            'rsi': prepared_df['RSI'].iloc[-1],
            'macd': prepared_df['MACD'].iloc[-1],
            'stochastic': prepared_df['SO_K'].iloc[-1]
        },
        'metrics': inv_metrics
    }
    
    with st.spinner('Performing AI analysis...'):
        recommendation = get_stock_recommendation(analysis_data, stock_symbol, language)
        st.markdown(recommendation)
    
    # Trading signals
    st.subheader("Trading Signals")
    signals = generate_trading_signals(prepared_df)
    for signal in signals:
        st.info(signal)
    
    # Download button for data
    st.download_button(
        label="Download Historical Data",
        data=df.to_csv().encode('utf-8'),
        file_name=f"{stock_symbol}_historical_data.csv",
        mime='text/csv'
    )

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please enter a valid stock symbol and try again.")
