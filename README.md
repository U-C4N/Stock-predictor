# Stock Market Analysis & Prediction Platform 📈

## Overview
An advanced stock market analysis platform that combines historical data visualization with ML-based price predictions and investment analytics. The platform integrates multiple prediction models, technical indicators, and AI-powered recommendations.

## Features 🚀
- Multi-model ML price forecasting (Linear Regression, Random Forest, XGBoost, SVR)
- Interactive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Investment metrics analysis (Sharpe Ratio, Beta, Alpha)
- AI-powered trading recommendations in multiple languages
- Automated trading signals
- Real-time model performance comparison

## Installation 🛠️
```bash
# Clone the repository
git clone https://github.com/U-C4N/Stock-predictor.git
cd stock-market-analysis

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
Create .env file with:
GEMINI_API_KEY=your_gemini_api_key
```

## Usage 💡
1. Run the application:
```bash
streamlit run main.py
```
2. Enter stock symbol (e.g., NVDA, AAPL)
3. Select time period and model
4. View analysis and predictions

## Dependencies 📚
- Python 3.11+
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Plotly
- YFinance
- Google Gemini AI

## Technical Details 🔧
- Data processing: data_handler.py
- ML models: model.py
- Visualization: visualizations.py
- Investment analytics: utils.py
- AI analysis: ai_analyzer.py

