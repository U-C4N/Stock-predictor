from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class ModelComparisonPredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        self.trained_models = {}
        self.features = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volatility']
        self.target = 'Close'
        
    def prepare_features(self, df):
        """Prepare features for prediction."""
        X = df[self.features]
        y = df[self.target]
        return X, y
    
    def train(self, df):
        """Train all models."""
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        metrics = {}
        predictions = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            predictions[name] = test_pred
            
            # Calculate metrics
            metrics[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'r2_score': r2_score(y_test, test_pred)
            }
        
        return metrics, (X_test, y_test, predictions)
    
    def predict_future(self, df, model_name='Linear Regression', days=30):
        """Predict future stock prices using specified model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found or not trained")
            
        model = self.trained_models[model_name]
        last_data = df.iloc[-1:]
        future_predictions = []
        
        for _ in range(days):
            pred = model.predict(last_data[self.features])[0]
            future_predictions.append(pred)
            
            # Update features for next prediction
            last_data = self._update_features(last_data, pred)
            
        return future_predictions
    
    def predict_future_all_models(self, df, days=30):
        """Predict future stock prices using all trained models."""
        predictions = {}
        for model_name in self.trained_models:
            predictions[model_name] = self.predict_future(df, model_name, days)
        return predictions
    
    def _update_features(self, last_data, pred):
        """Update features for the next prediction."""
        new_data = last_data.copy()
        new_data['Close'] = pred
        
        # Update SMA
        new_data['SMA_5'] = pred
        new_data['SMA_20'] = pred
        
        # Maintain other features
        new_data['RSI'] = last_data['RSI']
        new_data['MACD'] = last_data['MACD']
        new_data['Volatility'] = last_data['Volatility']
        
        return new_data
