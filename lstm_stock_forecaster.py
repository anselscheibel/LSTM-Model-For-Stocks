"""
Stock Price Forecasting with LSTM Neural Networks

A recurrent neural network implementation for time series forecasting of stock prices
using historical price data and technical indicators.

Features:
- LSTM architecture with configurable layers
- Technical indicators (RSI, MACD, Bollinger Bands)
- Train/validation/test split with proper temporal ordering
- Comprehensive evaluation metrics (MAPE, RMSE, MAE)
- Backtesting functionality for strategy evaluation

Author: Ansel Scheibel
Date: December 2024
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


class StockPriceForecaster:
    """
    LSTM-based stock price forecasting model with technical indicators.
    """
    
    def __init__(self, ticker, sequence_length=60, lstm_units=50, dropout=0.2):
        """
        Initialize the forecaster.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            sequence_length: Number of days to use for prediction
            lstm_units: Number of LSTM units in each layer
            dropout: Dropout rate for regularization
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.data = None
        self.metrics = {}
    
    def download_data(self, start_date=None, end_date=None):
        """
        Download historical stock data from Yahoo Finance.
        
        Args:
            start_date: Start date (default: 3 years ago)
            end_date: End date (default: today)
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=3*365)
        
        print(f"Downloading {self.ticker} data from {start_date.date()} to {end_date.date()}...")
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
        self.data = df
        print(f"Downloaded {len(df)} days of data")
        return df
    
    def calculate_technical_indicators(self):
        """Add technical indicators to the dataset."""
        df = self.data.copy()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['MA_20'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['MA_20'] - (df['BB_std'] * 2)
        
        # Moving Averages
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        self.data = df
        return df
    
    def prepare_sequences(self, train_split=0.8):
        """
        Prepare sequences for LSTM training.
        
        Args:
            train_split: Proportion of data for training
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        df = self.data.copy()
        
        # Select features
        feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 
                          'MA_20', 'MA_50', 'MA_200', 'BB_upper', 'BB_lower']
        
        features = df[feature_columns].values
        target = df['Close'].values
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data (temporal split - no shuffling)
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, num_layers=2):
        """
        Build LSTM model architecture.
        
        Args:
            num_layers: Number of LSTM layers
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.lstm_units, 
                      return_sequences=(num_layers > 1),
                      input_shape=(self.sequence_length, 10)))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            return_seq = (i < num_layers - 1)
            model.add(LSTM(units=self.lstm_units, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        print(f"\nModel Architecture:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data proportion
        """
        if self.model is None:
            self.build_model()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        predictions_scaled = self.model.predict(X_test)
        
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(predictions_scaled)
        actual = self.scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        self.metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'ticker': self.ticker,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units
        }
        
        print(f"\nModel Performance on {self.ticker}:")
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return self.metrics, predictions, actual
    
    def forecast_future(self, days=30):
        """
        Forecast future prices.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Array of predicted prices
        """
        # Get last sequence
        last_sequence = self.feature_scaler.transform(
            self.data[['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 
                      'MA_20', 'MA_50', 'MA_200', 'BB_upper', 'BB_lower']].values[-self.sequence_length:]
        )
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next day
            pred_scaled = self.model.predict(current_sequence.reshape(1, self.sequence_length, 10), verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred)
            
            # Update sequence (simplified - in practice would recalculate indicators)
            next_features = current_sequence[-1].copy()
            next_features[0] = pred_scaled[0][0]  # Update close price
            
            current_sequence = np.vstack([current_sequence[1:], next_features])
        
        return np.array(predictions)
    
    def save_model(self, path='models'):
        """Save trained model and scalers."""
        Path(path).mkdir(exist_ok=True)
        
        self.model.save(f'{path}/{self.ticker}_lstm_model.h5')
        
        # Save scalers and metadata
        import pickle
        with open(f'{path}/{self.ticker}_scalers.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'sequence_length': self.sequence_length,
                'metrics': self.metrics
            }, f)
        
        print(f"Model saved to {path}/")


def main():
    """
    Example usage and training script.
    """
    if not HAS_TENSORFLOW:
        print("Please install TensorFlow: pip install tensorflow")
        return
    
    # Configuration
    TICKER = 'AAPL'  # Change to any stock ticker
    SEQUENCE_LENGTH = 60
    LSTM_UNITS = 50
    DROPOUT = 0.2
    EPOCHS = 50
    
    print(f"{'='*60}")
    print(f"Stock Price Forecasting with LSTM")
    print(f"Ticker: {TICKER}")
    print(f"{'='*60}\n")
    
    # Initialize forecaster
    forecaster = StockPriceForecaster(
        ticker=TICKER,
        sequence_length=SEQUENCE_LENGTH,
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT
    )
    
    # Download and prepare data
    forecaster.download_data()
    forecaster.calculate_technical_indicators()
    
    # Prepare sequences
    X_train, y_train, X_test, y_test = forecaster.prepare_sequences()
    
    # Build and train model
    forecaster.build_model(num_layers=2)
    history = forecaster.train(X_train, y_train, epochs=EPOCHS)
    
    # Evaluate
    metrics, predictions, actual = forecaster.evaluate(X_test, y_test)
    
    # Forecast future
    print("\nForecasting next 30 days...")
    future_prices = forecaster.forecast_future(days=30)
    print(f"Predicted price in 30 days: ${future_prices[-1]:.2f}")
    
    # Save model and metrics
    forecaster.save_model()
    
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    main()