#!/usr/bin/env python3
"""
Real-Time Stock Price Prediction System
======================================

This script provides real-time stock price prediction using live market data.
Features:
- Live data fetching from Yahoo Finance API
- Real-time data preprocessing and feature engineering
- LSTM and Random Forest models for prediction
- Continuous model updates and predictions
- Real-time visualization with Plotly
- Model persistence and prediction storage
- Buy/Sell recommendation system
"""

import os
import time
import json
import pickle
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    import tensorflow as tf  # Optional for deployment; app can run without TF
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealTimeStockPredictor:
    """
    Main class for real-time stock price prediction system
    """
    
    def __init__(self, symbol: str = "AAPL", prediction_horizon: int = 5):
        """
        Initialize the stock predictor
        
        Args:
            symbol: Stock symbol (default: AAPL)
            prediction_horizon: Number of days to predict ahead (default: 5)
        """
        self.symbol = symbol.upper()
        self.prediction_horizon = prediction_horizon
        self.data = pd.DataFrame()
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.rf_model = None
        self.feature_columns = []
        self.target_column = 'Close'
        self.sequence_length = 60  # Number of time steps for LSTM
        self.predictions_history = []
        self.last_update = None
        
        # Create directories for storing models and data
        self.create_directories()
        
        # Initialize models
        self.initialize_models()
        
        print(f"Initialized Real-Time Stock Predictor for {self.symbol}")
    
    def create_directories(self):
        """Create necessary directories for storing models and data"""
        directories = ['models', 'data', 'predictions', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def fetch_live_data(self) -> pd.DataFrame:
        """
        Fetch live market data from Yahoo Finance
        
        Returns:
            DataFrame with live market data
        """
        try:
            # Fetch live data for the last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                raise ValueError(f"No data received for {self.symbol}")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            
            print(f"Fetched {len(data)} records for {self.symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer technical indicators and features
        
        Args:
            data: Raw market data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            
            # Basic price features
            df['Price_Change'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            
            # Volume features
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Moving averages
            for window in [5, 10, 20, 50, 200]:
                df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
            
            # Technical indicators using ta library
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
            
            # Stochastic oscillator
            df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
            df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
            
            # ATR (Average True Range)
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Williams %R
            df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
            
            # Momentum indicators
            df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
            df['TSI'] = ta.momentum.TSIIndicator(df['Close']).tsi()
            
            # Volatility indicators
            df['NATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
            # Time-based features
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Year'] = df['Date'].dt.year
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
                df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Define feature columns (exclude Date and target column)
            self.feature_columns = [col for col in df.columns if col not in ['Date', self.target_column]]
            
            print(f"Engineered {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            print(f"Error engineering features: {str(e)}")
            return data
    
    def prepare_lstm_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model
        
        Args:
            data: DataFrame with features
            
        Returns:
            Tuple of (X, y) arrays for LSTM
        """
        try:
            # Select features and target
            features = data[self.feature_columns].values
            target = data[self.target_column].values
            
            # Scale the data
            features_scaled = self.scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(self.sequence_length, len(features_scaled)):
                X.append(features_scaled[i-self.sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing LSTM data: {str(e)}")
            return np.array([]), np.array([])
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is not available in this environment")

        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            LSTM(units=100, return_sequences=False),
            Dropout(0.2),
            Dense(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model
    
    def train_models(self, data: pd.DataFrame):
        """
        Train both LSTM and Random Forest models
        
        Args:
            data: DataFrame with engineered features
        """
        try:
            print("Training models...")
            
            # Prepare data for Random Forest
            X_rf = data[self.feature_columns].values
            y_rf = data[self.target_column].values
            
            # Train Random Forest
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_rf, y_rf)
            
            # Prepare and optionally train LSTM (if TensorFlow is available)
            if TENSORFLOW_AVAILABLE:
                X_lstm, y_lstm = self.prepare_lstm_data(data)
                if len(X_lstm) > 0:
                    # Build and train LSTM
                    self.lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
                    # Split data for training
                    split_idx = int(0.8 * len(X_lstm))
                    X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
                    y_train, y_val = y_lstm[:split_idx], y_lstm[split_idx:]
                    # Train LSTM
                    self.lstm_model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        verbose=1
                    )
            
            # Save models
            self.save_models()
            
            print("Models trained and saved successfully")
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
    
    def predict_next_days(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Make predictions for the next few days
        
        Args:
            data: Latest market data
            
        Returns:
            Dictionary with predictions from both models
        """
        try:
            predictions = {
                'lstm': [],
                'random_forest': [],
                'ensemble': []
            }
            
            # Get the latest data for prediction
            latest_features = data[self.feature_columns].iloc[-1:].values
            
            # Random Forest prediction
            if self.rf_model is not None:
                rf_pred = self.rf_model.predict(latest_features)[0]
                predictions['random_forest'].append(rf_pred)
            
            # LSTM prediction (only if TensorFlow available and model exists)
            if TENSORFLOW_AVAILABLE and self.lstm_model is not None and len(data) >= self.sequence_length:
                # Prepare sequence for LSTM
                latest_sequence = data[self.feature_columns].iloc[-self.sequence_length:].values
                latest_sequence_scaled = self.scaler.transform(latest_sequence)
                lstm_input = latest_sequence_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
                
                lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
                predictions['lstm'].append(lstm_pred)
            
            # Ensemble prediction (average of both models)
            if predictions['lstm'] and predictions['random_forest']:
                ensemble_pred = (predictions['lstm'][0] + predictions['random_forest'][0]) / 2
                predictions['ensemble'].append(ensemble_pred)
            
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return {'lstm': [], 'random_forest': [], 'ensemble': []}

    def generate_trading_signals(self, data: pd.DataFrame, predictions: Dict[str, List[float]]) -> Dict[str, any]:
        """
        Generate comprehensive buy/sell trading signals based on technical indicators and predictions
        
        Args:
            data: Latest market data with technical indicators
            predictions: Model predictions
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        try:
            if data.empty:
                return {}
            
            current_price = data['Close'].iloc[-1]
            signals = {
                'current_price': current_price,
                'signals': {},
                'overall_recommendation': 'HOLD',
                'confidence': 0.0,
                'reasons': []
            }
            
            # 1. RSI Signal
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if rsi < 30:
                    signals['signals']['rsi'] = 'BUY'
                    signals['reasons'].append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    signals['signals']['rsi'] = 'SELL'
                    signals['reasons'].append(f"RSI overbought ({rsi:.1f})")
                else:
                    signals['signals']['rsi'] = 'NEUTRAL'
            
            # 2. MACD Signal
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_Signal'].iloc[-1]
                if macd > macd_signal:
                    signals['signals']['macd'] = 'BUY'
                    signals['reasons'].append("MACD bullish crossover")
                elif macd < macd_signal:
                    signals['signals']['macd'] = 'SELL'
                    signals['reasons'].append("MACD bearish crossover")
                else:
                    signals['signals']['macd'] = 'NEUTRAL'
            
            # 3. Bollinger Bands Signal
            if 'BB_Position' in data.columns:
                bb_pos = data['BB_Position'].iloc[-1]
                if bb_pos < 0.2:
                    signals['signals']['bollinger'] = 'BUY'
                    signals['reasons'].append("Price near lower Bollinger Band")
                elif bb_pos > 0.8:
                    signals['signals']['bollinger'] = 'SELL'
                    signals['reasons'].append("Price near upper Bollinger Band")
                else:
                    signals['signals']['bollinger'] = 'NEUTRAL'
            
            # 4. Moving Average Signals
            ma_signals = []
            for window in [20, 50, 200]:
                if f'MA_{window}' in data.columns:
                    ma = data[f'MA_{window}'].iloc[-1]
                    if current_price > ma:
                        ma_signals.append('BUY')
                    else:
                        ma_signals.append('SELL')
            
            if ma_signals.count('BUY') > ma_signals.count('SELL'):
                signals['signals']['moving_averages'] = 'BUY'
                signals['reasons'].append("Price above major moving averages")
            else:
                signals['signals']['moving_averages'] = 'SELL'
                signals['reasons'].append("Price below major moving averages")
            
            # 5. Stochastic Oscillator Signal
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                stoch_k = data['Stoch_K'].iloc[-1]
                stoch_d = data['Stoch_D'].iloc[-1]
                if stoch_k < 20 and stoch_d < 20:
                    signals['signals']['stochastic'] = 'BUY'
                    signals['reasons'].append("Stochastic oversold")
                elif stoch_k > 80 and stoch_d > 80:
                    signals['signals']['stochastic'] = 'SELL'
                    signals['reasons'].append("Stochastic overbought")
                else:
                    signals['signals']['stochastic'] = 'NEUTRAL'
            
            # 6. Volume Signal
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                if volume_ratio > 1.5:
                    signals['signals']['volume'] = 'BUY' if current_price > data['Close'].iloc[-2] else 'SELL'
                    signals['reasons'].append(f"High volume ({volume_ratio:.1f}x average)")
                else:
                    signals['signals']['volume'] = 'NEUTRAL'
            
            # 7. Price Prediction Signal
            if predictions.get('ensemble'):
                predicted_price = predictions['ensemble'][0]
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                if price_change_pct > 2.0:
                    signals['signals']['prediction'] = 'BUY'
                    signals['reasons'].append(f"Predicted {price_change_pct:+.1f}% increase")
                elif price_change_pct < -2.0:
                    signals['signals']['prediction'] = 'SELL'
                    signals['reasons'].append(f"Predicted {price_change_pct:+.1f}% decrease")
                else:
                    signals['signals']['prediction'] = 'NEUTRAL'
            
            # 8. Williams %R Signal
            if 'Williams_R' in data.columns:
                williams_r = data['Williams_R'].iloc[-1]
                if williams_r < -80:
                    signals['signals']['williams_r'] = 'BUY'
                    signals['reasons'].append("Williams %R oversold")
                elif williams_r > -20:
                    signals['signals']['williams_r'] = 'SELL'
                    signals['reasons'].append("Williams %R overbought")
                else:
                    signals['signals']['williams_r'] = 'NEUTRAL'
            
            # Calculate overall recommendation
            buy_signals = sum(1 for signal in signals['signals'].values() if signal == 'BUY')
            sell_signals = sum(1 for signal in signals['signals'].values() if signal == 'SELL')
            total_signals = len(signals['signals'])
            
            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals
                
                if buy_ratio > 0.6:
                    signals['overall_recommendation'] = 'BUY'
                    signals['confidence'] = buy_ratio
                elif sell_ratio > 0.6:
                    signals['overall_recommendation'] = 'SELL'
                    signals['confidence'] = sell_ratio
                else:
                    signals['overall_recommendation'] = 'HOLD'
                    signals['confidence'] = max(buy_ratio, sell_ratio)
            
            return signals
            
        except Exception as e:
            print(f"Error generating trading signals: {str(e)}")
            return {}
    
    def generate_trading_signals(self, data: pd.DataFrame, predictions: Dict[str, List[float]]) -> Dict[str, any]:
        """
        Generate comprehensive buy/sell trading signals based on technical indicators and predictions
        
        Args:
            data: Latest market data with technical indicators
            predictions: Model predictions
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        try:
            if data.empty:
                return {}
            
            current_price = data['Close'].iloc[-1]
            signals = {
                'current_price': current_price,
                'signals': {},
                'overall_recommendation': 'HOLD',
                'confidence': 0.0,
                'reasons': []
            }
            
            # 1. RSI Signal
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if rsi < 30:
                    signals['signals']['rsi'] = 'BUY'
                    signals['reasons'].append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    signals['signals']['rsi'] = 'SELL'
                    signals['reasons'].append(f"RSI overbought ({rsi:.1f})")
                else:
                    signals['signals']['rsi'] = 'NEUTRAL'
            
            # 2. MACD Signal
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_Signal'].iloc[-1]
                if macd > macd_signal:
                    signals['signals']['macd'] = 'BUY'
                    signals['reasons'].append("MACD bullish crossover")
                elif macd < macd_signal:
                    signals['signals']['macd'] = 'SELL'
                    signals['reasons'].append("MACD bearish crossover")
                else:
                    signals['signals']['macd'] = 'NEUTRAL'
            
            # 3. Bollinger Bands Signal
            if 'BB_Position' in data.columns:
                bb_pos = data['BB_Position'].iloc[-1]
                if bb_pos < 0.2:
                    signals['signals']['bollinger'] = 'BUY'
                    signals['reasons'].append("Price near lower Bollinger Band")
                elif bb_pos > 0.8:
                    signals['signals']['bollinger'] = 'SELL'
                    signals['reasons'].append("Price near upper Bollinger Band")
                else:
                    signals['signals']['bollinger'] = 'NEUTRAL'
            
            # 4. Moving Average Signals
            ma_signals = []
            for window in [20, 50, 200]:
                if f'MA_{window}' in data.columns:
                    ma = data[f'MA_{window}'].iloc[-1]
                    if current_price > ma:
                        ma_signals.append('BUY')
                    else:
                        ma_signals.append('SELL')
            
            if ma_signals.count('BUY') > ma_signals.count('SELL'):
                signals['signals']['moving_averages'] = 'BUY'
                signals['reasons'].append("Price above major moving averages")
            else:
                signals['signals']['moving_averages'] = 'SELL'
                signals['reasons'].append("Price below major moving averages")
            
            # 5. Stochastic Oscillator Signal
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                stoch_k = data['Stoch_K'].iloc[-1]
                stoch_d = data['Stoch_D'].iloc[-1]
                if stoch_k < 20 and stoch_d < 20:
                    signals['signals']['stochastic'] = 'BUY'
                    signals['reasons'].append("Stochastic oversold")
                elif stoch_k > 80 and stoch_d > 80:
                    signals['signals']['stochastic'] = 'SELL'
                    signals['reasons'].append("Stochastic overbought")
                else:
                    signals['signals']['stochastic'] = 'NEUTRAL'
            
            # 6. Volume Signal
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                if volume_ratio > 1.5:
                    signals['signals']['volume'] = 'BUY' if current_price > data['Close'].iloc[-2] else 'SELL'
                    signals['reasons'].append(f"High volume ({volume_ratio:.1f}x average)")
                else:
                    signals['signals']['volume'] = 'NEUTRAL'
            
            # 7. Price Prediction Signal
            if predictions.get('ensemble'):
                predicted_price = predictions['ensemble'][0]
                price_change_pct = ((predicted_price - current_price) / current_price) * 100
                
                if price_change_pct > 2.0:
                    signals['signals']['prediction'] = 'BUY'
                    signals['reasons'].append(f"Predicted {price_change_pct:+.1f}% increase")
                elif price_change_pct < -2.0:
                    signals['signals']['prediction'] = 'SELL'
                    signals['reasons'].append(f"Predicted {price_change_pct:+.1f}% decrease")
                else:
                    signals['signals']['prediction'] = 'NEUTRAL'
            
            # 8. Williams %R Signal
            if 'Williams_R' in data.columns:
                williams_r = data['Williams_R'].iloc[-1]
                if williams_r < -80:
                    signals['signals']['williams_r'] = 'BUY'
                    signals['reasons'].append("Williams %R oversold")
                elif williams_r > -20:
                    signals['signals']['williams_r'] = 'SELL'
                    signals['reasons'].append("Williams %R overbought")
                else:
                    signals['signals']['williams_r'] = 'NEUTRAL'
            
            # Calculate overall recommendation
            buy_signals = sum(1 for signal in signals['signals'].values() if signal == 'BUY')
            sell_signals = sum(1 for signal in signals['signals'].values() if signal == 'SELL')
            total_signals = len(signals['signals'])
            
            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals
                
                if buy_ratio > 0.6:
                    signals['overall_recommendation'] = 'BUY'
                    signals['confidence'] = buy_ratio
                elif sell_ratio > 0.6:
                    signals['overall_recommendation'] = 'SELL'
                    signals['confidence'] = sell_ratio
                else:
                    signals['overall_recommendation'] = 'HOLD'
                    signals['confidence'] = max(buy_ratio, sell_ratio)
            
            return signals
            
        except Exception as e:
            print(f"Error generating trading signals: {str(e)}")
            return {}
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            # Save Random Forest model
            if self.rf_model is not None:
                with open(f'models/rf_model_{self.symbol}.pkl', 'wb') as f:
                    pickle.dump(self.rf_model, f)
            
            # Save LSTM model
            if self.lstm_model is not None:
                self.lstm_model.save(f'models/lstm_model_{self.symbol}.h5')
            
            # Save scaler
            with open(f'models/scaler_{self.symbol}.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(f'models/features_{self.symbol}.json', 'w') as f:
                json.dump(self.feature_columns, f)
            
            print("Models saved successfully")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load Random Forest model
            rf_path = f'models/rf_model_{self.symbol}.pkl'
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
            
            # Load LSTM model (only if TensorFlow available)
            lstm_path = f'models/lstm_model_{self.symbol}.h5'
            if TENSORFLOW_AVAILABLE and os.path.exists(lstm_path):
                self.lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Load scaler
            scaler_path = f'models/scaler_{self.symbol}.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load feature columns
            features_path = f'models/features_{self.symbol}.json'
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def save_predictions(self, predictions: Dict[str, List[float]], trading_signals: Dict[str, any] = None):
        """
        Save predictions and trading signals to disk with timestamp
        
        Args:
            predictions: Dictionary with model predictions
            trading_signals: Dictionary with trading signals
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prediction_data = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'predictions': predictions,
                'current_price': self.data['Close'].iloc[-1] if not self.data.empty else None,
                'trading_signals': trading_signals
            }
            
            # Save to JSON file
            filename = f'predictions/predictions_{self.symbol}_{datetime.now().strftime("%Y%m%d")}.json'
            
            # Load existing predictions or create new list
            all_predictions = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Check if file is not empty
                            all_predictions = json.loads(content)
                            if not isinstance(all_predictions, list):
                                print(f"Warning: Invalid prediction file format, starting fresh")
                                all_predictions = []
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Warning: Corrupted prediction file, starting fresh: {str(e)}")
                    all_predictions = []
            
            all_predictions.append(prediction_data)
            
            # Save updated predictions with proper error handling
            try:
                # Create a temporary file first
                temp_filename = filename + '.tmp'
                with open(temp_filename, 'w', encoding='utf-8') as f:
                    json.dump(all_predictions, f, indent=2, ensure_ascii=False)
                
                # Atomic move to replace the original file
                if os.path.exists(filename):
                    os.replace(temp_filename, filename)
                else:
                    os.rename(temp_filename, filename)
                
                # Store in memory for visualization
                self.predictions_history.append(prediction_data)
                
            except Exception as save_error:
                print(f"Error saving predictions to file: {str(save_error)}")
                # Clean up temp file if it exists
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")
    
    def create_real_time_plot(self) -> go.Figure:
        """
        Create real-time visualization of stock data and predictions
        
        Returns:
            Plotly figure object
        """
        try:
            if self.data.empty:
                return go.Figure()
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Stock Price & Predictions', 'Volume', 'Technical Indicators', 'Trading Signals'),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Stock price data
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=self.data['Close'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages
            for window in [20, 50]:
                if f'MA_{window}' in self.data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=self.data['Date'],
                            y=self.data[f'MA_{window}'],
                            mode='lines',
                            name=f'MA {window}',
                            line=dict(dash='dash')
                        ),
                        row=1, col=1
                    )
            
            # Add predictions if available
            if self.predictions_history:
                latest_pred = self.predictions_history[-1]
                if latest_pred['predictions']['ensemble']:
                    # Create future dates for predictions
                    last_date = self.data['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(self.prediction_horizon)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=[latest_pred['predictions']['ensemble'][0]] * self.prediction_horizon,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='red', width=2, dash='dot')
                        ),
                        row=1, col=1
                    )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=self.data['Date'],
                    y=self.data['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=3, col=1
                )
                
                # Add RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Trading signals visualization
            if self.predictions_history and self.predictions_history[-1].get('trading_signals'):
                signals = self.predictions_history[-1]['trading_signals']
                if signals.get('overall_recommendation'):
                    recommendation = signals['overall_recommendation']
                    confidence = signals.get('confidence', 0) * 100
                    
                    # Add signal indicator
                    fig.add_annotation(
                        x=self.data['Date'].iloc[-1],
                        y=self.data['Close'].iloc[-1],
                        text=f"{recommendation}<br>Confidence: {confidence:.1f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='green' if recommendation == 'BUY' else 'red' if recommendation == 'SELL' else 'orange',
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1,
                        row=1, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f'Real-Time Stock Analysis: {self.symbol}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=1000,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            return go.Figure()
    
    def update_data_and_predictions(self):
        """Update data and make new predictions"""
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating data and predictions...")
            
            # Fetch latest data
            new_data = self.fetch_live_data()
            if new_data.empty:
                print("No new data available")
                return
            
            # Engineer features
            processed_data = self.engineer_features(new_data)
            if processed_data.empty:
                print("Error processing data")
                return
            
            # Update stored data
            self.data = processed_data
            self.last_update = datetime.now()
            
            # Make predictions
            predictions = self.predict_next_days(processed_data)
            
            # Generate trading signals
            trading_signals = self.generate_trading_signals(processed_data, predictions)
            
            # Save predictions and signals
            self.save_predictions(predictions, trading_signals)
            
            # Print current status
            current_price = processed_data['Close'].iloc[-1]
            print(f"Current {self.symbol} price: ${current_price:.2f}")
            
            if predictions['ensemble']:
                predicted_price = predictions['ensemble'][0]
                change_pct = ((predicted_price - current_price) / current_price) * 100
                print(f"Predicted price: ${predicted_price:.2f} ({change_pct:+.2f}%)")
            
            # Print trading recommendation
            if trading_signals:
                recommendation = trading_signals['overall_recommendation']
                confidence = trading_signals['confidence'] * 100
                reasons = trading_signals['reasons']
                
                print(f"\n🎯 TRADING RECOMMENDATION: {recommendation}")
                print(f"Confidence: {confidence:.1f}%")
                print("Reasons:")
                for reason in reasons:
                    print(f"  • {reason}")
                
                # Color-coded recommendation
                if recommendation == 'BUY':
                    print("🟢 BUY SIGNAL - Consider purchasing shares")
                elif recommendation == 'SELL':
                    print("🔴 SELL SIGNAL - Consider selling shares")
                else:
                    print("🟡 HOLD - Monitor for better opportunities")
            
            # Create and save plot
            fig = self.create_real_time_plot()
            fig.write_html(f'data/stock_analysis_{self.symbol}.html')
            
            print("Update completed successfully")
            
        except Exception as e:
            print(f"Error in update: {str(e)}")
    
    def initialize_models(self):
        """Initialize models by loading existing ones or training new ones"""
        try:
            # Try to load existing models
            self.load_models()
            
            # If models don't exist, fetch data and train
            if self.rf_model is None or self.lstm_model is None:
                print("No existing models found. Training new models...")
                data = self.fetch_live_data()
                if not data.empty:
                    processed_data = self.engineer_features(data)
                    if not processed_data.empty:
                        self.train_models(processed_data)
                        self.data = processed_data
            else:
                # Load latest data for predictions
                data = self.fetch_live_data()
                if not data.empty:
                    self.data = self.engineer_features(data)
                
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
    
    def start_real_time_monitoring(self, update_interval_minutes: int = 60):
        """
        Start real-time monitoring with scheduled updates
        
        Args:
            update_interval_minutes: Interval between updates in minutes
        """
        try:
            print(f"Starting real-time monitoring for {self.symbol}")
            print(f"Updates every {update_interval_minutes} minutes")
            
            # Initial update
            self.update_data_and_predictions()
            
            # Schedule regular updates
            schedule.every(update_interval_minutes).minutes.do(self.update_data_and_predictions)
            
            # Run the scheduler
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in monitoring: {str(e)}")

def main():
    """Main function to run the stock predictor"""
    # Initialize predictor
    symbol = input("Enter stock symbol (default: AAPL): ").strip().upper() or "AAPL"
    predictor = RealTimeStockPredictor(symbol=symbol)
    
    # Start real-time monitoring
    try:
        predictor.start_real_time_monitoring(update_interval_minutes=60)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
