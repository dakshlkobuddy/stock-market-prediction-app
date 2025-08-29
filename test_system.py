#!/usr/bin/env python3
"""
Test Script for Real-Time Stock Prediction System
================================================

This script tests the main components of the stock prediction system
to ensure everything is working correctly.
"""

import sys
import os
import time
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        import ta
        import plotly.graph_objects as go
        import streamlit as st
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetching():
    """Test data fetching from Yahoo Finance"""
    print("\nTesting data fetching...")
    
    try:
        import yfinance as yf
        
        # Test with a well-known stock
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="30d")
        
        if not data.empty:
            print(f"✅ Data fetched successfully: {len(data)} records")
            print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No data received")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\nTesting feature engineering...")
    
    try:
        import yfinance as yf
        import pandas as pd
        import ta
        
        # Fetch some data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="60d")
        data.reset_index(inplace=True)
        
        # Test basic feature engineering
        df = data.copy()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Check if features were created
        if 'RSI' in df.columns and 'MACD' in df.columns and 'MA_20' in df.columns:
            print(f"✅ Feature engineering successful: {len(df.columns)} features")
            return True
        else:
            print("❌ Feature engineering failed")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def test_model_creation():
    """Test model creation and basic training"""
    print("\nTesting model creation...")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Test Random Forest
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        # Test prediction
        prediction = rf_model.predict(X[:1])
        
        if prediction is not None:
            print("✅ Random Forest model created and tested successfully")
            return True
        else:
            print("❌ Model prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_lstm_model():
    """Test LSTM model creation"""
    print("\nTesting LSTM model...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        import numpy as np
        
        # Create dummy sequence data
        X = np.random.rand(50, 60, 10)  # 50 samples, 60 time steps, 10 features
        y = np.random.rand(50)
        
        # Create LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 10)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Test training
        history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
        
        # Test prediction
        prediction = model.predict(X[:1], verbose=0)
        
        if prediction is not None:
            print("✅ LSTM model created and tested successfully")
            return True
        else:
            print("❌ LSTM prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ LSTM model error: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\nTesting visualization...")
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        prices = np.random.rand(30) * 100 + 100
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Test Data'
        ))
        
        fig.update_layout(
            title='Test Chart',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        
        # Test saving
        fig.write_html('test_chart.html')
        
        if os.path.exists('test_chart.html'):
            os.remove('test_chart.html')  # Clean up
            print("✅ Visualization test successful")
            return True
        else:
            print("❌ Chart saving failed")
            return False
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_main_system():
    """Test the main prediction system"""
    print("\nTesting main prediction system...")
    
    try:
        from real_time_stock_predictor import RealTimeStockPredictor
        
        # Initialize predictor (this will test the full pipeline)
        predictor = RealTimeStockPredictor(symbol="AAPL")
        
        if predictor.data is not None and not predictor.data.empty:
            print(f"✅ Main system initialized successfully")
            print(f"   Data points: {len(predictor.data)}")
            print(f"   Features: {len(predictor.feature_columns)}")
            return True
        else:
            print("❌ Main system initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Main system error: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 Running Stock Prediction System Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Feature Engineering", test_feature_engineering),
        ("Random Forest Model", test_model_creation),
        ("LSTM Model", test_lstm_model),
        ("Visualization", test_visualization),
        ("Main System", test_main_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test mode
        print("🚀 Quick test mode - testing essential components only")
        quick_tests = [
            ("Imports", test_imports),
            ("Data Fetching", test_data_fetching),
            ("Feature Engineering", test_feature_engineering),
        ]
        
        for test_name, test_func in quick_tests:
            try:
                result = test_func()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:.<30} {status}")
            except Exception as e:
                print(f"{test_name:.<30} ❌ FAIL (Error: {e})")
    else:
        # Full test mode
        run_all_tests()

if __name__ == "__main__":
    main()
