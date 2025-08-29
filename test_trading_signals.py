#!/usr/bin/env python3
"""
Test Trading Signals Functionality
=================================

This script demonstrates the new buy/sell recommendation system
by testing it with a sample stock.
"""

from real_time_stock_predictor import RealTimeStockPredictor
import time

def test_trading_signals():
    """Test the trading signals functionality"""
    print("🧪 Testing Trading Signals Functionality")
    print("=" * 50)
    
    # Initialize predictor with a well-known stock
    symbol = "AAPL"
    print(f"Testing with {symbol}...")
    
    try:
        # Create predictor
        predictor = RealTimeStockPredictor(symbol=symbol)
        
        if predictor.data.empty:
            print("❌ No data available for testing")
            return False
        
        print(f"✅ Data loaded: {len(predictor.data)} records")
        
        # Make predictions
        predictions = predictor.predict_next_days(predictor.data)
        
        if not predictions['ensemble']:
            print("❌ No predictions available")
            return False
        
        print(f"✅ Predictions generated")
        
        # Generate trading signals
        trading_signals = predictor.generate_trading_signals(predictor.data, predictions)
        
        if not trading_signals:
            print("❌ No trading signals generated")
            return False
        
        # Display results
        print("\n📊 TRADING SIGNALS ANALYSIS")
        print("=" * 50)
        
        current_price = trading_signals['current_price']
        recommendation = trading_signals['overall_recommendation']
        confidence = trading_signals['confidence'] * 100
        reasons = trading_signals['reasons']
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence:.1f}%")
        
        print(f"\n📋 Reasons:")
        for reason in reasons:
            print(f"  • {reason}")
        
        print(f"\n🔍 Individual Signals:")
        signals = trading_signals['signals']
        for signal_name, signal_value in signals.items():
            status_icon = "✅" if signal_value == 'BUY' else "❌" if signal_value == 'SELL' else "➖"
            print(f"  {status_icon} {signal_name.replace('_', ' ').title()}: {signal_value}")
        
        # Color-coded recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        if recommendation == 'BUY':
            print("🟢 BUY SIGNAL - Consider purchasing shares")
        elif recommendation == 'SELL':
            print("🔴 SELL SIGNAL - Consider selling shares")
        else:
            print("🟡 HOLD - Monitor for better opportunities")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_multiple_stocks():
    """Test trading signals with multiple stocks"""
    print("\n🧪 Testing Multiple Stocks")
    print("=" * 50)
    
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    for symbol in stocks:
        print(f"\nTesting {symbol}...")
        try:
            predictor = RealTimeStockPredictor(symbol=symbol)
            
            if not predictor.data.empty:
                predictions = predictor.predict_next_days(predictor.data)
                trading_signals = predictor.generate_trading_signals(predictor.data, predictions)
                
                if trading_signals:
                    recommendation = trading_signals['overall_recommendation']
                    confidence = trading_signals['confidence'] * 100
                    current_price = trading_signals['current_price']
                    
                    print(f"  Price: ${current_price:.2f}")
                    print(f"  Signal: {recommendation} ({confidence:.1f}% confidence)")
                else:
                    print(f"  ❌ No signals generated")
            else:
                print(f"  ❌ No data available")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

def main():
    """Main test function"""
    print("🚀 Trading Signals Test Suite")
    print("=" * 60)
    
    # Test single stock
    success = test_trading_signals()
    
    if success:
        print("\n✅ Single stock test completed successfully!")
        
        # Test multiple stocks
        test_multiple_stocks()
        
        print("\n🎉 All tests completed!")
        print("\n💡 To run the full system:")
        print("   python real_time_stock_predictor.py")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n❌ Tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
