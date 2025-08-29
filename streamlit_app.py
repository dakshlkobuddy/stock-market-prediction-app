#!/usr/bin/env python3
"""
Streamlit Web Application for Real-Time Stock Prediction
=======================================================

This Streamlit app provides a web interface for the real-time stock prediction system.
Features:
- Interactive stock symbol selection
- Real-time data visualization
- Model performance metrics
- Prediction history tracking
- Technical indicators display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os
from datetime import datetime, timedelta
import time
from real_time_stock_predictor import RealTimeStockPredictor

# Page configuration
st.set_page_config(
    page_title="Real-Time Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_prediction_history(symbol: str) -> list:
    """Load prediction history from JSON files"""
    try:
        predictions = []
        data_dir = "predictions"
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.startswith(f"predictions_{symbol}_") and filename.endswith(".json"):
                    filepath = os.path.join(data_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            file_content = f.read().strip()
                            if file_content:  # Check if file is not empty
                                daily_predictions = json.loads(file_content)
                                if isinstance(daily_predictions, list):
                                    predictions.extend(daily_predictions)
                                else:
                                    st.warning(f"Invalid format in {filename}: expected list, got {type(daily_predictions)}")
                            else:
                                st.warning(f"Empty file: {filename}")
                    except json.JSONDecodeError as e:
                        st.warning(f"Corrupted JSON file {filename}: {str(e)}")
                        # Try to backup and remove corrupted file
                        try:
                            backup_path = filepath + '.backup'
                            if os.path.exists(filepath):
                                os.rename(filepath, backup_path)
                                st.info(f"Backed up corrupted file to {backup_path}")
                        except Exception as backup_error:
                            st.error(f"Failed to backup corrupted file: {str(backup_error)}")
                    except Exception as file_error:
                        st.warning(f"Error reading {filename}: {str(file_error)}")
        return predictions
    except Exception as e:
        st.error(f"Error loading prediction history: {str(e)}")
        return []

def create_prediction_accuracy_chart(predictions: list) -> go.Figure:
    """Create chart showing prediction accuracy over time"""
    if not predictions:
        return go.Figure()
    
    try:
        # Extract data
        dates = []
        actual_prices = []
        predicted_prices = []
        
        for pred in predictions:
            if pred.get('current_price') and pred.get('predictions', {}).get('ensemble'):
                dates.append(datetime.strptime(pred['timestamp'], "%Y-%m-%d %H:%M:%S"))
                actual_prices.append(pred['current_price'])
                predicted_prices.append(pred['predictions']['ensemble'][0])
        
        if not dates:
            return go.Figure()
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Actual': actual_prices,
            'Predicted': predicted_prices
        })
        df = df.sort_values('Date')
        
        # Calculate accuracy metrics
        df['Error'] = abs(df['Actual'] - df['Predicted'])
        df['Error_Pct'] = (df['Error'] / df['Actual']) * 100
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Predictions vs Actual', 'Prediction Error (%)'),
            vertical_spacing=0.1
        )
        
        # Price comparison
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Actual'],
                mode='lines+markers',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Predicted'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # Error percentage
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Error_Pct'],
                mode='lines+markers',
                name='Error %',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Prediction Accuracy Over Time',
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating accuracy chart: {str(e)}")
        return go.Figure()

def create_model_comparison_chart(predictions: list) -> go.Figure:
    """Create chart comparing different model predictions"""
    if not predictions:
        return go.Figure()
    
    try:
        # Extract data
        dates = []
        lstm_preds = []
        rf_preds = []
        ensemble_preds = []
        actual_prices = []
        
        for pred in predictions:
            if pred.get('current_price') and pred.get('predictions'):
                dates.append(datetime.strptime(pred['timestamp'], "%Y-%m-%d %H:%M:%S"))
                actual_prices.append(pred['current_price'])
                
                preds = pred['predictions']
                lstm_preds.append(preds.get('lstm', [None])[0] if preds.get('lstm') else None)
                rf_preds.append(preds.get('random_forest', [None])[0] if preds.get('random_forest') else None)
                ensemble_preds.append(preds.get('ensemble', [None])[0] if preds.get('ensemble') else None)
        
        if not dates:
            return go.Figure()
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Actual': actual_prices,
            'LSTM': lstm_preds,
            'Random Forest': rf_preds,
            'Ensemble': ensemble_preds
        })
        df = df.sort_values('Date')
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Actual'],
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='black', width=3)
        ))
        
        if df['LSTM'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['LSTM'],
                mode='lines+markers',
                name='LSTM Prediction',
                line=dict(color='blue', width=2)
            ))
        
        if df['Random Forest'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Random Forest'],
                mode='lines+markers',
                name='Random Forest Prediction',
                line=dict(color='green', width=2)
            ))
        
        if df['Ensemble'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Ensemble'],
                mode='lines+markers',
                name='Ensemble Prediction',
                line=dict(color='red', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title='Model Comparison: Predictions vs Actual',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating model comparison chart: {str(e)}")
        return go.Figure()

def calculate_model_metrics(predictions: list) -> dict:
    """Calculate model performance metrics"""
    if not predictions:
        return {}
    
    try:
        metrics = {}
        
        for model_name in ['lstm', 'random_forest', 'ensemble']:
            actual_prices = []
            predicted_prices = []
            
            for pred in predictions:
                if pred.get('current_price') and pred.get('predictions', {}).get(model_name):
                    actual_prices.append(pred['current_price'])
                    predicted_prices.append(pred['predictions'][model_name][0])
            
            if actual_prices and predicted_prices:
                actual = np.array(actual_prices)
                predicted = np.array(predicted_prices)
                
                # Calculate metrics
                mse = np.mean((actual - predicted) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(actual - predicted))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                metrics[model_name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'Count': len(actual_prices)
                }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def main():
    """Main Streamlit application"""
    
    # Add deployment info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Deployed App**")
    st.sidebar.markdown("Stock Market Prediction System")
    
    # Header
    st.markdown('<h1 class="main-header">📈 Real-Time Stock Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter the stock symbol (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Update interval
    update_interval = st.sidebar.slider(
        "Update Interval (minutes)",
        min_value=1,
        max_value=60,
        value=5,
        help="How often to update predictions"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh",
        value=True,
        help="Automatically refresh data and predictions"
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Live Analysis: {symbol}")
        
        # Initialize predictor
        try:
            predictor = RealTimeStockPredictor(symbol=symbol)
            
            # Display current data
            if not predictor.data.empty:
                current_price = predictor.data['Close'].iloc[-1]
                price_change = predictor.data['Close'].pct_change().iloc[-1] * 100
                
                # Price metrics
                col1_1, col1_2, col1_3 = st.columns(3)
                
                with col1_1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col1_2:
                    if predictor.data['Volume'].iloc[-1]:
                        st.metric(
                            "Volume",
                            f"{predictor.data['Volume'].iloc[-1]:,.0f}"
                        )
                
                with col1_3:
                    if 'RSI' in predictor.data.columns:
                        rsi = predictor.data['RSI'].iloc[-1]
                        st.metric(
                            "RSI",
                            f"{rsi:.1f}",
                            "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                        )
                
                # Real-time chart
                st.subheader("📈 Real-Time Chart")
                fig = predictor.create_real_time_plot()
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No data available. Please check the stock symbol and try again.")
                
        except Exception as e:
            st.error(f"Error initializing predictor: {str(e)}")
    
    with col2:
        st.subheader("🎯 Predictions & Trading Signals")
        
        try:
            # Make predictions
            if not predictor.data.empty:
                predictions = predictor.predict_next_days(predictor.data)
                
                if predictions.get('ensemble'):
                    predicted_price = predictions['ensemble'][0]
                    current_price = predictor.data['Close'].iloc[-1]
                    change_pct = ((predicted_price - current_price) / current_price) * 100
                    
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric(
                        "Next Day Prediction",
                        f"${predicted_price:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate trading signals
                    trading_signals = predictor.generate_trading_signals(predictor.data, predictions)
                    
                    if trading_signals:
                        recommendation = trading_signals['overall_recommendation']
                        confidence = trading_signals['confidence'] * 100
                        reasons = trading_signals['reasons']
                        
                        # Display trading recommendation
                        st.subheader("📊 Trading Recommendation")
                        
                        # Color-coded recommendation display
                        if recommendation == 'BUY':
                            st.success(f"🟢 **BUY** - Confidence: {confidence:.1f}%")
                            st.info("Consider purchasing shares")
                        elif recommendation == 'SELL':
                            st.error(f"🔴 **SELL** - Confidence: {confidence:.1f}%")
                            st.info("Consider selling shares")
                        else:
                            st.warning(f"🟡 **HOLD** - Confidence: {confidence:.1f}%")
                            st.info("Monitor for better opportunities")
                        
                        # Display reasons
                        st.subheader("📋 Signal Analysis")
                        for reason in reasons:
                            st.write(f"• {reason}")
                        
                        # Display individual signals
                        st.subheader("🔍 Technical Signals")
                        signals = trading_signals['signals']
                        for signal_name, signal_value in signals.items():
                            if signal_value == 'BUY':
                                st.write(f"✅ {signal_name.replace('_', ' ').title()}: BUY")
                            elif signal_value == 'SELL':
                                st.write(f"❌ {signal_name.replace('_', ' ').title()}: SELL")
                            else:
                                st.write(f"➖ {signal_name.replace('_', ' ').title()}: NEUTRAL")
                    
                    # Model predictions comparison
                    st.subheader("🤖 Model Predictions")
                    
                    if predictions.get('lstm'):
                        st.metric("LSTM", f"${predictions['lstm'][0]:.2f}")
                    
                    if predictions.get('random_forest'):
                        st.metric("Random Forest", f"${predictions['random_forest'][0]:.2f}")
                    
                    if predictions.get('ensemble'):
                        st.metric("Ensemble", f"${predictions['ensemble'][0]:.2f}")
                
                else:
                    st.warning("No predictions available")
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
    
    # Historical Analysis Section
    st.subheader("📚 Historical Analysis")
    
    # Load prediction history
    predictions_history = load_prediction_history(symbol)
    
    if predictions_history:
        # Model performance metrics
        st.subheader("📊 Model Performance")
        metrics = calculate_model_metrics(predictions_history)
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            for i, (model_name, model_metrics) in enumerate(metrics.items()):
                with [col1, col2, col3][i]:
                    st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                    st.metric("RMSE", f"${model_metrics['RMSE']:.2f}")
                    st.metric("MAPE", f"{model_metrics['MAPE']:.2f}%")
                    st.metric("Predictions", model_metrics['Count'])
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Accuracy")
            accuracy_fig = create_prediction_accuracy_chart(predictions_history)
            st.plotly_chart(accuracy_fig, use_container_width=True)
        
        with col2:
            st.subheader("🤖 Model Comparison")
            comparison_fig = create_model_comparison_chart(predictions_history)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Prediction history table
        st.subheader("📋 Recent Predictions")
        
        # Create DataFrame for display
        history_data = []
        for pred in predictions_history[-10:]:  # Show last 10 predictions
            if pred.get('current_price') and pred.get('predictions', {}).get('ensemble'):
                history_data.append({
                    'Timestamp': pred['timestamp'],
                    'Actual Price': f"${pred['current_price']:.2f}",
                    'Predicted Price': f"${pred['predictions']['ensemble'][0]:.2f}",
                    'Error': f"${abs(pred['current_price'] - pred['predictions']['ensemble'][0]):.2f}",
                    'Error %': f"{abs((pred['current_price'] - pred['predictions']['ensemble'][0]) / pred['current_price']) * 100:.2f}%"
                })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
    
    else:
        st.info("No historical predictions found. Start the prediction system to generate data.")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(update_interval * 60)
        st.rerun()

if __name__ == "__main__":
    main()
