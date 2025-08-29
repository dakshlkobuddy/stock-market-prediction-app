# Real-Time Stock Price Prediction System

A comprehensive Python-based system for real-time stock price prediction using machine learning and deep learning models. The system fetches live market data, engineers features, trains multiple models, and provides continuous predictions with visualization.

## 🚀 Features

- **Real-time Data Fetching**: Connects to Yahoo Finance API for live market data
- **Advanced Feature Engineering**: 50+ technical indicators including RSI, MACD, Bollinger Bands, etc.
- **Multiple ML Models**: LSTM (Deep Learning) and Random Forest (Ensemble) models
- **Trading Signals**: Comprehensive buy/sell recommendations based on technical analysis
- **Continuous Monitoring**: Automatic updates at configurable intervals
- **Real-time Visualization**: Interactive charts with Plotly and Streamlit
- **Model Persistence**: Save and load trained models
- **Prediction Storage**: Historical prediction tracking and analysis
- **Web Interface**: Beautiful Streamlit dashboard with trading recommendations
- **Comprehensive Logging**: Detailed logging and error handling

## 📋 Requirements

- Python 3.8+
- Internet connection for data fetching
- 4GB+ RAM (8GB+ recommended)
- GPU support optional (for faster LSTM training)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd stock-market-prediction
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Initialize the system:**
```bash
python config.py
```

## 🚀 Quick Start

### Command Line Interface

1. **Run the main prediction system:**
```bash
python real_time_stock_predictor.py
```

2. **Enter stock symbol when prompted (default: AAPL)**

3. **The system will:**
   - Fetch historical data
   - Train models (if not already trained)
   - Start real-time monitoring
   - Update predictions every hour

### Web Interface

1. **Launch the Streamlit dashboard:**
```bash
streamlit run streamlit_app.py
```

2. **Open your browser to the provided URL**

3. **Use the interactive interface to:**
   - Select different stock symbols
   - View real-time charts
   - Monitor predictions
   - Analyze historical performance

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Preprocessing  │───▶│  Feature Eng.   │
│ (Yahoo Finance) │    │   & Cleaning    │    │  & Indicators   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Predictions   │◀───│  Model Training │◀───│  Model Storage  │
│   & Storage     │    │  & Inference    │    │   & Loading     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Visualization  │◀───│  Real-time      │◀───│  Scheduled      │
│  & Dashboard    │    │  Monitoring     │    │  Updates        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

The system is highly configurable through `config.py`:

### Key Settings

- **Update Interval**: How often to fetch new data (default: 60 minutes)
- **Prediction Horizon**: Number of days to predict ahead (default: 5)
- **Model Parameters**: LSTM layers, Random Forest settings, etc.
- **Technical Indicators**: RSI, MACD, Bollinger Bands parameters
- **Visualization**: Chart colors, sizes, update frequency

### Environment Variables

```bash
# Optional: Set environment for production
export ENVIRONMENT=production

# Optional: Email alerts (if enabled)
export SENDER_EMAIL=your-email@gmail.com
export SENDER_PASSWORD=your-app-password

# Optional: Webhook alerts (if enabled)
export WEBHOOK_URL=your-webhook-url
```

## 📈 Technical Indicators

The system calculates 50+ technical indicators:

### Price-based Indicators
- Moving Averages (5, 10, 20, 50, 200-day)
- Price Change, High/Low Ratio
- Bollinger Bands, ATR

### Volume Indicators
- Volume Change, Volume MA
- Volume Ratio, Volume Standard Deviation

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R, ROC, TSI

### Time-based Features
- Day of Week, Month, Quarter, Year
- Lag features (1, 2, 3, 5, 10-day)

## 🤖 Machine Learning Models

### LSTM (Long Short-Term Memory)
- **Architecture**: 3 LSTM layers with dropout
- **Input**: 60-day sequence of engineered features
- **Output**: Next day price prediction
- **Training**: Adam optimizer, MSE loss

### Random Forest
- **Ensemble**: 100 decision trees
- **Features**: All engineered indicators
- **Output**: Next day price prediction
- **Advantages**: Handles non-linear relationships

### Ensemble Method
- **Combination**: Average of LSTM and Random Forest predictions
- **Benefits**: Reduces overfitting, improves accuracy

## 🎯 Trading Signals System

The system provides comprehensive buy/sell recommendations based on multiple technical indicators:

### Technical Analysis Signals
- **RSI (Relative Strength Index)**: Oversold/overbought conditions
- **MACD**: Bullish/bearish momentum crossovers
- **Bollinger Bands**: Price position relative to volatility bands
- **Moving Averages**: Trend analysis (20, 50, 200-day)
- **Stochastic Oscillator**: Momentum and reversal signals
- **Volume Analysis**: Volume spikes and trends
- **Williams %R**: Momentum oscillator
- **Price Predictions**: ML model forecast integration

### Signal Aggregation
- **Confidence Scoring**: Weighted combination of all signals
- **Recommendation Levels**: BUY, SELL, or HOLD
- **Reason Analysis**: Detailed explanation of each signal
- **Real-time Updates**: Continuous signal monitoring

## 📁 File Structure

```
stock-market-prediction/
├── real_time_stock_predictor.py  # Main prediction system
├── streamlit_app.py              # Web dashboard
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── test_system.py                # Test script for system functionality
├── test_trading_signals.py       # Test script for trading signals
├── models/                       # Trained models storage
├── data/                         # Market data and charts
├── predictions/                  # Historical predictions
├── logs/                         # System logs
└── reports/                      # Generated reports
```

## 📊 Usage Examples

### Basic Usage

```python
from real_time_stock_predictor import RealTimeStockPredictor

# Initialize predictor
predictor = RealTimeStockPredictor(symbol="AAPL")

# Make predictions
predictions = predictor.predict_next_days(predictor.data)
print(f"Predicted price: ${predictions['ensemble'][0]:.2f}")

# Generate trading signals
trading_signals = predictor.generate_trading_signals(predictor.data, predictions)
if trading_signals:
    recommendation = trading_signals['overall_recommendation']
    confidence = trading_signals['confidence'] * 100
    print(f"Trading Recommendation: {recommendation} ({confidence:.1f}% confidence)")

# Start real-time monitoring
predictor.start_real_time_monitoring(update_interval_minutes=30)
```

### Custom Configuration

```python
# Modify configuration
from config import MODEL_CONFIG, API_CONFIG

MODEL_CONFIG['prediction_horizon'] = 10
API_CONFIG['update_interval_minutes'] = 15

# Use custom settings
predictor = RealTimeStockPredictor(symbol="GOOGL")
```

### Data Analysis

```python
# Access processed data
data = predictor.data
print(f"Features: {len(predictor.feature_columns)}")
print(f"Data points: {len(data)}")

# View technical indicators
print(data[['RSI', 'MACD', 'BB_Position']].tail())
```

## 🔍 Monitoring and Logs

### Log Files
- **Location**: `logs/stock_predictor.log`
- **Level**: INFO (configurable)
- **Rotation**: 10MB max size, 5 backups

### Key Metrics Tracked
- Data fetch success/failure
- Model training progress
- Prediction accuracy
- System performance
- Error handling

### Performance Monitoring
```bash
# View recent logs
tail -f logs/stock_predictor.log

# Check system status
python -c "from real_time_stock_predictor import RealTimeStockPredictor; p = RealTimeStockPredictor('AAPL'); print(f'Last update: {p.last_update}')"
```

## 🚨 Troubleshooting

### Common Issues

1. **No data received**
   - Check internet connection
   - Verify stock symbol is valid
   - Check API rate limits

2. **Model training fails**
   - Ensure sufficient data (100+ points minimum)
   - Check memory availability
   - Verify TensorFlow installation

3. **Streamlit app won't start**
   - Check port availability
   - Verify Streamlit installation
   - Check file permissions

### Performance Optimization

1. **Reduce update frequency** in `config.py`
2. **Use GPU** for LSTM training (set `enable_gpu=True`)
3. **Increase memory limit** if available
4. **Enable caching** for repeated data fetches

## 📈 Model Performance

### Typical Accuracy Metrics
- **RMSE**: $2-5 (varies by stock volatility)
- **MAPE**: 2-8% (percentage error)
- **Training Time**: 5-15 minutes (first run)
- **Prediction Time**: <1 second

### Model Comparison
| Model | Pros | Cons |
|-------|------|------|
| LSTM | Captures temporal patterns | Requires more data |
| Random Forest | Handles non-linear relationships | Less temporal awareness |
| Ensemble | Best overall accuracy | More complex |

## 🔮 Future Enhancements

- [ ] Support for additional data sources (Alpha Vantage, Twelve Data)
- [ ] More ML models (XGBoost, Transformer models)
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization
- [ ] Mobile app interface
- [ ] Cloud deployment support
- [ ] Advanced alerting system
- [ ] Backtesting framework

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

This system is for educational and research purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the logs for error details
- Review the configuration settings
- Ensure all dependencies are installed

---

**Happy Trading! 📈**
