#!/usr/bin/env python3
"""
Configuration file for Real-Time Stock Prediction System
=======================================================

This file contains all configurable parameters for the stock prediction system.
Modify these settings to customize the behavior of the system.
"""

import os
from datetime import timedelta

# API Configuration
API_CONFIG = {
    'default_symbol': 'AAPL',
    'data_source': 'yahoo_finance',  # Options: yahoo_finance, alpha_vantage, twelve_data
    'update_interval_minutes': 60,
    'data_lookback_days': 730,  # 2 years of historical data
}

# Model Configuration
MODEL_CONFIG = {
    'prediction_horizon': 5,  # Number of days to predict ahead
    'sequence_length': 60,  # Number of time steps for LSTM
    'train_test_split': 0.8,  # Training data ratio
    
    # LSTM Model Parameters
    'lstm_units': [100, 100, 100],  # Number of units in each LSTM layer
    'lstm_dropout': 0.2,
    'lstm_learning_rate': 0.001,
    'lstm_epochs': 50,
    'lstm_batch_size': 32,
    
    # Random Forest Parameters
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    'rf_random_state': 42,
    
    # Feature Engineering
    'moving_averages': [5, 10, 20, 50, 200],
    'lag_features': [1, 2, 3, 5, 10],
    'rolling_windows': [5, 10, 20],
}

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'williams_period': 14,
    'atr_period': 14,
}

# File Paths Configuration
PATHS = {
    'models_dir': 'models',
    'data_dir': 'data',
    'predictions_dir': 'predictions',
    'logs_dir': 'logs',
    'reports_dir': 'reports',
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'chart_height': 800,
    'chart_width': None,  # Auto-width
    'color_scheme': {
        'actual_price': '#1f77b4',
        'predicted_price': '#ff7f0e',
        'lstm_prediction': '#2ca02c',
        'rf_prediction': '#d62728',
        'volume': '#7f7f7f',
        'rsi': '#9467bd',
        'macd': '#8c564b',
    },
    'update_charts': True,
    'save_charts': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/stock_predictor.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_ttl_hours': 24,
    'max_workers': 4,
    'memory_limit_gb': 8,
    'enable_gpu': False,  # Set to True if GPU is available
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': False,
    'price_change_threshold': 0.05,  # 5% price change
    'volume_spike_threshold': 2.0,  # 2x average volume
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'alert_methods': ['console', 'email'],  # Options: console, email, webhook
}

# Email Configuration (if alerts are enabled)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.getenv('SENDER_EMAIL', ''),
    'sender_password': os.getenv('SENDER_PASSWORD', ''),
    'recipient_emails': [],
}

# Webhook Configuration (if alerts are enabled)
WEBHOOK_CONFIG = {
    'webhook_url': os.getenv('WEBHOOK_URL', ''),
    'webhook_headers': {
        'Content-Type': 'application/json',
    },
}

# Validation Configuration
VALIDATION_CONFIG = {
    'min_data_points': 100,
    'max_missing_data_pct': 0.1,  # 10%
    'outlier_threshold': 3.0,  # Standard deviations
    'validate_predictions': True,
}

# Backup Configuration
BACKUP_CONFIG = {
    'enable_backup': True,
    'backup_interval_hours': 24,
    'backup_retention_days': 30,
    'backup_compression': True,
}

# Security Configuration
SECURITY_CONFIG = {
    'api_key_required': False,
    'rate_limit_requests': 100,  # Requests per hour
    'rate_limit_window': 3600,  # 1 hour in seconds
    'enable_ssl': True,
}

# Environment-specific configurations
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # Production settings
    API_CONFIG['update_interval_minutes'] = 15
    MODEL_CONFIG['lstm_epochs'] = 100
    PERFORMANCE_CONFIG['enable_gpu'] = True
    ALERT_CONFIG['enable_alerts'] = True
    LOGGING_CONFIG['log_level'] = 'WARNING'
    
elif ENVIRONMENT == 'testing':
    # Testing settings
    API_CONFIG['update_interval_minutes'] = 5
    MODEL_CONFIG['lstm_epochs'] = 10
    MODEL_CONFIG['rf_n_estimators'] = 10
    PERFORMANCE_CONFIG['enable_caching'] = False
    LOGGING_CONFIG['log_level'] = 'DEBUG'

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the application"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

# Validate configuration
def validate_config():
    """Validate the configuration settings"""
    errors = []
    
    # Check required directories
    for name, path in PATHS.items():
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {path}: {str(e)}")
    
    # Validate model parameters
    if MODEL_CONFIG['prediction_horizon'] <= 0:
        errors.append("prediction_horizon must be positive")
    
    if MODEL_CONFIG['sequence_length'] <= 0:
        errors.append("sequence_length must be positive")
    
    if not (0 < MODEL_CONFIG['train_test_split'] < 1):
        errors.append("train_test_split must be between 0 and 1")
    
    # Validate API configuration
    if API_CONFIG['update_interval_minutes'] <= 0:
        errors.append("update_interval_minutes must be positive")
    
    if API_CONFIG['data_lookback_days'] <= 0:
        errors.append("data_lookback_days must be positive")
    
    # Validate technical indicators
    for indicator, value in TECHNICAL_INDICATORS.items():
        if value <= 0:
            errors.append(f"{indicator} period must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Initialize configuration
if __name__ == "__main__":
    create_directories()
    validate_config()
    print("Configuration validated successfully!")
