#!/bin/bash

# UPI Fraud Detection System - Startup Script

echo "🛡️  UPI Fraud Detection System"
echo "=============================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip install --break-system-packages -r requirements.txt
fi

echo "🚀 Starting UPI Fraud Detection System..."
echo ""
echo "📍 Server will be available at: http://localhost:5000"
echo ""
echo "🔧 Features available:"
echo "   • Dashboard: Real-time overview and system features"
echo "   • Fraud Detection: Interactive transaction analysis"
echo "   • Model Training: ML model training with different dataset sizes"
echo "   • Analytics: Comprehensive fraud statistics and visualizations"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python3 app.py