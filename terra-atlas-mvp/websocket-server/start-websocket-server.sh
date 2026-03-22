#!/bin/bash

# Start WebSocket Server for Terra Atlas

echo "🚀 Starting Terra Atlas WebSocket Server..."
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running (pub/sub enabled)"
    else
        echo "⚠️  Redis not running (running without pub/sub)"
    fi
else
    echo "ℹ️  Redis not installed (running without pub/sub)"
fi

# Start the server
echo ""
echo "🌐 Starting WebSocket server on port 8002..."
echo "==========================================="
echo "WebSocket URL: ws://localhost:8002/ws/{client_id}"
echo "Health Check: http://localhost:8002/"
echo "Statistics: http://localhost:8002/stats"
echo ""

# Run the server
python server.py