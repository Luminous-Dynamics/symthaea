#!/usr/bin/env bash

echo "🚀 Starting Terra Atlas ML Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export ML_BACKEND_PORT=8001

# Start the FastAPI server
echo "🌟 Starting ML server on port $ML_BACKEND_PORT..."
uvicorn main:app --host 0.0.0.0 --port $ML_BACKEND_PORT --reload