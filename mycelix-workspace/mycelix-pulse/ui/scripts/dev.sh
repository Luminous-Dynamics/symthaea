#!/bin/bash

# Start both backend and frontend in development mode

echo "🚀 Starting Mycelix-Mail development servers..."

# Check if concurrently is installed
if ! npm list -g concurrently &> /dev/null; then
    echo "Installing concurrently..."
    npm install -g concurrently
fi

# Start both servers
npm run dev
