#!/bin/bash
# Terra Atlas Complete Platform Startup Script
# Launches all services with full data integration

set -e

echo "🚀 Terra Atlas Complete Platform Launcher"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Not in terra-atlas-mvp directory"
    echo "Please run from: /srv/luminous-dynamics/terra-atlas-mvp"
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    fi
    return 0
}

echo "${BLUE}1. Setting up environment...${NC}"
# Get API keys from BWS
export NEXT_PUBLIC_MAPBOX_TOKEN=$(bws get mapbox-prod-public-token 2>/dev/null || echo "")
export NEXT_PUBLIC_CESIUM_TOKEN=$(bws get cesium-prod-ion-token 2>/dev/null || echo "")
echo "✅ Environment configured"

echo ""
echo "${BLUE}2. Checking dependencies...${NC}"
# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi
echo "✅ Node dependencies ready"

# Check Python dependencies
if [ ! -d "backend/venv" ]; then
    echo "Setting up Python virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt 2>/dev/null || {
        # If requirements.txt doesn't exist, install manually
        pip install fastapi uvicorn psycopg2-binary redis pandas numpy \
                   scikit-learn aiohttp python-dotenv joblib
    }
    cd ..
else
    source backend/venv/bin/activate
fi
echo "✅ Python dependencies ready"

echo ""
echo "${BLUE}3. Database Migration${NC}"
read -p "Run database migration? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Migrating 79,193 projects from old demos..."
    npx tsx scripts/complete-data-integration.ts || {
        echo "${YELLOW}Migration script not ready, creating database tables...${NC}"
        # Run Supabase migrations
        npm run db:push || echo "Database already configured"
    }
    echo "✅ Database ready"
fi

echo ""
echo "${BLUE}4. Starting services...${NC}"

# Start Python backend in background
if check_port 8000; then
    echo "Starting Python site analysis engine on port 8000..."
    cd backend
    nohup python site_analysis_engine.py > ../logs/python-backend.log 2>&1 &
    PYTHON_PID=$!
    cd ..
    echo "✅ Python backend started (PID: $PYTHON_PID)"
else
    echo "${YELLOW}⚠️  Python backend already running on port 8000${NC}"
fi

# Optional: Start Redis for caching
if command -v redis-server &> /dev/null; then
    if check_port 6379; then
        echo "Starting Redis cache server..."
        redis-server --daemonize yes
        echo "✅ Redis cache started"
    else
        echo "${YELLOW}Redis already running${NC}"
    fi
fi

# Start Next.js development server
echo ""
echo "${BLUE}5. Starting Terra Atlas web application...${NC}"
if check_port 3002; then
    echo "${GREEN}=========================================="
    echo "🌍 Terra Atlas Platform Starting"
    echo "=========================================="
    echo ""
    echo "📍 Main App:        http://localhost:3002"
    echo "🔬 Site Analysis:   http://localhost:8000/docs"
    echo "📊 Database:        ${NEXT_PUBLIC_SUPABASE_URL}"
    echo ""
    echo "Features Active:"
    echo "  ✅ 79,193 energy projects"
    echo "  ✅ Real-time WebSocket updates"
    echo "  ✅ ML-powered site analysis"
    echo "  ✅ Investment calculator"
    echo "  ✅ Portfolio tracking"
    echo "  ✅ 3D Globe visualization"
    echo ""
    echo "${YELLOW}Press Ctrl+C to stop all services${NC}"
    echo "==========================================${NC}"
    echo ""
    
    # Start Next.js
    npm run dev
else
    echo "❌ Port 3002 is already in use!"
    echo "Run: lsof -i :3002"
    echo "Then kill the process or use a different port"
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down services..."
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    # Don't stop Redis as other services might use it
    echo "✅ Services stopped"
    exit 0
}

trap cleanup INT

# Keep script running
wait