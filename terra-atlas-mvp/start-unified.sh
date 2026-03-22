#!/usr/bin/env bash

# Terra Atlas MVP - The ONE Unified Version with 3D Globe
# This is the main version we improve, not create new demos

echo "🌍 Terra Atlas MVP - Starting the ONE unified version"
echo "===================================================="
echo ""
echo "✨ Features:"
echo "  • 3D Globe visualization (the one you liked!)"
echo "  • Real database with 79,193 projects"
echo "  • Modern Next.js 15 architecture"
echo "  • Authentication ready (90% complete)"
echo "  • All improvements in ONE place"
echo ""

# Check if database exists
if [ ! -f "data/terra-atlas-local.db" ]; then
    echo "📦 Setting up database with 79,193 projects..."
    if [ -f "../terra-atlas-old-demos/data/terra-atlas-local.db" ]; then
        cp ../terra-atlas-old-demos/data/terra-atlas-local.db data/
        echo "✅ Database copied from old demos"
    else
        echo "⚠️  Database not found. Please run setup-database.js"
    fi
fi

# Check dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Set port to avoid Grafana conflict
export PORT=3002
echo "🚀 Starting on port $PORT (avoiding Grafana on 3000)"
echo ""
echo "📍 Access points:"
echo "  • Main app: http://localhost:$PORT"
echo "  • API: http://localhost:$PORT/api/projects"
echo "  • Stats: http://localhost:$PORT/api/projects (POST)"
echo ""

# Start Next.js development server
npm run dev -- --port $PORT