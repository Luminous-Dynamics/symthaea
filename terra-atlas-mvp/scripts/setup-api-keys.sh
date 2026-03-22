#!/bin/bash

# Terra Atlas API Key Setup Helper
# This script helps users register for required API keys

echo "🌍 Terra Atlas API Key Setup"
echo "============================"
echo ""
echo "This guide will help you register for the API keys needed"
echo "to fetch real environmental data for Terra Atlas."
echo ""

# Function to open URL
open_url() {
    if command -v xdg-open > /dev/null; then
        xdg-open "$1"
    elif command -v open > /dev/null; then
        open "$1"
    elif command -v start > /dev/null; then
        start "$1"
    else
        echo "Please open this URL manually: $1"
    fi
}

# NASA FIRMS
echo "1️⃣  NASA FIRMS (Fire Information)"
echo "-----------------------------------"
echo "NASA FIRMS provides real-time active fire data from satellites."
echo ""
echo "📝 Registration Steps:"
echo "  1. Visit: https://firms.modaps.eosdis.nasa.gov/api/area/"
echo "  2. Click 'Get MAP key'"
echo "  3. Fill out the registration form"
echo "  4. Check your email for the API key"
echo ""
read -p "Press Enter to open NASA FIRMS registration page..."
open_url "https://firms.modaps.eosdis.nasa.gov/api/area/"
echo ""

# OpenWeatherMap
echo "2️⃣  OpenWeatherMap (Weather Data)"
echo "-----------------------------------"
echo "OpenWeatherMap provides global weather data and forecasts."
echo ""
echo "📝 Registration Steps:"
echo "  1. Visit: https://openweathermap.org/api"
echo "  2. Click 'Sign Up' (if you don't have an account)"
echo "  3. Go to 'API keys' in your account"
echo "  4. Copy your default API key or create a new one"
echo ""
echo "💡 Free tier includes:"
echo "  - 60 calls/minute"
echo "  - 1,000,000 calls/month"
echo "  - Current weather, forecasts, historical data"
echo ""
read -p "Press Enter to open OpenWeatherMap registration page..."
open_url "https://openweathermap.org/api"
echo ""

# USGS Earthquakes
echo "3️⃣  USGS Earthquake Data"
echo "------------------------"
echo "✅ No API key required!"
echo "USGS provides free, open access to earthquake data."
echo "Documentation: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php"
echo ""

# Optional: Mapbox
echo "4️⃣  Mapbox (Optional - Enhanced Maps)"
echo "--------------------------------------"
echo "Mapbox provides advanced map features and styles."
echo ""
echo "📝 Registration Steps:"
echo "  1. Visit: https://www.mapbox.com/"
echo "  2. Click 'Sign up' for a free account"
echo "  3. Go to your account dashboard"
echo "  4. Copy your default public token"
echo ""
echo "💡 Free tier includes:"
echo "  - 50,000 map loads/month"
echo "  - All map styles"
echo "  - Geocoding, directions, and more"
echo ""
read -p "Press Enter to open Mapbox registration page (or skip)..."
open_url "https://www.mapbox.com/"
echo ""

# Create .env.local file
echo "5️⃣  Setting up your .env.local file"
echo "------------------------------------"

if [ -f .env.local ]; then
    echo "⚠️  .env.local already exists. Backing up to .env.local.backup"
    cp .env.local .env.local.backup
fi

echo "Creating .env.local file..."
cat > .env.local << 'EOF'
# Terra Atlas Environment Variables
# Add your API keys below

# NASA FIRMS API
FIRMS_API_KEY=

# OpenWeatherMap API
OPENWEATHER_API_KEY=

# Mapbox (Optional)
NEXT_PUBLIC_MAPBOX_TOKEN=

# Environment
NODE_ENV=development
EOF

echo "✅ Created .env.local file"
echo ""
echo "📝 Next Steps:"
echo "1. Edit .env.local and add your API keys"
echo "2. Run: python3 scripts/fetch-real-data.py"
echo "3. Restart the dev server: npm run dev"
echo ""
echo "Your .env.local file is ready at:"
echo "  $(pwd)/.env.local"
echo ""
echo "🔒 Security Note:"
echo ".env.local is already in .gitignore and will not be committed"
echo ""
echo "Happy mapping! 🌍"