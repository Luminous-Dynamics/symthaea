#!/bin/bash

# Terra Atlas Discovery API Deployment Script
# Deploys the Discovery API to production on Vercel

echo "🚀 Deploying Terra Atlas Discovery API"
echo "======================================"
echo

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Not in Terra Atlas project directory"
    exit 1
fi

# 1. Test the API endpoints locally first
echo "📋 Testing API endpoints locally..."
echo "-----------------------------------"

# Start dev server in background for testing
npm run dev > /dev/null 2>&1 &
DEV_PID=$!
echo "Dev server started (PID: $DEV_PID)"

# Wait for server to start
sleep 5

# Test each endpoint
echo
echo "Testing Discovery API endpoints:"

# Test main endpoint
echo -n "  /api/discovery - "
curl -s http://localhost:3000/api/discovery > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test similar projects
echo -n "  /api/discovery/similar - "
curl -s "http://localhost:3000/api/discovery/similar?type=solar&capacity=100&state=TX" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test transmission
echo -n "  /api/discovery/transmission - "
curl -s "http://localhost:3000/api/discovery/transmission?lat=31.9686&lng=-99.9018&radius=50" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test corridors
echo -n "  /api/discovery/corridors - "
curl -s "http://localhost:3000/api/discovery/corridors?state=TX&capacity=500" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test queue intelligence
echo -n "  /api/discovery/queue-intelligence - "
curl -s "http://localhost:3000/api/discovery/queue-intelligence?region=ERCOT&position=847" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Kill dev server
kill $DEV_PID 2>/dev/null
echo
echo "✅ Local tests complete"
echo

# 2. Build the project
echo "🔨 Building project..."
echo "---------------------"
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Fix errors before deploying."
    exit 1
fi

echo "✅ Build successful"
echo

# 3. Deploy to Vercel
echo "🌐 Deploying to Vercel..."
echo "------------------------"

# Deploy to production
npx vercel --prod

echo
echo "✅ Deployment complete!"
echo

# 4. Test production endpoints
echo "🧪 Testing production endpoints..."
echo "---------------------------------"
echo

PROD_URL="https://atlas.luminousdynamics.io"

echo "Testing Discovery API on production:"

# Test main endpoint
echo -n "  $PROD_URL/api/discovery - "
curl -s $PROD_URL/api/discovery > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test similar projects
echo -n "  $PROD_URL/api/discovery/similar - "
curl -s "$PROD_URL/api/discovery/similar?type=solar&capacity=100&state=TX" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test transmission
echo -n "  $PROD_URL/api/discovery/transmission - "
curl -s "$PROD_URL/api/discovery/transmission?lat=31.9686&lng=-99.9018&radius=50" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test corridors
echo -n "  $PROD_URL/api/discovery/corridors - "
curl -s "$PROD_URL/api/discovery/corridors?state=TX&capacity=500" > /dev/null && echo "✅ OK" || echo "❌ Failed"

# Test queue intelligence
echo -n "  $PROD_URL/api/discovery/queue-intelligence - "
curl -s "$PROD_URL/api/discovery/queue-intelligence?region=ERCOT&position=847" > /dev/null && echo "✅ OK" || echo "❌ Failed"

echo
echo "=========================================="
echo "🎉 Terra Atlas Discovery API is LIVE!"
echo "=========================================="
echo
echo "📚 API Documentation: $PROD_URL/api/discovery"
echo "🌐 Production URL: $PROD_URL"
echo
echo "📊 Available Endpoints:"
echo "  • /api/discovery/similar - Find similar successful projects"
echo "  • /api/discovery/transmission - Find available transmission"
echo "  • /api/discovery/corridors - Find corridor opportunities"
echo "  • /api/discovery/queue-intelligence - Analyze FERC queue"
echo
echo "🔗 Share with developers:"
echo "  Documentation: https://github.com/terra-atlas/discovery-api"
echo "  Contact: developers@terraatlas.com"
echo
echo "💡 Next Steps:"
echo "  1. Share API with stuck FERC developers"
echo "  2. Collect feedback on usefulness"
echo "  3. Add more real-time data sources"
echo "  4. Build developer dashboard"
echo

# Create API usage example
cat > discovery-api-example.js << 'EOF'
// Terra Atlas Discovery API - Quick Start Example
const TERRA_ATLAS_API = 'https://atlas.luminousdynamics.io/api/discovery';

// Find similar successful solar projects in Texas
async function findSimilarProjects() {
  const response = await fetch(`${TERRA_ATLAS_API}/similar?type=solar&capacity=100&state=TX`);
  const data = await response.json();
  
  console.log('Similar Projects:', data.similarProjects);
  console.log('Success Rate:', data.insights.success_rate);
  console.log('Average Cost:', data.insights.average_interconnection_cost);
}

// Find available transmission capacity near a location
async function findTransmission(lat, lng) {
  const response = await fetch(`${TERRA_ATLAS_API}/transmission?lat=${lat}&lng=${lng}&radius=50`);
  const data = await response.json();
  
  console.log('Nearby Lines:', data.nearbyTransmission);
  console.log('Substations:', data.substations);
}

// Run examples
findSimilarProjects();
findTransmission(31.9686, -99.9018); // West Texas coordinates
EOF

echo "📝 Created discovery-api-example.js for quick testing"
echo
echo "✨ Discovery API deployment complete!"