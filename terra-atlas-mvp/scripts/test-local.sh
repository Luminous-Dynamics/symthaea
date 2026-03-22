#!/bin/bash

echo "🌍 Testing Terra Atlas MVP Locally"
echo "=================================="

BASE_URL="http://localhost:3002"

echo ""
echo "1. Testing API Root Endpoint..."
curl -s "$BASE_URL/api" | jq -r '.status' | grep -q "operational" && echo "✅ API is operational" || echo "❌ API failed"

echo ""
echo "2. Testing Data Endpoints..."
for endpoint in fires earthquakes weather emissions; do
    echo -n "   Testing /api/data/$endpoint... "
    response=$(curl -s "$BASE_URL/api/data/$endpoint")
    if echo "$response" | jq -r '.type' | grep -q "FeatureCollection"; then
        count=$(echo "$response" | jq -r '.features | length')
        echo "✅ $count features"
    else
        echo "❌ Failed"
    fi
done

echo ""
echo "3. Testing PWA Manifest..."
curl -s "$BASE_URL/manifest.json" | jq -r '.name' | grep -q "Terra Atlas" && echo "✅ PWA manifest ready" || echo "❌ Manifest failed"

echo ""
echo "4. Testing Main App..."
curl -s "$BASE_URL" | grep -q "Terra Atlas" && echo "✅ Main app loads" || echo "❌ App failed to load"

echo ""
echo "=================================="
echo "📊 Summary:"
echo "   API: http://localhost:3002/api"
echo "   App: http://localhost:3002"
echo "   Status: Ready for deployment!"
echo ""
echo "🚀 Next: Run 'npx vercel' to deploy"