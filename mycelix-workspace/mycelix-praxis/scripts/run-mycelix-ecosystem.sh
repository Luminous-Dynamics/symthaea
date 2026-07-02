#!/usr/bin/env bash
# Run Mycelix Ecosystem - Shared Conductor for All Apps
# Usage: ./scripts/run-mycelix-ecosystem.sh

set -euo pipefail

echo "🌐 Mycelix Ecosystem - Shared Conductor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Create databases directory if it doesn't exist
mkdir -p conductor/databases

echo "🔧 Configuration:"
echo "  • Admin Interface: localhost:4444"
echo "  • App Interface: localhost:8888"
echo "  • Database: ./conductor/databases"
echo "  • Config: conductor/mycelix-ecosystem-conductor.yaml"
echo ""

# Check if DNA bundles exist
EDUNET_DNA="dna/edunet.dna"
FL_DNA="../mycelix-core/0TML/mycelix_fl/holochain/dna/fl.dna"
CORE_DNA="../mycelix-core/0TML/core.dna"

echo "📦 Checking DNA bundles..."
if [ ! -f "$EDUNET_DNA" ]; then
    echo "⚠️  EduNet DNA not found at $EDUNET_DNA"
    echo "   Building EduNet DNA..."
    ./scripts/build-dna.sh
fi

if [ -f "$FL_DNA" ]; then
    echo "  ✓ FL DNA found"
fi

if [ -f "$CORE_DNA" ]; then
    echo "  ✓ Core DNA found"
fi

echo ""
echo "🚀 Starting shared conductor..."
echo ""
echo "To install apps, run (in another terminal):"
echo "  hc app install --admin-port 4444 --app-id mycelix-edunet --bundle $EDUNET_DNA"
echo ""
echo "Press Ctrl+C to stop the conductor"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run shared conductor
holochain -c conductor/mycelix-ecosystem-conductor.yaml
