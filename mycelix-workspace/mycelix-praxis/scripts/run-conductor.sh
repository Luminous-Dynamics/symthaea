#!/usr/bin/env bash
# Run Holochain conductor for development
# Usage: ./scripts/run-conductor.sh

set -euo pipefail

echo "🎓 Mycelix EduNet - Development Conductor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Create databases directory if it doesn't exist
mkdir -p conductor/databases

# Check if DNA exists
if [ ! -f dna/edunet.dna ]; then
    echo "⚠️  DNA bundle not found at dna/edunet.dna"
    echo ""
    echo "Building DNA first..."
    ./scripts/build-dna.sh
fi

echo "🚀 Starting Holochain conductor..."
echo ""
echo "Configuration:"
echo "  • Admin port: 4444"
echo "  • App port: 8888"
echo "  • Database: ./conductor/databases"
echo ""
echo "Press Ctrl+C to stop the conductor"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run conductor with development configuration
holochain -c conductor/dev-conductor-config.yaml
