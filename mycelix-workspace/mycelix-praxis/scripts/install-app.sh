#!/usr/bin/env bash
# Install Mycelix EduNet app to running conductor
# Usage: ./scripts/install-app.sh

set -euo pipefail

echo "🎓 Mycelix EduNet - App Installation Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if DNA exists
if [ ! -f dna/edunet.dna ]; then
    echo "❌ Error: DNA bundle not found at dna/edunet.dna"
    echo ""
    echo "Please run: ./scripts/build-dna.sh"
    exit 1
fi

APP_ID="${1:-mycelix-edunet}"
ADMIN_PORT="${2:-4444}"

echo "Installing app..."
echo "  • App ID: $APP_ID"
echo "  • Admin port: $ADMIN_PORT"
echo ""

# Create app bundle configuration
cat > /tmp/edunet-app-bundle.json << EOF
{
  "app_name": "$APP_ID",
  "agent_key": null,
  "roles": [
    {
      "name": "edunet",
      "dna": {
        "path": "$(pwd)/dna/edunet.dna"
      }
    }
  ]
}
EOF

# Install app using hc CLI
echo "📦 Installing app to conductor..."
hc app install --admin-port "$ADMIN_PORT" /tmp/edunet-app-bundle.json

echo ""
echo "✅ App installed successfully!"
echo ""
echo "Next steps:"
echo "  • Start web client: cd apps/web && npm run dev"
echo "  • Connect to conductor via WebSocket: ws://localhost:8888"
echo ""
