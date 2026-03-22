#!/usr/bin/env bash
# Terra Atlas - Clean Dev Restart Script
# Ensures fresh cache and proper server restart

set -e

echo "🧹 Cleaning development environment..."

# Kill existing dev server
echo "  → Stopping existing dev server..."
pkill -f "next dev" 2>/dev/null || true
sleep 1

# Clear Next.js cache
echo "  → Clearing .next cache..."
rm -rf .next

# Clear node_modules cache (optional, uncomment if needed)
# echo "  → Clearing node_modules/.cache..."
# rm -rf node_modules/.cache

# Clear browser cache instruction
echo ""
echo "📱 Browser Cache:"
echo "  → Chrome/Edge: Ctrl+Shift+Delete → Clear cached images/files"
echo "  → Firefox: Ctrl+Shift+Delete → Cache"
echo "  → Or use: Ctrl+F5 for hard refresh"
echo ""

# Start dev server
echo "🚀 Starting fresh dev server..."
npm run dev

# Note: This will run in foreground. Use Ctrl+C to stop.
