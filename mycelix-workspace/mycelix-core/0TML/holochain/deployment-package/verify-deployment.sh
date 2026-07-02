#!/usr/bin/env bash
# Verify Holochain Deployment
# Run this on server after deployment

echo "🔍 Holochain Deployment Verification"
echo "====================================="
echo ""

PASS=0
FAIL=0

# Check holochain installed
echo -n "Checking holochain binary... "
if command -v holochain &> /dev/null; then
    VERSION=$(holochain --version 2>&1 | head -1)
    echo "✅ $VERSION"
    ((PASS++))
else
    echo "❌ Not found"
    ((FAIL++))
fi

# Check service exists
echo -n "Checking systemd service... "
if systemctl list-unit-files | grep -q holochain-zerotrustml; then
    echo "✅ Exists"
    ((PASS++))
else
    echo "❌ Not found"
    ((FAIL++))
fi

# Check service status
echo -n "Checking service running... "
if systemctl is-active --quiet holochain-zerotrustml; then
    echo "✅ Running"
    ((PASS++))
else
    echo "❌ Not running"
    ((FAIL++))
fi

# Check port listening
echo -n "Checking port 8888... "
if ss -tlnp 2>/dev/null | grep -q :8888; then
    echo "✅ Listening"
    ((PASS++))
else
    echo "❌ Not listening"
    ((FAIL++))
fi

# Check config file
echo -n "Checking config file... "
if [ -f /opt/zerotrustml/holochain/conductor-config-minimal.yaml ]; then
    echo "✅ Exists"
    ((PASS++))
else
    echo "❌ Not found"
    ((FAIL++))
fi

# Check data directory
echo -n "Checking data directory... "
if [ -d /var/lib/holochain/zerotrustml ]; then
    echo "✅ Exists"
    ((PASS++))
else
    echo "❌ Not found"
    ((FAIL++))
fi

# Check bundles
echo -n "Checking hApp bundle... "
if [ -f /opt/zerotrustml/holochain/happ/zerotrustml.happ ]; then
    SIZE=$(du -h /opt/zerotrustml/holochain/happ/zerotrustml.happ | cut -f1)
    echo "✅ Exists ($SIZE)"
    ((PASS++))
else
    echo "❌ Not found"
    ((FAIL++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: ✅ $PASS passed, ❌ $FAIL failed"

if [ $FAIL -eq 0 ]; then
    echo "🎉 Deployment verified successfully!"
    exit 0
else
    echo "⚠️  Some checks failed. See README.md for troubleshooting."
    exit 1
fi
