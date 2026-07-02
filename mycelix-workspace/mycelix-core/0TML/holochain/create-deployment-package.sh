#!/usr/bin/env bash
# Create Holochain Deployment Package for Production Server
# Run this locally to create a tarball ready to transfer to your server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$SCRIPT_DIR/deployment-package"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
PACKAGE_NAME="holochain-zerotrustml-deploy-${TIMESTAMP}.tar.gz"

echo "📦 Creating Holochain Deployment Package"
echo "========================================="
echo ""

# Clean and create deployment directory
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

echo "✅ Collecting files..."

# Copy essential files
cp "$SCRIPT_DIR/conductor-config-minimal.yaml" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/start-conductor.sh" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/DEPLOY_TO_SERVER.md" "$DEPLOY_DIR/README.md"
cp "$SCRIPT_DIR/PRODUCTION_DEPLOYMENT.md" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/CONFIG_SOLUTION.md" "$DEPLOY_DIR/"

# Copy bundles
mkdir -p "$DEPLOY_DIR/bundles"
if [ -f "$SCRIPT_DIR/dna/zerotrustml.dna" ]; then
    cp "$SCRIPT_DIR/dna/zerotrustml.dna" "$DEPLOY_DIR/bundles/"
    echo "  ✓ DNA bundle ($(du -h "$SCRIPT_DIR/dna/zerotrustml.dna" | cut -f1))"
fi

if [ -f "$SCRIPT_DIR/happ/zerotrustml.happ" ]; then
    cp "$SCRIPT_DIR/happ/zerotrustml.happ" "$DEPLOY_DIR/bundles/"
    echo "  ✓ hApp bundle ($(du -h "$SCRIPT_DIR/happ/zerotrustml.happ" | cut -f1))"
fi

# Create quick setup script for server
cat > "$DEPLOY_DIR/quick-setup.sh" << 'SETUP_EOF'
#!/usr/bin/env bash
# Quick Setup Script for Holochain on Server
# Run this after extracting the package

set -e

echo "🚀 Holochain ZeroTrustML Quick Setup"
echo "================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo "⚠️  Please don't run as root. Use sudo for individual commands."
   exit 1
fi

# Create user
echo "Creating holochain user..."
sudo useradd -r -s /bin/bash -d /var/lib/holochain holochain 2>/dev/null || echo "User already exists"

# Create directories
echo "Creating directories..."
sudo mkdir -p /opt/zerotrustml/holochain
sudo mkdir -p /var/lib/holochain/zerotrustml
sudo mkdir -p /var/log/holochain

# Copy files
echo "Copying files..."
sudo cp conductor-config-minimal.yaml /opt/zerotrustml/holochain/
sudo cp start-conductor.sh /opt/zerotrustml/holochain/
sudo chmod +x /opt/zerotrustml/holochain/start-conductor.sh

# Copy bundles
if [ -d "bundles" ]; then
    sudo mkdir -p /opt/zerotrustml/holochain/dna /opt/zerotrustml/holochain/happ
    [ -f bundles/zerotrustml.dna ] && sudo cp bundles/zerotrustml.dna /opt/zerotrustml/holochain/dna/
    [ -f bundles/zerotrustml.happ ] && sudo cp bundles/zerotrustml.happ /opt/zerotrustml/holochain/happ/
fi

# Set permissions
echo "Setting permissions..."
sudo chown -R holochain:holochain /opt/zerotrustml/holochain
sudo chown -R holochain:holochain /var/lib/holochain
sudo chown -R holochain:holochain /var/log/holochain

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install Holochain (see README.md)"
echo "2. Update config paths in /opt/zerotrustml/holochain/conductor-config-minimal.yaml"
echo "3. Create systemd service (see README.md)"
echo "4. Start service: sudo systemctl start holochain-zerotrustml"
echo ""
echo "See README.md for complete deployment guide."
SETUP_EOF

chmod +x "$DEPLOY_DIR/quick-setup.sh"

# Create verification script
cat > "$DEPLOY_DIR/verify-deployment.sh" << 'VERIFY_EOF'
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
VERIFY_EOF

chmod +x "$DEPLOY_DIR/verify-deployment.sh"

# Create systemd service template
cat > "$DEPLOY_DIR/holochain-zerotrustml.service" << 'SERVICE_EOF'
[Unit]
Description=Holochain ZeroTrustML Conductor
Documentation=https://developer.holochain.org/
After=network.target

[Service]
Type=simple
User=holochain
Group=holochain
WorkingDirectory=/opt/zerotrustml/holochain

# IMPORTANT: Update this path to match your installation
# If using Nix: run "which holochain" to find the correct path
# If using Cargo: /home/holochain/.cargo/bin/holochain
ExecStart=/usr/bin/holochain --config-path /opt/zerotrustml/holochain/conductor-config-minimal.yaml

Restart=always
RestartSec=10
StartLimitBurst=5
StartLimitInterval=300

StandardOutput=append:/var/log/holochain/conductor.log
StandardError=append:/var/log/holochain/conductor.error.log

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/holochain/zerotrustml
ReadWritePaths=/var/log/holochain

LimitNOFILE=65535
MemoryMax=2G

[Install]
WantedBy=multi-user.target
SERVICE_EOF

echo "  ✓ Scripts created"
echo "  ✓ Systemd service template"
echo ""

# Create tarball
echo "Creating tarball..."
cd "$(dirname "$DEPLOY_DIR")"
tar czf "$PACKAGE_NAME" "$(basename "$DEPLOY_DIR")"
PACKAGE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment package created!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Package: $PACKAGE_NAME"
echo "Size: $PACKAGE_SIZE"
echo "Location: $(dirname "$DEPLOY_DIR")/$PACKAGE_NAME"
echo ""
echo "📤 Transfer to server:"
echo "  scp $PACKAGE_NAME user@your-server:/tmp/"
echo ""
echo "📥 On server, extract and run:"
echo "  cd /tmp"
echo "  tar xzf $PACKAGE_NAME"
echo "  cd deployment-package"
echo "  ./quick-setup.sh"
echo "  # Then follow README.md for complete setup"
echo ""
echo "Contents:"
tar tzf "$PACKAGE_NAME" | head -20
if [ $(tar tzf "$PACKAGE_NAME" | wc -l) -gt 20 ]; then
    echo "  ... and $(($(tar tzf "$PACKAGE_NAME" | wc -l) - 20)) more files"
fi
echo ""
echo "🎉 Ready to deploy!"
