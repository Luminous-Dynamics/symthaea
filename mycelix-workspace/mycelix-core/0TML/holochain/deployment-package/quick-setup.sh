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
