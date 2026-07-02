#!/usr/bin/env bash
# Install mycelix-space-screener as a systemd service
# Usage: ./install.sh [--enable]
#
# Builds the release binary, installs the service unit, and optionally enables it.
# Run with --enable to start immediately and persist across reboots.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_FILE="$SCRIPT_DIR/mycelix-space-screener.service"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "Building screening daemon (release)..."
cd "$SPACE_DIR"
CARGO_TARGET_DIR=target-check cargo build --release -p mycelix-space-screener

BINARY="$SPACE_DIR/target-check/release/mycelix-space-screener"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi
echo "Binary: $BINARY ($(du -h "$BINARY" | cut -f1))"

# Install as user service (no sudo needed)
mkdir -p "$SYSTEMD_DIR"
cp "$SERVICE_FILE" "$SYSTEMD_DIR/mycelix-space-screener.service"

# Patch the service file for user mode (remove User/Group, adjust paths)
sed -i '/^User=/d' "$SYSTEMD_DIR/mycelix-space-screener.service"
sed -i '/^Group=/d' "$SYSTEMD_DIR/mycelix-space-screener.service"
sed -i 's|^WantedBy=multi-user.target|WantedBy=default.target|' "$SYSTEMD_DIR/mycelix-space-screener.service"

systemctl --user daemon-reload
echo "Service installed at $SYSTEMD_DIR/mycelix-space-screener.service"

if [ "${1:-}" = "--enable" ]; then
    systemctl --user enable --now mycelix-space-screener.service
    echo "Service enabled and started."
    echo "Check status: systemctl --user status mycelix-space-screener"
    echo "View logs:    journalctl --user -u mycelix-space-screener -f"
else
    echo ""
    echo "To start:     systemctl --user start mycelix-space-screener"
    echo "To enable:    systemctl --user enable mycelix-space-screener"
    echo "To check:     systemctl --user status mycelix-space-screener"
    echo "To logs:      journalctl --user -u mycelix-space-screener -f"
    echo ""
    echo "Or run directly:"
    echo "  $BINARY --celestrak-url 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle' --protected-objects 25544 --once"
fi
