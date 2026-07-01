#!/bin/sh
# Sovereign Scan — one-liner installer
# Usage: curl -sL install.nixforhumanity.org/scan.sh | sh
#
# Downloads and runs the system scanner, then opens the browser.
# No installation needed — runs once and exits.

set -e

REPO="Luminous-Dynamics/nixforhumanity"
BINARY="sovereign-scan"

# Detect OS and arch
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$ARCH" in
    x86_64|amd64) ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
    linux) TARGET="${BINARY}-${ARCH}-unknown-linux-gnu" ;;
    darwin) TARGET="${BINARY}-${ARCH}-apple-darwin" ;;
    *) echo "Unsupported OS: $OS (use Windows installer instead)"; exit 1 ;;
esac

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   Sovereign Scan — NixOS Migration Aid   ║"
echo "  ║   install.nixforhumanity.org             ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# Download
TMPDIR="${TMPDIR:-/tmp}"
DEST="$TMPDIR/$BINARY"

echo "Downloading scanner for $OS/$ARCH..."
if command -v curl >/dev/null 2>&1; then
    curl -sL "https://github.com/${REPO}/releases/latest/download/${TARGET}" -o "$DEST"
elif command -v wget >/dev/null 2>&1; then
    wget -q "https://github.com/${REPO}/releases/latest/download/${TARGET}" -O "$DEST"
else
    echo "Error: curl or wget required"
    exit 1
fi

chmod +x "$DEST"

echo "Scanning your system..."
echo ""

# Run in serve mode (browser connects via WebSocket)
if [ "$1" = "--json" ]; then
    exec "$DEST"
elif [ "$1" = "--pretty" ]; then
    exec "$DEST" --pretty
else
    # Default: serve mode — open browser
    "$DEST" --serve &
    SCAN_PID=$!
    sleep 1

    # Try to open browser
    URL="https://install.nixforhumanity.org"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" 2>/dev/null || true
    elif command -v open >/dev/null 2>&1; then
        open "$URL" 2>/dev/null || true
    else
        echo "Open $URL in your browser"
    fi

    echo "Scanner running. Browser should open automatically."
    echo "Press Ctrl+C to stop."
    wait $SCAN_PID
fi
