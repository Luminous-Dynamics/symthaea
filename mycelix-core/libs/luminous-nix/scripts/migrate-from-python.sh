#!/usr/bin/env bash
# Migration script: Python ask-nix → Rust ask-nix
#
# This script:
# 1. Backs up the old Python version
# 2. Installs the new Rust binary
# 3. Updates symlinks
#
# Usage: ./migrate-from-python.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_CRATE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_DIR="/srv/luminous-dynamics/11-meta-consciousness/luminous-nix"
INSTALL_DIR="$HOME/.local/bin"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Luminous Nix Migration: Python → Rust (Symthaea HDC)      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Build Rust version
echo "→ Building Rust version..."
cd "$RUST_CRATE_DIR"
cargo build --release
echo "  ✓ Built: target/release/ask-nix"
echo ""

# Step 2: Backup Python version
if [ -f "$PYTHON_DIR/bin/ask-nix" ]; then
    echo "→ Backing up Python version..."
    BACKUP_DIR="$PYTHON_DIR/bin/.backup-$(date +%Y%m%d)"
    mkdir -p "$BACKUP_DIR"
    cp "$PYTHON_DIR/bin/ask-nix" "$BACKUP_DIR/ask-nix-python"
    cp "$PYTHON_DIR/bin/nix-tui" "$BACKUP_DIR/nix-tui-python" 2>/dev/null || true
    echo "  ✓ Backed up to: $BACKUP_DIR"
else
    echo "→ No Python version found, skipping backup"
fi
echo ""

# Step 3: Install Rust binary
echo "→ Installing Rust binary..."
mkdir -p "$INSTALL_DIR"
cp "$RUST_CRATE_DIR/target/release/ask-nix" "$INSTALL_DIR/ask-nix"
chmod +x "$INSTALL_DIR/ask-nix"
echo "  ✓ Installed: $INSTALL_DIR/ask-nix"
echo ""

# Step 4: Create convenience symlinks
echo "→ Creating symlinks..."
ln -sf "$INSTALL_DIR/ask-nix" "$INSTALL_DIR/luminous-nix" 2>/dev/null || true
echo "  ✓ luminous-nix → ask-nix"
echo ""

# Step 5: Verify installation
echo "→ Verifying installation..."
if command -v ask-nix &> /dev/null; then
    VERSION=$(ask-nix --version 2>/dev/null || echo "unknown")
    echo "  ✓ ask-nix is available: $VERSION"
else
    echo "  ⚠ ask-nix not in PATH. Add $INSTALL_DIR to your PATH:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Migration Complete!                                        ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  New binary: $INSTALL_DIR/ask-nix"
echo "║  Old backup: $BACKUP_DIR (if existed)"
echo "║                                                              ║"
echo "║  Try: ask-nix \"search firefox\"                             ║"
echo "║       ask-nix --help                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
