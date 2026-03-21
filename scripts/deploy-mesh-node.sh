#!/usr/bin/env bash
# deploy-mesh-node.sh — Deploy Symthaea consciousness + Mycelix mesh to a remote node
#
# Usage:
#   ./scripts/deploy-mesh-node.sh <target-host> [options]
#
# Examples:
#   ./scripts/deploy-mesh-node.sh pi@mesh-node-1.local
#   ./scripts/deploy-mesh-node.sh pi@192.168.1.50 --transport meshtastic
#   ./scripts/deploy-mesh-node.sh pi@garden-node.local --dim 4096 --hz 10
#
# Options:
#   --transport <type>   Transport: meshtastic|espnow|reticulum|loopback (default: meshtastic)
#   --dim <n>            HDC dimension (default: 8192)
#   --hz <n>             Target cycle rate (default: 15)
#   --arch <target>      Cross-compile target (default: aarch64-unknown-linux-gnu)
#   --build-only         Build but don't deploy
#   --no-build           Deploy pre-built binaries without rebuilding
#
# Prerequisites:
#   - nix develop environment active (provides cross-compiler)
#   - SSH key access to target host
#   - Target runs Linux (RPi OS, NixOS, Ubuntu, etc.)

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_HOST="${1:?Usage: $0 <target-host> [options]}"
shift

TRANSPORT="meshtastic"
DIM=8192
HZ=15
ARCH="aarch64-unknown-linux-gnu"
BUILD_ONLY=false
NO_BUILD=false
DEPLOY_DIR="/opt/symthaea-mesh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --transport) TRANSPORT="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --hz) HZ="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --no-build) NO_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/symthaea/target/$ARCH/release"

echo "═══════════════════════════════════════════════════════════════"
echo "  Symthaea Mesh Node Deployment"
echo "═══════════════════════════════════════════════════════════════"
echo "  Target:    $TARGET_HOST"
echo "  Transport: $TRANSPORT"
echo "  HDC dim:   $DIM"
echo "  Target Hz: $HZ"
echo "  Arch:      $ARCH"
echo "  Deploy to: $DEPLOY_DIR"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Build binaries
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$NO_BUILD" == "false" ]]; then
    echo ""
    echo "Building spore-mesh-daemon for $ARCH..."

    # Determine transport feature flag
    TRANSPORT_FEATURES=""
    case "$TRANSPORT" in
        meshtastic) TRANSPORT_FEATURES="--features meshtastic" ;;
        espnow)     TRANSPORT_FEATURES="--features espnow" ;;
        reticulum)  TRANSPORT_FEATURES="--features reticulum" ;;
        mqtt)       TRANSPORT_FEATURES="--features mqtt" ;;
        loopback)   TRANSPORT_FEATURES="" ;;
        *) echo "Unknown transport: $TRANSPORT"; exit 1 ;;
    esac

    # Build consciousness daemon
    (cd "$ROOT_DIR/symthaea" && \
        cargo build --release --target "$ARCH" \
            -p symthaea-spore --bin spore-mesh-daemon)
    echo "  ✓ spore-mesh-daemon built"

    # Build mesh-bridge
    (cd "$ROOT_DIR/mycelix-workspace/mesh-bridge" && \
        cargo build --release --target "$ARCH" $TRANSPORT_FEATURES)
    echo "  ✓ mesh-bridge built"

    echo ""
    echo "Binary sizes:"
    ls -lh "$BUILD_DIR/spore-mesh-daemon" 2>/dev/null || echo "  (spore-mesh-daemon not found at $BUILD_DIR)"
    ls -lh "$ROOT_DIR/mycelix-workspace/mesh-bridge/target/$ARCH/release/mesh-bridge" 2>/dev/null || echo "  (mesh-bridge: check target dir)"
fi

if [[ "$BUILD_ONLY" == "true" ]]; then
    echo "Build complete (--build-only). Skipping deployment."
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Deploy to target
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "Deploying to $TARGET_HOST:$DEPLOY_DIR..."

# Create deployment directory
ssh "$TARGET_HOST" "sudo mkdir -p $DEPLOY_DIR/bin $DEPLOY_DIR/config $DEPLOY_DIR/data"

# Copy binaries
scp "$BUILD_DIR/spore-mesh-daemon" "$TARGET_HOST:$DEPLOY_DIR/bin/" 2>/dev/null || \
    echo "  ⚠ spore-mesh-daemon not found — run without --no-build first"

BRIDGE_BIN="$ROOT_DIR/mycelix-workspace/mesh-bridge/target/$ARCH/release/mesh-bridge"
scp "$BRIDGE_BIN" "$TARGET_HOST:$DEPLOY_DIR/bin/" 2>/dev/null || \
    echo "  ⚠ mesh-bridge not found — run without --no-build first"

# Generate environment config
cat <<EOF | ssh "$TARGET_HOST" "sudo tee $DEPLOY_DIR/config/mesh-node.env > /dev/null"
# Symthaea Mesh Node Configuration
# Generated: $(date -Iseconds)

# Consciousness daemon
SPORE_DIM=$DIM
SPORE_HZ=$HZ
SPORE_LAYERS=2
SPORE_NEURONS=32
SPORE_PHI_INTERVAL=5
SPORE_JSON_TELEMETRY=1
SPORE_TELEMETRY_INTERVAL=50

# Mesh bridge
MESH_TRANSPORT=$TRANSPORT
CONDUCTOR_URL=ws://localhost:8888
POLL_INTERVAL_SECS=30
HEALTH_PORT=9100

# Meshtastic (if applicable)
MESHTASTIC_PORT=/dev/ttyUSB0
MESHTASTIC_BAUD=115200
MESHTASTIC_CHANNEL=0

# ESP-NOW (if applicable)
ESPNOW_PORT=/dev/ttyACM0
ESPNOW_BAUD=921600

# Reticulum (if applicable)
RETICULUM_HOST=127.0.0.1
RETICULUM_PORT=37428

# Encryption (generate a shared key for your mesh)
# MESH_ENCRYPTION_KEY=<64 hex chars>
EOF
echo "  ✓ Configuration deployed"

# Generate systemd service files
cat <<'EOF' | ssh "$TARGET_HOST" "sudo tee /etc/systemd/system/spore-mesh-daemon.service > /dev/null"
[Unit]
Description=Symthaea Spore Consciousness Daemon
After=network.target
Wants=mesh-bridge.service

[Service]
Type=simple
EnvironmentFile=/opt/symthaea-mesh/config/mesh-node.env
ExecStart=/opt/symthaea-mesh/bin/spore-mesh-daemon
WorkingDirectory=/opt/symthaea-mesh/data
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
MemoryMax=512M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

cat <<'EOF' | ssh "$TARGET_HOST" "sudo tee /etc/systemd/system/mesh-bridge.service > /dev/null"
[Unit]
Description=Mycelix Mesh Bridge
After=network.target
Wants=holochain-conductor.service

[Service]
Type=notify
EnvironmentFile=/opt/symthaea-mesh/config/mesh-node.env
ExecStart=/opt/symthaea-mesh/bin/mesh-bridge
WorkingDirectory=/opt/symthaea-mesh/data
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
WatchdogSec=120
NotifyAccess=main

[Install]
WantedBy=multi-user.target
EOF
echo "  ✓ Systemd services deployed"

# Enable and start services
ssh "$TARGET_HOST" "sudo systemctl daemon-reload && \
    sudo systemctl enable spore-mesh-daemon mesh-bridge && \
    echo '  ✓ Services enabled'"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Start services:"
echo "    ssh $TARGET_HOST 'sudo systemctl start spore-mesh-daemon mesh-bridge'"
echo ""
echo "  Check status:"
echo "    ssh $TARGET_HOST 'sudo systemctl status spore-mesh-daemon'"
echo "    ssh $TARGET_HOST 'curl -s localhost:9100/health | jq'"
echo ""
echo "  View consciousness:"
echo "    ssh $TARGET_HOST 'journalctl -u spore-mesh-daemon -f'"
echo ""
echo "  View mesh traffic:"
echo "    ssh $TARGET_HOST 'journalctl -u mesh-bridge -f'"
echo ""
