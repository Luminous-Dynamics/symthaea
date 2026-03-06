#!/usr/bin/env bash
# Start a local Holochain conductor for development/testing
#
# Usage:
#   ./scripts/start-conductor.sh          # Start conductor
#   ./scripts/start-conductor.sh stop     # Stop conductor
#   ./scripts/start-conductor.sh status   # Check status
#
# Prerequisites:
#   - holochain 0.6.x binary
#   - lair-keystore 0.6.x binary (MUST match the version that initialized the keystore)
#
# Environment variables:
#   HOLOCHAIN_BIN  - Path to holochain binary (auto-detected from nix store or PATH)
#   LAIR_BIN       - Path to lair-keystore binary (auto-detected from nix store or PATH)
#   CONDUCTOR_DIR  - Data directory (default: /tmp/mycelix-conductor)
#   ADMIN_PORT     - Admin websocket port (default: 4444)
#   APP_PORT       - App websocket port (default: 4445)
#   PASSPHRASE     - Lair passphrase (default: test-passphrase)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDUCTOR_DIR="${CONDUCTOR_DIR:-$SCRIPT_DIR/.conductor}"
ADMIN_PORT="${ADMIN_PORT:-4444}"
APP_PORT="${APP_PORT:-4445}"
PASSPHRASE="${PASSPHRASE:-test-passphrase}"
PIDS_FILE="$CONDUCTOR_DIR/pids"

# Auto-detect binaries
find_binary() {
    local name="$1"
    local env_var="$2"
    local nix_pattern="$3"

    # Check env var first
    if [ -n "${!env_var:-}" ] && [ -x "${!env_var}" ]; then
        echo "${!env_var}"; return
    fi

    # Check PATH
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"; return
    fi

    # Search nix store
    local found
    found=$(find /nix/store -maxdepth 2 -name "$name" -path "$nix_pattern" 2>/dev/null | head -1)
    if [ -n "$found" ] && [ -x "$found" ]; then
        echo "$found"; return
    fi

    echo "ERROR: Cannot find $name. Set $env_var or add to PATH." >&2
    return 1
}

HOLOCHAIN_BIN="${HOLOCHAIN_BIN:-$(find_binary holochain HOLOCHAIN_BIN '*/holochain-*/bin/*')}"
LAIR_BIN="${LAIR_BIN:-$(find_binary lair-keystore LAIR_BIN '*/lair-keystore-*/bin/*')}"

stop_conductor() {
    if [ -f "$PIDS_FILE" ]; then
        source "$PIDS_FILE"
        kill "${HC_PID:-0}" 2>/dev/null && echo "Stopped conductor (PID: $HC_PID)" || true
        kill "${LAIR_PID:-0}" 2>/dev/null && echo "Stopped lair (PID: $LAIR_PID)" || true
        rm -f "$PIDS_FILE"
    fi
    # Clean up any orphans
    pkill -f "lair-keystore.*${CONDUCTOR_DIR}" 2>/dev/null || true
    pkill -f "holochain.*${CONDUCTOR_DIR}" 2>/dev/null || true
}

status_conductor() {
    if [ -f "$PIDS_FILE" ]; then
        source "$PIDS_FILE"
        if kill -0 "${HC_PID:-0}" 2>/dev/null; then
            echo "Conductor: RUNNING (PID: $HC_PID, admin: $ADMIN_PORT)"
        else
            echo "Conductor: STOPPED (stale PID file)"
        fi
        if kill -0 "${LAIR_PID:-0}" 2>/dev/null; then
            echo "Lair: RUNNING (PID: $LAIR_PID)"
        else
            echo "Lair: STOPPED"
        fi
    else
        echo "No conductor running (no PID file at $PIDS_FILE)"
    fi
}

case "${1:-start}" in
    stop)
        stop_conductor
        exit 0
        ;;
    status)
        status_conductor
        exit 0
        ;;
    start)
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac

# Stop any existing conductor
stop_conductor
sleep 1

# Initialize if needed
if [ ! -f "$CONDUCTOR_DIR/ks/lair-keystore-config.yaml" ]; then
    echo "Initializing new conductor at $CONDUCTOR_DIR..."
    mkdir -p "$CONDUCTOR_DIR/ks"

    # Init lair keystore
    echo "$PASSPHRASE" | "$LAIR_BIN" --lair-root "$CONDUCTOR_DIR/ks" init --piped

    # Get connection URL
    CONN_URL=$(grep connectionUrl "$CONDUCTOR_DIR/ks/lair-keystore-config.yaml" | awk '{print $2}')

    # Write conductor config
    cat > "$CONDUCTOR_DIR/conductor-config.yaml" <<YAML
data_root_path: $CONDUCTOR_DIR
keystore:
  type: lair_server
  connection_url: $CONN_URL
admin_interfaces:
  - driver:
      type: websocket
      port: $ADMIN_PORT
      allowed_origins: "*"
network:
  bootstrap_url: https://dev-test-bootstrap2.holochain.org/
  signal_url: wss://dev-test-bootstrap2.holochain.org/
  relay_url: wss://dev-test-bootstrap2.holochain.org/
  target_arc_factor: 1
  advanced:
    tx5Transport:
      signalAllowPlainText: true
db_sync_strategy: Resilient
YAML
    echo "Conductor initialized."
else
    echo "Using existing conductor at $CONDUCTOR_DIR"
fi

# Clean stale socket
rm -f "$CONDUCTOR_DIR/ks/socket" "$CONDUCTOR_DIR/ks/pid_file" 2>/dev/null

# Start lair
echo "$PASSPHRASE" | "$LAIR_BIN" --lair-root "$CONDUCTOR_DIR/ks" server --piped \
    > "$CONDUCTOR_DIR/lair.log" 2>&1 &
LAIR_PID=$!
sleep 2

if ! kill -0 $LAIR_PID 2>/dev/null; then
    echo "ERROR: Lair failed to start"
    cat "$CONDUCTOR_DIR/lair.log"
    exit 1
fi
echo "Lair running (PID: $LAIR_PID)"

# Start conductor (lair-keystore must be on PATH for holochain)
export PATH="$(dirname "$LAIR_BIN"):$(dirname "$HOLOCHAIN_BIN"):$PATH"
echo "$PASSPHRASE" | "$HOLOCHAIN_BIN" -c "$CONDUCTOR_DIR/conductor-config.yaml" --piped \
    > "$CONDUCTOR_DIR/conductor.log" 2>&1 &
HC_PID=$!
sleep 4

if ! kill -0 $HC_PID 2>/dev/null; then
    echo "ERROR: Conductor failed to start"
    cat "$CONDUCTOR_DIR/conductor.log"
    kill $LAIR_PID 2>/dev/null
    exit 1
fi

echo "LAIR_PID=$LAIR_PID" > "$PIDS_FILE"
echo "HC_PID=$HC_PID" >> "$PIDS_FILE"

echo ""
echo "Holochain conductor running!"
echo "  Admin port: $ADMIN_PORT"
echo "  Data dir:   $CONDUCTOR_DIR"
echo "  PID file:   $PIDS_FILE"
echo "  Logs:       $CONDUCTOR_DIR/conductor.log"
echo ""
echo "Install a hApp:"
echo "  echo '$PASSPHRASE' | hc sandbox --piped call --running $ADMIN_PORT install-app --app-id <name> <path.happ>"
echo "  echo '$PASSPHRASE' | hc sandbox --piped call --running $ADMIN_PORT enable-app <name>"
