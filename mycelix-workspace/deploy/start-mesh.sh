#!/usr/bin/env bash
# Start a 3-node Mycelix mesh for integration testing.
# Requires: Docker, docker-compose v2
# Usage: bash start-mesh.sh [--resilience|--unified]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
HAPP_MODE="${1:---resilience}"

echo "=== Mycelix Mesh Startup ==="
echo "Workspace: $WORKSPACE_DIR"
echo "Mode: $HAPP_MODE"

# 1. Verify prerequisites
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi
if ! docker compose version &>/dev/null && ! docker-compose version &>/dev/null; then
    echo "ERROR: docker compose not found."
    exit 1
fi

COMPOSE_CMD="docker compose"
if ! docker compose version &>/dev/null; then
    COMPOSE_CMD="docker-compose"
fi

# 2. Check .env exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Creating .env with auto-generated passphrase..."
    PASSPHRASE=$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | xxd -p | head -c 32)
    cat > "$SCRIPT_DIR/.env" <<EOF
LAIR_PASSPHRASE=$PASSPHRASE
HAPP_PATH=/happ/mycelix-resilience.happ
BOOTSTRAP_URL=https://dev-test-bootstrap2.holochain.org/
SIGNAL_URL=wss://dev-test-bootstrap2.holochain.org/
EOF
    echo ".env created."
fi

# 3. Select compose files
COMPOSE_FILES="-f $SCRIPT_DIR/docker-compose.prod.yml -f $SCRIPT_DIR/docker-compose.multinode.yml"

# 4. Start the mesh
echo ""
echo "Starting 3-node conductor mesh..."
cd "$SCRIPT_DIR"
$COMPOSE_CMD $COMPOSE_FILES up -d

# 5. Wait for health
echo ""
echo "Waiting for conductors to be healthy..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTHY=0
    for PORT in 4444 4446 4447; do
        if curl -sf "http://localhost:$PORT/healthcheck" &>/dev/null || \
           timeout 2 bash -c "echo > /dev/tcp/localhost/$PORT" &>/dev/null; then
            HEALTHY=$((HEALTHY + 1))
        fi
    done
    if [ $HEALTHY -ge 3 ]; then
        echo "All 3 conductors are up (${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... ($WAITED/${MAX_WAIT}s, $HEALTHY/3 healthy)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "WARNING: Not all conductors healthy after ${MAX_WAIT}s"
    echo "Check logs: $COMPOSE_CMD $COMPOSE_FILES logs"
fi

# 6. Install hApp on all nodes
if [ -f "$SCRIPT_DIR/install-happ.sh" ]; then
    echo ""
    echo "Installing hApp on all nodes..."
    for PORT in 4444 4446 4447; do
        echo "  Installing on conductor at admin port $PORT..."
        HC_ADMIN_PORT=$PORT bash "$SCRIPT_DIR/install-happ.sh" 2>/dev/null || \
            echo "  WARNING: install failed on port $PORT (may need nix shell)"
    done
fi

# 7. Status summary
echo ""
echo "=== Mesh Status ==="
echo "Node 1: admin=4444 app=8888"
echo "Node 2: admin=4446 app=8889"
echo "Node 3: admin=4447 app=8890"
echo ""
echo "To check: bash $SCRIPT_DIR/check-mesh.sh"
echo "To stop:  bash $SCRIPT_DIR/stop-mesh.sh"
