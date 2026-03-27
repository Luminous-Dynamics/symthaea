#!/usr/bin/env bash
# Stop the Mycelix mesh and collect logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMPOSE_CMD="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
fi

COMPOSE_FILES="-f $SCRIPT_DIR/docker-compose.prod.yml -f $SCRIPT_DIR/docker-compose.multinode.yml"

echo "=== Collecting logs ==="
mkdir -p "$SCRIPT_DIR/logs"
$COMPOSE_CMD $COMPOSE_FILES logs --no-color > "$SCRIPT_DIR/logs/mesh-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true

echo "=== Stopping mesh ==="
cd "$SCRIPT_DIR"
$COMPOSE_CMD $COMPOSE_FILES down

echo "Mesh stopped. Logs saved to $SCRIPT_DIR/logs/"
