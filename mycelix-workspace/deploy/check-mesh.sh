#!/usr/bin/env bash
# Check health of the 3-node Mycelix mesh.
set -euo pipefail

echo "=== Mycelix Mesh Health Check ==="
echo ""

NODES=("Node1:4444:8888" "Node2:4446:8889" "Node3:4447:8890")
ALL_OK=true

for node_spec in "${NODES[@]}"; do
    IFS=':' read -r name admin_port app_port <<< "$node_spec"
    echo "--- $name (admin=$admin_port, app=$app_port) ---"

    # Check admin port
    if timeout 2 bash -c "echo > /dev/tcp/localhost/$admin_port" 2>/dev/null; then
        echo "  Admin port: OK"
    else
        echo "  Admin port: UNREACHABLE"
        ALL_OK=false
        continue
    fi

    # Check app port
    if timeout 2 bash -c "echo > /dev/tcp/localhost/$app_port" 2>/dev/null; then
        echo "  App port:   OK"
    else
        echo "  App port:   UNREACHABLE"
        ALL_OK=false
    fi
    echo ""
done

if $ALL_OK; then
    echo "All nodes healthy."
else
    echo "WARNING: Some nodes are unhealthy. Check docker logs."
    exit 1
fi
