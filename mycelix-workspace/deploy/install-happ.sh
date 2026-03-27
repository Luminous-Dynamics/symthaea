#!/usr/bin/env bash
# Install Mycelix hApp on all running conductor nodes.
#
# Usage:
#   ./install-happ.sh [happ-path]
#
# Default: installs mycelix-resilience.happ from the host happs/ directory.
# Requires: docker, node:22-slim image, @holochain/client in sdk-ts/node_modules.

set -euo pipefail

HAPP_PATH="${1:-/srv/luminous-dynamics/mycelix-workspace/happs/mycelix-resilience.happ}"
SDK_MODULES="/srv/luminous-dynamics/mycelix-workspace/sdk-ts/node_modules"
APP_ID="mycelix-test"

if [ ! -f "$HAPP_PATH" ]; then
    echo "ERROR: hApp not found at $HAPP_PATH"
    exit 1
fi

echo "=== Mycelix hApp Installer ==="
echo "hApp: $HAPP_PATH ($(du -h "$HAPP_PATH" | cut -f1))"
echo ""

for NODE in mycelix-conductor mycelix-conductor-2 mycelix-conductor-3; do
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${NODE}$"; then
        echo "[$NODE] SKIPPED (not running)"
        continue
    fi

    echo "[$NODE] Installing..."

    # Copy hApp to persistent volume
    docker cp "$HAPP_PATH" "${NODE}:/data/holochain/install.happ" 2>/dev/null || true

    # Use sidecar Node container sharing network namespace
    docker run --rm --network="container:${NODE}" \
        -v "${SDK_MODULES}:/app/node_modules" \
        -w /app \
        node:22-slim node -e "
async function main() {
    const { AdminWebsocket } = await import('@holochain/client');
    const admin = await AdminWebsocket.connect({
        url: 'ws://127.0.0.1:4444',
        wsClientOptions: { origin: 'localhost' }
    });

    const apps = await admin.listApps({});
    if (apps.some(a => a.installed_app_id === '${APP_ID}')) {
        await admin.enableApp({ installed_app_id: '${APP_ID}' });
        console.log('Already installed, enabled');
        process.exit(0);
    }

    const agent = await admin.generateAgentPubKey();
    console.log('Agent generated, installing via path...');

    try {
        const appInfo = await admin.installApp({
            path: '/data/holochain/install.happ',
            installed_app_id: '${APP_ID}',
            agent_key: agent,
        });
        console.log('Installed! Roles:', Object.keys(appInfo.cell_info).join(', '));
        await admin.enableApp({ installed_app_id: '${APP_ID}' });
        console.log('Enabled!');
    } catch(e) {
        if (e.message?.includes('deserialize')) {
            console.log('Path install not supported, trying sandbox approach...');
            process.exit(2);
        }
        throw e;
    }
    process.exit(0);
}
main().catch(e => { console.error(e.message?.slice(0,200)); process.exit(1); });
" 2>&1

    RESULT=$?
    if [ $RESULT -eq 2 ]; then
        echo "[$NODE] Falling back to hc sandbox generate..."
        docker exec "${NODE}" sh -c "
            echo \"\$LAIR_PASSPHRASE\" | timeout 300 hc sandbox --piped generate \
                --in-process-lair /data/holochain/install.happ 2>&1 | tail -3
        " || echo "[$NODE] Sandbox install timed out (may still be processing)"
    elif [ $RESULT -ne 0 ]; then
        echo "[$NODE] FAILED (exit $RESULT)"
    fi

    echo ""
done

echo "=== Checking all nodes ==="
for NODE in mycelix-conductor mycelix-conductor-2 mycelix-conductor-3; do
    if docker ps --format '{{.Names}}' | grep -q "^${NODE}$"; then
        APPS=$(docker run --rm --network="container:${NODE}" \
            -v "${SDK_MODULES}:/app/node_modules" \
            -w /app node:22-slim node -e "
async function main() {
    const { AdminWebsocket } = await import('@holochain/client');
    const admin = await AdminWebsocket.connect({
        url: 'ws://127.0.0.1:4444',
        wsClientOptions: { origin: 'localhost' }
    });
    const apps = await admin.listApps({});
    console.log(apps.map(a => a.installed_app_id + '(' + JSON.stringify(a.status).slice(0,20) + ')').join(', ') || 'none');
    process.exit(0);
}
main().catch(() => { console.log('unreachable'); process.exit(0); });
" 2>&1)
        echo "[$NODE] Apps: $APPS"
    fi
done
