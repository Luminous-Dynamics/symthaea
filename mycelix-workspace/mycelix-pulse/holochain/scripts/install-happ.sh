#!/usr/bin/env bash
# Install the Mycelix Mail hApp on the local conductor.
#
# Prerequisites:
#   - Conductor running with admin port on 33800
#   - mycelix_mail.happ in the holochain/ directory
#
# Uses the Holochain admin WebSocket API (MessagePack over WebSocket)
# Since hc CLI has interpreter issues, we use a Node.js script.

set -euo pipefail
cd "$(dirname "$0")/.."

ADMIN_PORT=33800
APP_PORT=8888
HAPP_FILE="mycelix_mail.happ"
APP_ID="mycelix_mail"

echo "=== Mycelix Mail hApp Installer ==="
echo "Admin port: $ADMIN_PORT"
echo "App port: $APP_PORT"
echo "hApp: $HAPP_FILE"
echo ""

if [ ! -f "$HAPP_FILE" ]; then
    echo "ERROR: $HAPP_FILE not found. Build it first:"
    echo "  cargo build --workspace --target wasm32-unknown-unknown --release"
    echo "  hc dna pack dna/ -o dna/mycelix_mail_dna.dna"
    echo "  hc app pack . -o mycelix_mail.happ"
    exit 1
fi

# Use the TypeScript client to install
cd client

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install 2>/dev/null || yarn install 2>/dev/null || echo "WARN: npm/yarn not found"
fi

# Write install script
cat > /tmp/install-happ.mjs << 'EOF'
import { AdminWebsocket } from '@holochain/client';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const ADMIN_PORT = parseInt(process.env.ADMIN_PORT || '33800');
const APP_PORT = parseInt(process.env.APP_PORT || '8888');
const APP_ID = process.env.APP_ID || 'mycelix_mail';
const HAPP_PATH = resolve(process.env.HAPP_PATH || '../mycelix_mail.happ');

async function main() {
    console.log(`Connecting to admin port ${ADMIN_PORT}...`);
    const admin = await AdminWebsocket.connect({ url: new URL(`ws://localhost:${ADMIN_PORT}`) });
    console.log('Connected to admin API.');

    // Generate agent key
    console.log('Generating agent key...');
    const agentKey = await admin.generateAgentPubKey();
    console.log(`Agent key: ${Buffer.from(agentKey).toString('base64url').slice(0, 16)}...`);

    // Check if app already installed
    const apps = await admin.listApps({});
    const existing = apps.find(a => a.installed_app_id === APP_ID);
    if (existing) {
        console.log(`App '${APP_ID}' already installed (status: ${JSON.stringify(existing.status)})`);
        // Ensure app interface exists
        const interfaces = await admin.listAppInterfaces();
        console.log(`App interfaces: ${JSON.stringify(interfaces.map(i => i.port))}`);

        // Issue auth token for the frontend
        const token = await admin.issueAppAuthenticationToken({ installed_app_id: APP_ID });
        console.log(`Auth token: ${Buffer.from(token.token).toString('base64url').slice(0, 20)}...`);
        console.log('DONE: App is ready.');
        await admin.client.close();
        return;
    }

    // Read hApp bundle
    console.log(`Reading hApp from ${HAPP_PATH}...`);
    const happBundle = readFileSync(HAPP_PATH);
    console.log(`hApp size: ${(happBundle.length / 1024 / 1024).toFixed(1)} MB`);

    // Install
    console.log('Installing hApp...');
    const result = await admin.installApp({
        agent_key: agentKey,
        installed_app_id: APP_ID,
        membrane_proofs: {},
        bundle: happBundle,
    });
    console.log(`Installed: ${result.installed_app_id}`);

    // Enable app
    console.log('Enabling app...');
    await admin.enableApp({ installed_app_id: APP_ID });
    console.log('App enabled.');

    // Attach app interface on the expected port
    try {
        await admin.attachAppInterface({ port: APP_PORT, allowed_origins: '*' });
        console.log(`App interface attached on port ${APP_PORT}`);
    } catch (e) {
        console.log(`App interface: ${e.message} (may already be attached)`);
    }

    // Issue auth token
    const token = await admin.issueAppAuthenticationToken({ installed_app_id: APP_ID });
    console.log(`Auth token issued: ${Buffer.from(token.token).toString('base64url').slice(0, 20)}...`);

    console.log('\nDONE: Mycelix Mail hApp is installed and ready!');
    console.log(`Frontend should connect to ws://localhost:${APP_PORT}`);

    await admin.client.close();
}

main().catch(e => { console.error('FATAL:', e.message); process.exit(1); });
EOF

echo "Running installer..."
ADMIN_PORT=$ADMIN_PORT APP_PORT=$APP_PORT APP_ID=$APP_ID HAPP_PATH="$(pwd)/../$HAPP_FILE" \
    node /tmp/install-happ.mjs

echo ""
echo "Next steps:"
echo "  1. Open https://mail.mycelix.net"
echo "  2. The app should auto-detect the conductor and switch from mock to real data"
echo "  3. If not, check browser console for '[Mail]' logs"
