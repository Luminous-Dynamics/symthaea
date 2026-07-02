// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
const WebSocket = require('ws');
const msgpack = require('@msgpack/msgpack');

const APP_PORT = 9021;
const DNA_HASH = Buffer.from('842d248c173d552c84c8c8c7c2e629632e1f922361efcf1ffbf2c1ae5a95f4d75d5d3e127456dc', 'hex');
const AGENT_KEY = Buffer.from('842024be50763a7319b5f4f0c635931c946460213d698f4a9b460866d2d8810130ee2ab51c7134', 'hex');

async function main() {
    console.log("Connecting to app WebSocket...");
    
    const ws = new WebSocket(`ws://127.0.0.1:${APP_PORT}`, {
        headers: { 'Origin': 'http://localhost' }
    });

    ws.on('error', (err) => {
        console.error("WebSocket error:", err.message);
    });

    ws.on('close', (code, reason) => {
        console.log("WebSocket closed:", code, reason.toString());
    });

    ws.on('open', async () => {
        console.log("Connected!");

        // Try sending an app_info request
        const appInfo = {
            type: "app_info",
            data: { installed_app_id: "byzantine-fl-v06" }
        };

        const request = {
            id: 1,
            type: "request",
            data: msgpack.encode(appInfo)
        };

        console.log("Sending app_info request...");
        ws.send(msgpack.encode(request));
    });

    ws.on('message', (data) => {
        const response = msgpack.decode(data);
        console.log("Response:", response);
    });

    // Wait for response
    await new Promise(resolve => setTimeout(resolve, 5000));
    ws.close();
}

main().catch(console.error);
