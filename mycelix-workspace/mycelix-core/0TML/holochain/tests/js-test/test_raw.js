// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
const WebSocket = require('ws');

// Using msgpack-lite as it's commonly available
let msgpack;
try {
    msgpack = require('@msgpack/msgpack');
} catch {
    console.log('Installing @msgpack/msgpack...');
    require('child_process').execSync('npm install @msgpack/msgpack', { cwd: __dirname });
    msgpack = require('@msgpack/msgpack');
}

const ADMIN_PORT = 9020;
const APP_PORT = 9021;

// Cell ID from list-apps (hex decoded)
const DNA_HASH = Buffer.from('842d248c173d552c84c8c8c7c2e629632e1f922361efcf1ffbf2c1ae5a95f4d75d5d3e127456dc', 'hex');
const AGENT_KEY = Buffer.from('842024be50763a7319b5f4f0c635931c946460213d698f4a9b460866d2d8810130ee2ab51c7134', 'hex');

async function testAdmin() {
    return new Promise((resolve, reject) => {
        console.log('\n1. Testing admin WebSocket...');
        const ws = new WebSocket(`ws://127.0.0.1:${ADMIN_PORT}`, {
            headers: { 'Origin': 'http://localhost' }
        });

        ws.on('open', () => {
            console.log('   Admin connected!');

            const request = {
                id: 1,
                type: 'request',
                data: msgpack.encode({ type: 'list_apps', value: { status_filter: null } })
            };

            ws.send(msgpack.encode(request));
        });

        ws.on('message', (data) => {
            const response = msgpack.decode(data);
            console.log('   Response type:', response.type);

            if (response.data) {
                const decoded = msgpack.decode(response.data);
                console.log('   Decoded type:', decoded.type);
                if (decoded.type === 'apps_listed') {
                    console.log('   Apps:', decoded.value.length);
                }
            }

            ws.close();
            resolve();
        });

        ws.on('error', (err) => {
            console.log('   Error:', err.message);
            reject(err);
        });
    });
}

async function testAppWithDifferentFormats() {
    console.log('\n2. Testing app WebSocket with different formats...');

    const cellId = [DNA_HASH, AGENT_KEY];

    // Try multiple formats sequentially
    const formats = [
        {
            name: 'Standard format',
            msg: {
                id: 1,
                type: 'request',
                data: msgpack.encode({
                    type: 'call_zome',
                    value: {
                        cell_id: cellId,
                        zome_name: 'gradient_storage',
                        fn_name: 'get_statistics',
                        payload: msgpack.encode(null),
                        provenance: AGENT_KEY,
                        cap_secret: null
                    }
                })
            }
        },
        {
            name: 'AppInfo first',
            msg: {
                id: 1,
                type: 'request',
                data: msgpack.encode({
                    type: 'app_info',
                    value: null
                })
            }
        },
        {
            name: 'With app_authentication header',
            msg: {
                type: 'AppAuthentication',
                token: Buffer.from([]),
            }
        }
    ];

    for (const format of formats) {
        await new Promise((resolve) => {
            console.log(`\n   Trying: ${format.name}`);

            const ws = new WebSocket(`ws://127.0.0.1:${APP_PORT}`, {
                headers: { 'Origin': 'http://localhost' }
            });

            let timeout = setTimeout(() => {
                console.log('   Timeout - closing');
                ws.close();
                resolve();
            }, 3000);

            ws.on('open', () => {
                console.log('   Connected, sending...');
                try {
                    ws.send(msgpack.encode(format.msg));
                } catch (e) {
                    console.log('   Send error:', e.message);
                }
            });

            ws.on('message', (data) => {
                clearTimeout(timeout);
                try {
                    const response = msgpack.decode(data);
                    console.log('   SUCCESS! Response:', response);
                } catch (e) {
                    console.log('   Response (raw):', data);
                }
                ws.close();
                resolve();
            });

            ws.on('close', (code, reason) => {
                clearTimeout(timeout);
                console.log(`   Closed: ${code} ${reason}`);
                resolve();
            });

            ws.on('error', (err) => {
                clearTimeout(timeout);
                console.log('   Error:', err.message);
                resolve();
            });
        });
    }
}

async function main() {
    console.log('='.repeat(60));
    console.log('Byzantine Defense - Raw WebSocket Testing');
    console.log('='.repeat(60));

    try {
        await testAdmin();
        await testAppWithDifferentFormats();
    } catch (e) {
        console.error('Fatal error:', e);
    }

    console.log('\n' + '='.repeat(60));
}

main();
