// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
const { AdminWebsocket, AppWebsocket } = require('@holochain/client');

const ADMIN_PORT = 9020;
const APP_PORT = 9021;
const APP_ID = 'byzantine-fl-v06';

async function main() {
    console.log('='.repeat(60));
    console.log('Byzantine Defense Zome Call Testing (Official JS Client)');
    console.log('='.repeat(60));

    try {
        // Connect to admin
        console.log('\n1. Connecting to admin WebSocket...');
        const adminWs = await AdminWebsocket.connect({
            url: new URL(`ws://127.0.0.1:${ADMIN_PORT}`),
            wsClientOptions: { origin: 'http://localhost' }
        });
        console.log('   Admin connected!');

        // List apps
        console.log('\n2. Listing apps...');
        const apps = await adminWs.listApps({});
        console.log(`   Found ${apps.length} app(s)`);

        const app = apps.find(a => a.installed_app_id === APP_ID);
        if (!app) {
            console.log(`   ERROR: App ${APP_ID} not found`);
            process.exit(1);
        }
        console.log(`   App status: ${app.status.type}`);

        // Get cell info
        const cellInfo = Object.values(app.cell_info)[0][0];
        const cell = cellInfo.value;
        console.log(`   Cell: ${cell.name}`);

        // Connect to app
        console.log('\n3. Connecting to app WebSocket...');
        try {
            const appWs = await AppWebsocket.connect({
                url: new URL(`ws://127.0.0.1:${APP_PORT}`),
                wsClientOptions: { origin: 'http://localhost' },
                installedAppId: APP_ID
            });
            console.log('   App connected!');

            // Get app info
            console.log('\n4. Getting app info...');
            const appInfo = await appWs.appInfo();
            console.log(`   App: ${appInfo.installed_app_id}`);
            console.log(`   Status: ${appInfo.status.type}`);

            // Make zome call
            console.log('\n5. Calling gradient_storage::get_statistics...');
            const result = await appWs.callZome({
                role_name: 'fl_node',
                zome_name: 'gradient_storage',
                fn_name: 'get_statistics',
                payload: null
            });
            console.log('   Result:', result);

        } catch (appErr) {
            console.log(`   App connection error: ${appErr.message}`);
            console.log('   Full error:', appErr);
        }

    } catch (err) {
        console.error('Error:', err.message);
        console.error('Stack:', err.stack);
    }

    console.log('\n' + '='.repeat(60));
    console.log('Test complete!');
    console.log('='.repeat(60));
}

main().catch(console.error);
