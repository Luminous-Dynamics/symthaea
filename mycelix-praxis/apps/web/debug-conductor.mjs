#!/usr/bin/env node
/**
 * Debug conductor cell info structure
 */

import { AdminWebsocket, AppWebsocket } from '@holochain/client';

const ADMIN_PORT = process.argv[2] || 45961;
const APP_PORT = process.argv[3] || 8888;

async function main() {
  console.log('🔍 Debugging Holochain Conductor...\n');

  // Connect to admin
  const admin = await AdminWebsocket.connect({
    url: new URL(`ws://localhost:${ADMIN_PORT}`),
    wsClientOptions: { origin: 'debug' }
  });
  console.log('✅ Admin connected\n');

  // List apps with full details
  const apps = await admin.listApps({});
  console.log('📋 Apps:');
  console.log(JSON.stringify(apps, null, 2));
  console.log('\n');

  const appId = apps[0]?.installed_app_id;

  // Get app token
  const { token } = await admin.issueAppAuthenticationToken({
    installed_app_id: appId
  });
  console.log('🔑 Token issued\n');

  // Connect to app
  const app = await AppWebsocket.connect({
    url: new URL(`ws://localhost:${APP_PORT}`),
    wsClientOptions: { origin: 'debug' },
    token
  });
  console.log('✅ App connected\n');

  // Get app info
  const info = await app.appInfo();
  console.log('📊 App Info:');
  console.log(JSON.stringify(info, null, 2));
  console.log('\n');

  // Extract cell ID properly
  if (info?.cell_info?.edunet) {
    const cellInfo = info.cell_info.edunet[0];
    console.log('📦 Cell Info (raw):');
    console.log(JSON.stringify(cellInfo, null, 2));

    // Check different possible structures
    let cellId = null;
    if (cellInfo.provisioned) {
      console.log('\n  Found "provisioned" key');
      cellId = cellInfo.provisioned.cell_id;
    } else if (cellInfo.cloned) {
      console.log('\n  Found "cloned" key');
      cellId = cellInfo.cloned.cell_id;
    } else if (cellInfo.stem) {
      console.log('\n  Found "stem" key');
      cellId = cellInfo.stem.cell_id;
    } else if (cellInfo.cell_id) {
      console.log('\n  Found direct "cell_id" key');
      cellId = cellInfo.cell_id;
    }

    if (cellId) {
      console.log('\n🔐 Authorizing signing credentials...');
      console.log('Cell ID:', JSON.stringify(cellId));
      try {
        await admin.authorizeSigningCredentials(cellId);
        console.log('✅ Authorized!\n');

        // Try zome call again
        console.log('🔧 Testing zome call...');
        const result = await app.callZome({
          role_name: 'edunet',
          zome_name: 'learning_coordinator',
          fn_name: 'list_courses',
          payload: null,
          cap_secret: null
        });
        console.log('✅ Zome call succeeded!');
        console.log('Result:', JSON.stringify(result));
      } catch (err) {
        console.error('❌ Error:', err.message);
      }
    } else {
      console.log('❌ Could not find cell ID');
    }
  }

  await admin.client.close();
  await app.client.close();
}

main().catch(console.error);
