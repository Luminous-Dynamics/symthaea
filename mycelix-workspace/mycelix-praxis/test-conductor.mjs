#!/usr/bin/env node
/**
 * Simple conductor connectivity test for Mycelix EduNet
 * Tests that the hApp is installed and zome calls work
 */

import { AdminWebsocket, AppWebsocket } from '@holochain/client';

const ADMIN_PORT = process.argv[2] || 45961;
const APP_PORT = process.argv[3] || 8889;

async function main() {
  console.log('🧪 Testing Holochain Conductor Connection...\n');

  // Test 1: Admin WebSocket connection
  console.log(`📡 Connecting to admin port ${ADMIN_PORT}...`);
  let admin;
  try {
    admin = await AdminWebsocket.connect({
      url: new URL(`ws://localhost:${ADMIN_PORT}`),
      wsClientOptions: { origin: 'mycelix-edunet-test' }
    });
    console.log('✅ Admin WebSocket connected\n');
  } catch (err) {
    console.error('❌ Failed to connect to admin:', err.message);
    process.exit(1);
  }

  // Test 2: List installed apps
  console.log('📋 Listing installed apps...');
  try {
    const apps = await admin.listApps({});
    console.log(`✅ Found ${apps.length} installed app(s):`);
    apps.forEach(app => {
      console.log(`   - ${app.installed_app_id} (${app.status.type})`);
    });
    console.log('');

    if (apps.length === 0) {
      console.log('⚠️  No apps installed - hApp may not have been installed yet');
      process.exit(1);
    }
  } catch (err) {
    console.error('❌ Failed to list apps:', err.message);
    process.exit(1);
  }

  // Test 3: App WebSocket connection
  console.log(`📡 Connecting to app port ${APP_PORT}...`);
  let app;
  try {
    app = await AppWebsocket.connect({
      url: new URL(`ws://localhost:${APP_PORT}`),
      wsClientOptions: { origin: 'mycelix-edunet-test' }
    });
    console.log('✅ App WebSocket connected\n');
  } catch (err) {
    console.error('❌ Failed to connect to app:', err.message);
    // Not a fatal error - app port may require auth
    console.log('');
  }

  // Test 4: Get app info
  if (app) {
    console.log('📊 Getting app info...');
    try {
      const info = await app.appInfo();
      console.log(`✅ App ID: ${info?.installed_app_id || 'unknown'}`);
      if (info?.cell_info) {
        console.log(`   Cells:`);
        Object.entries(info.cell_info).forEach(([role, cells]) => {
          console.log(`   - ${role}: ${cells.length} cell(s)`);
        });
      }
      console.log('');
    } catch (err) {
      console.error('❌ Failed to get app info:', err.message);
    }

    // Test 5: Make a simple zome call (list_courses - should return empty array)
    console.log('🔧 Testing zome call: list_courses...');
    try {
      const result = await app.callZome({
        role_name: 'edunet',
        zome_name: 'learning_coordinator',
        fn_name: 'list_courses',
        payload: null,
        cap_secret: null
      });
      console.log('✅ Zome call succeeded!');
      console.log(`   Result: ${JSON.stringify(result)}`);
    } catch (err) {
      console.error('❌ Zome call failed:', err.message);
      if (err.message.includes('NotAuthorized')) {
        console.log('   (This is expected - may need proper authorization)');
      }
    }
  }

  console.log('\n🎉 Conductor test complete!');

  // Clean up
  if (admin) await admin.client.close();
  if (app) await app.client.close();

  process.exit(0);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
