#!/usr/bin/env node
/**
 * Conductor connectivity test for Mycelix EduNet
 * Tests that the hApp is installed and zome calls work
 */

import { AdminWebsocket, AppWebsocket } from '@holochain/client';

const ADMIN_PORT = process.argv[2] || 45961;
const APP_PORT = process.argv[3] || 8888;

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
  let appId;
  let cellId;
  try {
    const apps = await admin.listApps({});
    console.log(`✅ Found ${apps.length} installed app(s):`);
    apps.forEach(app => {
      console.log(`   - ${app.installed_app_id} (${app.status.type})`);
      appId = app.installed_app_id;

      // Get cell ID from the first app - Holochain 0.6 format: { type, value }
      if (app.cell_info && app.cell_info.edunet) {
        const cellInfo = app.cell_info.edunet[0];
        if (cellInfo.type === 'provisioned' && cellInfo.value?.cell_id) {
          cellId = cellInfo.value.cell_id;
          console.log(`   Cell ID found for role 'edunet'`);
        }
      }
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

  // Test 3: Issue app auth token via admin API
  console.log('🔑 Issuing app authentication token...');
  let appToken;
  try {
    const response = await admin.issueAppAuthenticationToken({
      installed_app_id: appId
    });
    appToken = response.token;
    console.log('✅ App token issued\n');
  } catch (err) {
    console.error('❌ Failed to issue token:', err.message);
    process.exit(1);
  }

  // Test 4: Authorize signing credentials for the cell
  if (cellId) {
    console.log('🔐 Authorizing signing credentials...');
    try {
      await admin.authorizeSigningCredentials(cellId);
      console.log('✅ Signing credentials authorized\n');
    } catch (err) {
      console.error('❌ Failed to authorize signing:', err.message);
      process.exit(1);
    }
  } else {
    console.error('❌ No cell ID found - cannot authorize signing');
    process.exit(1);
  }

  // Test 5: App WebSocket connection with token
  console.log(`📡 Connecting to app port ${APP_PORT}...`);
  let app;
  try {
    app = await AppWebsocket.connect({
      url: new URL(`ws://localhost:${APP_PORT}`),
      wsClientOptions: { origin: 'mycelix-edunet-test' },
      token: appToken
    });
    console.log('✅ App WebSocket connected\n');
  } catch (err) {
    console.error('❌ Failed to connect to app:', err.message);
    process.exit(1);
  }

  // Test 6: Get app info
  console.log('📊 Getting app info...');
  try {
    const info = await app.appInfo();
    console.log(`✅ App ID: ${info?.installed_app_id || 'unknown'}`);
    if (info?.cell_info) {
      Object.entries(info.cell_info).forEach(([role, cells]) => {
        console.log(`   - Role '${role}': ${cells.length} cell(s)`);
      });
    }
    console.log('');
  } catch (err) {
    console.error('❌ Failed to get app info:', err.message);
  }

  // Test 7: Make a simple zome call (list_courses - should return empty array)
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
    console.log('');
  } catch (err) {
    console.error('❌ Zome call failed:', err.message);
    console.log('');
  }

  // Test 8: Create a course to verify write operations
  console.log('📝 Testing zome call: create_course...');
  try {
    // Course struct from the integrity zome
    const now = Date.now();
    const course = {
      course_id: `test-course-${now}`,  // CourseId is a newtype around String
      title: 'Test Course from Node.js',
      description: 'A test course created via WebSocket API',
      creator: 'test-agent',
      tags: ['test', 'nodejs'],
      model_id: null,
      created_at: now,
      updated_at: now,
      metadata: { test: true }
    };

    const result = await app.callZome({
      role_name: 'edunet',
      zome_name: 'learning_coordinator',
      fn_name: 'create_course',
      payload: course,
      cap_secret: null
    });
    console.log('✅ Create course succeeded!');
    console.log(`   Action hash: ${JSON.stringify(result).slice(0, 60)}...`);
    console.log('');
  } catch (err) {
    console.error('❌ Create course failed:', err.message);
    console.log('');
  }

  // Test 9: List courses again to verify the new course exists
  console.log('🔧 Verifying course creation: list_courses...');
  try {
    const result = await app.callZome({
      role_name: 'edunet',
      zome_name: 'learning_coordinator',
      fn_name: 'list_courses',
      payload: null,
      cap_secret: null
    });
    console.log('✅ List courses succeeded!');
    console.log(`   Found ${result?.length || 0} course(s)`);
    console.log('');
  } catch (err) {
    console.error('❌ List courses failed:', err.message);
    console.log('');
  }

  console.log('🎉 Conductor test complete!');

  // Clean up
  if (admin) await admin.client.close();
  if (app) await app.client.close();

  process.exit(0);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
