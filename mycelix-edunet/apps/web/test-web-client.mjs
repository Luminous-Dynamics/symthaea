#!/usr/bin/env node
/**
 * Test script to verify web client can connect to Holochain conductor
 *
 * Usage: node test-web-client.mjs [admin_port] [app_port]
 *
 * This tests the same connection path the React app will use.
 */

import { AdminWebsocket, AppWebsocket } from '@holochain/client';

const ADMIN_PORT = process.argv[2] || '42473';
const APP_PORT = process.argv[3] || '8888';
const APP_ID = '9999';

console.log('🔧 Testing Web Client Connection...\n');
console.log(`   Admin Port: ${ADMIN_PORT}`);
console.log(`   App Port: ${APP_PORT}`);
console.log(`   App ID: ${APP_ID}`);
console.log('');

async function testConnection() {
  let adminWs = null;
  let appWs = null;

  try {
    // Step 1: Connect to Admin
    console.log('📡 Step 1: Connecting to admin port...');
    adminWs = await AdminWebsocket.connect({
      url: new URL(`ws://localhost:${ADMIN_PORT}`),
      wsClientOptions: { origin: 'test-client' },
    });
    console.log('✅ Admin WebSocket connected\n');

    // Step 2: List apps
    console.log('📋 Step 2: Listing installed apps...');
    const apps = await adminWs.listApps({});
    console.log(`✅ Found ${apps.length} installed app(s)`);

    const app = apps.find(a => a.installed_app_id === APP_ID);
    if (!app) {
      throw new Error(`App ${APP_ID} not found`);
    }
    console.log(`   - ${app.installed_app_id} (${app.status.type})`);

    // Extract cell info
    const role = app.cell_info?.edunet?.[0];
    if (!role) {
      throw new Error('No edunet role found');
    }

    // Handle Holochain 0.6 cell info structure
    let cellId;
    if (role.type === 'provisioned') {
      cellId = role.value?.cell_id;
    } else if (role.provisioned) {
      cellId = role.provisioned.cell_id;
    } else {
      cellId = role.cell_id;
    }

    if (!cellId) {
      console.log('Cell info structure:', JSON.stringify(role, null, 2));
      throw new Error('Could not extract cell ID');
    }
    console.log('   Cell ID found ✓\n');

    // Step 3: Issue app token
    console.log('🔑 Step 3: Issuing app authentication token...');
    const { token } = await adminWs.issueAppAuthenticationToken({
      installed_app_id: APP_ID,
    });
    console.log('✅ App token issued\n');

    // Step 4: Authorize signing credentials
    console.log('🔐 Step 4: Authorizing signing credentials...');
    await adminWs.authorizeSigningCredentials(cellId);
    console.log('✅ Signing credentials authorized\n');

    // Step 5: Connect to app port
    console.log('📡 Step 5: Connecting to app port...');
    appWs = await AppWebsocket.connect({
      url: new URL(`ws://localhost:${APP_PORT}`),
      wsClientOptions: { origin: 'test-client' },
      token,
    });
    console.log('✅ App WebSocket connected\n');

    // Step 6: Get app info
    console.log('📊 Step 6: Getting app info...');
    const appInfo = await appWs.appInfo();
    console.log(`✅ App ID: ${appInfo.installed_app_id}`);
    console.log(`   Roles: ${Object.keys(appInfo.cell_info || {}).join(', ')}\n`);

    // Step 7: Test list_courses
    console.log('🔧 Step 7: Testing learning_coordinator.list_courses...');
    const courses = await appWs.callZome({
      role_name: 'edunet',
      zome_name: 'learning_coordinator',
      fn_name: 'list_courses',
      payload: null,
    });
    console.log(`✅ list_courses returned ${courses.length} course(s)\n`);

    // Step 8: Test create_course
    // The Rust Course struct has: course_id, title, description, creator, tags, model_id, created_at, updated_at, metadata
    console.log('📝 Step 8: Testing learning_coordinator.create_course...');
    const now = Date.now();
    const newCourse = {
      course_id: `web-test-${now}`,
      title: 'Web Client Test Course',
      description: 'Created by web client test',
      creator: 'Test Agent',
      tags: ['test', 'web-client'],
      model_id: null,
      created_at: now,
      updated_at: now,
      metadata: null,
    };

    const actionHash = await appWs.callZome({
      role_name: 'edunet',
      zome_name: 'learning_coordinator',
      fn_name: 'create_course',
      payload: newCourse,
    });
    console.log(`✅ create_course succeeded!`);
    console.log(`   Action hash: [${actionHash.slice(0, 8).join(',')}...]\n`);

    // Step 9: Verify course creation
    console.log('🔧 Step 9: Verifying course was created...');
    const updatedCourses = await appWs.callZome({
      role_name: 'edunet',
      zome_name: 'learning_coordinator',
      fn_name: 'list_courses',
      payload: null,
    });
    console.log(`✅ Now have ${updatedCourses.length} course(s)\n`);

    console.log('═══════════════════════════════════════════════════════════');
    console.log('🎉 ALL TESTS PASSED! Web client connection is working.');
    console.log('═══════════════════════════════════════════════════════════');
    console.log('\nThe React app should be able to connect successfully.');
    console.log('Start the dev server with: npm run dev');
    console.log('');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error('');
    console.error('Troubleshooting:');
    console.error('1. Is the conductor running?');
    console.error(`   Check: curl -I ws://localhost:${ADMIN_PORT}`);
    console.error('2. Are the ports correct?');
    console.error('   Look for "admin_port" and "app_ports" in conductor output');
    console.error('3. Is the hApp installed?');
    console.error(`   App ID should be: ${APP_ID}`);
    process.exit(1);
  } finally {
    if (appWs) {
      try { await appWs.close(); } catch {}
    }
    if (adminWs) {
      try { await adminWs.close(); } catch {}
    }
  }
}

testConnection();
