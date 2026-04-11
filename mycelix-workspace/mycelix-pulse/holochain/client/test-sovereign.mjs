const { AdminWebsocket, AppWebsocket } = await import('@holochain/client');

const admin = await AdminWebsocket.connect({ url: new URL('ws://localhost:33800') });
console.log('Admin connected');

const apps = await admin.listApps({});
console.log('Apps:', apps.map(a => `${a.installed_app_id} (${a.status.type})`).join(', '));

const identityApp = apps.find(a => a.installed_app_id === 'identity-test');
if (!identityApp) { console.error('No identity-test app'); process.exit(1); }

const { token } = await admin.issueAppAuthenticationToken({ installed_app_id: 'identity-test' });
const app = await AppWebsocket.connect({ url: new URL('ws://localhost:8888'), token });
console.log('App connected');

const result = await app.callZome({
  role_name: 'identity',
  zome_name: 'identity_bridge',
  fn_name: 'issue_sovereign_credential',
  payload: 'did:mycelix:test-sovereign-e2e',
});

console.log('\n=== SOVEREIGN CREDENTIAL ISSUED ===');
console.log(JSON.stringify(result, null, 2));
process.exit(0);
