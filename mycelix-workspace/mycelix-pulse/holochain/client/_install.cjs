const { AdminWebsocket } = require('@holochain/client');
const { readFileSync } = require('fs');
const { resolve } = require('path');

const ADMIN_PORT = parseInt(process.env.ADMIN_PORT || '33800');
const APP_PORT = parseInt(process.env.APP_PORT || '8888');
const APP_ID = process.env.APP_ID || 'mycelix_mail';
const HAPP_PATH = resolve(process.env.HAPP_PATH || '../mycelix_mail.happ');

async function main() {
    console.log(`Connecting to admin port ${ADMIN_PORT}...`);
    const admin = await AdminWebsocket.connect({ url: new URL(`ws://localhost:${ADMIN_PORT}`) });
    console.log('Connected to admin API.');

    console.log('Generating agent key...');
    const agentKey = await admin.generateAgentPubKey();
    console.log(`Agent key generated (${agentKey.length} bytes)`);

    const apps = await admin.listApps({});
    console.log(`Installed apps: ${apps.length}`);
    for (const app of apps) {
        console.log(`  - ${app.installed_app_id} (${JSON.stringify(app.status)})`);
    }

    const existing = apps.find(a => a.installed_app_id === APP_ID);
    if (existing) {
        console.log(`\nApp '${APP_ID}' already installed!`);
        await admin.client.close();
        return;
    }

    console.log(`\nReading hApp from ${HAPP_PATH}...`);
    const happBundle = readFileSync(HAPP_PATH);
    console.log(`hApp size: ${(happBundle.length / 1024 / 1024).toFixed(1)} MB`);

    console.log('Installing hApp (this takes ~30s)...');
    const result = await admin.installApp({
        agent_key: agentKey,
        installed_app_id: APP_ID,
        membrane_proofs: {},
        bundle: happBundle,
    });
    console.log(`Installed: ${result.installed_app_id}`);

    console.log('Enabling app...');
    await admin.enableApp({ installed_app_id: APP_ID });
    console.log('App enabled.');

    try {
        await admin.attachAppInterface({ port: APP_PORT, allowed_origins: '*' });
        console.log(`App interface on :${APP_PORT}`);
    } catch (e) {
        console.log(`Interface: ${e.message}`);
    }

    console.log(`\nDONE: ${APP_ID} ready on ws://localhost:${APP_PORT}`);
    await admin.client.close();
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
