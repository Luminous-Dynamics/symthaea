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
    console.log(`Installed apps: ${apps.length}`);
    const existing = apps.find(a => a.installed_app_id === APP_ID);
    if (existing) {
        console.log(`App '${APP_ID}' already installed (status: ${JSON.stringify(existing.status)})`);
        for (const [role, cells] of Object.entries(existing.cell_info || {})) {
            console.log(`  Role: ${role} (${cells.length} cell(s))`);
        }
        await admin.client.close();
        return;
    }

    // Read hApp bundle
    console.log(`Reading hApp from ${HAPP_PATH}...`);
    const happBundle = readFileSync(HAPP_PATH);
    console.log(`hApp size: ${(happBundle.length / 1024 / 1024).toFixed(1)} MB`);

    // Install
    console.log('Installing hApp (this may take a minute)...');
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

    // Attach app interface
    try {
        await admin.attachAppInterface({ port: APP_PORT, allowed_origins: '*' });
        console.log(`App interface attached on port ${APP_PORT}`);
    } catch (e) {
        console.log(`App interface: ${e.message} (may already exist)`);
    }

    // List interfaces
    const interfaces = await admin.listAppInterfaces();
    console.log(`Active interfaces: ${JSON.stringify(interfaces.map(i => i.port))}`);

    console.log(`\nDONE: ${APP_ID} installed and ready on ws://localhost:${APP_PORT}`);
    await admin.client.close();
}

main().catch(e => { console.error('FATAL:', e.message || e); process.exit(1); });
