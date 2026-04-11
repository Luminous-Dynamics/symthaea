#!/usr/bin/env bash
# DSID Multi-Agent Sandbox — creates two agents for testing email between DSIDs.
#
# Usage:
#   ./scripts/dsid-sandbox.sh setup    # Create Alice + Bob sandboxes
#   ./scripts/dsid-sandbox.sh run      # Run both conductors
#   ./scripts/dsid-sandbox.sh test     # Send test email Alice → Bob
#   ./scripts/dsid-sandbox.sh clean    # Remove sandboxes
#
# Requires: hc (Holochain CLI), holochain, node (for client tests)

set -euo pipefail

HC_LD="/nix/store/jms7zxzm7w1whczwny5m3gkgdjghmi2r-glibc-2.42-51/lib64/ld-linux-x86-64.so.2"
HC="$HC_LD /home/tstoltz/.cargo/bin/hc"
HAPP="/srv/luminous-dynamics/mycelix-mail/holochain/mycelix_mail_simple.happ"
WORKDIR="/tmp/dsid-sandbox"

case "${1:-help}" in
  setup)
    echo "=== Setting up DSID sandbox (Alice + Bob) ==="
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"

    # Generate two sandboxes with in-memory transport
    printf "\n" | $HC sandbox generate \
      --in-process-lair \
      -n 2 \
      -d alice,bob \
      --root "$WORKDIR" \
      "$HAPP" \
      network mem 2>&1 || true

    echo ""
    echo "=== Sandbox created ==="
    echo "Alice: $WORKDIR/alice"
    echo "Bob:   $WORKDIR/bob"
    echo ""
    echo "Run with: $0 run"
    ;;

  run)
    echo "=== Running DSID sandbox ==="
    cd "$WORKDIR"

    # Run both sandboxes
    printf "\n" | $HC sandbox run \
      --root "$WORKDIR" \
      -p 8401,8402 \
      0 1 &

    sleep 10
    echo ""
    echo "=== Conductors running ==="
    echo "Alice admin: ws://localhost:8401"
    echo "Bob admin:   ws://localhost:8402"
    echo ""
    echo "Test with: $0 test"
    wait
    ;;

  test)
    echo "=== Testing Alice → Bob email ==="
    cd /srv/luminous-dynamics/mycelix-mail/holochain/client

    node -e "
const { AdminWebsocket, AppWebsocket } = require('@holochain/client');

async function test() {
    // Connect Alice (port 8401)
    const aliceAdmin = await AdminWebsocket.connect({url: new URL('ws://localhost:8401'), wsClientOptions: {origin: 'http://localhost'}});
    const aliceToken = await aliceAdmin.issueAppAuthenticationToken({installed_app_id: 'mycelix_mail'});
    // Find Alice's app port
    const aliceAppPorts = await aliceAdmin.listAppInterfaces();
    console.log('Alice app ports:', aliceAppPorts);

    const aliceApp = await AppWebsocket.connect({url: new URL('ws://localhost:' + (aliceAppPorts[0]?.port || 8888)), wsClientOptions: {origin: 'http://localhost'}, token: aliceToken.token});
    const aliceInfo = await aliceApp.appInfo();
    await aliceAdmin.authorizeSigningCredentials(aliceInfo.cell_info['main'][0].value.cell_id);
    console.log('Alice connected:', aliceInfo.installed_app_id);
    console.log('Alice agent:', Buffer.from(aliceInfo.agent_pub_key).toString('base64').substring(0, 20) + '...');

    // Connect Bob (port 8402)
    const bobAdmin = await AdminWebsocket.connect({url: new URL('ws://localhost:8402'), wsClientOptions: {origin: 'http://localhost'}});
    const bobToken = await bobAdmin.issueAppAuthenticationToken({installed_app_id: 'mycelix_mail'});
    const bobAppPorts = await bobAdmin.listAppInterfaces();

    const bobApp = await AppWebsocket.connect({url: new URL('ws://localhost:' + (bobAppPorts[0]?.port || 8889)), wsClientOptions: {origin: 'http://localhost'}, token: bobToken.token});
    const bobInfo = await bobApp.appInfo();
    await bobAdmin.authorizeSigningCredentials(bobInfo.cell_info['main'][0].value.cell_id);
    console.log('Bob connected:', bobInfo.installed_app_id);
    console.log('Bob agent:', Buffer.from(bobInfo.agent_pub_key).toString('base64').substring(0, 20) + '...');

    // Alice creates profile
    await aliceApp.callZome({role_name: 'main', zome_name: 'mail_profiles', fn_name: 'set_profile', payload: {name: 'Alice', email: '', avatar_url: '', bio: 'First test user'}});
    console.log('Alice profile created');

    // Bob creates profile
    await bobApp.callZome({role_name: 'main', zome_name: 'mail_profiles', fn_name: 'set_profile', payload: {name: 'Bob', email: '', avatar_url: '', bio: 'Second test user'}});
    console.log('Bob profile created');

    // Alice sends email to Bob
    console.log('\\nAlice sending email to Bob...');
    try {
        const result = await aliceApp.callZome({
            role_name: 'main',
            zome_name: 'mail_messages',
            fn_name: 'send_email',
            payload: {
                recipients: [bobInfo.agent_pub_key],
                cc: [],
                bcc: [],
                encrypted_subject: Array.from(Buffer.from('Hello Bob!')),
                encrypted_body: Array.from(Buffer.from('This is the first DSID-to-DSID email on Mycelix Pulse.')),
                encrypted_attachments: [],
                ephemeral_pubkey: Array.from(Buffer.alloc(32)),
                nonce: Array.from(Buffer.alloc(24)),
                signature: Array.from(Buffer.alloc(64)),
                crypto_suite: 'PQC-AES256-Ed25519',
                message_id: '<test-001@mycelix.net>',
                in_reply_to: '',
                references: [],
                priority: 'Normal',
                read_receipt_requested: false,
                expires_at: null,
            },
        });
        console.log('Email sent! Hash:', JSON.stringify(result).substring(0, 80));
    } catch(e) {
        console.log('Send error:', e.message?.substring(0, 200));
    }

    // Wait for DHT propagation
    console.log('Waiting for DHT propagation...');
    await new Promise(r => setTimeout(r, 5000));

    // Bob checks inbox
    try {
        const inbox = await bobApp.callZome({
            role_name: 'main',
            zome_name: 'mail_messages',
            fn_name: 'get_inbox',
            payload: {limit: 10},
        });
        console.log('\\nBob inbox:', JSON.stringify(inbox).substring(0, 200));
        if (Array.isArray(inbox) && inbox.length > 0) {
            console.log('\\n✅ SUCCESS: Bob received email from Alice!');
        } else {
            console.log('\\n⏳ Inbox empty — DHT may need more propagation time');
        }
    } catch(e) {
        console.log('Inbox error:', e.message?.substring(0, 200));
    }

    await aliceAdmin.client.close();
    await aliceApp.client.close();
    await bobAdmin.client.close();
    await bobApp.client.close();
}

test().catch(e => console.error('Test failed:', e.message));
" 2>&1
    ;;

  clean)
    echo "=== Cleaning DSID sandbox ==="
    rm -rf "$WORKDIR"
    echo "Removed $WORKDIR"
    ;;

  *)
    echo "DSID Multi-Agent Sandbox"
    echo ""
    echo "Usage: $0 {setup|run|test|clean}"
    echo ""
    echo "  setup  Create Alice + Bob sandboxes with in-memory transport"
    echo "  run    Start both conductors"
    echo "  test   Send test email Alice → Bob"
    echo "  clean  Remove sandbox files"
    ;;
esac
