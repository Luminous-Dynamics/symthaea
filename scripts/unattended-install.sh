#!/usr/bin/env bash
# Unattended NixOS install via NixForHumanity relay.
#
# Usage:
#   ./unattended-install.sh <relay-url> <token> <config.json>
#
# Example:
#   ./unattended-install.sh ws://192.168.1.100:8094 sovereign my-config.json
#
# The config.json should contain all install fields:
# {
#   "disk": "/dev/sda",
#   "layout": "single",
#   "hostname": "my-nixos",
#   "desktop": "gnome",
#   "gpu_driver": "nvidia",
#   "timezone": "America/Chicago",
#   "keyboard": "us",
#   "secure_boot": false,
#   "tpm2_unlock": false,
#   "fido2_unlock": false,
#   "configuration_nix": "...",
#   "flake_nix": "...",
#   "user_password": "changeme"
# }
#
# Requires: python3, python3-websockets (pip install websockets)

set -euo pipefail

RELAY="${1:?Usage: $0 <relay-url> <token> <config.json>}"
TOKEN="${2:?Missing token}"
CONFIG="${3:?Missing config.json}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found"
    exit 1
fi

if ! python3 -c "import websockets" 2>/dev/null; then
    echo "Error: python3 websockets module required. Install with: pip install websockets"
    exit 1
fi

echo "NixForHumanity Unattended Install"
echo "  Relay: $RELAY"
echo "  Config: $CONFIG"
echo ""

python3 -c "
import asyncio, json, sys

async def install():
    import websockets

    config = json.load(open('$CONFIG'))

    async with websockets.connect('$RELAY', ping_timeout=1800) as ws:
        # Authenticate
        await ws.send(json.dumps({'action': 'auth', 'token': '$TOKEN'}))
        r = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if r.get('type') not in ('authed', 'authenticated'):
            print(f'Auth failed: {r}', file=sys.stderr)
            sys.exit(1)
        print('Authenticated.')

        # Connect (relay connects to local sshd)
        await ws.send(json.dumps({'action': 'connect', 'host': '127.0.0.1', 'port': 22, 'username': 'root', 'password': 'sovereign'}))
        r = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if r.get('type') == 'error':
            print(f'Connect failed: {r.get(\"message\", r)}', file=sys.stderr)
            sys.exit(1)
        print('Connected to target.')

        # Probe hardware (required before install)
        await ws.send(json.dumps({'action': 'probe_hardware'}))
        while True:
            r = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if r.get('type') == 'hardware_probe':
                print('Hardware probed.')
                break
            elif r.get('type') == 'error':
                print(f'Probe failed: {r.get(\"message\", r)}', file=sys.stderr)
                sys.exit(1)

        # Discover disks
        await ws.send(json.dumps({'action': 'discover_disks'}))
        while True:
            r = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if r.get('type') == 'disks':
                print('Disks discovered.')
                break
            elif r.get('type') == 'error':
                print(f'Disk discovery failed: {r.get(\"message\", r)}', file=sys.stderr)
                sys.exit(1)

        # Start install
        config['action'] = 'install'
        await ws.send(json.dumps(config))
        print('Install started. Streaming output...')
        print('---')

        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 1800))
            msg_type = msg.get('type', '')

            if msg_type == 'progress':
                pct = msg.get('percentage', 0)
                stage = msg.get('stage', '')
                print(f'[{pct}%] {stage}')

            elif msg_type == 'output':
                data = msg.get('data', '')
                if data.strip():
                    print(f'  {data}')

            elif msg_type == 'exit':
                code = msg.get('code', -1)
                print('---')
                if code == 0:
                    print('Install COMPLETE. Reboot the target machine.')
                    sys.exit(0)
                else:
                    print(f'Install FAILED (exit code {code})')
                    sys.exit(1)

            elif msg_type == 'error':
                print(f'Error: {msg.get(\"message\", msg)}', file=sys.stderr)
                sys.exit(1)

asyncio.run(install())
"
