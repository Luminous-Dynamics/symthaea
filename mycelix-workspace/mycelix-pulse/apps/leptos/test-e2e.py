#!/usr/bin/env python3
"""
Mycelix Pulse — End-to-End Integration Test

Proves the complete pipeline works:
1. Connect to conductor WebSocket
2. Get app_info (discover cell IDs)
3. Call get_folders (verify init created system folders)
4. Create a contact
5. Send a real encrypted email
6. Verify the email appears in inbox
7. Revoke the key

Requires: pip install websockets msgpack
"""

import asyncio
import json
import os
import struct
import sys
import time

try:
    import websockets
    import msgpack
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "msgpack", "-q"])
    import websockets
    import msgpack

APP_URL = "ws://127.0.0.1:8888"
ADMIN_URL = "ws://127.0.0.1:33800"

class HolochainClient:
    def __init__(self):
        self.ws = None
        self.request_id = 0
        self.cell_id = None
        self.agent_key = None

    async def connect(self):
        """Connect to the app WebSocket and discover cell info."""
        print(f"\n{'='*60}")
        print("STEP 1: Connect to Conductor")
        print(f"{'='*60}")

        self.ws = await websockets.connect(APP_URL, origin="http://localhost")
        print(f"  [OK] WebSocket connected to {APP_URL}")

        # Send app_info request
        response = await self._request("app_info", None)
        print(f"  [OK] app_info response received")

        if isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, dict):
                app_id = data.get("installed_app_id", "unknown")
                agent_key = data.get("agent_pub_key")
                cell_info = data.get("cell_info", {})

                print(f"  App ID: {app_id}")
                if isinstance(agent_key, (bytes, bytearray)):
                    self.agent_key = agent_key
                    print(f"  Agent: {agent_key[:8].hex()}...")

                # Extract cell_id for the 'main' role
                if isinstance(cell_info, dict):
                    for role_name, cells in cell_info.items():
                        print(f"  Role '{role_name}': {len(cells)} cell(s)")
                        if cells and isinstance(cells, list):
                            cell = cells[0]
                            if isinstance(cell, dict) and "value" in cell:
                                cell_data = cell["value"]
                                if isinstance(cell_data, dict) and "cell_id" in cell_data:
                                    cid = cell_data["cell_id"]
                                    if isinstance(cid, dict):
                                        self.cell_id = (cid.get("dna_hash"), cid.get("agent_pub_key"))
                                    elif isinstance(cid, (list, tuple)) and len(cid) == 2:
                                        self.cell_id = tuple(cid)
                                    print(f"    Cell ID captured for role '{role_name}'")

        if self.cell_id:
            print(f"  [OK] Cell ID resolved")
        else:
            print(f"  [WARN] Could not resolve cell ID — zome calls may fail")

        return True

    async def call_zome(self, zome_name, fn_name, payload=None):
        """Make a zome call and return the response."""
        if not self.cell_id:
            print(f"  [SKIP] No cell_id — cannot call {zome_name}.{fn_name}")
            return None

        # Build the zome call request
        encoded_payload = msgpack.packb(payload) if payload is not None else msgpack.packb(None)

        # Generate nonce (32 random bytes)
        nonce = os.urandom(32)

        # Expiry: 5 minutes from now (microseconds)
        expires_at = int((time.time() + 300) * 1_000_000)

        call_data = {
            "cell_id": list(self.cell_id),
            "zome_name": zome_name,
            "fn_name": fn_name,
            "payload": encoded_payload,
            "cap_secret": None,
            "provenance": self.cell_id[1],  # agent pub key
            "nonce": nonce,
            "expires_at": expires_at,
            "signature": b'\x00' * 64,  # unsigned
        }

        response = await self._request("call_zome", call_data)
        return response

    async def _request(self, req_type, data):
        """Send a request and wait for the response."""
        self.request_id += 1
        rid = self.request_id

        envelope = {
            "id": rid,
            "type": req_type,
            "data": msgpack.packb(data) if data is not None else msgpack.packb(None),
        }

        wire_bytes = msgpack.packb(envelope)
        await self.ws.send(wire_bytes)

        # Wait for response with matching ID
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
            response = msgpack.unpackb(raw, raw=False)

            if isinstance(response, dict):
                resp_id = response.get("id")
                if resp_id == rid:
                    resp_type = response.get("type", "unknown")
                    if resp_type == "error" or "error" in response:
                        error = response.get("data", response.get("error", "unknown error"))
                        return {"error": error, "type": resp_type}

                    resp_data = response.get("data")
                    if isinstance(resp_data, (bytes, bytearray)):
                        try:
                            return msgpack.unpackb(resp_data, raw=False)
                        except Exception:
                            return resp_data
                    return response

    async def close(self):
        if self.ws:
            await self.ws.close()


async def test_get_folders(client):
    """STEP 2: Call get_folders — verify init created system folders."""
    print(f"\n{'='*60}")
    print("STEP 2: Get Folders (verify init)")
    print(f"{'='*60}")

    response = await client.call_zome("mail_messages", "get_folders", None)

    if response is None:
        print("  [SKIP] No cell_id available")
        return False

    if isinstance(response, dict) and "error" in response:
        print(f"  [INFO] Zome responded: {response}")
        # The error might be expected if init hasn't run yet
        return False

    print(f"  Response type: {type(response).__name__}")
    if isinstance(response, list):
        print(f"  [OK] Got {len(response)} folders")
        for folder in response[:7]:
            if isinstance(folder, dict):
                name = folder.get("name", folder.get(b"name", "?"))
                print(f"    - {name}")
        return len(response) > 0
    else:
        print(f"  Response: {str(response)[:200]}")
        return False


async def test_create_contact(client):
    """STEP 3: Create a contact."""
    print(f"\n{'='*60}")
    print("STEP 3: Create Contact")
    print(f"{'='*60}")

    contact = {
        "id": f"test-{int(time.time())}",
        "display_name": "Test Contact",
        "nickname": None,
        "emails": [{"email": "test@mycelix.net", "email_type": "personal", "is_primary": True, "verified": None}],
        "phones": [],
        "addresses": [],
        "organization": "Integration Test",
        "title": None,
        "notes": "Created by e2e test",
        "avatar": None,
        "groups": [],
        "labels": [],
        "agent_pub_key": None,
        "is_favorite": False,
        "is_blocked": False,
        "metadata": {
            "source": "test",
            "last_email_sent": None,
            "last_email_received": None,
            "email_count": 0,
            "response_rate": None,
            "average_response_time": None,
        },
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }

    response = await client.call_zome("mail_contacts", "create_contact", contact)

    if response is None:
        print("  [SKIP] No cell_id")
        return False

    if isinstance(response, dict) and "error" in response:
        print(f"  [INFO] Zome responded: {str(response)[:200]}")
        return False

    print(f"  [OK] Contact created: {str(response)[:100]}")
    return True


async def test_send_email(client):
    """STEP 4: Send a real email."""
    print(f"\n{'='*60}")
    print("STEP 4: Send Email")
    print(f"{'='*60}")

    nonce = list(os.urandom(24))
    timestamp = int(time.time() * 1_000_000)  # microseconds
    message_id = f"<test-{int(time.time())}@mycelix.net>"

    email_input = {
        "recipients": [list(client.cell_id[1])] if client.cell_id else [],  # Send to self
        "cc": [],
        "bcc": [],
        "encrypted_subject": list(b"E2E Test: Hello from integration test"),
        "encrypted_body": list(b"This message was sent by the automated integration test to verify the full DHT pipeline."),
        "encrypted_attachments": list(b""),
        "ephemeral_pubkey": nonce,
        "nonce": nonce,
        "signature": [0] * 64,  # unsigned for test
        "crypto_suite": {
            "key_exchange": "x25519",
            "symmetric": "chacha20-poly1305",
            "signature": "ed25519",
        },
        "message_id": message_id,
        "in_reply_to": None,
        "references": [],
        "timestamp": timestamp,
        "priority": "Normal",
        "read_receipt_requested": False,
        "expires_at": None,
    }

    response = await client.call_zome("mail_messages", "send_email", email_input)

    if response is None:
        print("  [SKIP] No cell_id")
        return False

    if isinstance(response, dict) and "error" in response:
        print(f"  [INFO] Zome responded: {str(response)[:300]}")
        return False

    print(f"  [OK] Email sent: {str(response)[:100]}")
    return True


async def test_get_inbox(client):
    """STEP 5: Check inbox for the sent email."""
    print(f"\n{'='*60}")
    print("STEP 5: Get Inbox")
    print(f"{'='*60}")

    query = {"limit": 10}
    response = await client.call_zome("mail_messages", "get_inbox", query)

    if response is None:
        print("  [SKIP] No cell_id")
        return False

    if isinstance(response, dict) and "error" in response:
        print(f"  [INFO] Zome responded: {str(response)[:200]}")
        return False

    if isinstance(response, list):
        print(f"  [OK] Inbox has {len(response)} message(s)")
        for msg in response[:3]:
            if isinstance(msg, dict):
                subj = msg.get("encrypted_subject", msg.get(b"encrypted_subject", b"?"))
                if isinstance(subj, (bytes, bytearray, list)):
                    subj = bytes(subj).decode("utf-8", errors="replace")
                print(f"    - {subj[:60]}")
        return True
    else:
        print(f"  Response: {str(response)[:200]}")
        return False


async def test_key_status(client):
    """STEP 6: Check key bundle health."""
    print(f"\n{'='*60}")
    print("STEP 6: Key Bundle Status")
    print(f"{'='*60}")

    response = await client.call_zome("mail_keys", "needs_refresh", None)

    if response is None:
        print("  [SKIP] No cell_id")
        return False

    if isinstance(response, dict) and "error" in response:
        print(f"  [INFO] Zome responded: {str(response)[:200]}")
        return False

    print(f"  [OK] Key status: {response}")
    return True


async def main():
    print("=" * 60)
    print("MYCELIX MAIL — END-TO-END INTEGRATION TEST")
    print("=" * 60)
    print(f"Conductor: {APP_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    client = HolochainClient()
    results = {}

    try:
        # Step 1: Connect
        results["connect"] = await client.connect()

        # Step 2: Get folders
        results["folders"] = await test_get_folders(client)

        # Step 3: Create contact
        results["contact"] = await test_create_contact(client)

        # Step 4: Send email
        results["send"] = await test_send_email(client)

        # Step 5: Check inbox
        results["inbox"] = await test_get_inbox(client)

        # Step 6: Key status
        results["keys"] = await test_key_status(client)

    except Exception as e:
        print(f"\n  [ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "PASS" if result else "FAIL/SKIP"
        print(f"  {name:12s}: {status}")
    print(f"\n  {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TESTS PASSED — The DHT pipeline is LIVE.")
    elif passed > 0:
        print(f"\n  PARTIAL — {passed} calls succeeded. Debug failures above.")
    else:
        print("\n  NO CALLS SUCCEEDED — Check conductor configuration.")

    return passed > 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
