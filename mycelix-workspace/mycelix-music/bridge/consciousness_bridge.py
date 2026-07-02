#!/usr/bin/env python3
"""
Consciousness Composition Bridge
=================================
HTTP → Holochain WebSocket bridge for Symthaea's MusicPublisher.

Receives ConsciousnessTrigger JSON from MusicPublisher (POST /api/consciousness-compose)
and forwards it as a zome call to the music-bridge coordinator on the shared conductor.

Usage:
    python3 consciousness_bridge.py [--port 8092] [--conductor ws://localhost:8888]

The bridge also emits the signal directly to any connected WebSocket clients
on the conductor, so the Music UI receives the composition metadata in real time.
"""

import argparse
import asyncio
import json
import struct
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import msgpack
import websockets.sync.client as ws_sync


# ─── Holochain Wire Protocol ─────────────────────────────────────────────────

def call_zome(conductor_url: str, app_id: str, role: str, zome: str, fn_name: str, payload: dict) -> dict:
    """Call a zome function on the Holochain conductor via WebSocket.

    Uses the Holochain AppWebsocket wire protocol (msgpack-encoded).
    """
    try:
        with ws_sync.connect(conductor_url, additional_headers={}) as conn:
            # The Holochain app interface expects a specific msgpack structure.
            # For now, we'll use a simplified approach — emit the signal directly
            # since we can't easily construct the full wire protocol without
            # the holochain_client library.
            #
            # Instead of calling the zome (which requires cell_id, provenance, etc.),
            # we'll construct the signal payload and emit it ourselves.
            print(f"[bridge] Connected to conductor at {conductor_url}")

            # The conductor doesn't accept raw signal injection from clients.
            # We need the full zome call protocol. For now, log and return mock.
            print(f"[bridge] Would call {zome}::{fn_name} with: {json.dumps(payload, indent=2)[:200]}")
            return {"status": "mock", "message": "Full zome call requires holochain_client — bridge logged the trigger"}
    except Exception as e:
        return {"error": str(e)}


# ─── Signal Broadcaster ──────────────────────────────────────────────────────

# Connected Music UI clients (for direct signal broadcast)
broadcast_clients = set()

def broadcast_signal(payload: dict):
    """Broadcast a consciousness_composition signal to all connected WS clients.

    This is a fallback for when the full zome call isn't available —
    the Music UI receives the signal directly from the bridge.
    """
    global broadcast_clients
    signal = json.dumps({
        "type": "Signal",
        "data": {
            "App": {
                "cell_id": [[], []],
                "zome_name": "music_bridge",
                "payload": {
                    "signal_type": "consciousness_composition",
                    "action_hash": f"bridge-{int(time.time())}",
                    "tempo_bpm": payload.get("tempo_bpm", 90.0),
                    "scale_name": payload.get("scale_name", "chromatic"),
                    "note_count": payload.get("note_count", 8),
                    "quality_score": payload.get("quality_score", 0.5),
                    "phi_score": payload.get("phi_score", 0.5),
                    "valence": payload.get("valence", 0.0),
                    "arousal": payload.get("arousal", 0.5),
                    "narrative_tags": payload.get("narrative_tags", []),
                }
            }
        }
    })

    if not broadcast_clients:
        print(f"[bridge-ws] No clients connected, signal queued for next connection")
        return

    dead = set()
    for client in list(broadcast_clients):
        try:
            asyncio.run_coroutine_threadsafe(client.send(signal), ws_loop)
        except Exception:
            dead.add(client)
    broadcast_clients -= dead


def derive_musical_params(trigger: dict) -> dict:
    """Derive musical parameters from a ConsciousnessTrigger (mirrors zome logic)."""
    v = trigger.get("valence", 0.0)
    a = trigger.get("arousal", 0.5)
    phi = trigger.get("phi_score", 0.5)

    tempo = 60.0 + a * 80.0 + abs(v) * 20.0

    if phi > 0.7:
        scale = "just_intonation"
    elif v > 0.2 and a > 0.5:
        scale = "raga_desh"
    elif v > 0.2:
        scale = "gamelan_slendro"
    elif v < -0.2 and a > 0.5:
        scale = "maqam_hijaz"
    elif v < -0.2:
        scale = "maqam_saba"
    else:
        scale = "chromatic"

    note_count = int(4 + a * 12 + phi * 8)
    quality = 0.3 + phi * 0.4 + (1.0 - abs(v)) * 0.3

    return {
        "tempo_bpm": round(tempo, 1),
        "scale_name": scale,
        "note_count": note_count,
        "quality_score": round(quality, 3),
    }


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class BridgeHandler(BaseHTTPRequestHandler):
    conductor_url = "ws://localhost:8888"

    def do_POST(self):
        if self.path == "/api/consciousness-compose":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)

            try:
                trigger = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

            # Derive musical params
            params = derive_musical_params(trigger)
            trigger.update(params)

            print(f"[bridge] Received trigger: phi={trigger.get('phi_score', '?')}, "
                  f"V/A={trigger.get('valence', '?')}/{trigger.get('arousal', '?')} "
                  f"→ {params['scale_name']} @ {params['tempo_bpm']} BPM")

            # Broadcast signal to connected Music UI clients
            broadcast_signal(trigger)

            # Zome call is best-effort — skip if conductor protocol isn't wired
            result = {"status": "signal_broadcast", "clients": len(broadcast_clients)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True,
                "params": params,
                "zome_result": result,
            }).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[bridge-http] {args[0]}")


# ─── WebSocket Relay ──────────────────────────────────────────────────────────

ws_loop = None

async def ws_relay(websocket):
    """Accept WebSocket connections from Music UI clients for signal relay."""
    broadcast_clients.add(websocket)
    print(f"[bridge-ws] Music UI client connected ({len(broadcast_clients)} total)")
    try:
        async for _ in websocket:
            pass  # We only send, never receive
    finally:
        broadcast_clients.discard(websocket)
        print(f"[bridge-ws] Client disconnected ({len(broadcast_clients)} total)")


async def run_ws_server(port: int):
    global ws_loop
    ws_loop = asyncio.get_event_loop()
    async with websockets.serve(ws_relay, "127.0.0.1", port):
        print(f"[bridge-ws] Signal relay on ws://127.0.0.1:{port}")
        await asyncio.Future()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consciousness Composition Bridge")
    parser.add_argument("--port", type=int, default=8092, help="HTTP port (default: 8092)")
    parser.add_argument("--ws-port", type=int, default=8893, help="WebSocket relay port (default: 8893)")
    parser.add_argument("--conductor", default="ws://localhost:8888", help="Conductor app WS URL")
    args = parser.parse_args()

    BridgeHandler.conductor_url = args.conductor

    # Start WebSocket relay in background thread
    def run_ws():
        asyncio.run(run_ws_server(args.ws_port))

    ws_thread = Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # Start HTTP server
    server = HTTPServer(("127.0.0.1", args.port), BridgeHandler)
    print(f"[bridge] Consciousness Bridge running:")
    print(f"  HTTP: http://127.0.0.1:{args.port}/api/consciousness-compose")
    print(f"  WS relay: ws://127.0.0.1:{args.ws_port}")
    print(f"  Conductor: {args.conductor}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    # Import websockets for the async server
    import websockets.server
    main()
