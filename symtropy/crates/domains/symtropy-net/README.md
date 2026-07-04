# symtropy-net

Deterministic lockstep networking for Symtropy.

## What's real today

- **Lockstep protocol** (`src/lockstep.rs`, ~922 LOC): leader rotation,
  state-hash divergence detection, and resync. Tested — see
  `lockstep_two_peers_stay_in_sync_and_resync_on_divergence`.
- **`LoopbackTransport`** (`src/loopback.rs`): in-memory, single-process
  transport. This is the only transport in the crate proven to move
  bytes between peers today.

## What's partial

- **`RelayTransport` / `signaling.rs`** (behind the `webrtc` feature):
  a WebSocket-based signaling client and a transport that relays game
  data over that same WebSocket. Compiles cleanly (fixed 2026-07-04 —
  previously the feature didn't even build, since `tokio-tungstenite`
  wasn't a declared dependency), but has not been validated end-to-end
  against a live signaling server; no integration test drives it.

## What's a stub

- **`IrohTransport`** (`src/iroh_transport.rs`): no `iroh` dependency, no
  QUIC endpoint, no NAT traversal — every method just pushes/drains
  in-memory queues on `self`. See that module's doc comment for the full
  honest breakdown and what a real follow-up would require. The real
  Iroh integration code (~2,143 LOC) lives separately in
  `symthaea/src/swarm/iroh/`, not here.

## Not implemented

There is no Holochain DHT state-sync in this crate. `symtropy-holochain-relay`
is a dependency (`src/peer.rs`) only for its `ConnectionState` enum used in
peer bookkeeping.

## License
AGPL-3.0-or-later. For permissive core types, see `symtropy-net-core`.
