# Bincode Trust Boundary Audit

Date: 2026-06-21

`bincode` remains in the workspace and is currently allowed by audit policy because several subsystems use it for compact internal state. It must not be treated as a safe parser for arbitrary unauthenticated input.

## Current Use Classes

| Class | Examples | Risk |
| --- | --- | --- |
| Local checkpoints and model state | LTC snapshots, HDC persistence, swarm checkpoints | Acceptable if files are local/trusted and versioned |
| SQLite blobs | Causal link vectors in `sqlite_client.rs` | Acceptable for data written by Symthaea itself |
| Network payloads | swarm service, Iroh bridge, TCP backend, RDP frames | Higher risk unless size-bounded and authenticated before decode |
| WASM/artifact payloads | Broca signed artifacts | Acceptable only when signature verification occurs before trusting decoded content |

## Policy

- Do not use `bincode::deserialize` on unauthenticated network bytes.
- Apply a maximum frame size before decode.
- Authenticate or verify signatures before decoded content is trusted.
- Prefer explicit, versioned envelopes for durable files.
- Prefer JSON/MessagePack/CBOR or a schema format for user-provided interchange.

## Follow-Up Targets

The next hardening pass should focus on network-facing decode sites first:

- `src/swarm/service.rs`
- `src/swarm/iroh/mod.rs`
- `src/swarm/iroh/bridge.rs`
- `src/swarm/federated_network/tcp_backend.rs`
- `src/swarm/rdp_protocol.rs`
- `src/swarm/rdp_wire.rs`

Internal checkpoint paths can remain on bincode if they stay local-only, bounded, and versioned.
