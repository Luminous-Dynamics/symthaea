# Symtropy Multiplayer Demo

An 8-player deterministic multiplayer demo using Lightyear 0.26 and Iroh QUIC transport.

## Features
- **8-Player "Village" Layout**: Each player spawned in a circular formation.
- **Deterministic Rollback**: Groundwork for Morton-code-based spatial partitioning.
- **Spatial Authority**: Dynamic authority management based on proximity.
- **Iroh QUIC Integration**: Low-latency P2P transport.

## Running the Demo

To run as host:
```bash
cargo run -- --host 0
```

To run as client (P1 - P7):
```bash
cargo run -- --client 1
```
