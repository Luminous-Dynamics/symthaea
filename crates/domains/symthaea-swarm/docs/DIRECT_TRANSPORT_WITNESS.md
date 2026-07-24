# Direct transport native witness

This witness proves that the direct ALPN crosses a real Iroh connection between
separate operating-system processes. It exercises both delivery primitives:

- an endpoint-signed reliable packet with remote validation and local-queue ACK;
- a same-operation retry that must return a duplicate ACK without redelivery;
- a conflicting same-operation payload that must be rejected;
- an endpoint-signed QUIC datagram with explicit best-effort semantics.

It is intentionally separate from the gossip witness. A green gossip witness
does not prove that the direct physics/control data plane works, and vice versa.

## Run

From the full Symthaea workspace, open two terminals.

Host:

```bash
cargo run -p symthaea-swarm --example direct_transport_witness -- \
  host /tmp/symthaea-direct.addr
```

Joiner:

```bash
cargo run -p symthaea-swarm --example direct_transport_witness -- \
  join /tmp/symthaea-direct.addr
```

Expected terminal markers:

```text
HOST_WITNESS_OK
JOIN_WITNESS_OK
```

The endpoint-address file is written atomically and contains a bounded bincode
encoding of `EndpointAddr`. It is only a local witness exchange mechanism, not a
production invitation format. Production session admission should use a signed,
expiring invitation or an authenticated rendezvous ledger.

## Assertions

The witness fails unless all of the following hold:

1. the joiner completes an authenticated Iroh handshake under `DIRECT_ALPN`;
2. the host validates the reliable packet's endpoint signature;
3. the packet author equals the authenticated QUIC peer;
4. the host queues the reliable packet and returns the matching message-ID ACK;
5. the host sends a reliable response and receives the joiner's queue ACK;
6. an identical retry receives a duplicate ACK and creates no second application event;
7. reuse of the same operation UUID with different bytes is rejected as an operation conflict;
8. the host receives the joiner's datagram through the same direct connection;
9. neither side emits an unexpected structured packet rejection.

The reliable ACK proves queue admission, not domain application or persistence.
A crash after ACK but before domain commit remains an application concern.

## CI lanes

Use three increasingly realistic jobs:

- **loopback:** two processes on one runner, required on every change;
- **relay-only:** direct UDP disabled and both peers forced through the configured
  relay, required before a release candidate;
- **two-host:** peers on separate networks/NATs, required before production.

The relay-only and two-host jobs should record endpoint IDs, connection path,
latency, reconnect count, direct metrics snapshots, and exact crate/git revision.
Do not record endpoint secret keys or unredacted application payloads.
