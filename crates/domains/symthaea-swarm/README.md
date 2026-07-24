# symthaea-swarm

Transport-independent swarm messages and aggregation rules with two optional,
authenticated Iroh networking planes:

- **gossip control plane** for many-to-many state, proof, law, and governance
  dissemination;
- **direct data plane** for peer-to-peer reliable streams and unreliable QUIC
  datagrams.

Both planes retain explicit capability reports and intentionally avoid claiming
more delivery certainty than they provide.

## Gossip control plane

`networking::TelepathicSocket` provides:

- endpoint-signed origin envelopes;
- signed, expiring invitations containing real `EndpointAddr` bootstrap data;
- protocol versioning, expiry, payload bounds, duplicate suppression, and
  session-aware sequence replay protection;
- one-to-one binding between an application `Uuid` and an Iroh `EndpointId`;
- trust-on-first-use and fail-closed pinned identity profiles;
- bounded per-neighbor message and byte rates;
- explicit lifecycle, neighbor events, structured rejections, metrics, and
  local broadcast receipts;
- fail-fast durable local backpressure.

A gossip broadcast receipt proves local gossip acceptance. It is not remote
receipt, application commit, or persistence evidence.

## Direct data plane

`direct::DirectTransport` provides:

- a separate `luminous/direct/2` ALPN for the shared Iroh router;
- endpoint-signed generic lane envelopes;
- authenticated peer-to-peer connections with deterministic simultaneous-dial
  resolution;
- reliable bidirectional streams with remote validation and local-queue
  ACK/NACK responses;
- unreliable datagrams with path-MTU checks and receive-rate limits;
- replay windows, expiry, bounded framing, peer caps, and pinned-only admission;
- signed, lane/payload-bound idempotency keys for duplicate-safe reliable retries;
- persistent pinned-peer books and dual-signed endpoint-key rollover;
- maintained peer sessions with bounded exponential reconnect and jitter;
- connection-health and transport-metrics snapshots.

A reliable ACK proves that the remote protocol validated and queued the frame.
It does not prove domain application or durable commit. Datagram receipts prove
local QUIC acceptance only.

## Real-time authority and safety

`realtime` adds the Symtropy/robotics admission contract above direct transport:

- authority issuers grant bounded controller leases;
- authority epochs prevent stale controller resurrection;
- frame ticks and expiry ticks bound simulation freshness;
- lane and delivery requirements prevent authority changes from using
  datagrams;
- trusted safety issuers can latch and clear emergency stops;
- player input and robotics commands are blocked while stopped;
- bounded reorder windows admit rollback-friendly traffic without accepting
  duplicates or arbitrarily stale ticks;
- monotonic world clocks and staged authority handoffs prevent premature control
  transitions;
- explicit subject retirement cannot erase staged leases or safety stops.

The module contains no physics types and is intended to move with `direct` into
a future project-neutral `luminous-iroh-transport` crate.

## Shared router setup

```rust,ignore
use iroh::{
    Endpoint,
    address_lookup::memory::MemoryLookup,
    endpoint::presets,
    protocol::Router,
};
use symthaea_swarm::{
    direct::{DIRECT_ALPN, DirectEvent, DirectTransport},
    networking::{SwarmNetworkEvent, TelepathicSocket},
};
use tokio::sync::mpsc;

let lookup = MemoryLookup::new();
let endpoint = Endpoint::builder(presets::N0)
    .address_lookup(lookup.clone())
    .bind()
    .await?;

let (swarm_tx, mut swarm_rx) = mpsc::channel::<SwarmNetworkEvent>(256);
let swarm = TelepathicSocket::new_authenticated(
    endpoint.clone(),
    [7u8; 32],
    swarm_tx,
)
.await?;

let (direct_tx, mut direct_rx) = mpsc::channel::<DirectEvent>(512);
let direct = DirectTransport::new(endpoint.clone(), direct_tx)?;

let router = Router::builder(endpoint)
    .accept(iroh_gossip::net::GOSSIP_ALPN, swarm.gossip_protocol())
    .accept(DIRECT_ALPN, direct.protocol_handler())
    .spawn();

let swarm_task = tokio::spawn(swarm.clone().run());
```

Keep the router alive for the node lifetime. During shutdown, stop the swarm and
direct handles, await the swarm task, and then await `router.shutdown()`.

## Witnesses and migration

- [gossip networking migration](docs/NETWORKING_MIGRATION.md)
- [gossip native witness](docs/TWO_PROCESS_WITNESS.md)
- [direct transport migration](docs/DIRECT_TRANSPORT_MIGRATION.md)
- [direct native witness](docs/DIRECT_TRANSPORT_WITNESS.md)
- [production profile and extraction boundary](docs/PRODUCTION_PROFILE.md)
- [round-four hardening](docs/ROUND4_HARDENING.md)
- [Symtropy / Lightyear adapter contract](docs/SYMTROPY_LIGHTYEAR_INTEGRATION.md)
