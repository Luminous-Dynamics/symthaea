# Direct transport and real-time migration

## Why the direct plane is separate

Iroh Gossip remains appropriate for many-to-many discovery, proofs, laws,
governance, and eventual state announcements. It is not the right primitive for
per-tick input, rollback snapshots, robotics commands, or peer-specific bulk
state.

Round three added `direct::DirectTransport` under a separate ALPN. Both protocols
can share one endpoint and router without making the gossip socket own endpoint
lifecycle.

## Router registration

Register the direct handler with the same endpoint used by gossip:

```rust,ignore
let direct = DirectTransport::new(endpoint.clone(), direct_events_tx)?;
let router = Router::builder(endpoint)
    .accept(iroh_gossip::net::GOSSIP_ALPN, swarm.gossip_protocol())
    .accept(DIRECT_ALPN, direct.protocol_handler())
    .spawn();
```

Constructing `DirectTransport` does not install an accept loop. Dropping the
router ends inbound protocol handling.

## Connecting peers

`connect(EndpointAddr)` establishes and supervises one direct connection. When
both endpoints dial simultaneously, the lower endpoint ID deterministically
retains its outgoing connection and the higher endpoint retains that same
physical connection as incoming. This prevents each side from selecting a
different duplicate.

For peers that should remain connected:

```rust,ignore
let maintained = direct
    .maintain_connection(peer_addr, ReconnectPolicy::default())
    .await?;
```

The reconnect task uses bounded exponential backoff with deterministic
per-session jitter. Pinned-only admission failures, self-connections, identity
mismatches, and capacity failures stop rather than retry forever.

## Reliable delivery semantics

```rust,ignore
let receipt = direct
    .send_reliable(peer, DirectLane::CONTROL, payload)
    .await?;
assert!(receipt.remote_queue_accepted);
```

The receiver verifies:

1. signed envelope version, expiry, and size;
2. envelope signature;
3. claimed author equals the authenticated QUIC peer;
4. delivery primitive matches the signed envelope;
5. message ID and sequence are not replayed;
6. the local application event queue has capacity.

Only then does it return the matching message-ID ACK. For retryable domain
operations, use a stable operation UUID:

```rust,ignore
let receipt = direct
    .send_reliable_idempotent(peer, DirectLane::CONTROL, operation_id, payload)
    .await?;
```

If the first ACK is lost after queue admission, a retry with the same operation
UUID and identical lane/payload is acknowledged with `remote_duplicate = true`
without queueing the action a second time. Reusing that UUID for a different
lane or payload is rejected as `OperationConflict`. Durable effects still need a
domain-level commit/status protocol and persisted idempotency beyond the
transport process lifetime.

## Datagram semantics

```rust,ignore
let receipt = direct
    .send_datagram(peer, DirectLane::PLAYER_INPUT, payload)
    .await?;
```

The encoded frame must fit both the protocol cap and the active QUIC path MTU.
A receipt means local QUIC accepted the datagram. Delivery, order, uniqueness,
and application admission are not guaranteed.

Snapshots larger than the current datagram path maximum must be fragmented by a
domain protocol or sent over a reliable stream. The transport deliberately does
not perform invisible fragmentation that could create head-of-line blocking.

## Pinned production profile

```rust,ignore
direct.enroll_peer(expected_endpoint).await?;
direct.set_peer_policy(DirectPeerPolicy::PinnedOnly).await;
```

Switching to pinned-only immediately closes currently connected peers that are
not enrolled. Removing an enrollment also closes that peer while pinned-only is
active.

Persist pinned identities with `DirectEnrollmentBook`. Endpoint replacement can
use `DirectPeerRolloverProof`, which expires and requires signatures from both
the old and new endpoint keys. Organizational authorization, revocation
publication, and recovery remain external policy.

## Symtropy real-time frames

Use `RealtimeFrame` instead of sending opaque physics/control bytes directly:

```rust,ignore
let frame = RealtimeFrame::new(
    world_id,
    vehicle_id,
    authority_epoch,
    tick,
    tick + 2,
    RealtimeFrameKind::PlayerInput,
    encoded_input,
)?;
direct.send_realtime(peer, frame).await?;
```

`send_realtime` chooses the required lane and primitive:

| Frame kind | Lane | Primitive |
|---|---|---|
| authority lease/revoke | control | reliable |
| safety stop/resume | control | reliable |
| player input | player input | datagram |
| state delta | state snapshot | datagram |
| checkpoint | state snapshot | reliable |
| telemetry | telemetry | datagram |
| robotics command | robotics | reliable |

The receiver passes authenticated direct messages through
`RealtimeAdmissionGuard::admit_direct`. The guard validates authority issuers,
controller endpoint, epoch, lease range, frame freshness, duplicate/reorder
window, and safety-stop state before domain dispatch.

## Remaining extraction work

The direct and real-time modules are intentionally implemented here first so
they can be proven against Symthaea's Iroh 0.97 endpoint. They should next move
into `luminous-iroh-transport` with:

- persistent endpoint-key loading;
- shared endpoint-builder profiles;
- signed project-neutral connection tickets;
- exporter integration for readiness and metrics;
- metrics exporter integration;
- large-object transfer;
- compatibility and wire-version test vectors.
