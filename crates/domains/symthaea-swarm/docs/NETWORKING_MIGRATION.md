# Authenticated networking migration

## Wire protocol v2

The original transport serialized `SwarmMessage` directly. Round one introduced a
signed envelope. Round two advances that envelope to wire protocol v2 with a
signed process `session_id` and session-local sequence number.

All peers on a topic must use the same wire version. Protocol v2 deliberately
rejects v1 rather than attempting an ambiguous downgrade.

The session identifier allows a node with a persistent Iroh key to restart its
sequence counter without weakening replay protection. Within a session, duplicate
or excessively old sequence values are rejected even when an attacker changes an
otherwise valid timestamp.

## API migration

Preferred construction remains:

```rust,ignore
let socket = TelepathicSocket::new_authenticated(endpoint, topic, events_tx).await?;
let task = tokio::spawn(socket.clone().run());
```

The legacy constructor remains temporarily available but discards authenticated
author, neighbor, lifecycle, and rejection metadata.

Use:

```rust,ignore
let receipt = socket.broadcast_tracked(message).await?;
```

when the application needs the signed message ID and sequence for an outbox,
acknowledgement, audit, or idempotency ledger. `broadcast(message)` remains as a
compatibility wrapper that discards the local receipt.

## Router requirement

Register `socket.gossip_protocol()` under `iroh_gossip::net::GOSSIP_ALPN` on the
same endpoint before expecting inbound connections. Constructing a gossip actor
alone does not install an endpoint accept loop.

## Signed invitations

Invitations use their own schema version (`INVITE_PROTOCOL_VERSION = 1`) so future envelope changes do not unnecessarily invalidate otherwise compatible invitation tooling.


A host now exports a signed and expiring invitation:

```rust,ignore
let invite = socket.invite_online(DEFAULT_ONLINE_WAIT).await?;
let encoded = invite.encode()?;
```

The issuer's Iroh endpoint key signs the topic, time bounds, and complete bootstrap
address set. A rendezvous channel can still suppress an invitation, but it cannot
silently rewrite those fields without detection.

A joining node must:

1. decode and verify the invitation;
2. construct a `MemoryLookup`;
3. install a clone on its endpoint builder;
4. pass the same lookup to `from_invite_authenticated`;
5. register the gossip protocol on a router;
6. start the socket run loop;
7. call `wait_for_neighbors` before assuming a usable swarm connection.

Distribution through QR, Mycelix/Holochain, a rendezvous service, or LAN discovery
belongs above this crate.

## Identity binding

Two profiles are available:

- `TrustOnFirstUse` records the first valid endpoint/UUID mapping.
- `PinnedOnly` rejects every mapping not explicitly enrolled in the loaded
  `IdentityBook`.

`IdentityBook::encode` and `IdentityBook::decode` provide a bounded versioned
snapshot for application-managed persistence. Protect that file with appropriate
filesystem permissions and integrity controls. This snapshot is not itself a key
rotation protocol.

A production enrollment system should record signed old-key/new-key rotation
authorizations, revocations, and operator policy separately.

## Queue behavior

- state, haptic, macro, curvature, social-Phi, rejection, and similar freshness
  events are best effort locally;
- proof, law, aid, weight-update, lag, and topology events are durable locally;
- a full best-effort queue increments `best_effort_dropped`;
- a full durable queue increments `durable_queue_full` and fails the socket
  immediately;
- the network receive loop never waits five seconds on a blocked application
  consumer.

The application should choose an adequately sized inbound channel and drain it in
an independent task.

## Abuse controls and observability

The adapter applies a bounded message and byte budget to each immediate gossip neighbor before deserialization. `RateLimitConfig` can change the window and limits before `run()` starts; zero or otherwise invalid limits are rejected. Structured rejection reasons distinguish
rate limits, decode failures, invalid signatures, replayed sequences, oversized
messages, and identity-binding failures.

Configuration setters share a startup gate with `run()`, preventing a bootstrap, identity-policy, identity-book, or rate-limit update from racing subscription startup.

Metrics now include received/sent bytes, send failures, rate limiting, replay rejections, durable queue failures, best-effort drops, and gossip lag.

## Capability truth

`socket.capabilities()` currently reports:

- signed origins: yes
- expiring signed invitations: yes
- gossip broadcast: yes
- authenticated direct streams: no
- unreliable datagrams: no
- end-to-end acknowledgements: no

Do not use gossip-local durability claims as evidence of remote delivery.

## Round-three direct data plane

Round three does not change gossip wire protocol v2. It adds a separate
`luminous/direct/2` protocol in `crate::direct` and a simulation/robotics
admission layer in `crate::realtime`.

`TelepathicSocket::capabilities()` continues to describe only the gossip object,
so its direct-stream and datagram fields remain false. Use
`DirectTransport::capabilities()` for the separate data plane.

See [direct transport migration](DIRECT_TRANSPORT_MIGRATION.md) before moving
per-tick or robotics traffic off the legacy in-memory Symtropy adapter.
