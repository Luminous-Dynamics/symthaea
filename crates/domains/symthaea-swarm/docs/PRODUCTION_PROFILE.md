# Production profile and extraction boundary

## Current ownership

`symthaea-swarm` now contains three deliberately separated layers:

1. Symthaea domain messages and aggregation;
2. authenticated Iroh gossip control plane;
3. project-neutral direct stream/datagram transport plus real-time admission.

The direct implementation lives here as an incubation and proof point. The
long-term dependency direction should move it into a shared crate rather than
making Symtropy, Xenia, or Mycelix depend on Symthaea domain code.

## What remains Symthaea-specific

- consciousness state announcements;
- proof and law dissemination;
- mutual-aid and governance coordination;
- application `Uuid` to endpoint identity binding;
- swarm aggregation and social-Φ policies.

## What should move to `luminous-iroh-transport`

- persistent endpoint-key loading with atomic creation and strict permissions;
- endpoint builder profiles and shared router lifecycle;
- signed endpoint/session invitations reusable across projects;
- direct connection supervision and reconnect/backoff;
- endpoint-signed generic direct envelopes;
- acknowledged reliable streams and unreliable datagrams;
- bounded framing, replay windows, quotas, and transport metrics;
- real-time authority leases, tick admission, and safety-stop primitives;
- large-object transfer and fault-injection support.

## Capability truth

The gossip socket's capability report still says that *the gossip adapter* does
not provide direct streams, datagrams, or end-to-end ACKs. That remains true.
The separate `DirectTransportCapabilities::DATA_PLANE_V1` reports:

- endpoint-authenticated connections: yes;
- signed packets: yes;
- reliable streams: yes;
- unreliable datagrams: yes;
- remote queue acknowledgements: yes;
- domain-apply acknowledgements: no;
- peer discovery: no.

These reports describe separate protocol objects and are not contradictory.

## Symtropy mapping

| Symtropy need | Iroh lane |
|---|---|
| session discovery and world announcements | signed gossip |
| authority lease/revoke and safety state | acknowledged reliable direct stream |
| player input and high-rate state deltas | signed direct datagram |
| rollback checkpoint | acknowledged reliable direct stream |
| large initial state or asset bundle | future dedicated stream/blob protocol |
| governance and economic records | gossip announcement plus durable application protocol |

## Required production gates

Before governance or economic deployment:

- gossip two-process and two-host witnesses;
- persistent endpoint keys;
- pinned identity enrollment and tested key rotation;
- durable application outbox/inbox ledgers;
- domain acknowledgements and idempotent effects;
- malformed-input, idempotency-conflict, and rate-limit stress tests;
- relay-only and reconnect tests;
- clock-skew policy tests;
- security review of invitation distribution;
- metrics export and alert thresholds.

Before robotics or physics deployment, additionally require:

- direct loopback, relay-only, and two-host witnesses;
- datagram loss/reorder/duplication fault injection;
- authority-lease and safety-stop model checking;
- latency and stale-command budgets;
- checkpoint/recovery and reconnect convergence tests;
- hardware-in-the-loop emergency-stop tests;
- explicit fail-safe behavior when event queues, clocks, or authority ledgers
  become unavailable.

The current code establishes the protocol invariants and witness harnesses. It
has not yet passed all of these production gates.
