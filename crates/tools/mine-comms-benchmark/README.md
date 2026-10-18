# MINE-COMMS-001 deterministic benchmark kernel

This standard-library-only Rust crate is the first executable slice of the underground communications benchmark. It models route selection over a declared graph and typed message receipts. Its graph and link values are synthetic inputs, not a mine propagation model.

## Run

From a Rust toolchain compatible with edition 2024:

```sh
cargo test --offline --manifest-path crates/tools/mine-comms-benchmark/Cargo.toml
cargo run --offline --manifest-path crates/tools/mine-comms-benchmark/Cargo.toml --bin mine-comms-benchmark
```

## What is checked

- Stable deterministic route selection from explicit directed links.
- A fixed channel-preference baseline compared with a link-quality/latency/energy-weighted adaptive policy.
- Down, unknown, zero-quality and stale observations are not routed as healthy.
- Receiver acknowledgement requires a usable return path; local transmitter acceptance is never reported as end-to-end delivery.
- Message expiry is evaluated against modelled arrival time; messages already expired at evaluation are not sent, deliveries are never timestamped before evaluation time, and ACK routes are rechecked for freshness at receiver arrival.
- Unauthenticated-origin distress can still be received and acknowledged, but its origin remains explicitly unverified.
- Each receipt carries a typed lifecycle trace; missing routes and unacknowledged reception are distinct outcomes.
- The CLI prints synthetic comparisons for nominal routing, backbone severance, local-relay loss, stale TTE observations, a missing acknowledgement path and an expired message.
- Bounded queue planning puts distress/evacuation messages ahead of telemetry/bulk and removes expired entries; this is message-count prioritization only, not a radio airtime or contention model.
- A single-channel airtime model serializes frame bytes from an explicit bit rate, adds propagation delay, consumes airtime for bounded injected failed attempts, enforces message expiry, and reports queue wait and receiver-arrival time. It uses deterministic attempt traces, not random loss or physical propagation. It assumes a single strict-priority sender with no contention, backoff or acknowledgement airtime.
- Bounded failure-domain-diverse route search accounts for declared cable/shaft/power/water dependencies and shared internal relays. Missing/incomplete domain evidence cannot support a redundancy claim; candidate enumeration is explicitly capped.

## Claim ceiling and next step

The next protocol design should evaluate standards-based store-carry-forward (IETF BPv7/RFC 9171) and bundle integrity/confidentiality (BPSec/RFC 9172), rather than assuming a custom protocol is justified. The harness does not yet implement these standards.

The harness does not simulate electromagnetic propagation, stochastic packet loss, routing contention/medium access, receiver-acknowledgement airtime, general multi-hop queues, physical radio energy, mine safety or regulatory permissibility. Link quality, latency and energy are supplied synthetic measurements. The next increment should add frozen traces and reproducible fault schedules, then calibrate against instrumented link measurements. HDC/CfC is not presumed better; it must beat simpler policies on held-out scenarios.

See [the MINE-COMMS-001 specification](../../../docs/research/MINE_COMMUNICATIONS_BENCHMARK_V0_1.md) and [issue #8010](https://github.com/Luminous-Dynamics/symthaea/issues/8010).


## Physical TTE characterization

[MINE-TTE-CHAR-001 #8047](https://github.com/Luminous-Dynamics/symthaea/issues/8047) defines calibrated bench measurements and held-out physical validation to replace purely synthetic channel assumptions. It is a research/test protocol, not permission to operate unapproved transmitters in a mine.


Time semantics: the message evaluator rejects messages already expired when evaluated, never assigns retroactive transmission/arrival times, and rechecks ACK-link freshness when the receiver gets the message.

## Deterministic fault-campaign evaluator

The kernel now generates a reproducible suite containing one nominal case, one case per link failure, and one case per declared shared failure domain. The generator requires complete, non-empty dependency inventories and fails rather than silently truncating the suite when its case cap would be exceeded.

\`evaluate_fault_campaign\` runs each declared fault snapshot at a common evaluation time and reports four distinct outcomes: on-time delivery, deadline miss, no route, and expiry before delivery. Scenarios can fail links, fail shared domains, or make selected observations stale. Unknown references, conflicting faults, duplicate scenario IDs, incomplete domain inventories for domain-outage cases and malformed deadlines are rejected.

This is **snapshot-based graph/latency evaluation**, not yet a time-evolving multi-hop network simulation. It does not model faults that occur while a message is in flight, per-hop airtime queues, random loss, contention or store-carry-forward. Its results are synthetic benchmark evidence only. The next increment should replay frozen time-series link traces with per-hop queue/ACK budgets, then compare calibrated predictions against held-out measured traces.

