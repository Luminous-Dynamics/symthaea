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
- Message expiry is evaluated against modelled arrival time.
- Unauthenticated-origin distress can still be received and acknowledged, but its origin remains explicitly unverified.
- Each receipt carries a typed lifecycle trace; missing routes and unacknowledged reception are distinct outcomes.
- The CLI prints synthetic comparisons for nominal routing, backbone severance, local-relay loss, stale TTE observations, a missing acknowledgement path and an expired message.
- Bounded queue planning puts distress/evacuation messages ahead of telemetry/bulk and removes expired entries; this is message-count prioritization only, not a radio airtime or contention model.
- A single-channel airtime model serializes frame bytes from an explicit bit rate, adds propagation delay, consumes airtime for bounded injected failed attempts, enforces message expiry, and reports queue wait and receiver-arrival time. It uses deterministic attempt traces, not random loss or physical propagation. It assumes a single strict-priority sender with no contention, backoff or acknowledgement airtime.

## Claim ceiling and next step

The next protocol design should evaluate standards-based store-carry-forward (IETF BPv7/RFC 9171) and bundle integrity/confidentiality (BPSec/RFC 9172), rather than assuming a custom protocol is justified. The harness does not yet implement these standards.

The harness does not simulate electromagnetic propagation, stochastic packet loss, routing contention/medium access, receiver-acknowledgement airtime, general multi-hop queues, physical radio energy, mine safety or regulatory permissibility. Link quality, latency and energy are supplied synthetic measurements. The next increment should add frozen traces and reproducible fault schedules, then calibrate against instrumented link measurements. HDC/CfC is not presumed better; it must beat simpler policies on held-out scenarios.

See [the MINE-COMMS-001 specification](../../../docs/research/MINE_COMMUNICATIONS_BENCHMARK_V0_1.md) and [issue #8010](https://github.com/Luminous-Dynamics/symthaea/issues/8010).
