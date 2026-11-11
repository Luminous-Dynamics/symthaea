# MACE-CoS release ledger v0.1

Research-only, dependency-free Go reference for cumulative cross-request declassification budgets, exact grant binding, one-shot consumption, revocation generations, idempotency, and unknown-outcome reconciliation.

## Bundle

- [Source archive](https://raw.githubusercontent.com/Luminous-Dynamics/symthaea/research/mace-release-ledger-v0.1/research/mace-release-ledger-v0.1.tar.gz)
- SHA-256: `b6f9679e78b3d7eb274f8a11b59aca0c6cfbe7d1e837f4e76b25b8cd5417e22f`
- Companion ZIP SHA-256: `acc5a0598b7c47b6838026e2dc219f46a350d86af0ca879f10fe4b13092d2b1e`

The archive contains the complete Go source, tests, schema, threat model, benchmark protocol, latency note, results, checksum manifest and `verify.sh`.

## Local verification

Go 1.23.2, linux/amd64; standard library only.

- 23 named test functions.
- Normal tests, race detector, `go vet`, `gofmt`, and fresh-extraction checksum verification pass.
- 10,000 fixed-seed sequential budget attempts checked against an independent budget oracle.
- 128 concurrent workers against one one-shot grant: exactly one dispatch permit.
- 200 concurrent requests against a shared budget of 37: exactly 37 accepted.
- 84.9% statement coverage.
- Five 50,000-operation single-thread in-memory consume microbenchmarks: median p50 2.263 µs, p95 6.971 µs, p99 42.312 µs.

## Claim boundary

This is an in-memory, single-process reference state machine—not a durable or multi-instance broker. It does not establish crash consistency, database transactions, remote gateway fencing, signature or issuer authentication, reconciliation evidence authenticity, a differential-privacy guarantee, or production deployment readiness. The latency figures exclude persistence, network calls and external actions. See the archive's `README.md`, `THREAT_MODEL.md`, `RESULTS.md`, and `BENCHMARK_PROTOCOL.md`.

Related work: [#11619](https://github.com/Luminous-Dynamics/symthaea/issues/11619), [MACE-CoS #11557](https://github.com/Luminous-Dynamics/symthaea/issues/11557), [paired-world oracle #11601](https://github.com/Luminous-Dynamics/symthaea/issues/11601), and [action authority #9490](https://github.com/Luminous-Dynamics/symthaea/issues/9490).
