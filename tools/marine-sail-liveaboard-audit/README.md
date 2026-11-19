# Marine Sail-Liveaboard Audit

Standalone Rust CLI for the P0 cross-owner mission-composition contract in [Symthaea #12884](https://github.com/Luminous-Dynamics/symthaea/issues/12884).

This tool does not generate routes, simulate vessel physics, calculate energy, schedule services, authorize operation or certify a vessel. It audits manifest semantics and compares whether two policy cases share the exact mission, configuration, environment and service inputs.

## Commands

```bash
cargo test --manifest-path tools/marine-sail-liveaboard-audit/Cargo.toml
cargo run --manifest-path tools/marine-sail-liveaboard-audit/Cargo.toml -- audit-case docs/research/marine/fixtures/synthetic-sail-liveaboard-case-0001.json
cargo run --manifest-path tools/marine-sail-liveaboard-audit/Cargo.toml -- compare-pair baseline.json candidate.json
cargo clippy --manifest-path tools/marine-sail-liveaboard-audit/Cargo.toml --all-targets -- -D warnings
```

## Dispositions

- `INVALID`: malformed identity, digest, time or gate structure.
- `BLOCKED`: future-information leakage, failed/unresolved gate, expired/revoked evidence or other hard blocker.
- `INCOMPLETE`: missing required owner roles, resource dimensions, or insufficient evidence class.
- `SYNTHETIC_ONLY`: internally consistent synthetic example; never a physical-performance claim.
- `ELIGIBLE_FOR_COMPARISON`: manifest is structurally eligible for a matched comparison under its stated inputs; not a finding that a policy is better or safe.

Pair comparison returns `INCOMPARABLE` when mission/configuration/service/forecast/hard-gate inputs differ, when evidence inputs differ (except the candidate-specific sail-route output), or when a deployable policy is compared with the perfect-foresight evaluator-only arm.

## Invariants

- Forecast issue time <= availability time <= decision time for deployable policies.
- Realized environment is kept outside deployable decision input.
- Every case includes the complete liveaboard resource-dimension roster.
- Every required owner role and hard gate is explicit.
- Unknown, partial, expired, revoked and synthetic evidence cannot be promoted into a non-synthetic comparison. A route that automatically relaxes a hard comfort requirement is BLOCKED; a relaxed soft preference remains INCOMPLETE. Any fallback/no-route outcome must include an evidence reference or the manifest is INVALID.
- Failed or unresolved hard gates cannot be compensated by energy or cost improvements.
- Matching is by exact identifiers/digests and service/mission fields, not inferred rescaling.

A SHA-256-shaped string is only a syntax/identity check. This CLI does not authenticate the signer, source publisher, physical instrument, calibration or underlying measurement.
