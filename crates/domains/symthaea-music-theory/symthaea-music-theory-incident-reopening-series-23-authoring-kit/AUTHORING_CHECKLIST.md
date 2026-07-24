# Series 23 authoring checklist

## Base and replay

- Begin from the exact demonstrated Series 22 final tree produced from the Series 21 base.
- Verify the Series 21 archive and its internal `SHA256SUMS`.
- Preserve the complete Series 20–21 incident, recovery, re-entry, quarantine, authority, and closure lineage.
- Convert each plan into one intentional mail patch with stable subject, rationale, tests, and non-claims.
- Record authored commit and tree identities; replay with sanitized Git configuration and compare final trees.

## Required implementation gates

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

- Run canonical Nix verification.
- Run examples and shell-free external-verifier adapters.
- Run frozen positive, boundary, replay, tampering, stale-state, concurrency, and resource-limit fixtures.
- Reproduce every public archive byte-for-byte.
- Generate claims only from observed evidence.

## Authority discipline

- Do not trust policy, thresholds, freshness, or expected identities merely because they are embedded in an artifact.
- Reauthenticate all mutable authority at the exact state-changing boundary.
- Do not allow telemetry, alerts, documentation, or operator convenience to satisfy evidence authority.
- Preserve rejected and superseded history append-only.
