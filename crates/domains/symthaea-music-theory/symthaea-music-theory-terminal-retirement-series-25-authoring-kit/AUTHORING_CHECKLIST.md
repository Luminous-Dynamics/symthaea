# Series 25 authoring checklist

## Grounding

- Begin from the exact implemented and verified Series 24 final tree.
- Preserve all Series 20–23 incident, recovery, closure, segment, resumption, challenge, and reopening evidence.
- Convert each plan into one intentional mail patch.
- Record exact authored commit and tree identities.
- Replay with sanitized Git configuration and compare final trees.

## Mandatory implementation gates

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

- Run the canonical Nix lane.
- Run independent-verifier fixtures.
- Run replay, staleness, wrong-lineage, race, rollback, privacy, and resource-limit cases.
- Rebuild public archives byte-for-byte.
- Generate claims only from observed evidence.

## Authority discipline

- Every policy and expected identity comes from trusted caller configuration.
- Every state-changing operation reauthenticates at commit time.
- No telemetry, alert, documentation, challenge count, or operator assertion becomes evidence authority.
- Historical decisions remain append-only after later transitions.
