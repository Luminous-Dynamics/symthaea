# Series 27 authoring checklist

## Grounding and scope

- Begin from the exact implemented and verified Series 26 final tree.
- Verify every prerequisite archive and the real Series 21 final tree.
- Consolidate duplicate plans instead of mechanically preserving authoring-kit numbering.
- Record every revision, deferral, rejection, and consolidation in the implementation ledger.
- Do not add new authority semantics unless executable evidence exposes a concrete missing invariant.

## Mandatory execution gates

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

- Run minimal/default/all-feature lanes.
- Run every public binary, example, and doctest under declared features.
- Run canonical Nix verification.
- Run independent-verifier conformance.
- Run deterministic transaction, replay, hostile-input, privacy, and archive tests.
- Replay the real mail series from a clean checkout and reproduce the final tree.
- Rebuild public archives byte-for-byte.

## Claim discipline

- Missing evidence is `not-run` or `unsupported`, never passed.
- Telemetry, documentation, checksums, and artifact-supplied policy are never authority.
- Preserve original bytes and append-only history.
