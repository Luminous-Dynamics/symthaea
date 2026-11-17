# subterra-operation-contract

A dependency-free Rust reference state machine for one bounded physical operation.

## Contract

The crate keeps these events distinct:

1. immutable work-package preflight;
2. command issuance;
3. device acknowledgement;
4. actuator telemetry;
5. measured physical delta;
6. independent postcondition verification;
7. accepted closeout or explicit rejection/insufficient evidence.

A timeout after command issuance creates `OutcomeUnknown`. The affected work scope remains locked until an independent reconciliation establishes either that the operation was not executed or that it executed and still requires physical inspection.

An insufficient verification does not become an execution-unknown state. It remains locked while additional evidence is gathered and a new verifier receipt is assessed.

## Verify locally

```sh
cargo fmt --manifest-path Cargo.toml --check
cargo test --manifest-path Cargo.toml --all-targets
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
```

## Explicit limitations

- No hardware I/O or robot command transport.
- No cryptographic digest calculation, signature verification or identity authentication.
- No geotechnical/structural adequacy model.
- No proof that sensor declarations reflect real physical independence.
- No durable journal or crash-consistency guarantee.
- No claim of mine safety, equipment certification or field qualification.

This crate validates state-machine semantics only. The independent evidence and authority layers must be supplied by reviewed systems.
