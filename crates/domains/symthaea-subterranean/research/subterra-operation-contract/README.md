# subterra-operation-contract

A dependency-free Rust reference state machine for one bounded physical operation.

## Contract

The crate keeps these events distinct:

1. immutable work-package preflight bound to work-package, design, coordinate-frame, calibration and authority digests;
2. command issuance and an idempotency key;
3. device acknowledgement;
4. actuator telemetry bound to the calibration digest;
5. measured physical delta bound to the work package, design, coordinate frame and calibration;
6. independent postcondition verification bound to the exact physical-delta digest;
7. accepted closeout or explicit rejection / insufficient evidence.

A timeout after command issuance creates `OutcomeUnknown`. The affected work scope remains locked until an independent reconciliation establishes either that the operation was not executed or that it executed and still requires physical inspection.

Idempotency keys are reserved across attempts in one registry, not just within a single attempt. A new attempt cannot reuse an earlier command key.

An insufficient verification does not become an execution-unknown state. It remains locked while additional evidence is gathered and a new verifier receipt is assessed.

A failed postcondition enters `ReworkReview`. The work scope remains locked until a responsible authority supplies a new, higher-revision work package, design, and authority evidence. The registry then permits only an attempt matching that exact rework authorization.

Transition history retains typed events and evidence references, with a hard per-attempt event cap. At the cap, the attempt remains locked; the registry does not discard history to make progress.

## Independent verifier boundary

The reference contract requires:
- a different verifier identity and sensor-chain identity;
- matching work-package, design, coordinate-frame and calibration revisions and digests;
- adequate declared coverage and no unobserved required regions;
- a non-empty common-cause declaration with no declared overlap against the executor;
- a non-empty common-cause assessment digest;
- a verifier receipt bound to the exact physical-delta digest.

These are structural checks on supplied declarations. The crate does **not** prove real-world sensor independence, validate signatures, or establish that the supplied assessments are truthful.

## Verify locally

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

A dedicated GitHub Actions workflow runs these exact commands with Rust 1.96.0 when this package changes.

## Explicit limitations

- No hardware I/O or robot command transport.
- No cryptographic digest calculation, signature verification or identity authentication.
- No geotechnical/structural adequacy model.
- No proof that sensor declarations reflect real physical independence.
- No durable journal or crash-consistency guarantee.
- No claim of mine safety, equipment certification or field qualification.

This crate validates state-machine semantics only. The independent evidence and authority layers must be supplied by reviewed systems.
