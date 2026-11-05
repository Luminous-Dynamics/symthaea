# SYM-RSI benchmark receipt composer

This standalone Rust library composes multiple canonical benchmark-run receipts into a
multi-dimensional candidate evidence view. It consumes the existing
`benchmark-run-receipt-validator` crate; it does not define or persist another receipt
schema. The candidate identity digest is derived from the declared system name, arm,
version, source revision, executable digest, model ID, configuration digest and prompt
digest. It is **not** a source-tree digest and must not be substituted for one.

## Contract

- Every input byte string is passed through `verify_receipt_json` before projection.
- Every metric pins benchmark ID/version/revision, dataset-manifest digest, scoring-formula ID, grader version and a frozen score transform.
- All metric runs for one candidate must cover the exact same eligible task universe,
  share the same candidate system identity, and use compatible cohort, runner, comparison,
  protocol and budget context.
- Every eligible task remains in the output. Grade-less results retain their attempt state
  and terminal disposition; they are never silently dropped.
- The summary reports graded-only mean, graded coverage, and coverage-adjusted yield as
  distinct quantities.
- A candidate capability vector contains the score, coverage, and coverage-adjusted yield
  dimensions for every metric. A preregistered minimum-coverage floor can make the vector
  unavailable; it cannot cause task rows to be removed.
- Parent/candidate comparison retains incomplete rows and reports paired coverage and
  cluster means. It does not perform inferential statistics or promotion.

## Evidence ceiling

The receipt validator proves schema and accounting consistency relative to its contract.
It does not authenticate a producer, qualify runner isolation, prove grader truth, or
verify independent regrade. This composer always reports that lower assurance ceiling.
It does not establish source-tree identity; bind the actual source tree through its own
evidence and applicable attestation/regrade path. Use the independent attestation, trust
and isolation owners before confirmatory claims.

The cluster mapping and experiment preregistration digest are supplied by the caller. The
composer checks exact coverage and binds them into an in-memory context digest, but does
not authenticate the external manifest.

## Run

From this directory:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
```

Tests use the shared validator's explicitly synthetic example. They qualify composition
mechanics only; they are not benchmark results.
