# Institutional Intent Compiler — structural validator v0.1

This is a small, standalone Rust research tool. It validates the **structure** of a source-grounded institutional requirement manifest. It does not extract requirements from prose, determine whether a statement is true, verify customer authority, interpret law, establish title, certify compliance, or qualify safety.

## Run

From this directory:

```sh
cargo test --locked
cargo run -- fixtures/valid.json
```

The CLI reads one JSON manifest and emits a deterministic JSON report. Exit code 0 means the manifest passed this structural contract. Exit code 2 means parsing/validation failed. The status `STRUCTURALLY_VALID` is intentionally not named `APPROVED`, `COMPLIANT`, `SAFE` or `VERIFIED`.

## Contract enforced

- Strict JSON structs reject unknown fields.
- Source and requirement identifiers must be unique.
- Source/quote digests must have SHA-256-shaped hexadecimal encodings.
- Source spans must fit the declared source byte length.
- Accepted requirements must have source-span or approved-derivation traceability.
- Accepted requirements must have a matching reviewer decision for the current baseline.
- A change between declared and proposed modality requires a distinct modality-change approval.
- Accepted requirements need an assigned owner and an acceptance condition.
- `not_yet_verifiable` requires an explicit limitation and separately referenced exception decision.
- Accepted dependencies cannot point to unaccepted requirements.
- Conflicts between accepted requirements require a matching conflict-resolution decision.
- Diagnostics are sorted deterministically.

## Deliberate limitations

The tool does not load source documents. It can validate that span references and digests are well-shaped and within a declared byte length; it cannot prove that a quoted span exists in the source or that its digest matches the underlying bytes.

Reviewer and authority references are opaque strings. The validator does not authenticate signatures, resolve role assignments, establish that a reviewer had legal authority, or confirm that a client approved the decision.

Conflict links are supplied evidence. The validator can reject an unresolved declared conflict; it cannot discover every semantic contradiction in natural language.

The tool does not score requirement atomicity, feasibility, necessity, correctness, completeness, legal meaning, or domain validity. Those require domain experts, source-grounded analysis and independent review.

No client-private source text belongs in public fixtures. The included fixture is synthetic.

## Evidence state

The source and tests in this directory are an implementation proposal on an open draft PR. No compile/test result or CI pass is claimed until the exact head has actually run the commands above.
