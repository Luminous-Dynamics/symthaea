# Culinary validator corpus v1

This directory contains the frozen, machine-readable fixtures for the first
deterministic validator benchmark. The runner lives at
`examples/culinary_validation_receipt.rs`.

## Run

From a clean checkout of the Symthaea repository, set `EXPECTED_HEAD` to the
**independently known full commit SHA being evaluated** (for CI, use the event's
trusted commit SHA; for local review, use the SHA supplied by the PR/revision
being audited). Do not derive the expected SHA from the checkout being tested,
because that makes the identity comparison tautological.

```bash
EXPECTED_HEAD=<full-commit-sha-under-review>
cargo run -p symthaea-culinary --example culinary_validation_receipt -- \
  --expected-head "$EXPECTED_HEAD" \
  --output target/culinary-validation-receipt.json
```

Without `--expected-head`, or if the expected SHA differs from `HEAD`, the
runner returns `not_comparable` and exits non-zero.

The command records its exact git commit, checks the requested head if supplied,
requires a clean checkout, hashes the raw JSON bytes with SHA-256, executes each
fixture through the production `CulinarySpec` validator, and writes one JSON
receipt. Exit code 0 means the receipt status is `pass`; non-zero means the
result must not be promoted as passing. Inspect the receipt status rather than
inferring success from process logs alone.

## Status semantics

- `pass`: the complete pinned corpus ran at the exact identifiable commit and
  every case matched its preregistered expectation, including valid positive
  controls.
- `fail`: the corpus and run identity are comparable, but one or more
  validator outcomes did not match expected outcomes.
- `incomplete`: a required fixture is absent, duplicated, malformed, or not
  attempted; no partial run can pass.
- `not_comparable`: the manifest digest/version differs from the pin, the
  commit cannot be identified, the working tree is dirty, `--expected-head`
  is absent, or the expected SHA does not match the checkout.

A GitHub Actions queue, an empty checks response, or a green unrelated commit
is not evidence that this corpus passed. Bind any claimed result to the commit
SHA inside the receipt.

## Change control

The manifest SHA-256 and required case IDs are independently pinned in the Rust
runner. Intentional corpus changes must review both fixture semantics and
expected outcomes, update the benchmark version and schema as needed, then
update the manifest digest pin in the same reviewed change. Never weaken a
scientific threshold merely to make a test pass. A changed model parameter
requires its own reference and review.

The manifest encodes special floating-point values as JSON strings such as
`"NaN"`, `"Infinity"`, or `"MAX"`. These strings are deliberately converted
to their floating-point equivalents by the Rust runner; the on-disk corpus
remains valid JSON and its digest is reproducible.

## Scope limits

This corpus tests implemented validator behavior and numeric-domain guards. A
`pass` establishes only that the software agrees with these fixed cases. It
does not establish that the underlying physical parameters are universally
valid, that a simulated trajectory is achievable in a kitchen, or that a dish
is microbiologically safe. In particular, the pasteurization fixtures exercise
numerical failure handling for a simplified model, not a real-world cooking
protocol or food-safety certification.
