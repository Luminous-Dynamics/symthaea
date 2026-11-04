# PITB SorryDB evidence-closure verifier

This standalone Rust crate validates the *bytes behind* a claimed SorryDB proof-run receipt. It extends (rather than duplicates) `tools/pitb-sorrydb-audit`.

The audit crate validates that a proof-run row refers to a task in the pinned SorryDB evaluation split and that its declared gate fields and artifact digests are well-formed. This crate opens the referenced artifacts and verifies that they actually match those digests and each other.

## Evidence bundle layout

One directory per `run_id`:

```text
evidence-root/
  <run_id>/
    source-before.lean
    source-after.lean
    proof.txt
    replacement.patch
    source-reproduction.log
    lake-manifest.json
    checker.log
    semantic-audit.log
    source-binding-receipt.json
    checker-receipt.json
    semantic-gate-receipt.json
```

The JSONL proof-run receipt continues to use `pitb.sorrydb-proof-run.v0` from the SorryDB audit crate. The three sidecar receipts use versioned schemas in this crate.

## What is verified

For each `verified_success` the verifier checks:

1. The task ID, repository commit, Lean version, source path/range and goal digest match the exact record in the frozen SorryDB evaluation split.
2. The source-before digest matches the proof-run record and the exact recorded coordinates select the literal `sorry` token.
3. The source-after file is byte-for-byte the exact single-span substitution of `proof.txt` for that token. No unrelated source edits are allowed.
4. The proof and replacement-patch bytes match their digests in the proof-run receipt.
5. The source-binding, checker and semantic-gate receipt bytes match the corresponding proof-run digests and parse against strict, versioned receipt types.
6. Each receipt binds the same run/task/commit/toolchain/source/goal/signature and agrees on the required gates.
7. The Lake manifest, source-reproduction log, checker log and semantic-review log bytes match the digests in their sidecar receipts.
8. The semantic gate explicitly marks source correspondence, hypothesis audit, vacuity audit and reviewer decision as `pass`.

A success row whose referenced bundle is missing, altered, unresolved or internally contradictory fails the whole evidence-closure run. Non-success attempts remain visible but do not require a proof-artifact bundle.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-evidence-closure/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-evidence-closure/Cargo.toml
cargo run --locked --manifest-path tools/pitb-sorrydb-evidence-closure/Cargo.toml -- --self-test

# Produce a complete synthetic evidence root, proof-run JSONL, and closure report:
cargo run --locked --manifest-path tools/pitb-sorrydb-evidence-closure/Cargo.toml -- \
  --self-test-report \
  /tmp/sorrydb-evidence-closure-report.json \
  /tmp/sorrydb-evidence-bundle

# Verify measured attempts against the pinned 1,000-task evaluation split:
cargo run --locked --manifest-path tools/pitb-sorrydb-evidence-closure/Cargo.toml -- \
  --verify \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  /path/to/proof-runs.jsonl \
  /path/to/evidence-root \
  /path/to/evidence-closure-report.json
```

The verifier uses Git's `hash-object --stdin` to check that the supplied evaluation split is the frozen SorryDB blob. The synthetic self-test deliberately bypasses the real-snapshot pin, but the live `--verify` path does not.

## Report interpretation

The report distinguishes:
- all attempts;
- rows claiming `verified_success`;
- successes whose full evidence bundle was byte-closed;
- unique tasks with closed success bundles;
- non-success attempts.

Ten retries that close for one task remain one unique task solved. A digest-closed bundle is stronger than a syntactically valid receipt, but it is not the same as independently rerunning Lean or authenticating a reviewer.

## Explicit trust boundary

This tool verifies content-addressed byte closure and cross-binding. It does **not**:
- execute Lean or independently repeat the kernel check;
- build the source repository or re-run the source-to-goal reproduction command;
- verify a checker's binary provenance or sandbox isolation;
- authenticate the reviewer identity or the reviewer's attestation signature;
- prove that the formal proposition expresses the intended mathematical statement.

The `reviewer_attestation_sha256` is currently a digest reference, not a verified signature. Production admission should add a trusted reviewer-key registry and a cryptographic signature (for example, Ed25519) over a canonical semantic-gate receipt, and independently replay source reproduction and checker commands in a hermetic environment.

This is an evidence-closure gate, not a truth oracle. The proof kernel, source binding, semantic review and reproducibility checks remain separate assurance layers.
