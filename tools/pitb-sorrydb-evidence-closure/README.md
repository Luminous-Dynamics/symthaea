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
    reviewer-attestation.json
  reviewer-key-registry.json
```

The JSONL proof-run receipt continues to use `pitb.sorrydb-proof-run.v0` from the SorryDB audit crate. Source-binding, checker and semantic-gate receipts use versioned schemas. A separate reviewer-attestation file signs the semantic decision; the public-key registry is supplied independently and its digest is pinned out of band.

## What is verified

For each `verified_success` the verifier checks:

1. The task ID, repository commit, Lean version, source path/range and goal digest match the exact record in the frozen SorryDB evaluation split.
2. The source-before digest matches the proof-run record and the exact recorded coordinates select the literal `sorry` token.
3. The source-after file is byte-for-byte the exact single-span substitution of `proof.txt` for that token. No unrelated source edits are allowed.
4. The proof and replacement-patch bytes match their digests in the proof-run receipt.
5. The source-binding, checker, semantic-gate and reviewer-attestation bytes match their receipt digests and parse against strict, versioned schemas.
6. Each receipt binds the same run/task/commit/toolchain/source/goal/signature and agrees on the required gates.
7. The Lake manifest, source-reproduction log, checker log and semantic-review log bytes match the digests in their sidecar receipts.
8. The semantic gate explicitly marks source correspondence, hypothesis audit, vacuity audit and reviewer decision as `pass`.
9. The reviewer attestation signs the exact semantic decision, source-binding receipt digest, target goal/signature identity and audit-log digest with Ed25519.
10. The reviewer key ID resolves to an active reviewer identity in a strict key registry whose exact raw bytes match a separately supplied SHA-256 pin.

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
  /path/to/reviewer-key-registry.json \
  <pinned-reviewer-registry-sha256> \
  /path/to/evidence-closure-report.json
```

The verifier uses Git's `hash-object --stdin` to check that the supplied evaluation split is the frozen SorryDB blob. Production `--verify` requires a reviewer key registry file and an out-of-band SHA-256 pin; it rejects a changed registry, unknown key, revoked key, reviewer-identity mismatch or invalid Ed25519 signature. The synthetic self-test generates a disposable local Ed25519 keypair and registry only to exercise the cryptographic path; its private key is deleted before the bundle is retained. Never use that synthetic identity for production reviews.

## Report interpretation

The report distinguishes:
- all attempts;
- rows claiming `verified_success`;
- successes whose full evidence bundle was byte-closed;
- unique tasks with closed success bundles;
- non-success attempts.

Ten retries that close for one task remain one unique task solved. A digest-closed and signature-verified bundle is stronger than a syntactically valid receipt, but it is not the same as independently rerunning Lean or proving informal mathematical truth.

## Explicit trust boundary

This tool verifies content-addressed byte closure and cross-binding. It does **not**:
- execute Lean or independently repeat the kernel check;
- build the source repository or re-run the source-to-goal reproduction command;
- verify a checker's binary provenance or sandbox isolation;
- independently rerun Lean or the checker in a hermetic execution environment;
- prove that the formal proposition expresses the intended mathematical statement.

Reviewer identity is now authenticated by an Ed25519 signature over a canonical JSON payload (`serde-json-map-sorted-compact-v0`). The key registry is strict, rejects duplicate/unsorted IDs and revoked keys, and is trusted only when its raw SHA-256 matches the out-of-band pin supplied to `--verify`. Registry rotation is a governance event: publish the new registry digest through a trusted channel, revoke old keys explicitly, and preserve the historical registry bytes with each scored run. The report records the registry digest it used. This authenticates which registered reviewer signed the review; it does not prove the reviewer's judgment is true. Independently replay source reproduction and checker commands in a hermetic environment.

This is an evidence-closure gate, not a truth oracle. The proof kernel, source binding, semantic review and reproducibility checks remain separate assurance layers.
