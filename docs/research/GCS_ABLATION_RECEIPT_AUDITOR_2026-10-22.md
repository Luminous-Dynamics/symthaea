# GCS Ablation Receipt Auditor (2026-10-22)

## Why this exists

The paired-ablation producer writes a JSON receipt, but a producer-generated summary should not be accepted solely because it agrees with itself. This standalone Rust binary is a separate read path that recomputes summary arithmetic from individual runs and rejects missing, duplicate, mismatched, or altered receipt fields.

The goal is *proof-carrying measurement*: make an evaluation artifact independently challengeable, not merely plausible.

## Checks performed

- Schema and benchmark identifier; valid source-revision and BLAKE3 digest shapes.
- Ordered task-manifest digest recomputation from task IDs and per-task digests.
- Unique task IDs and nonzero expected test counts.
- Reference-pass and real empty-patch-fail preflight controls for every manifest task.
- Exactly two modes per task/repeat; no duplicate or missing off/on run; expected run count; consistent test counts.
- Fail-closed behavioral score recomputed from compile status, simulation status, runtime errors, and complete expected test counts.
- Oracle-completeness classification recomputed separately from correctness. A complete run that fails one behavioral test is an assessed failure, not an incomplete oracle.
- Alternating paired order checks and recomputation of GCS-on wins, GCS-off wins, ties, net pass delta, per-mode rates, timings, and energy totals.
- Optional external pins for the exact source SHA and frozen task-manifest digest.

## Run

Build and execute the ablation producer from the target checkout, binding the receipt to that exact revision:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- \
  --repeats 3 --json-out /tmp/symthaea-gcs-ablation.json
```

Run the independent auditor:

```bash
cargo run --release --bin coding_gcs_ablation_verify -- \
  /tmp/symthaea-gcs-ablation.json \
  --expected-source-sha "$(git rev-parse HEAD)"
```

For publication or comparisons over time, additionally pass `--expected-manifest-blake3` with the digest captured in the pre-registration/benchmark protocol (not copied from the receipt currently being audited):

```bash
cargo run --release --bin coding_gcs_ablation_verify -- \
  /tmp/symthaea-gcs-ablation.json \
  --expected-source-sha "$PINNED_SOURCE_SHA" \
  --expected-manifest-blake3 "$PINNED_TASK_MANIFEST_BLAKE3"
```

A failure exits nonzero and identifies all detected consistency problems. A clean result prints `RECEIPT CONSISTENT`.

## Scope and security boundary

This tool audits **receipt consistency**, not the truth of every claimed execution. It cannot independently prove that a compiler ran, that the runner was uncompromised, that a model label is accurate, or that the benchmark tasks are representative. The source SHA is metadata, not a signature; BLAKE3 digests do not authenticate a producer. The verifier intentionally does not accept a self-reported "pass" as proof of correctness: for behavioral outcomes it recomputes the claim from the receipt's raw counts and flags.

For stronger provenance, preserve CI/runner logs and raw artifacts, run the auditor in an independently managed environment, pre-register source and task-manifest hashes outside the receipt, and sign the resulting audit record using an independently controlled key. Do not interpret receipt consistency as independent test execution, cryptographic attestation, or a SOTA result.

## Current validation status

The source and documentation were committed by API-backed repository edits. The authoring environment has no `rustc` or `cargo`; formatting, compilation, unit tests, integration against a real receipt, and CI are therefore **not claimed as passing**. Require exact-head Rust validation before merge.
