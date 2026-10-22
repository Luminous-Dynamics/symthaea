# GCS Ablation Receipt Auditor (2026-10-22)

## Why this exists

The paired-ablation producer writes a JSON receipt, but a producer-generated summary should not be accepted solely because it agrees with itself. This standalone Rust binary is a separate read path that recomputes summary arithmetic from individual runs and rejects missing, duplicate, mismatched, or altered receipt fields.

The goal is *proof-carrying measurement*: make an evaluation artifact independently challengeable, not merely plausible.

## Checks performed

- Schema and benchmark identifier; valid source-revision and BLAKE3 digest shapes.
- Ordered task-manifest digest recomputation from task IDs and per-task digests.
- The receipt includes task description, expected symbol, reference source, test fixture and semantic mutants. The auditor recomputes a length-prefixed task digest, independently hashes reference/test/mutant source content, and checks the declared test count against the included fixture.
- Unique task IDs and nonzero expected test counts.
- Reference-pass, real rustc empty-patch compile rejection, and mutation-adequacy preflight controls for every manifest task. Every semantic mutant must compile and be rejected by one or more tests under a complete non-simulated oracle; a compiler error is not credited as a killed behavioral mutant.
- Exactly two modes per task/repeat; no duplicate or missing off/on run; expected run count; consistent test counts.
- Fail-closed behavioral score recomputed from compile status, simulation status, runtime errors, and complete expected test counts.
- Oracle-completeness classification recomputed separately from correctness. A complete run that fails one behavioral test is an assessed failure, not an incomplete oracle.
- Alternating paired order checks and recomputation of GCS-on wins, GCS-off wins, ties, net pass delta, per-mode rates, timings, and energy totals. Pairs count as comparable only if both arm oracles completed; unresolved pairs are reported separately and are not silently counted as ties.
- Prints a deterministic 10,000-resample percentile bootstrap interval over task-identity clusters. Repeated trials are averaged within each task before resampling; the seed derives from the frozen manifest digest. Small task-cluster counts are explicitly labelled descriptive.
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

A failure exits nonzero and identifies all detected consistency problems. A clean result prints `RECEIPT CONSISTENT` and the equal-task paired effect with a task-cluster bootstrap interval when at least one complete paired outcome is available.

## Scope and security boundary

The eight-task pilot uses public fixtures and ten hand-authored semantic mutants; this is a small oracle-discrimination screen, not a comprehensive mutation score, hidden-test or contamination-resistant evaluation. Its task-cluster bootstrap has only eight task identities at most and the tasks share a toy-function task family, so its interval is descriptive and does not justify broad generalization. Larger studies should preserve repository and task-family identifiers and resample at the independence unit relevant to the claim. For a publication-grade benchmark, split public development tasks from held-out/private tasks and preserve the exact task-manifest pin independently from the receipt.

This tool audits **receipt consistency**, not the truth of every claimed execution. It cannot independently prove that a compiler ran, that the runner was uncompromised, that a model label is accurate, or that the benchmark tasks are representative. The source SHA is metadata, not a signature; BLAKE3 digests do not authenticate a producer. Test counting intentionally mirrors the producer’s simple `#[test]` text scan and is not a Rust parser; prefer structured test metadata or AST-based extraction for a larger heterogeneous suite. Test-function counting intentionally mirrors the producer's simple `#[test]` text scan and is not a Rust parser; use structured test metadata or AST-based extraction for a larger heterogeneous suite. The verifier intentionally does not accept a self-reported "pass" as proof of correctness: for behavioral outcomes it recomputes the claim from the receipt's raw counts and flags.

For stronger provenance, preserve CI/runner logs and raw artifacts, run the auditor in an independently managed environment, pre-register source and task-manifest hashes outside the receipt, and sign the resulting audit record using an independently controlled key. Do not interpret receipt consistency as independent test execution, cryptographic attestation, or a SOTA result.

## Current validation status

The source and documentation were committed by API-backed repository edits. The authoring environment has no `rustc` or `cargo`; formatting, compilation, unit tests, integration against a real receipt, and CI are therefore **not claimed as passing**. Require exact-head Rust validation before merge.
