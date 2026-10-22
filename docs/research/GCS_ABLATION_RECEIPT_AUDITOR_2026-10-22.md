# GCS Ablation Receipt Auditor (2026-10-22)

## Why this exists

The paired-ablation producer writes a JSON receipt, but a producer-generated summary should not be accepted solely because it agrees with itself. This standalone Rust binary is a separate read path that recomputes summary arithmetic from individual runs and rejects missing, duplicate, mismatched, or altered receipt fields.

The goal is *proof-carrying measurement*: make an evaluation artifact independently challengeable, not merely plausible.

## Checks performed

- Schema and benchmark identifier; valid source-revision and BLAKE3 digest shapes.
- Requested backend/model metadata must be non-empty and internally consistent. Schema v4 separates selected tier decisions from actual `backend_attempts` returned by dispatcher telemetry. The auditor validates selected-tier values independently; it validates each backend attempt's tier, backend name and success flag; enforces known implementation-to-tier mappings; and checks actual backend attempts against the configured backend mode. Fallback routes are visible rather than misclassified from the initially selected tier. Both selected-tier and actual-attempt summaries are recomputed from individual run records.
- Ordered task-manifest digest recomputation from task IDs and per-task digests.
- The receipt includes task description, expected symbol, reference source, explicit expected test count, test fixture and semantic mutants. Schema v2 binds the count into the length-prefixed task digest; schema v3 adds requested model identity; schema v4 adds selected-tier decisions and actual backend-attempt records. The auditor independently hashes reference/test/mutant source content and checks the declared count against each run's observed pass+fail total; the producer's reference/mutant preflight checks test execution before trials.
- Unique task IDs and nonzero expected test counts.
- Reference-pass, real rustc empty-patch compile rejection, and mutation-adequacy preflight controls for every manifest task. Every semantic mutant must compile and be rejected by one or more tests under a complete non-simulated oracle; a compiler error is not credited as a killed behavioral mutant. A nonzero test binary exit is not itself evidence of an incomplete oracle when the full expected test count is present.
- Exactly two modes per task/repeat; no duplicate or missing off/on run; expected run count; consistent test counts.
- Fail-closed behavioral score recomputed from compile status, simulation status, runtime errors, and complete expected test counts. Oracle completeness is separate: CodeExecutor may put a failed test binary exit in `runtime_error` for an ordinary assertion failure, so exact full test counts with real non-simulated execution make the oracle complete while success remains false.
- Oracle-completeness classification recomputed separately from correctness. A complete run that fails one behavioral test is an assessed failure, not an incomplete oracle.
- Alternating paired order checks and recomputation of GCS-on wins, GCS-off wins, ties, net pass delta, per-mode rates, timing totals/means and nearest-rank p50/p95 agent and verification latency, energy totals, selected-tier decisions, actual backend attempts by tier and backend name, attempt success/failure totals, per-attempt latency total and nearest-rank p50/p95, external-LLM and simulated attempts, and runs with no backend calls. Pairs count as comparable only if both arm oracles completed; unresolved pairs are reported separately and are not silently counted as ties.
- Prints a deterministic 10,000-resample percentile bootstrap interval over task-identity clusters. Repeated trials are averaged within each task before resampling; the seed derives from the frozen manifest digest. Small task-cluster counts are explicitly labelled descriptive.
- Optional external pins for the exact source SHA and frozen task-manifest digest.

## Pre-register an external manifest pin

Generate the frozen source/task digest without running oracle controls or model trials:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run --release \
  --example coding_gcs_ablation_benchmark \
  --features 'code_generation geodesic_synthesis' -- --print-manifest
```

Capture the printed source SHA and task-manifest BLAKE3 digest in a separately reviewed, versioned preregistration record. Do not derive the supposedly external pin from the receipt under audit.

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

Selected tier decisions and actual backend-attempt telemetry are reported by the application, not independent attestation of the model endpoint/provider. Schema v4 preserves both: `selected_generation_tiers` captures the dispatcher's pre-fallback decision, while `backend_attempts` records implementation-reported invocations in order, including failed calls and actual fallback attempts. The benchmark refuses `--cloud` before trials if `ANTHROPIC_API_KEY` is absent/empty and resolves model labels from backend constructor defaults, but it cannot prove a provider honored the requested model or that an API call reached that provider. This tool audits **receipt consistency**, not the truth of every claimed execution. It cannot independently prove that a compiler ran, that the runner was uncompromised, that a model label is accurate, or that benchmark tasks are representative. The source SHA is metadata, not a signature; BLAKE3 digests do not authenticate a producer. Schema v2 removes text-based `#[test]` counting; schema v3 adds requested model identity; schema v4 separates selected tier decisions from actual backend attempts and recomputes both arm summaries. The eight-task pilot declares test counts explicitly in `TaskCase` and binds each count into the task digest; real oracle execution must account for exactly that many tests before the benchmark can proceed. This is safer for the fixed fixtures, but it does not solve arbitrary repository task ingestion: a larger task bank needs a machine-readable test-discovery/execution contract and independent checks against zero-test or misconfigured runs. Schema v1/v2/v3 receipts are not interchangeable with schema v4. The verifier does not accept a self-reported "pass" as proof of correctness: for behavioral outcomes it recomputes claims from raw counts and flags.

For stronger provenance, preserve CI/runner logs and raw artifacts, run the auditor in an independently managed environment, pre-register source and task-manifest hashes outside the receipt, and sign the resulting audit record using an independently controlled key. Do not interpret receipt consistency as independent test execution, cryptographic attestation, or a SOTA result.

## Current validation status

The source and documentation were committed by API-backed repository edits. The authoring environment has no `rustc` or `cargo`; formatting, compilation, unit tests, integration against a real receipt, and CI are therefore **not claimed as passing**. Require exact-head Rust validation before merge.
