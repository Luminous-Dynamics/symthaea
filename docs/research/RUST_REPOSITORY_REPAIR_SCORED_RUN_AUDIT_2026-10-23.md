# Rust Repository-Repair Scored-Run Audit (2026-10-23)

## Purpose

`rust-repair-score-audit` is the independent scoring layer for the Rust repository-repair benchmark. It is deliberately separate from `rust-repair-manifest`:

- The task-manifest auditor decides whether an upstream task has a valid, discriminating oracle and sufficiently complete baseline/gold evidence.
- The scored-run auditor decides whether one attempted coding-agent patch for each locked task earns a pass, a fail, or an unresolved outcome.

A valid task manifest does **not** prove that a scored agent rollout used the correct patch, ran in the intended container, completed the tests, or generated the status artifact that is being scored.

## Inputs and immutable identity

Before running an agent, pin the SHA-256 of the locked task-manifest file in the preregistration record. Supply that digest as `--expected-task-manifest-sha256`; do not derive it after seeing the trial outcomes. The scorer checks that external pin, the full candidate-JSONL SHA-256, candidate row count, candidate ID digest, eligible task ID digest, task-record digest and each included candidate row's raw-line SHA-256.

The prediction file is a JSON object keyed by `instance_id`, with each value containing `model_patch`. The file SHA-256 is included in the output report. A task absent from the prediction map, or with no string patch payload, is an **unresolved assigned attempt**, not a dropped denominator row. IDs outside the locked manifest are rejected as a protocol violation.

## Scoring contract

For every task in the locked manifest, the tool verifies:

1. Task validation status JSON, baseline log, aggregate gold log, and `validation_evidence.json` hashes against the locked manifest.
2. Individual baseline and three gold-control logs against the hashes in the runtime evidence sidecar.
3. Per-run roles, test-patch/gold-patch application confirmations, command exits, test IDs/statuses, container IDs and actual Docker image IDs; the three-run aggregate must reconstruct the recorded gold test map.
4. Candidate test-patch identity and the independently recomputed task oracle: explicit baseline `fail → pass` tests plus every baseline-passing regression guard.
5. Scored prediction patch SHA-256, test-patch SHA-256, scored raw-log SHA-256, scored status JSON versus the sidecar's parsed status, test-ID completeness, patch-application confirmations, command exit codes and observed Docker identity.
6. The scored container must use the same immutable image ID as task/gold validation but a distinct container ID.

The required test set is the union of explicit fail-to-pass tests and all baseline-passing tests. A missing or skipped required test is unresolved; a failed required test is a failure. Failures in tests outside the required set are counted separately, since those tests may already have failed on the baseline and are not part of the registered F2P/P2P oracle. A nonzero test-command exit without a parsed failing test to explain it is unresolved. The output retains the actual test exit code and count of non-required failures to make that distinction auditable.

## Denominator and reporting

The denominator is the **full locked task manifest**. Results are task-level `pass`, `fail`, or `unresolved`.

- `pass_at_1` is emitted only if every assigned task has a resolved verdict. It is the number of passing tasks divided by the full locked task count.
- `completed_run_success_rate` is a diagnostic rate among resolved tasks only. If any run is unresolved, it must not be presented as pass@1 or as a leaderboard result.
- The report contains the external task-manifest pin, candidate-data digest, prediction-file digest, per-task patch/log/runtime evidence digests, required-test counts, runtime image ID, exit code, and non-required failure count.

The primary comparison should use one rollout per task. Retries, pass@k, web-enabled or retrieval-enabled runs must be separately preregistered and separately reported. Never retry a task until it passes and then present the result as pass@1.

## Invocation

First run the pinned task validation and create the lockable task manifest using the commands in [the task-audit protocol](SWE_BENCH_LIVE_RUST_TASK_AUDIT_2026-10-23.md). Then run:

```bash
cargo test --locked --bin rust-repair-score-audit --features rust-repair-manifest

cargo run --locked --bin rust-repair-score-audit \
  --features rust-repair-manifest -- \
  --task-manifest build/rust-repair/eligible-task-manifest.json \
  --expected-task-manifest-sha256 <PRE-REGISTERED-64-HEX-DIGEST> \
  --candidate-jsonl /path/to/full-pinned-rust-candidate-split.jsonl \
  --predictions-file build/rust-repair/predictions.json \
  --task-validation-dir /path/to/upstream-validation-output \
  --evaluation-dir /path/to/scored-evaluation-output \
  --report-out build/rust-repair/scored-run-audit.json
```

The prediction file must be a direct record of the one assigned patch per task; the evaluator's output directory must contain each task's `evaluation_evidence.json`, `status.json`, and `post_patch_log.txt`. Do not edit either after hashing; preserve originals and rerun the audit if any artifact changes.

## Evidence boundary

The implementation has 13 authored Rust unit tests. This authoring environment has no `rustc`, `cargo`, or `rustfmt`; those tests and the formatter/Clippy checks have not been executed locally. The dedicated GitHub Actions workflow is configured to check the pinned upstream overlay, task auditor and scored-run auditor, but queued/in-progress runs are not passes.

Docker API IDs, command exit codes and hashes are useful execution evidence, not cryptographic host attestation. The tool does not run the containers itself, prove that a trusted runner generated the sidecars, or establish a state-of-the-art result. Before publishing a result, an independent party should reproduce the environment and grading from the preserved artifacts.
