# Independent Fresh-Image Regrade Audit (2026-10-24)

**Protocol ID:** `symthaea-rust-regrade-audit-v1`  
**Status:** verifier implementation; actual primary/regrade runs remain required.

## Purpose

A passing score from the same mutable agent workspace and evaluator process is not a strong independent check. This stage compares the primary scored-run report with a separately generated regrade report using the same locked task manifest, candidate split, prediction artifact, immutable image IDs and per-task patches. The second grade must be produced by a new execution in a fresh container/workspace and its report retained independently.

The binary `rust-repair-regrade-audit` does not execute Docker or tests. It validates the two existing `rust-repair-score-audit` JSON reports and produces an explicit parity/disagreement report.

## Required provenance

Supply both report file hashes from separate retained artifact receipts or an immutable artifact manifest. Computing the expected digest ad hoc from the same mutable file is not independent provenance.

```bash
cargo test --locked \
  --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
  --bin rust-repair-regrade-audit

cargo run --locked \
  --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
  --bin rust-repair-regrade-audit -- \
  --primary-report build/rust-repair/primary/score-report.json \
  --expected-primary-report-sha256 <INDEPENDENTLY-RETAINED-64-HEX> \
  --regrade-report build/rust-repair/fresh-regrade/score-report.json \
  --expected-regrade-report-sha256 <INDEPENDENTLY-RETAINED-64-HEX> \
  --task-manifest build/rust-repair/locked-task-manifest.json \
  --expected-task-manifest-sha256 <INDEPENDENTLY-RETAINED-MANIFEST-FILE-SHA256> \
  --report-out build/rust-repair/fresh-regrade/parity-audit.json
```

## Fail-closed checks

- Both input reports must be regular files whose bytes match their externally supplied SHA-256 values; they must be distinct files and distinct digests.
- Both reports must identify protocol `symthaea-rust-repair`, format version 1, and the exact pinned task-manifest SHA-256. The same externally retained digest is checked against the exact locked manifest file bytes via `--task-manifest`.
- The verifier parses the locked manifest, checks its full task-record schema, recomputes the canonical sorted `task_records_sha256`, and requires its exact task-ID set, candidate-split digest, per-task immutable image IDs and repository HEADs to agree with the two grade reports. The expected manifest SHA-256 is the digest of the complete JSON file, not the inner task-record digest.
- Candidate JSONL SHA, predictions-file SHA, task count, and exact task-ID set must match across reports.
- Aggregates (`success_count`, `failure_count`, `unresolved_count`, `scored_count`, `lockable`, and `pass_at_1`) are recomputed from task rows and compared with the claimed top-level values.
- Each task's exact prediction patch digest must match between grades. When both sides provide image IDs or repository HEADs, those values must match. Resolved task rows must carry patch/log/evidence hashes, immutable `sha256:` image ID, Linux platform, network mode `none`, clean-workspace result, observed repository HEAD, and required-test counts; a passing row also needs all required tests passing and exit code zero.
- If both grades resolve a task, the evaluation-evidence hash must differ. Identical test-log hashes are permitted because deterministic tests can produce byte-identical output on independent executions; the execution-specific evidence still needs to differ. Reusing the same runtime receipt cannot be presented as an independent run.
- The output names the relation for every task: `concordant_pass`, `concordant_fail`, `primary_only_pass`, `regrade_only_pass`, or an unresolved relation. Both directions of disagreement remain visible.

## Metrics and interpretation

The audit reports each grade's rate over the complete task denominator, a conservative double-verified pass rate, disagreement counts, possible primary false-positive fraction among primary passes, possible primary false-negative fraction among regrade passes, and unresolved coverage.

Strict `pass_at_1` is omitted if any task is unresolved in either grade. A pass requires both grades to say pass. A failure disagreement is not silently resolved in favor of the higher score: it is reported for investigation and the conservative double-verified rate counts it as not passed. An unresolved task is not converted to either success or definitive failure.

A full concordant report supports evidence repeatability for the matched artifacts. It does not authenticate the machine that produced them, prove that the model provider served the exact requested model, or by itself establish state-of-the-art. Preserve the actual container IDs, full tool trajectories, evaluator logs, image digests and artifact receipts outside the runner; use a separate runner or security boundary for the regrade if possible.

## Why this matters for comparison

SWE-Bench Pro V2's September 2026 methodology explicitly regrades agent diffs on pristine images and publishes both grades. The release notes describe examples where the primary diff's apparent success depended on tampering with a Go module checksum or editing a module cache. This implementation adopts the independent-grade comparison principle without claiming that report comparison alone is equivalent to a fully isolated, externally attested grader.

References:

- [SWE-Bench Pro V2 methodology (2026-09-22)](https://labs.scale.com/blog/swe-bench-pro-v2)
- [SWE-Bench Pro V2 leaderboard and methodology](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)
- [SWE-bench-Live trajectory verification policy](https://swe-bench-live.github.io/)
- [Epoch AI review of Terminal-Bench 4.0.0 (2026-09-04)](https://epoch.ai/benchmarks/terminal-bench-4/review)
