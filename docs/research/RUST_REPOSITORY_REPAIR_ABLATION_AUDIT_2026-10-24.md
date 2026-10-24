# Paired Repository-Repair Ablation Audit (2026-10-24)

**Protocol ID:** `symthaea-rust-repair-ablation-v1`  
**Version:** `1`  
**Status:** analysis implementation; preregistration required before any scored comparison.

This is an analysis contract, not a result. It does not claim that GCS improves repair quality or that Symthaea is state of the art.

## Why this exists

A headline percentage is too easy to misread. A causal GCS-on/off comparison must preserve task assignment, compare matched environments, bind outcomes to the independent scored-run auditor, report unresolved work, and account for the fact that tasks from one repository are correlated.

`rust-repair-ablation-audit` is a separate Rust analysis stage. It does not run agents or tests. It validates preregistered design fields, verifies the raw digest of each referenced score report, checks each report's task-manifest digest and task-level outcomes, then computes paired effects and repository-clustered uncertainty.

## Preregister before trials

The preregistration JSON is hashed and passed as `--expected-preregistration-sha256` before trial results are scored. Its required fields are:

- `format_version=1`, `protocol_id=symthaea-rust-repair-ablation-v1`;
- `benchmark_manifest_sha256`, the exact task-manifest SHA-256 expected in all score reports;
- exact `model_id`, agent-harness, prompt and scaffold SHA-256 digests, one budget identifier, immutable Docker `container_image_id`, and `network_policy="none"`;
- `randomization_seed` as a pre-trial, externally pinned 64-hex-character seed, `bootstrap_seed`, and `bootstrap_resamples` (1,000–1,000,000);
- one assignment per unique task ID, including its repository cluster and `first_arm`.

The assigned first arm is derived deterministically, not hand-edited after looking at outcomes: compute SHA-256 over UTF-8 bytes of the 64-hex-character `randomization_seed`, a single NUL byte, then `instance_id`. The seed must be generated and its complete preregistration digest independently retained before trials begin. If the first digest byte's low bit is zero, the first arm is `control`; otherwise it is `gcs`. The tool recomputes this mapping. The preregistration and task manifest must be frozen before trial outcomes are inspected.

For each assigned task, the trial-results JSONL contains one record for each arm (`control` and `gcs`). Each record has `run_id`, `instance_id`, `arm`, `outcome`, `gcs_enabled`, `cost_usd`, `latency_ms`, `started_at_ms`, the common model/harness/prompt/scaffold/budget/image/network fields, and a relative `score_report_path` plus `score_report_sha256`. The two recorded start timestamps must be distinct and the earlier arm must match preregistered order.

The referenced score-report file must exist under `--score-reports-dir`. Its SHA-256 is recomputed, its `protocol_id` must be `symthaea-rust-repair`, its `task_manifest_sha256` must equal the preregistered pin, and its unique task row must report the same outcome as the trial record. Paths must be relative and cannot traverse outside the report directory. The scorer report may contain unresolved task rows; those outcomes remain unresolved here.

Example record shape (all hashes and values below are placeholders, not valid evidence):

```json
{
  "run_id": "run-task-001-gcs",
  "instance_id": "task-001",
  "arm": "gcs",
  "outcome": "pass",
  "gcs_enabled": true,
  "cost_usd": 0.42,
  "latency_ms": 183000,
  "started_at_ms": 1790000000000,
  "model_id": "provider/model@immutable-revision",
  "agent_harness_sha256": "<64-hex>",
  "prompt_sha256": "<64-hex>",
  "scaffold_sha256": "<64-hex>",
  "budget_id": "tokens=...,wall_seconds=...,tools=...",
  "container_image_id": "sha256:<64-hex>",
  "network_policy": "none",
  "score_report_path": "gcs/score-report.json",
  "score_report_sha256": "<64-hex>"
}
```

The record's `outcome` must match the selected task in the referenced independent score report. The score report path is relative to the directory supplied to `--score-reports-dir`.

## Invocation

After both arms' score reports have been generated and preserved:

```bash
cargo test --locked \
  --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
  --bin rust-repair-ablation-audit

cargo run --locked \
  --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
  --bin rust-repair-ablation-audit -- \
  --preregistration build/rust-repair/ablation-preregistration.json \
  --expected-preregistration-sha256 <PRE-REGISTERED-64-HEX> \
  --trial-results build/rust-repair/ablation-trials.jsonl \
  --expected-trial-results-sha256 <INDEPENDENTLY-RETAINED-64-HEX> \
  --score-reports-dir build/rust-repair/score-reports \
  --report-out build/rust-repair/ablation-audit.json
```

The full preregistration SHA, raw JSONL SHA, externally expected JSONL SHA, per-line JSONL hashes, report-file hashes, arm settings, outcome counts and bootstrap settings are included in the generated report. The executable rejects a trial-results file that does not match `--expected-trial-results-sha256`. Obtain that digest from a separately retained collection receipt or immutable artifact manifest; do not calculate it ad hoc from the same mutable file and claim that this adds independent authenticity. The executable prints the output report's SHA-256 so the report can be placed in an immutable artifact store or signed by a separate trusted system.

## Estimands and safeguards

- **Full-assignment verified-pass rate:** verified passes divided by all preregistered tasks. An unresolved or missing arm counts as not yet verified, never as a verified pass.
- **Resolved paired effect:** GCS minus control pass rate only among tasks with a resolved pass/fail outcome in *both* arms. Paired GCS wins, control wins and ties are reported. This estimate can be selection-biased when outcomes are unresolved, so it is never shown alone.
- **Unresolved sensitivity bounds:** the tool gives a worst/best-case interval for the full-assignment GCS-minus-control effect by assigning each unresolved arm outcome to fail/pass in the direction that minimizes/maximizes the effect.
- **Cluster bootstrap:** a deterministic, preregistered bootstrap resamples repository clusters with replacement, carrying all complete task pairs in each sampled repository. A 95% percentile interval is emitted only when at least 10 repositories have at least one fully resolved pair. With fewer clusters, the interval is withheld rather than manufacturing precision.
- **Economics:** cost per verified fix includes recorded spend on passes, failures and resolved/unresolved runs. If any assigned arm record is missing, its cost is unknown and cost-per-fix for that arm is withheld. p50/p95 latency is computed over observed records and is not imputed for missing runs.
- **Configuration integrity:** a run whose model, harness, prompt, scaffold, budget, image or network policy differs from preregistration is rejected. GCS-enabled must match the registered arm. Duplicate IDs, unknown tasks, multiple records per arm, invalid paths, score-report hash mismatches and outcome mismatches are protocol errors.

V1 intentionally supports one paired attempt per task; repeated-run reliability studies need a separate preregistered design rather than being silently mixed into the one-attempt primary comparison.

## Preregistered internal gates (targets, not industry standards)

**GCS superiority target:** resolved paired effect at least +5 absolute percentage points, repository-clustered 95% CI lower bound above zero, full-assignment unresolved sensitivity lower bound above zero, at least 10 resolved repository clusters, overall unresolved rate below 2%, and absolute unresolved-rate difference between arms below 1 percentage point.

**Cost-efficiency alternative:** at least 20% lower cost per verified fix, the same unresolved-rate/repository gates, and both the clustered 95% CI and full-assignment unresolved sensitivity lower bound no worse than -1.5 percentage points for correctness.

These are proposed internal decision thresholds, not evidence of a current improvement and not a claim of leaderboard superiority. If a gate is not met, report the estimate, interval (when estimable), sensitivity bounds, failures, unresolved outcomes, cost and latency. Do not tune the gate after seeing results.

## Evidence boundary

The ablation auditor verifies artifact hashes, report structure and task-level outcome agreement. It cannot prove that a runner produced a score report honestly, that a provider served the requested model, or that a randomization schedule was executed exactly as recorded. The upstream score audit itself relies on runtime-generated evidence, not cryptographic host attestation. Preserve raw tool trajectories, immutable image digests, scorer outputs and runner provenance; arrange independent reproduction before a causal or state-of-the-art claim.
