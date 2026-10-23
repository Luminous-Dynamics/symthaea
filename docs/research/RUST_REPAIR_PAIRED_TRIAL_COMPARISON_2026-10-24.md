# Paired Rust Repair Trial Comparison — 2026-10-24

**Status:** implementation proposal + executable comparison tool; not a measured model result, and not an SOTA claim.

## Why this exists

A validated task manifest and a validated prediction run establish that a reported outcome is auditable. They do **not** establish that an intervention caused an improvement, that a small measured gain is larger than noise, or that a gain is economically worthwhile. The paired comparison tool closes part of that gap.

It consumes two outputs from `rust-repair-score-audit` (control and treatment), an externally hash-locked preregistration, and a complete per-task cost/receipt ledger. It refuses mismatched task universes, mismatched task-manifest/candidate digests, duplicate IDs, inconsistent summary counters, and missing cost rows.

## Invoke

After the task manifest and candidate split are locked, run both arms under the same frozen harness, then independently audit both prediction sets. Preserve all original evidence and the exact score reports.

```bash
cargo run --locked --bin rust-repair-trial-compare \
  --features rust-repair-manifest -- \
  --preregistration trial-preregistration.json \
  --expected-preregistration-sha256 <SHA256-locked-before-rollout> \
  --control-report control-score-report.json \
  --treatment-report treatment-score-report.json \
  --cost-ledger per-task-cost-ledger.json \
  --report-out paired-comparison-report.json
```

The digest supplied at runtime must be pinned outside the preregistration file and before rollouts begin (for example, in the signed experiment ticket or immutable run record). The tool checks the entire preregistration byte sequence against it.

## Preregistration contract

The following is a shape example, **not a runnable dataset**. Replace the placeholder hashes and task rows before hashing and before collecting either arm. Commit the resulting file or store it in an immutable experiment registry. Do not add, drop, relabel, or alter tasks after observing results.

```json
{
  "format_version": 1,
  "protocol_id": "symthaea-rust-repair-paired-comparison",
  "study_id": "rust-repair-gcs-ablation-2026-10-24",
  "control_arm": "gcs_off",
  "treatment_arm": "gcs_on",
  "control_model_scaffold": "record exact model ID, version, scaffold commit, and settings",
  "treatment_model_scaffold": "record exact model ID, version, scaffold commit, and settings",
  "task_manifest_sha256": "REPLACE_WITH_64_HEX_SHA256",
  "candidate_jsonl_sha256": "REPLACE_WITH_64_HEX_SHA256",
  "bootstrap_seed": "rust-repair-gcs-ablation-v1",
  "bootstrap_replicates": 10000,
  "min_tasks": 250,
  "min_repositories": 20,
  "max_unresolved_rate": 0.02,
  "max_unresolved_rate_difference": 0.01,
  "target_absolute_effect": 0.05,
  "cost_reduction_target": 0.20,
  "noninferiority_margin": 0.015,
  "tasks": [
    {
      "instance_id": "owner__repo-issue-123",
      "repo": "owner/repo",
      "stratum": "rust;bugfix;medium",
      "evaluation_order": "control_first"
    },
    {
      "instance_id": "owner__other-repo-issue-456",
      "repo": "owner/other-repo",
      "stratum": "rust;regression;hard",
      "evaluation_order": "treatment_first"
    }
  ]
}
```

The example's thresholds are **proposed internal gates**, not industry-wide standards. Pre-register the task sampling plan, repository/stratum balance, model and scaffold identity, tool and network policy, time/token/cost limits, retries, and exclusion rules alongside this file. `evaluation_order` records the randomized order assigned before the run; preserve timestamps that let a reviewer verify the execution order. One run per arm/task is the current supported comparison unit; repeated-trial and multi-seed analyses require a separate prespecified extension, not treating attempts as extra independent tasks.

## Observed cost ledger

Costs are measured outputs, so they belong in a separate ledger captured from raw usage receipts, not in the preregistration. Every task must be represented in both arms, including failures and unresolved attempts.

```json
{
  "format_version": 1,
  "study_id": "rust-repair-gcs-ablation-2026-10-24",
  "tasks": [
    {
      "instance_id": "owner__repo-issue-123",
      "control_cost_usd": 0.42,
      "treatment_cost_usd": 0.39,
      "control_receipt_sha256": "REPLACE_WITH_64_HEX_SHA256",
      "treatment_receipt_sha256": "REPLACE_WITH_64_HEX_SHA256"
    }
  ]
}
```

The implementation validates receipt hash shape and locks the hash of the entire cost-ledger file in the output. It does **not** authenticate provider receipts or fetch their contents. Preserve raw receipts and reconcile them independently; do not infer a served model/tier from a requested model name alone. Zero-dollar/local runs must use a consistently defined cost accounting policy and still preserve the compute, energy, hardware, and labor measures that matter to the claim.

## Statistical contract

### Primary outcome: conservative paired effect

Each task is paired across control and treatment. A resolved pass maps to 1; a resolved fail maps to 0. An unresolved attempt is not quietly scored as a fail or removed. Instead, its possible outcome widens the effect bounds:

- lower bound assigns unresolved treatment outcomes to failure and unresolved control outcomes to success;
- upper bound assigns unresolved treatment outcomes to success and unresolved control outcomes to failure.

The report gives the complete-case effect on pairs resolved in both arms **as a secondary descriptive statistic**, plus the conservative point-effect bounds. The decision interval is a 95% percentile interval generated by deterministic bootstrap resampling of whole **repository clusters**, not individual task rows. This is intended to reduce overconfidence caused by tasks from the same repository sharing architecture, tests, maintainers, and environment. It remains an approximate cluster-bootstrap procedure; it is not a randomization test or a substitute for independent review of the study design.

The bootstrap seed is derived from the preregistration byte hash and its locked `bootstrap_seed` string. Thus, the same preregistration and rows produce the same draws. Keep the raw per-task data and exact report; the summary alone is not enough to reproduce the experiment.

### Prespecified gates

With the example settings, the tool records these separate decisions:

1. **Evidence/quality gate:** at least 250 paired task IDs across at least 20 repositories; each arm's unresolved rate is at most 2%; the absolute arm difference in unresolved rates is at most 1 percentage point. An unresolved-rate imbalance outside the gate blocks a decision-grade comparison.
2. **Superiority target:** the complete-case treatment gain is at least 5 percentage points and the lower end of the conservative repository-bootstrap interval is above zero, after the evidence/quality gate passes.
3. **Cost target:** treatment cost per verified fix is at least 20% lower and the conservative interval's lower bound is no worse than -1.5 percentage points, after the evidence/quality gate passes.

The overall `decision_grade` field means only that the preregistered sample-size and unresolved-rate gates passed. It does **not** mean treatment won. The superiority and cost flags are separate. The cost denominator is the count of verified passing tasks; total cost includes every attempted task, including unresolved attempts, so spending on failed runs cannot disappear from the numerator. If an arm has zero verified fixes, cost per fix is null.

The cost gate is deliberately tied to the same preregistered outcome non-inferiority margin. Change the effect/cost thresholds only before looking at trial results. Report point estimates, conservative intervals, unresolved counts, cost totals, and every input hash—not only the favorable gate.

## Threats this still does not close

- **Model/provider identity:** the comparison tool accepts score reports assigned by command-line role. It does not cryptographically prove which model served a call. Bind provider request IDs, raw usage receipts, scaffold commit, runtime, tool-call trajectory, and settings into a separately reviewed run record.
- **Receipt fabrication:** well-formed hashes are not evidence that files are truthful. Preserve raw receipts, make the run bundle content-addressed, and support independent retrieval.
- **Prompt contamination / hidden solution leakage:** this comparator assumes the task-level audit and offline runtime contract were respected. It cannot prove a task image is uncontaminated.
- **Verifier defects:** it independently recounts report outcome labels against summary counters but does not re-run tests. Continue reference-patch-pass, empty-patch-fail, regression-guard, mutation, and independent task-validity reviews.
- **Repeated attempts / seed variance:** this version compares one scored attempt per task per arm. Do not use it to claim repeated-trial performance; a repeated design needs explicit attempt IDs, a locked aggregation rule, and inference that accounts for task and repository clusters.
- **Generalization:** repo-cluster bootstrap reflects the sampled repositories. A 20-repository threshold is not a guarantee that the sample represents the ecosystem; report repository and task families, perform held-out repository evaluation, and maintain a private contamination-resistant cohort.

## Industry comparison and why the bar should be evidence, not a single percentage

- SWE-bench-Live describes itself as a continually updating multi-language/multi-OS evaluation and states that submission trajectories are now required for verification. Its public headline result (71.0% on a 300-instance leaderboard row dated 8 October 2026) is mixed-language, not a Rust-only comparable score. Use a pinned Rust split and matched protocol, not that headline as an apples-to-apples target. <https://swe-bench-live.github.io/>
- Scale's SWE-bench Pro V2 announcement describes a refreshed 642-task public split across 11 repositories (down from 731 after removing 89 instances), network-locked evaluation, reference-patch/empty-patch checks, and regrading on pristine images. These are relevant design controls; a task count alone does not certify validity. <https://labs.scale.com/blog/swe-bench-pro-v2>
- Epoch AI's September 2026 review marked Terminal-Bench 4.0.0 flawed after identifying public scoring defects in 30 of 66 reviewed tasks in that version, including reward-hack paths and answer leakage. The benchmark authors separately describe fixes and semantic versioning for later releases. Treat the review as evidence for adversarial grader audits—not as evidence that every later version is flawed. <https://epoch.ai/benchmarks/terminal-bench-4/review> <https://www.tbench.ai/news/terminal-bench-4-0>

A credible "better than SOTA" claim requires the same frozen task universe, grader, environment, model/scaffold budgets, and scoring code; pinned versions and provenance; a private or held-out repository cohort; uncertainty intervals; cost/latency; independent grader audit; and replication. Publish all assigned tasks, including unresolved ones, plus exact hashes and exclusion reasons. Never compare a fresh Rust slice directly with another benchmark's mixed-language headline.

## Validation status

The source contains unit tests for conservative unresolved bounds, deterministic resampling, quantiles, duplicate IDs, summary-counter tampering, missing coverage, invalid hashes, and report inconsistency. The binary has been registered in Cargo and in the existing pinned-evaluator workflow for formatting, tests, and strict Clippy. Those checks must be run before treating the tool as verified. An authored test is not a passing test; a queued, skipped, or unobserved CI run is not a pass.
