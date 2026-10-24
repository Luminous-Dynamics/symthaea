# Rust Repository-Repair Paired Analysis (2026-10-24)

## Why this exists

The scored-run auditor establishes whether each attempt has a defensible task-level outcome. It does not estimate whether an intervention such as Geodesic Code Synthesis (GCS) caused an improvement. A headline pass rate alone cannot distinguish treatment effect, repository composition, correlated tasks, or differential missingness.

`rust-repair-paired-analysis` is a separate analysis stage. It consumes one locked task manifest and two independently audited score reports, such as `gcs_off` and `gcs_on`. It never executes agents or tests and never changes the scorer's verdicts.

## Input contract

The tool rejects the analysis if:

- either score report has a different task-manifest byte digest, candidate JSONL digest, protocol ID, format version, or task count;
- either report omits a locked task, duplicates a task ID, includes an unknown ID, or contains an outcome other than `pass`, `fail`, or `unresolved`;
- a manifest task has an empty ID or repository field.

The repository label comes from the locked manifest. It is not inferred from model output, filesystem paths, or the score report. Input and manifest SHA-256 digests are recorded in the output.

## Estimands and missingness

For tasks resolved in **both** arms, the primary descriptive contrast is:

`100 × (pass_rate_A − pass_rate_B)`

in percentage points. A task contributes +1 when A passes and B fails, −1 when A fails and B passes, and 0 when their binary outcomes agree. A task unresolved in either arm does not enter this complete-pair contrast; counts remain explicit.

The second estimate is a worst-case interval across the **entire locked manifest**, including unresolved tasks:

- lower bound: assign unresolved A outcomes to fail and unresolved B outcomes to pass;
- upper bound: assign unresolved A outcomes to pass and unresolved B outcomes to fail.

These bounds are sensitivity limits, not confidence intervals. They expose the strongest treatment-direction conclusion compatible with missing outcomes without deleting rows from the denominator. Asymmetric unresolved rates should halt interpretation until their cause is understood.

## Statistical uncertainty

The tool uses a deterministic percentile bootstrap that resamples repository clusters with replacement, keeping all complete task pairs from a selected repository together. The seed and iteration count are stored in the report (default seed `20261024`, default 20,000 iterations). It emits a 95% interval only when at least two repository clusters contribute complete pairs.

A repository count below 20 is flagged as exploratory. The proposed internal decision gate is at least 20 independent repository clusters, complete matched conditions, a pre-registered analysis, and a repository-cluster interval supporting the stated direction. Twenty is a project quality target, not an industry standard or a guarantee that a sample is well powered.

The bootstrap estimates uncertainty conditional on the represented repositories and current task set. It does **not** correct benchmark leakage, treatment noncompliance, poor task sampling, adaptive selection, or a misspecified target population. Repositories are the cluster unit because task outcomes from one repository are not plausibly independent draws.

## Example invocation

```bash
cargo run --locked \
  --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
  --bin rust-repair-paired-analysis -- \
  --manifest build/rust-repair/eligible-task-manifest.json \
  --arm-a build/rust-repair/gcs-off-score-report.json \
  --arm-b build/rust-repair/gcs-on-score-report.json \
  --arm-a-label gcs_off --arm-b-label gcs_on \
  --seed 20261024 --iterations 20000 \
  --out build/rust-repair/gcs-paired-analysis.json
```

Run one attempt per assigned task for the strict pass@1 comparison. Repeated trials, alternate sampling budgets, network-enabled variants, or best-of-k attempts need separate preregistrations and must not be pooled.

## Proposed research gates (not results)

1. Primary causal target: at least +5 percentage points for the intervention with a repository-clustered 95% interval excluding zero, under identical preregistered budgets and task assignment.
2. Alternative efficiency target: at least 20% lower measured cost per verified fix while correctness is within a preregistered 1.5 percentage-point non-inferiority margin.
3. Unresolved rate below 2% overall and between-arm difference below 1 percentage point as internal pause thresholds; investigate missingness rather than reclassifying it.
4. Report task count, repository count, task-family coverage, model/provider and version, scaffold/tool policy, time/token budget, latency, cost per verified fix, regressions, unresolved count and source hashes.
5. A “better than SOTA” claim still requires a named comparator on the same pinned split, grader revision, scaffold/tool access and resource budget, plus a held-out cohort for any generalization claim.

These thresholds are proposals to preregister, not achievements, universal criteria, or evidence that GCS currently improves repair capability.

## Validation and claim ceiling

The new binary has authored unit tests for directionality, unresolved outcomes, sensitivity bounds, deterministic sampling, bootstrap repeatability, cluster insufficiency, and percentile selection. GitHub Actions is configured to compile/test it and run strict Clippy in the standalone workspace, avoiding the unrelated broken root-workspace member that blocked the previous audit workflow.

Until that exact-head run succeeds, tests are authored but not runner-verified. A passing analysis binary only establishes the behavior of this analysis implementation against its tests. It does not establish causal improvement, evaluator validity, generalization, or SOTA.
