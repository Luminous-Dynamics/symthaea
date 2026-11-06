# PITB matched benchmark comparison (Rust)

A standalone Rust engine for paired, cluster-aware comparison of theorem-proving systems. It sits **after** the retrieval, graph, source-reproduction and proof-verification receipts. Its job is to determine whether two systems were tested on the same tasks under the same environment and resource policy, and to report success at several distinct assurance levels.

## The paradigm shift

A leaderboard should not collapse all outcomes into one pass percentage. A result can be:

1. a generated candidate;
2. a proof accepted by the Lean kernel for a pinned proposition;
3. a proof for a task whose source location, source commit, Lean toolchain and elaborated goal were independently reproduced;
4. a proof whose proposition has also passed specification-fidelity, vacuity and assumption audits.

Those are different claims. The report therefore separates:
- **raw kernel-verified success**;
- **source-reproduced success**;
- **assurance-qualified success**;
- **assurance-disqualified success** (for example, source-reproduction failure, explicit goal mismatch, vacuity finding or assumption finding);
- **assurance-unreviewed success** (the required evidence was not run or is incomplete).

A model must not get credit for a source-reproduced success when source reproduction was not performed. A kernel proof of the formal proposition does not itself establish that the proposition faithfully models the software or mathematical intent.

## Matched cohort contract

The cohort manifest freezes:
- benchmark, snapshot and test split IDs;
- split-manifest digest and exact task-ID-set digest;
- an explicit bootstrap cluster unit: `repository` or `repository_file`;
- one budget-policy digest and hard per-task caps for attempts, wall time and cost;
- which independent assurance gates are mandatory;
- exact arm identities (model/version, scaffold digest and implementation digest);
- deterministic cluster-bootstrap seed and replicate count;
- deterministic cluster sign-flip Monte Carlo replicate count (used only when exact enumeration is not practical).

Each JSONL run row binds the same task ID, repository revision, Lean version, source file, target statement digest, environment digest and budget policy across every arm. The analyzer requires exactly one result for every task/arm pair. It rejects:
- different task sets, snapshots, splits, environments, repository commits, Lean versions or budget policies;
- duplicate or missing task/arm rows;
- an undeclared model, changed scaffold or implementation;
- results outside the frozen resource envelope;
- `solved` without independent kernel verification;
- `did_not_compete` with nonzero attempts/resources;
- missing evidence digests for a gate marked passed or failed.

The cluster identifier is checked against the declared rule, not trusted as an arbitrary grouping label:
- `repository`: `cluster_id == repository`;
- `repository_file`: `cluster_id == repository + "::" + source_file`.

For SorryDB_2601, use `repository` as the primary cluster unit (78 clusters in its 1,000-row task-level evaluation split). Also publish a `repository_file` sensitivity analysis (339 clusters). The split contains all 78 projects from the full corpus, so neither form should be described as project-disjoint generalization.

## Statistics

Pairwise comparisons are exact-task paired: each arm receives the same tasks. The uncertainty interval uses a deterministic, repository-cluster bootstrap, resampling whole clusters rather than treating task rows as IID. The point estimate is task-weighted over the frozen test cohort. Record:
- challenger wins, reference wins and ties on assurance-qualified completion;
- paired success-rate difference;
- 95% percentile interval from the cluster bootstrap;
- number of tasks and clusters;
- fixed task-set/split/budget digests and bootstrap seed.

Use at least 10,000 bootstrap replicates for a public result (the contract permits 1,000 for fast development and caps at 100,000). Report the task-weighted point estimate and the cluster interval together. A positive point estimate whose interval crosses zero is not a demonstrated improvement.

The engine is deterministic for the same cohort, rows and seed. The bootstrap is an uncertainty estimate for the chosen benchmark cohort; it does not magically make the cohort project-disjoint or representative of all future repositories.

Pairwise inference also includes a paired **repository-cluster sign-flip test**. For up to 20 clusters, all sign assignments are enumerated exactly; larger cohorts use the declared deterministic Monte Carlo replicate count. When three or more arms create a family of pairwise tests, the report applies Holm's step-down family-wise error correction to the complete unordered comparison family. The test depends on cluster-level exchangeability/symmetry under the null; report the number of independent repositories and do not interpret a task-level sample size as the number of independent units. Exact sign-flip p-values with very few repositories are coarse by construction, which is why project-disjoint evaluation and additional independent projects remain necessary.

## Metrics

Primary metric:
**assurance-qualified exact-target completions / all tasks in the frozen test cohort** under the same per-task budget and the same pinned proof-checking policy.

Always publish the three distinct success rates:
- raw kernel-verified success rate;
- source-reproduced success rate;
- assurance-qualified success rate.

Also publish attempt coverage, did-not-compete count, attempt total, wall time, tokens, total cost, median and p95 cost per task, and cost per success at each assurance level. Cost is stored in integer micro-USD to avoid floating-point currency accumulation.

Do not use a denominator that excludes timeouts or non-competes without saying so. The default success rate is over all frozen task IDs; a did-not-compete is not a success. Comparisons across different model API pricing, hardware, test snapshots or stopping policies require a separately declared normalized cost protocol.

## SOTA benchmark portfolio and targets

These are *separate tracks*. Do not turn the numbers into one leaderboard or compare them as if they were the same task distribution.

| Track | Published reference point | PITB target for a matched evaluation |
|---|---|---|
| SorryDB_2601 real open proofs | The public 1,000-task task-level split reports 57.2% Pass@1 for the current public Aleph Prover leaderboard entry; the original paper reports 30.3% for its Gemini Flash 3 agentic strategy and 35.7% for its combined strategy. These are different systems/settings and are not a matched comparison. | First run exact same 1,000 task IDs, source commits, Lean versions, per-task caps and evaluator. Aim for +5 percentage points over the strongest reproducible baseline with the repository-cluster 95% interval above zero; then +10 points. Report project-disjoint and chronological generalization separately. |
| AutoformBench advanced formalization | The initial public 341-task release reports GPT-6 Astra at 281/341 (82.4%) and Opus 5 at 211/341 (61.9%) in the source-present setting. | First reproduce the chosen baseline/harness. +5 points requires at least 299/341; +10 points requires at least 316/341. Report cost and generated declarations as well as completion. The release is CC BY-NC 4.0 and explicitly not intended as training data; use it for evaluation, not model training. |
| Lean premise retrieval | On the shared 810-query MathlibQR subset, LeanGraphSearch reports nDCG@10 0.629 / Recall@10 0.794, compared with LeanSearch v2 standard 0.616 / 0.765; on MPR, complete-route coverage@50 is reported as 44.2% vs 41.3% for LeanSearch v2 Reasoning. | Target nDCG@10 >= 0.65, Recall@10 >= 0.82, MPR complete-route coverage@50 >= 0.50, then show a downstream +5-point improvement in kernel-verified exact-target completion under the same prover budget. Retrieval-only wins do not establish proof-solving improvements. |
| Proof assurance / benchmark validity | A 2026 audit reports thousands of findings across formal-verification benchmarks, including mechanically certified vacuity, counterexamples and unsound assumptions. | Zero known unapproved axioms or vacuous tasks admitted to the assurance-qualified denominator; 100% exact task/environment/receipt binding; independently replay every admitted proof and source reproduction. |

**Comparability rule:** the references above are targets, not evidence that PITB already beats them. Different benchmark versions, model access, harnesses, test budgets, proof checkers and source visibility prevent a fair direct comparison. Only a matched cohort accepted by this validator can support a paired claim.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-matched-benchmark-comparison/Cargo.toml
cargo test --locked --manifest-path tools/pitb-matched-benchmark-comparison/Cargo.toml
cargo run --locked --manifest-path tools/pitb-matched-benchmark-comparison/Cargo.toml -- --self-test

# Write a synthetic, schema-checkable cohort and paired run ledger:
cargo run --locked --manifest-path tools/pitb-matched-benchmark-comparison/Cargo.toml -- \
  --self-test-fixtures /tmp/pitb-cohort.json /tmp/pitb-runs.jsonl

# Compare real run results under the frozen cohort:
cargo run --locked --manifest-path tools/pitb-matched-benchmark-comparison/Cargo.toml -- \
  --compare /data/cohort.json /data/paired-runs.jsonl /reports/matched-comparison.json
```

The synthetic cohort contains two repository clusters and four tasks across two arms. It deliberately includes a raw kernel success that remains assurance-unreviewed, demonstrating why a single pass metric is misleading.

## Evidence boundary

The validator checks task/run identity, cohort completeness, budget identity, declared resource caps, gate-state/receipt-digest presence and statistical aggregation. It does **not** open or authenticate the referenced source/build/semantic-audit artifact bytes, rerun Lean, validate model checkpoints, or prove the completeness of semantic audits. An external receipt verifier must independently open and authenticate those bytes before public assurance-qualified leaderboard admission.

The report explicitly sets `scheduler_admission=false`. This is a research comparison engine, not production proof scheduling and not a claim of benchmark performance.
