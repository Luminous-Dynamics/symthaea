# PITB paired repository/file-cluster benchmark statistics

A standalone Rust tool for paired, assurance-adjusted comparisons of theorem-proving strategies on the frozen SorryDB evaluation split.

## Why not an IID bootstrap?

The pinned 1,000-task SorryDB evaluation split is clustered:
- 78 repositories;
- 107 repository/commit pairs;
- 339 repository/file pairs;
- up to 21 tasks in a repository and 20 tasks in one source file.

Tasks from one file share definitions, imports, notation, local lemmas and project-specific difficulty. Treating all 1,000 outcomes as IID underestimates uncertainty. The resampling unit must preserve the project/file structure.

## Sampling design

For each bootstrap replicate:

1. Sample the same number of repositories with replacement from the 78 repository clusters.
2. Within each selected repository draw, sample the same number of source-file clusters with replacement as that repository originally contained.
3. Retain every task in each selected source-file cluster.
4. Apply the exact same sampled repository/file/task draw to both strategies.
5. Calculate the difference in assurance-adjusted task pass rates over the resampled paired task set.

The report contains the observed paired difference, a percentile 95% interval over the seeded hierarchical bootstrap, and the fraction of replicates where the left strategy exceeds the right. The seed and replicate count are recorded.

This is a two-stage nested cluster bootstrap, not an IID task bootstrap. It preserves source-file clusters nested inside repositories. It does not replace independent model repetitions, and it does not remove confounding when different strategies have different models, source visibility, tools, scaffolds, hardware or budgets.

## Paired task and proof identity

The tool accepts the existing `pitb.sorrydb-proof-run.v0` JSONL receipt format from `tools/pitb-sorrydb-audit`. Each strategy must cover **every task in the frozen 1,000-task evaluation split exactly**. Missing tasks are rejected rather than silently shrinking the denominator.

Every run must bind the exact task ID, repository commit, Lean version, source path and coordinates, plus the SHA-256 of the exact SorryDB debug-goal bytes. Across strategies, each task must have the same source-file digest, goal digest and pre-proof declaration signature. A run labeled `verified_success` must pass all gates: exact sorry replacement, sorry absence, `Prop` target, project build, kernel, source binding, semantic gate, vacuity gate, zero axiom delta and zero new-sorry delta. A task is a success if at least one attempt is assurance-qualified; retries are collapsed to one task outcome, but all retry cost/time/tokens remain charged to that strategy.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml
cargo test --locked --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml
cargo run --locked --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml -- --self-test

# Generate a synthetic, explicitly labeled report for schema validation:
cargo run --locked --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml -- \
  --self-test-report /tmp/clustered-comparison.synthetic.json

# List exact strategy identities in a real paired-run file:
cargo run --locked --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml -- \
  --list-strategies \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  /path/to/proof-runs.jsonl

# Run a matched comparison (10,000 replicates; fixed seed):
cargo run --locked --manifest-path tools/pitb-clustered-benchmark-stats/Cargo.toml -- \
  --compare \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  /path/to/proof-runs.jsonl \
  <left-strategy-id> <right-strategy-id> \
  /reports/paired-cluster-comparison.json \
  10000 20261105
```

The `--compare` path re-verifies the evaluation split's Git blob against the SorryDB-2601 pinned blob. It records the raw split SHA-256 and proof-run JSONL SHA-256.

## Strategy IDs and denominator policy

Use `--list-strategies` to obtain the exact SHA-256 strategy IDs. Identity includes name, version, implementation digest, model ID and model version. Distinct model versions or implementation builds are distinct strategies and must not be merged into one score.

The tool rejects:
- tasks outside the frozen evaluation split;
- duplicate run IDs;
- missing unique task coverage for any strategy;
- source/goal/signature drift for the same task across strategies;
- a claimed success lacking any required proof or semantic gate;
- a non-success run that passes every success gate;
- malformed task, proof or strategy digests.

Retries for one strategy/task are combined with logical OR for verified success (one valid attempt solves the task), while every retry contributes to cost, time and token totals. The report distinguishes unique task solves from attempt counts and reports cost per qualified task.

## Interpretation and recommended use

The bootstrap interval describes uncertainty from the sampled repository/file task population under the pinned test split. Do not call a positive interval a causal model improvement unless model, prompt/scaffold, tool access, source visibility, hardware, wall/CPU/GPU budgets, token budget, verifier, and stopping policy are matched. For stochastic models, repeat the entire paired experiment across independent seeds/runs and report both between-run and cluster-sampling uncertainty.

For cross-system research compare, at minimum:
- no-retrieval + symbolic tactics;
- BM25;
- DAPS;
- LeanGraphSearch;
- LeanSearch v2;
- PITB typed graph + retrieval;
- ablation variants with support-bundle retrieval disabled.

Primary metric: assurance-adjusted unique exact-target completions / 1,000 tasks. Secondary metrics: retries, wall time, input/output tokens, cost per qualified task, source/goal replay coverage, verifier failures, abstention, and proof maintenance following upstream changes. Recall@K remains a diagnostic and must not substitute for verified completion.

A synthetic report is explicitly labeled `synthetic_fixture=true`; it is not a real 1,000-task result and must never be presented as a SorryDB score.
