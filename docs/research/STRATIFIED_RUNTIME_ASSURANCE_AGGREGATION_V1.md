# Stratified Runtime-Assurance Benchmark Aggregation v1

**Status:** analysis utility plus deterministic synthetic qualification fixture; not a controller benchmark result. The fixture establishes software behavior only.

## Research question

Does a candidate method improve outcomes across a fixed suite of heterogeneous tasks, or does one easy/high-scoring task dominate the headline number? This report builds a task-balanced aggregate from paired per-episode scores, while retaining each task's result.

## Input and normalization contract

The CLI accepts the strict, unquoted CSV schema:

```csv
task,seed,baseline_score,candidate_score
cartpole,1001,0.42,0.51
cartpole,1002,0.39,0.46
quadrotor,2001,0.64,0.62
quadrotor,2002,0.58,0.60
```

Scores must be finite and within [0, 1], with **higher always better**. Normalize each task using a predeclared, task-specific mapping based on fixed anchors or domain criteria before assembling the CSV. Do not normalize by the evaluated candidate's observed best/worst scores; that would make the scale depend on the method being measured. Do not directly pool reward, risk, latency, and other incompatible quantities. Safety violation rates and confidence bounds remain separate task/condition results in the episode-level safety report.

Each task needs at least two distinct paired episode seeds, and at least two tasks are required for aggregate output. Duplicate task/seed rows, quoted cells, absent headers, invalid scores and input/output path aliases fail closed.

Example:

```sh
cargo run --locked --release \
  --manifest-path tools/allostatic-control-bench/Cargo.toml \
  --bin allostatic-stratified-eval-bench -- \
  normalized-paired-scores.csv stratified-summary.csv 10000 17
```

The tool emits one `task` row per benchmark task plus one `aggregate` row. Task rows report paired count, baseline and candidate normalized means, candidate-minus-baseline mean delta, interquartile mean (IQM) of paired deltas, same-seed paired win rate, and within-task marginal probability of superiority from independent candidate/baseline episode draws (ties count as one half). These two win metrics are different estimands; the latter is not a paired-seed statistic.

## Aggregate estimands

All macro scores weight each task equally, regardless of task episode count:

- **Macro mean delta:** average of task-specific mean paired deltas.
- **Macro task-IQM delta:** average across tasks of the IQM of the paired score deltas within each task.
- **Task-balanced score IQM:** compute the interquartile mean over a weighted mixture of normalized baseline scores and candidate scores, with every task contributing equal total probability mass regardless of its episode count. Report baseline IQM, candidate IQM, their difference, and a 95% stratified-bootstrap interval for the difference. This is more directly comparable to RLiable's IQM discipline than the macro IQM of paired deltas, but remains conditional on our fixed task normalization/mixture.
- **Macro paired win rate:** task-equal average of the fraction of same-seed candidate wins, with ties worth one-half.
- **Macro marginal probability of superiority:** equal-task-weight mean of within-task (P(\text{candidate episode} > \text{baseline episode})) from independent draws, with ties worth one-half. This is distinct from paired win rate and should not be called RLiable's exact probability-of-improvement estimator.
- **Fraction of tasks improved:** share of tasks whose mean paired delta is strictly positive.

The 95% percentile confidence intervals come from a deterministic **stratified episode bootstrap**: within each task, sample that task's paired episodes with replacement, keep the task suite fixed, recompute task statistics, then average task statistics equally. The actual bootstrap seed and replicate count are emitted.

These intervals quantify resampling uncertainty conditional on the selected task suite. They do not quantify uncertainty over a population of unseen tasks, and they are not safety guarantees. Add new tasks or perform a separate task-level generalization analysis before claiming broad generalization. Preregister task inclusion, normalization anchors, method identities, primary estimands, seed partitions, and any multiplicity correction before final evaluation.

## Why this complements SafeRLEval and RLiable

Recent SafeRLEval work argues that expected cost alone can hide how often and how severely constraints are violated, and recommends reporting aggregate, distributional, and task/safety-bound-specific results jointly ([Spoor et al., 2026](https://arxiv.org/abs/2609.15315)). RLiable recommends uncertainty intervals, score distributions, interquartile mean, and probability-of-improvement measures for reliable multi-task comparisons ([Agarwal et al.](https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/)).

This utility borrows the *evaluation discipline*, not an assertion of equivalence to either framework. Because it uses paired outcomes, its win statistic is explicitly a **paired win rate**, not RLiable's independent-sample probability-of-improvement estimator. The benchmark should publish task rows alongside the aggregate so the aggregate cannot conceal task regressions.

## Qualification and limits

The repository qualification script runs a deterministic fixture twice, requires byte-identical summaries, checks the report's row/column schema, verifies the separate task and aggregate records, and includes source/output hashes in the provenance receipt. Passing that fixture qualifies only the tool contract. External performance claims require pinned physics-based task adapters and independently scored actual-plant outcomes.
