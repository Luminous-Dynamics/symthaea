# connectome-proofreading-policy

A dependency-free Rust policy primitive for allocating a finite human-review budget across connectome proofreading operations.

## What it does

`plan_reviews` ranks operations by expected avoidable error cost per reviewer-second:

```text
error_probability
× missed_error_cost_units
× review_resolution_probability
÷ review_seconds
```

It applies a deterministic greedy budget rule, skips cases that do not fit the remaining time, and uses ascending operation ID to break exact priority ties. The receipt reports selected/deferred IDs, spent time, expected baseline cost, expected avoided cost and expected residual cost.

`review_resolution_probability` is conditional on expert review and must be independently measured. A value of 1.0 represents an ideal-review upper bound, not an empirical guarantee.

## Explicit limitations

- This is a **greedy value-density heuristic**, not an exact 0/1 knapsack optimizer. It can be suboptimal when review durations differ; a unit test preserves a counterexample. Compare it with the exhaustive small-cohort enumeration oracle and other scheduling policies.
- The probabilities and costs are supplied by the caller. The crate does not train or calibrate a model, estimate causal effects, infer anatomical connectivity, or inspect raw EM.
- It does not know ground-truth labels and cannot calculate empirical review recall, precision, calibration, or human productivity. Those metrics belong to the evaluation harness.
- It allocates review time only. A review recommendation is not a completed edit.
- A production run must pin the model, calibration, cost profile, dataset revision, view contract and prediction artifact. See `research/connectomics/PROOFREADING_POLICY_EVALUATION_V0.1.md`.

## Exact small-cohort oracle

`plan_reviews_exact_small` exhaustively evaluates all review subsets for at most 20 candidates. It globally maximizes the supplied floating-point expected avoided-cost objective under the integer-second budget, then breaks ties by lower time and lexicographically smaller operation IDs. This is a regression oracle for small fixtures and sampled cohorts, not a large-dataset production scheduler.

## API sketch

```rust
use connectome_proofreading_policy::{
    plan_reviews, ProofreadingTask, ReviewBudget, ReviewCandidate,
};

let candidates = vec![
    ReviewCandidate {
        operation_id: "op-17".to_owned(),
        task: ProofreadingTask::MergeError,
        error_probability: 0.92,
        missed_error_cost_units: 3.0,
        review_resolution_probability: 0.98,
        review_seconds: 75,
    },
];

let plan = plan_reviews(&candidates, ReviewBudget { seconds: 300 })?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Tests

Run in an environment with Rust installed:

```sh
cargo test --locked --manifest-path tools/connectome-proofreading-policy/Cargo.toml
```

Sixteen unit tests are authored, including the review-budget known answer, cost-sensitive ranking, review-resolution adjustment, deterministic tie-breaking, zero budget, non-fitting cases, a greedy-suboptimality counterexample and malformed-input rejection. They have **not** been compiled or executed in the current authoring environment.

## Next implementation steps

1. Add randomized small-cohort property tests to quantify greedy regret and oracle agreement.
2. Add evaluator functions for review recall/precision at time budgets, with explicit label and missingness semantics.
3. Add cluster-bootstrap evaluation by volume/neuron.
4. Add a separate acquisition policy using decision-value-of-information, not entropy alone.
5. Add prospective edit-transaction replay and independent topology/lineage verification. Do not merge those claims into this review-only crate.
