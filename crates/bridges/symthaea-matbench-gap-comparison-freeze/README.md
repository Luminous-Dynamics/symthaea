# symthaea-matbench-gap-comparison-freeze

Pre-truth comparison freeze for the Matbench Benchmark Zero V0 policy ladder.

This crate exists to prevent a subtle but important form of post-hoc flexibility. Three individually replayable screening receipts are not yet one preregistered comparison if the shortlist budget, policy roster, random seed, or reported endpoints can still change after benchmark truth is inspected.

`freeze_comparison(plan, target, top_k, random_seed)` therefore constructs all three authored V0 policies together from the same #2653 retained universe:

```text
DeterministicRandomOrder(seed)  -> null ordering control
Legacy TargetDistance          -> composition-only heuristic
CompositionOnly Learned        -> grouped RF residual model
```

No benchmark truth is an input to this crate.

## Frozen comparison design

The receipt binds:

- exact truth-free source composition order;
- exact exposure partition;
- exact exposed-training snapshot;
- exact retained-universe digest;
- canonical candidate-set digest and candidate count;
- preregistered target window;
- preregistered top-k evaluation budget;
- deterministic random seed;
- fixed non-scalar endpoint vector;
- each policy's machine-readable method provenance;
- each policy's truth-free screening-subject digest;
- each policy's ordered ranking digest;
- one shared legacy prediction-surface digest.

The full exposure-plan digest is retained for provenance but deliberately excluded from the truth-free comparison-subject digest because the full plan transitively binds the exact truth-bearing source artifact.

## Legacy control theorem

The random-order null and legacy target-distance control use the same baseline predictor. This crate requires their complete candidate -> predicted-gap mappings to be bit-identical and records one order-independent prediction-surface SHA-256.

Thus their intended difference is ordering policy only, not predictor output.

## Candidate-universe theorem

All three policy receipts must:

1. rank exactly `candidate_count` entries;
2. contain the same candidate-id set;
3. bind the same source-plan identities;
4. use the same target.

Replay against the source plan additionally requires `candidate_count == plan.retained_composition_count` and regenerates all three rankings.

## Endpoint contract

V0 preregisters three directions without a weighted scalar score:

```text
target_regret_ev              minimize
top_k_hits                    maximize
mean_abs_prediction_error_ev  minimize
```

Precision and recall remain present in the underlying Benchmark Zero receipt, but with a common exact universe and common `k`, precision is a deterministic transform of hits and recall shares the same qualifying-truth denominator. The freeze does not define a hidden scalar winner function.

## Intended measurement sequence

```text
official pinned artifact
        -> exposure plan
        -> comparison freeze   (NO TRUTH INPUT)
        -> restrict truth to exact retained universe
        -> exact-universe measurement
        -> report all preregistered endpoints
```

## Authority boundary

A valid freeze establishes that the policy roster, candidate universe, target, shortlist budget, random seed, prediction surfaces, and ranking outputs were committed together before measurement.

It does not establish holdout cleanliness, model accuracy, statistical significance, scientific superiority, material novelty/feasibility, experiment authorization, or candidate promotion.
