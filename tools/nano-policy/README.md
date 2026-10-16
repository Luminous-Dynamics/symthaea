# Nano policy prototype

A dependency-free Rust research crate for comparing simple candidate-selection policies in nanotechnology studies. It is intentionally isolated from the root Symthaea Cargo workspace, and it is not a physical simulator or production model.

## Why it exists

Before claiming Symthaea improves material discovery, test the narrowest possible hypothesis: can a deterministic policy select more useful/diverse candidate evaluations under the same budget than random or cheapest-first baselines?

This crate provides four policy arms over the same predeclared feature vectors:

- `random`: deterministic pseudo-random baseline with an explicit seed;
- `cheapest`: lowest caller-estimated cost first;
- `max_min_diversity`: greedily maximize the nearest-distance to already selected candidates;
- `cost_aware_diversity`: the same coverage heuristic, discounted by declared relative cost.

If no prior references are supplied, the first diversity selection is chosen by distance from the normalized feature centroid. If reference feature vectors from previously evaluated candidates are supplied, the first selection maximizes distance from those references; later selections maximize minimum RMS distance from both references and new selections. Features are min-max normalized per dimension across the current candidates and references, so callers can preserve the same pool scale during sequential selection. The cost-aware score is:

`score = nearest_feature_distance * (candidate_cost / minimum_pool_cost)^(-cost_exponent)`

All feature similarity is just a *selection heuristic*. It is not energy, uncertainty, physical novelty, synthesize-ability, safety, or a probability of success.

## Build and test

From the repository root:

```sh
cargo test --locked --manifest-path tools/nano-policy/Cargo.toml
cargo clippy --locked --manifest-path tools/nano-policy/Cargo.toml --all-targets -- -D warnings
```

This crate has no third-party dependencies and a local lockfile. The environment used to author this change had no Rust toolchain, so these commands are not claimed to have passed locally. The dedicated GitHub Actions workflow is the execution gate; continue other research rather than waiting on it.

## API

The caller constructs a `SelectionRequest` containing:
- request schema version `nano-selection-request-v1`;
- immutable or versioned `feature_schema_id`, including feature transform and order;
- candidate IDs, numeric features and estimated cost;
- optional prior-evaluation feature vectors using the same schema (no target values), enabling sequential diversity sampling;
- policy, selection count, optional total cost budget, explicit seed and cost penalty.

The result is a typed `SelectionReceipt` containing ranked selections, policy scores, cost accounting and exactly one disposition for every input candidate. Oversized diversity batches, non-finite inputs, duplicate IDs, inconsistent feature dimensions, invalid budgets, and excessive work estimates are rejected.

The feature schema ID is caller-supplied, not verified by the crate. A benchmark harness still needs to audit feature generation, forbid target labels and future/outcome fields, freeze train/validation/test identities, validate budget semantics and independently recompute downstream task metrics. This crate does not parse candidate structures, invoke an oracle, or claim experimental outcomes. Keep any serializer/JSONL adapter outside the scientific core and give it its own versioned contract tests.

## Suggested benchmark use

For a frozen candidate pool:
1. Construct the same allowed feature vectors and cost estimates for every policy arm.
2. Freeze seed sets, selection count or total budget, candidate-group splits, and one primary downstream metric before outcomes are observed.
3. Run each policy with equal access to the unlabeled pool. Reveal the oracle result only after selection.
4. Record every selected, not-selected and over-budget candidate.
5. Compare against random and cheapest-first; then compare max-min and cost-aware diversity.
6. Measure task-level utility per budget (not just diversity scores), failure/invalid-query rate, time, and resource cost.
7. Use an independent checker to recompute results; retain null and failed runs.

For nanophotonics, downstream utility could be the best-so-far spectral error or Success@K under a frozen independent evaluation. For OC22, use the official IS2RE target only after source identity, input/target pair mapping and split reconciliation are verified. Never transfer metric thresholds across domains.

## Algorithm bounds

- Maximum 10,000 combined candidate and reference-feature vectors per request.
- Maximum 1,024 feature dimensions.
- Diversity work estimate is capped at 100,000,000 pair-dimension units and includes candidate-to-reference comparisons, within-batch distance comparisons, and feature-normalization work. Large histories or batches are rejected before distance evaluation if they exceed the cap.
- Candidate costs must be finite and positive. A total budget must be finite and nonnegative.
- The random baseline uses an internal SplitMix64 generator with rejection sampling for bounded indices; results are deterministic for a fixed seed and input order.

These limits make the prototype bounded and auditable, not optimized for large-scale production.
