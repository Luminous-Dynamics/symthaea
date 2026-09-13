# symthaea-matbench-gap-frozen-measurement

Post-freeze truth restriction and exact-universe Benchmark Zero measurement for the Matbench gap ladder.

The crate enforces a one-way protocol boundary:

```text
full official parsed dataset
        ↓
restricted truth receipt
        ↓
-------------------------------- truth handoff
        ↓
frozen comparison + restricted truth
        ↓
exact-universe measurement
```

No screening or model-fitting API is imported into the measurement path.

## 1. Official truth restriction after freeze

`restrict_official_truth_after_freeze(plan, freeze, dataset)` is the only operation in this crate that sees the complete parsed Matbench dataset.

Before selecting any values it requires:

- the comparison freeze to bind the supplied exposure plan;
- the plan to name the pinned official `matbench_expt_gap` dataset and compressed SHA-256;
- the parsed dataset compressed SHA-256 and row-order SHA-256 to equal the plan;
- the parsed truth provenance to bind the same dataset/content identity;
- source URI and license disclosure to equal the current official adapter constants;
- exact row count agreement;
- exact ordered `(candidate_id, composition)` parity between the parsed dataset and exposure plan;
- exact full truth-keyset equality with the plan's complete source candidate universe.

Only after those checks does it select the candidate IDs already committed in the frozen comparison.

The resulting `RestrictedTruthReceipt` records:

- source-plan identity;
- comparison-subject identity;
- parent compressed and row-order identities;
- exact parent truth provenance;
- retained candidate-universe digest/count;
- SHA-256 over sorted `(candidate_id, f64::to_bits(gap))` restricted truth records;
- a `BandgapTruthSet` whose split id is derived from the retained candidate-universe digest and whose content digest identifies the restricted truth values rather than pretending the full source artifact is the subset content.

## 2. Measurement without reranking

`measure_frozen_comparison(freeze, restricted_truth)` cannot access the full source dataset and does not call any screen/model API.

It invokes the exact-universe Benchmark Zero wrapper for the three already-frozen runs using the already-frozen `top_k`:

```text
random-order null
legacy target-distance
composition-only learned
```

The function cannot change the target, shortlist budget, candidate universe, random seed, policy roster, predictions, or ranking order.

## Endpoint reporting

The final receipt repeats the preregistered non-scalar endpoint vector and records the three corresponding values for each policy:

```text
target_regret_ev              minimize
top_k_hits                    maximize
mean_abs_prediction_error_ev  minimize
```

No scalar winner or weighted score is computed.

## Replay

Two separate replay functions exist:

- `verify_restricted_truth_receipt` re-derives the retained truth subset from the official parsed dataset;
- `verify_frozen_comparison_measurement` recomputes exact-universe measurements from the already-frozen runs and already-restricted truth.

Neither replay substitutes for exact source/tree/toolchain qualification.

## Authority boundary

A valid measurement receipt would establish that the preregistered policy outputs were measured over exactly the admissible frozen candidate population using truth values derived from the pinned official source artifact.

It would not establish global holdout cleanliness, absence of historical/manual prior knowledge, calibrated uncertainty, statistical significance, scientific superiority, material novelty/feasibility, experiment authorization, or candidate promotion.
