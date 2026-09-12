# symthaea-energy-benchmark-zero-promotion

Declared-criteria promotion assessment for Symthaea Energy Discovery Benchmark Zero.

This crate deliberately separates **measurement** from **promotion policy**. It does not define a universal threshold for scientific success and it does not convert benchmark performance into material, experimental, or deployment authority.

## What is compared

The gate executes and binds two matched evidence lines:

- the five-fold fixed-physics-baseline + blind-control suite;
- the five-fold crystal-ablated composition-RF benchmark.

Before any promotion criterion is evaluated, each learned fold must match the baseline/control fold on:

- fold identity;
- leakage-qualification SHA-256;
- target interval;
- top-k;
- ranked candidate count;
- truth candidate count;
- qualifying truth count.

A qualification mismatch is an integrity error, not a bad model score.

## No hidden aggregate rescue

The gate keeps every fold comparison and checks per-fold regression limits in addition to aggregate metrics. A favorable five-fold average therefore cannot hide a fold that violates the caller's declared maximum hit drop, MAE increase, or target-regret increase.

The five Matbench folds remain evaluation partitions of one dataset, not independent scientific replications.

## Explicit criteria only

`PromotionCriteria` intentionally has no `Default` implementation.

A caller must explicitly choose:

- minimum micro-precision delta vs the fixed baseline;
- minimum micro-recall delta;
- maximum candidate-weighted MAE delta;
- maximum macro target-regret delta;
- maximum per-fold top-k hit drop;
- maximum per-fold MAE increase;
- maximum per-fold target-regret increase;
- minimum folds where learned hits meet/exceed the blind-control mean;
- minimum folds where learned regret is at/below blind-control median.

Negative maximum MAE/regret deltas are allowed and mean an improvement is required.

The exact criteria receive a domain-separated SHA-256 and are bound into the promotion-plan identity.

## Outcome vocabulary

The gate returns only:

- `MeetsAllDeclaredCriteria`; or
- `ViolatesAtLeastOneDeclaredCriterion`.

It does **not** return labels such as `Validated`, `Discovered`, `ScientificallyValid`, `Certified`, or `ApprovedForDeployment`.

Meeting declared criteria means only that the exact benchmark evidence satisfies those declared thresholds.

## Chronology

The promotion plan can be emitted before benchmark execution and externally registered.

Its SHA-256 proves content identity, not chronology. If preregistration chronology matters, the caller must provide a separately immutable/timestamped registration evidence reference created before results are observed.

Providing that reference binds the string into the final receipt; this crate does not authenticate its timestamp.

## Criteria JSON

The CLI consumes a JSON object with all fields explicitly present, for example:

```json
{
  "min_micro_precision_delta": 0.0,
  "min_micro_recall_delta": 0.0,
  "max_weighted_mae_delta_ev": 0.0,
  "max_macro_regret_delta_ev": 0.0,
  "max_per_fold_top_k_hit_drop": 0,
  "max_per_fold_mae_increase_ev": 0.25,
  "max_per_fold_regret_increase_ev": 0.10,
  "min_folds_hits_ge_blind_mean": 4,
  "min_folds_regret_le_blind_median": 4
}
```

Those numbers are examples of file syntax, **not recommended scientific thresholds**.

## CLI

Create a content-addressed plan without reading Matbench truth:

`energy-benchmark-zero-promotion plan <target-min-eV> <target-max-eV> <top-k> <criteria.json>`

Execute later:

`energy-benchmark-zero-promotion run <matbench_expt_gap.json.gz> <target-min-eV> <target-max-eV> <top-k> <criteria.json> [registration-evidence-ref]`

The CLI performs no network fetch and omits host-local file paths from emitted evidence.

## Deliberate non-claims

This crate does not establish:

- scientific validation;
- statistical significance or independent replication;
- historical blindness of either model;
- calibrated uncertainty;
- material novelty or synthesizability;
- device performance;
- experimental validation;
- deployment fitness;
- material certification or physical authority.

The stacked branch remains unqualified until the inherited workspace lock is resolved by Cargo under the pinned toolchain and exact-head check/test/strict-Clippy execute successfully.
