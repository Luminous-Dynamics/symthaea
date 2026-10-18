# FDM cost-constrained spatial calibration benchmark v1

## Purpose

This benchmark extends equal-observation-budget sampling into **cost-constrained evidence acquisition**. The selector chooses proposed spatial measurement locations, and the evaluator asks how much spatial information and critical-region sensitivity that spend buys.

All results are synthetic. Cost units are abstract scenario parameters, not seconds, dollars, actual probe travel, or demonstrated savings. The module has no machine I/O, no G-code generation, and no live printer authority.

## Research basis

- ISO/ASTM 52902:2023 describes AM geometric-capability test artifacts and their quantitative/qualitative measurements, addresses measurement uncertainty, and does not impose one universal measurement method or machine-setting recipe: https://www.iso.org/standard/79683.html.
- ASTM's standard listing says the characterization applies to a particular AM setup and process parameters with a specific feedstock; performance needs to be reconsidered when those conditions change: https://store.astm.org/f3345-23.html.
- NIST's holistic data-fusion framework for metal powder-bed fusion combines measurements across builds with model/simulation information to support quality decisions; that metal PBF result motivates a multi-build evidence question, not direct FDM performance claims: https://www.nist.gov/publications/enhancing-part-quality-management-using-holistic-data-fusion-framework-metal-powder-bed.
- NIST's work on in-process observation selection provides related context for selecting process observations for quality management: https://www.nist.gov/publications/situ-observation-selection-quality-management-metal-additive-manufacturing.

## Frozen protocol

- Schema: `symthaea.fdm-cost-budget-synthetic.v1`.
- Evidence class: `synthetic`.
- Surface panel and selector logic reuse the preceding equal-budget synthetic sampler module.
- All arms start at the same five measurement locations in the same order.
- Candidate grid: normalized coordinates from 0.1 to 0.9 in 0.1 increments.
- Every selector produces a deterministic next-location sequence from prior observations and a declared cost schedule. A budget run takes a prefix of that sequence and stops before the first unaffordable measurement. This creates nested budget/quality curves for a given surface, selector, seed, and cost schedule family.
- Six fixed total budgets are evaluated: 17, 20, 23, 26, 29 and 32 abstract units.
- Under the demo schedule, the common five-point initial route costs approximately 16.125483 abstract units (2 setup + 5 acquisition + 7.725483 travel + 1.4 edge surcharge). Thus the 17-unit arm is intentionally a start-only boundary case; higher budgets allow additional actions. This is direct arithmetic on the declared fixture constants, not an executed Rust test or physical cost estimate.
- All surfaces and values remain synthetic; latent values and the evaluation grid are only consulted after the selector has proposed its next position.

## Explicit synthetic cost ledger

The demo schedule has these named components:

- **One-time setup:** 2.0 units, charged once.
- **Measurement acquisition:** 1.0 unit per observation.
- **Travel:** 2.0 units per normalized coordinate-distance unit, measured between consecutive sample locations (the first move starts at the normalized center).
- **Edge-zone surcharge:** 0.35 units for measurements within the declared 0.15 normalized boundary zone.

The edge surcharge is a synthetic proxy for a measurement's added difficulty, not an empirically validated printer/measurement cost. There is no modality change in v1, so changeover cost is zero by construction; instrument changes, specimen flips, fixture changeovers, repeat-measurement cost and actual dwell time belong in a later schedule version. Do not mix those omitted costs into a physical savings claim.

The cost schedule has its own domain-separated digest. Every observation records setup, acquisition, travel, edge surcharge, incremental total and cumulative cost. The report preserves the budget, spend, unused budget and individual components rather than hiding everything behind one weighted score.

## Strategies

1. `fixed_stratified`: use the next location in a frozen inner-grid plan.
2. `farthest_point`: choose the candidate farthest from existing observations.
3. `model_disagreement`: choose the greatest disagreement between nearest-neighbour and inverse-distance-weighted estimates.
4. `coverage_guarded_disagreement`: alternate space-filling exploration with model-disagreement exploitation.
5. `cost_normalized_disagreement`: choose the highest predictor disagreement per marginal synthetic measurement cost, with explicit spatial-distance and coordinate tie-breaks.
6. `seeded_random`: deterministic random-location baseline.

The cost-normalized rule is a simple, testable heuristic—not a claim of optimal expected value of information. A later version should compare selector decisions against an independent decision-impact oracle and test whether each observation can change a preregistered capability/release decision.

## Reported metrics

Each report includes:

- Declared budget, spent budget, unused budget and separate setup/acquisition/travel/edge cost totals.
- Observation count, additional observation count and normalized route distance.
- Mean and maximum distance from the dense evaluation grid to the nearest measured point.
- MAE and maximum absolute error for a nearest-observed-value estimate of the synthetic surface.
- Critical-region grid count and recall based on measured points crossing the frozen 0.10 mm threshold within a 0.08 normalized detection radius.
- False-alarm observation count, where measurement noise crosses the threshold although the latent synthetic value does not.
- Ordered observations, cost ledger, cost-schedule digest, and a domain-separated report digest.

The 0.10 mm critical threshold and 0.08 detection radius are fixture parameters, not universal dimensional tolerances. `normalized_route_distance` and all cost units are proxies, not physical time or money.

## Integrity and rejection checks

- A malformed, negative or non-finite cost schedule fails closed.
- If the shared initial design exceeds a budget, the arm fails closed instead of silently omitting initial measurements.
- No sample can make cumulative spend exceed the declared budget.
- Budget runs for one surface/strategy/seed have nested acquisition traces; greater budgets append observations instead of changing earlier decisions.
- Duplicate/out-of-range/non-lattice positions, altered initial design, broken order/identity, altered cost entries, invalid cumulative cost, inconsistent summary metrics and changed report digest are rejected.
- Same validated code, surface, seed, strategy, schedule and budget replay identically.

A SHA-256 digest binds the serialized identity; it is not a digital signature, a proof of trusted origin or an independent endorsement.

## Proposed next gates

1. Run workspace formatting, focused crate tests and clippy against the exact commit; skipped CI is not a pass.
2. Independently inspect whether the budget levels produce a useful spread of observation counts; revise the preregistered ladder if every strategy gets the same number at all budgets.
3. Add schedule families for realistic travel, dwell, repeat, setup change and higher-grade metrology, while keeping each component and its evidence provenance visible.
4. Report the Pareto frontier across worst-location error, critical-region recall, false-safe rate, uncertainty and cost; do not collapse the vectors into one scalar.
5. Repeat across seeds, candidate resolutions and surface families, then run a preregistered physical study with independently calibrated metrology, repeated builds, material lots, orientation/feature strata and complete failure/missingness lineage.

## Focused commands once an executable checkout is available

```sh
cargo fmt --check
cargo test -p symthaea-fabrication-kernel fdm_spatial_cost_budget
cargo clippy -p symthaea-fabrication-kernel --all-targets -- -D warnings
```

Record the exact SHA and command outputs. No performance, cost-saving, process-conformance or physical-capability claim follows from the synthetic benchmark alone.
