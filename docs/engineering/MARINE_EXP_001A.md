# MARINE-EXP-001A — deterministic experiment-selection crucible

Status: implementation started; Rust build/tests and hosted checks are not yet verified on this exact head.

Parent research issue: [MARINE-EXP-001 #7641](https://github.com/Luminous-Dynamics/symthaea/issues/7641)  
Interval-screening foundation: [PR #7629](https://github.com/Luminous-Dynamics/symthaea/pull/7629)  
Whole-system study: [MARINE-FREIGHT-001 #7444](https://github.com/Luminous-Dynamics/symthaea/issues/7444)

## Paradigm shift

Static screening answers whether today's evidence bounds establish a constraint. Experiment selection asks a more valuable follow-up question:

> Under a finite budget, which measurement or validation experiment should run next to resolve the most consequential uncertainty?

This is the first executable crucible for that idea. It does **not** claim calibrated expected information gain or Bayesian active learning. Until outcome distributions, bias, correlation, and coverage are empirically established, v1 uses explicit closed intervals and a synthetic measurement oracle.

## Implementation

- `src/experiment_selection.rs` — deterministic scenario generator, finite experiment catalog, four executable selection policies, exhaustive toy subset oracle, separate hard-constraint metrics, paired-seed summaries, and a SHA-256-addressed report receipt.
- `examples/experiment_selection_benchmark.rs` — runs the initial 1,000-scenario, 30-seed, 6-credit benchmark and prints a verifiable JSON receipt. Set `SYMTHAEA_SOURCE_REVISION` to the exact source-head SHA to bind that revision; otherwise the receipt says `unavailable`.
- Tests cover catalog validity, generated interval containment, target-only interval updates, robust constraint classification, budget ceilings, seeded reproducibility, oracle dominance in the toy world, receipt integrity/tamper detection, invalid seed inputs, and separate feasible/infeasible metrics.

## Synthetic decision model

The generator creates six hidden ground-truth parameters with initial closed ranges:

1. propulsion + auxiliary energy demand (kWh);
2. usable battery energy (kWh);
3. required water depth (m), representing draft plus declared clearance;
4. available water depth (m);
5. required battery-side charge power (kW);
6. available charging power (kW), limited by the synthetic terminal/grid envelope.

The evaluator independently classifies three constraints: battery-energy sufficiency, under-keel clearance, and charge-power sufficiency.

The finite catalog contains 12 experiment types representing evidence-gathering activities: telemetry/data review, instrumented propulsion trial, battery audit/capacity test, chart review/bathymetric survey, route-depth reference/survey, charge-schedule audit/instrumented cycle, and shore-power audit/load-bank test. Each has an explicit synthetic credit cost and retained-width factor.

When an experiment is selected, its simulator outcome narrows only the declared target interval and is centered on the hidden truth, so the truth remains inside. This is an intentionally idealized, optimistic toy oracle. It tests policy plumbing and decision logic, not realistic measurement noise, bias, correlation, or the value of a real test.

## Policies

- `cheapest_first` — deterministic lower-cost experiment selection.
- `round_robin` — rotates across parameter families, with the starting parameter rotated by paired seed. This avoids degenerating into the same order as cheapest-first because of catalog ordering.
- `seeded_random` — seeded baseline reproducible for paired comparisons.
- `constraint_criticality` — a transparent heuristic that prioritizes normalized interval width reduction per credit on currently indeterminate constraints.
- `exhaustive_toy_oracle` — enumerates every subset of the 12 catalog entries within the budget and selects the best resulting synthetic outcome by resolved hard constraints, normalized interval-width reduction, then lower cost. It sees hidden synthetic truth and is **not a deployable policy**; its bound is only for this finite toy outcome model.

The criticality policy is explicitly a heuristic, not expected information gain. The oracle is an idealized comparison, not evidence that any heuristic is optimal.

## Receipt and reproducibility

The report binds:
- benchmark and schema identity;
- experiment-catalog version;
- scenario-generator version;
- toy-oracle version;
- scenario count and ordered seed list;
- per-scenario budget;
- full experiment catalog;
- paired strategy summaries;
- scope limitations;
- hash of the generated scenario corpus;
- hash of the ordered policy/seed execution trace, including the experiment IDs selected for each scenario;
- SHA-256 of the serialized benchmark body.

The receipt structure is specified in `crates/domains/symthaea-marine-bench/schemas/marine_experiment_selection_receipt_v1.schema.json`. These digests are tamper-evident when recomputed, but SHA-256 is not a signature and does not authenticate who supplied data.

Receipt verification confirms the body digest matches. It does not authenticate measurement origins, validate scientific assumptions, or elevate synthetic evidence.

## Proposed benchmark gates

Run the frozen default suite: 1,000 scenarios × 30 paired seeds × 4 heuristic policies, compared with the exhaustive toy oracle. All policies receive the same scenario IDs, budget units, and experiment catalog.

Report separate:
- mean number of robustly resolved constraints (0–3);
- robust-feasible versus robust-infeasible counts (never merged into a single score);
- fraction of scenarios where all three questions are resolved;
- mean budget consumed;
- mean cost to full resolution among scenarios that reach it;
- normalized interval-width reduction;
- resolved-constraint regret against the clairvoyant toy oracle.

Research targets, not measured outcomes:
1. deterministic receipts reproduce byte-identically for the same inputs and tool generation;
2. no heuristic exceeds its declared per-scenario budget;
3. the exhaustive toy subset oracle resolves at least as many questions as any budget-matched heuristic on each generated scenario;
4. no policy converts overlapping intervals into robust feasibility or robust infeasibility;
5. rankings are compared over paired seeds and held-out synthetic scenarios, with differences and confidence intervals reported rather than only a winner label;
6. an experiment is not credited as useful merely because it shrinks an interval unrelated to a currently indeterminate hard constraint.

The synthetic simulator guarantees the hidden truth stays in nested intervals by construction. Therefore it cannot test real interval coverage or model-bias misses. Do not call those physical validation benchmarks.

## Research rationale and external anchors

Recent work in ship design optimization explores constraint-aware sample selection to reduce expensive surrogate-model evaluations: [Engineering Applications of Artificial Intelligence, 2026, DOI 10.1016/j.engappai.2025.112716](https://doi.org/10.1016/j.engappai.2025.112716). Maritime scenario-testing research also explores sampling compact, high-coverage encounter sets rather than indiscriminately testing large scenario collections: [Transportation Research Part C, DOI 10.1016/j.trc.2025.105246](https://doi.org/10.1016/j.trc.2025.105246). These papers motivate constraint-critical experiment selection, but do not validate this implementation.

Physical-model integration must use suitable external reference and validation procedures. The ITTC provides a [ship resistance/propulsion benchmark repository](https://www.ittc.info/benchmark-repository/), a [CFD validation database procedure for resistance and propulsion](https://ittc.info/search/?q=5415), and a [practical ship-CFD application guide](https://ittc.info/media/11958/75-03-02-03.pdf). The experiment-selection code does not execute a CFD solver.

## Next steps after the software crucible

1. Add frozen, content-addressed receipts for the exact synthetic case corpus and expected output.
2. Add hostile selectors and adversarial catalogs (unknown or zero cost, irrelevant interval reduction, stale provenance, unsupported instrument, test range that fails to enclose truth, and test attempts that should be rejected).
3. Measure paired-baseline improvements on frozen held-out synthetic scenarios; publish a benchmark table only from an executed exact head.
4. Introduce noisy and biased synthetic measurements with explicit truth-coverage metrics; ensure the oracle no longer assumes that every experiment encloses truth automatically.
5. Only then connect licensed, provenance-bound experimental records and validated solver cases. Report coverage, false-robust-feasibility rate, model-bias misses, and budget-to-resolution by operating envelope.
6. Keep physical engineering authority outside this crate. It cannot certify a hull, battery, route, terminal, docking operation, seaworthiness, or autonomous navigation.

## Explicit claim ceiling

A PASS can establish only that this exact experiment-selection benchmark reproduces the declared rules on its frozen synthetic scenarios and cost model. It cannot establish real-world expected information gain, calibrated uncertainty, faster ship-design cycles, improved vessel performance, safety, economic superiority, or seaworthiness.
