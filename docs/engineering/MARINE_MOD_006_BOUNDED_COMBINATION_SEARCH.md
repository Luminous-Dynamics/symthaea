# MARINE-MOD-006 — Bounded interaction-aware modular-interior search

**Status:** research prototype / synthetic contract oracle  
**Implementation:** `modular_interior_combinations.rs`  
**Example:** `cargo run -p symthaea-marine-bench --example modular_interior_combinations`  
**Frozen oracle:** `fixtures/modular-interiors/combination-search-oracle-v1.json`

## The paradigm shift

Do not treat an interior as a list of independently acceptable products. Treat it as a versioned, connected system whose behavior can change when modules are combined.

The preceding work supplies four layers: typed configuration validation, impact-based revalidation, interface-preserving single-module alternatives, and evidence-gated Pareto ranking. This increment adds a fifth layer: **bounded composition and interaction diagnostics**. It asks what becomes false only after two or more individually screened changes are composed, then requires fresh validation of the complete configuration.

This is not an optimizer and does not yet move placements, route services, prove naval-architectural feasibility, or create trustworthy objective estimates. It composes drop-in replacements at the baseline locations. Layout placement search is a distinct next step.

## Contract

1. **Bound the search before executing it.** Inputs are capped at 128 variants, candidate materializations at 10,000, and combination cardinality at 8 module types. Callers select stricter limits as appropriate. Invalid limits and duplicate variant IDs fail closed.
2. **Enumerate deterministically.** Combinations are cardinality-first and sorted by module target, variant ID, and replacement digest. No combination contains two alternatives for the same target module.
3. **Require each alternative to pass single-change interface preflight.** The existing drop-in generator is reused. A port/interface contract change is not silently adapted.
4. **Revalidate every complete combination.** The full configuration validator and change-impact analyzer run on materialized combinations. A single-module receipt is never reused as the combined receipt.
5. **Report emergent findings conservatively.** An interaction finding is a code/subject finding absent from the baseline and every constituent single-change validation. It is a diagnostic difference, not proof of causal mechanism. A missing interaction finding is not proof that no interaction exists.
6. **Make truncation explicit.** If more combinations exist than the configured candidate budget, the report exposes `search_complete=false`, `truncated_by_candidate_cap=true`, and `candidate_count_lower_bound=budget+1`. The report hashes the baseline validation receipt inside the report body. The deterministic output prefix must never be described as exhaustive.
7. **Bind output to exact inputs.** The report digest includes the baseline validation receipt, input-variant manifest, limits, completeness flags, and every generated candidate result. The verifier reruns the search and compares digests; this is integrity/recomputation, not a digital signature or source-authenticity claim.
8. **Keep safety non-compensatory.** Rejected and indeterminate designs stay distinguishable. There is no weighted safety score and no evidence carry-forward. A `candidate_only` validator disposition still does not mean safe, compliant, certified, seaworthy, or authorized to operate.

## Current synthetic oracle

The frozen JSON oracle contains two controls:

- Two distinct module targets yield two singleton candidates and one pair, in cardinality-first order.
- A synthetic +2,000 W increase to the galley and workspace AC-input ratings is designed so that each single change remains below the fixture's assumed AC-bus capacity while the pair crosses it. The expected pair finding is `SERVICE_CAPACITY_EXCEEDED|AC-BUS`.

The electrical values and vessel fixture are synthetic assumptions; the test asserts software behavior, not physical truth. The Rust test is authored into the crate, but until it is executed by a Rust toolchain it is not an executed test result.

## Benchmark protocol and target ladder

All values below are **targets to measure**, not claimed measurements.

| Tier | Workload | Required measurements / gate |
|---|---|---|
| Oracle correctness | Exhaustively enumerable spaces up to 12 variants, cardinality 1–3 | 100% exact match for candidate-set membership, ordering rules, and completeness/truncation flags |
| Determinism | 100 input permutations of the same catalog | 100% byte-identical report digest for semantically identical variant sets |
| Hard-gate sensitivity | Seeded geometry overlap, over-capacity service, invalid interface, stale revision, unknown safety gates | 100% detection of each frozen injected violation; 0 seeded violations silently promoted to eligible |
| Interaction detection | Frozen single-vs-pair cases and mutation-generated interactions | Report precision/recall separately by rule family; start at 100% recall on the frozen oracle, then report false-positive rate on hand-reviewed negatives |
| Budget behavior | Caps of 100, 1,000 and 10,000 candidates | Never emit more than the cap; exact completeness flag; no unbounded memory growth |
| Throughput | 10, 25 and 50 target modules; 1–3 variants per target; fixed host/toolchain | Report wall time, candidates/second, peak memory, p50/p95 over at least 30 runs; pin host, build mode, and input digest |
| Evidence generalization | Three distinct vessel data sets, with source-verified dimensions and mass properties where available | Per-domain coverage for geometry, services, mass/CG and safety gates; unresolved domains remain indeterminate |

A provisional engineering performance goal is to evaluate 10,000 candidates within 60 seconds on a pinned development host. Revisit that target after the first measured baseline; it is not an SLA and should not motivate weaker validation. The correct optimization is faster exact rejection and sound incremental computation, not skipping gates.

### Mandatory negative controls

- Duplicate variant IDs and duplicate target selections.
- Candidate budget that cuts off a known non-empty suffix.
- Interface profile, medium, direction, criticality, rating-unit, revision, and evidence-status mutation.
- A pairwise service overload invisible in either constituent single test.
- Indeterminate fire/egress/structural/stability gates that remain indeterminate after ranking.
- Tampered receipt, altered candidate ordering, stale baseline digest, and mismatch between a recomputed report and the declared search scope.

## Research context

Ship-arrangement optimization already treats cabins, compartments, equipment, adjacency, evacuation flow, and multiple objectives as coupled design variables. See the 2025 review, [Review of ship arrangement design using optimization methods](https://academic.oup.com/jcde/article/12/1/100/7922870), and the earlier [multi-objective optimization of ship internal arrangement](https://research.tudelft.nl/en/publications/multi-objective-optimisation-as-an-early-design-tool-for-smart-sh-2/). Recent subdivision research also combines search heuristics with geometric, hydrostatic, stability, and strength constraints: [D3QN-assisted NSGA-II ship subdivision optimization (2026)](https://www.sciencedirect.com/science/article/pii/S2092678226000348). Those works motivate measuring coverage, convergence and constraint preservation, but their reported results are not transferable performance claims for this crate. A closely aligned [October 7, 2026 arXiv preprint on procedural 3D ship internal arrangements](https://arxiv.org/abs/2610.09585) explores 50,000 generated concepts and uses voxel occupancy plus topological graphs to reason about space and connectivity. Its central design lesson for our roadmap is to couple spatial occupancy, connectivity, mass distribution, trim and stability rather than optimize module catalog swaps in isolation. Treat its dataset and reported results as an external benchmark candidate, not evidence that this crate reproduces the method.

Ship accommodation changes must retain vessel-specific professional review. IMO SOLAS materials call out fire containment, accommodation separation, and protection of evacuation routes; applicable obligations depend on vessel type and operation. See [IMO SOLAS overview](https://www.imo.org/en/about/conventions/pages/international-convention-for-the-safety-of-life-at-sea-%28solas%29%2C-1974.aspx) and [IMO fire-protection guidance](https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx).

## Next paradigm leaps

1. **Joint placement + module + interface search.** Add explicit placement variables, connector routing, access/egress graphs and service-path length/capacity. Preserve the exact baseline and require full revalidation after any placement change.
2. **Uncertainty-aware comparison.** Carry measurement/model intervals and source quality into objective evidence. Do not let apparent point-estimate gains outrank unresolved or fragile options.
3. **Value-of-information search.** When candidates are blocked by missing data, identify the next measurement or supplier document most likely to distinguish viable choices; produce the rationale and expected reduction in decision uncertainty.
4. **Incremental recomputation with independent checking.** Cache only by complete content-addressed dependencies, report every reused artifact, and compare optimized results against a slower independent recomputation on the frozen corpus.
5. **Human preference transparency.** Keep a Pareto set and let the owner express priorities (quiet, space, repairability, accessibility, range, cost) explicitly. Preserve safety as a hard gate and record how sensitive choices are to stated preferences.

## Qualification boundary

The implementation currently covers drop-in module combinations at fixed locations. It does not prove actual vessel stability, structure, evacuation compliance, fire performance, manufacturability, hydrodynamics, or safe operation. The fixture's service, mass, and platform numbers are synthetic. Rust compilation, formatting, unit-test execution, example execution, and exact-head hosted checks are separate verification gates; queued, skipped, absent, or stale checks are not passes.
