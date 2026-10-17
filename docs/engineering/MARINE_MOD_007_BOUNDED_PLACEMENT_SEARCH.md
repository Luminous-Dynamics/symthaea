# MARINE-MOD-007 — Bounded joint placement search

**Status:** research prototype / synthetic geometry oracle  
**Module:** crates/domains/symthaea-marine-bench/src/modular_interior_placement_search.rs  
**Example:** cargo run -p symthaea-marine-bench --example modular_interior_placement_search  
**Frozen oracle:** fixtures/modular-interiors/placement-search-oracle-v1.json

## Why this next step

MARINE-MOD-006 can combine module alternatives, but those replacements keep the baseline placement coordinates. Spatial layout is a separate decision variable. A module can be interface-compatible and still be badly placed; two movements that look acceptable in isolation can collide when composed.

This increment searches explicit, finite destination choices for several existing module instances. It uses the current whole-configuration validator as its geometry/systems screening oracle and runs the existing change-impact analyzer for each materialized candidate. It does not pretend that fixed-coordinate module swaps are full interior optimization.

## Contract and claim boundaries

- Inputs are explicit alternative placements for uniquely identified baseline instances. Each choice declares the deck, XYZ coordinate in millimetres, and Z rotation.
- Only the currently supported rotations 0, 90, 180 and 270 degrees can yield a geometry-valid candidate; unsupported rotations are allowed to reach the canonical validator so its failure is visible in the candidate receipt.
- Enumeration is cardinality-first; a candidate moves at most one choice per instance and never moves a given instance twice.
- Equivalent input-domain and choice permutations produce the same normalized input manifest, candidate order and report digest.
- Search limits fail closed: at most 50 placement domains, 32 choices per domain, 256 total choices, 10,000 generated candidates, and 4 simultaneously moved instances. Callers may specify tighter limits.
- If the cap is reached while at least one further candidate is found, the report includes search_complete=false, truncated_by_candidate_cap=true, and a candidate-count lower bound of budget + 1. The lower bound is not a total-count estimate.
- Every candidate gets a full configuration validation and impact analysis. Validation results are not transferred from single-move candidates to pair/multi-move candidates.
- An interaction finding is present in the combined receipt but absent from the baseline and every constituent single-move receipt. It is a diagnostic difference, not causal proof.
- The receipt binds the baseline validation receipt, normalized input manifest, limits, completeness data, and every candidate outcome. Verification repeats the bounded search. This is integrity/reproducibility, not a digital signature or proof that external evidence is authentic.
- The validator may report geometry, service-capacity and declared-constraint issues. This increment does not compute service routes, evacuation performance, hydrostatics, intact/damage stability, structural qualification, rule applicability, SOLAS compliance, seaworthiness or operational approval.

## Frozen synthetic interaction oracle

The tiny oracle models a 100 mm baseline gap between a sleeping pod and wet-room module:

- Move CABIN-01 from X=400 mm to X=600 mm. Its physical X envelope ends at X=3,000 mm while the wet room remains at X=3,100 mm; the two physical envelopes do not overlap.
- Move WETROOM-01 from X=3,100 mm to X=2,900 mm. Its physical X envelope begins at X=2,900 mm while the cabin remains ending at X=2,800 mm; those envelopes do not overlap.
- Compose both moves and their physical envelopes intersect by 100 mm. The expected new finding is MODULE_PHYSICAL_OVERLAP|CABIN-01 / WETROOM-01.

The frozen test expects exactly three candidates: each singleton and their pair. This oracle checks a known deterministic synthetic mutation; the test has been authored but not executed by Rust in this environment. The fixture is not a physical vessel model.

## Benchmark targets (not measured results)

| Benchmark | Proposed gate |
|---|---|
| Finite-space enumeration | 100% exact candidate-set and completeness agreement for exhaustive spaces up to 4 instances × 3 choices per instance and cardinality up to 3 |
| Determinism | 100% byte-identical report digests over 100 permutations of equivalent domain/choice lists |
| Geometry mutation sensitivity | 100% detection of the frozen overlap, out-of-bounds, invalid-rotation and reserved-clearance negative controls |
| Interaction sensitivity | 100% recall on the frozen oracle; then target ≥95% recall and >95% precision on a separately hand-reviewed synthetic corpus. Report per-rule counts and confidence bounds rather than only a single percentage |
| Candidate-budget integrity | 0 cap overruns; exact completeness flag; truthful lower bound whenever truncation is detected |
| Scaling | Measure 100, 1,000 and 10,000 candidates with 1, 2, 3 and 4 moved instances on a pinned host/toolchain |
| Runtime/memory | Report p50/p95 wall time, candidates/second and peak memory over ≥30 runs. Provisional goal: 10,000 small candidates under 60 seconds without weakening validator coverage |
| Generalization | Evaluate at least three independently sourced vessel/layout corpora before any real-world layout claim; report missing data, validator finding families and professional-review outcomes separately |

The proposed performance goal is not an observed result, an SLA or an engineering guarantee. Performance improvement should begin with profiling. Content-addressed incremental reuse is only acceptable when its full dependency identity is explicit and an independent full-recompute mode catches stale-cache defects.

## Research context

1. The [2025 review of ship arrangement design optimization](https://academic.oup.com/jcde/article/12/1/100/7922870) catalogs design variables and constraints across compartments, cabins, corridors and equipment placement. This supports explicit search domains and a constraint-oriented evaluation protocol.
2. A 2026 [case-based engine-room arrangement and passageway study](https://www.sciencedirect.com/science/article/pii/S2092678226000269) combines graph similarity, multi-objective optimization and automatic passageway candidates, emphasizing accessibility and connectivity beyond non-overlapping boxes.
3. A 2025 paper on [cruise-ship space allocation for evacuation safety](https://www.sciencedirect.com/science/article/pii/S0951832025008166) integrates evacuation performance into early space allocation rather than postponing it until after general arrangement. It reports case-study improvements; those figures are context, not targets transferable to this code.
4. The [October 2026 procedural 3D ship arrangement preprint](https://arxiv.org/abs/2610.09585) uses voxel occupancy and topological graphs for spatial exploration across vessel types. It motivates adding spatial-field and connectivity representations to the roadmap, and using an external arrangement corpus rather than just synthetic JSON fixtures.
5. The IMO's [SOLAS overview](https://www.imo.org/en/about/conventions/pages/international-convention-for-the-safety-of-life-at-sea-%28solas%29%2C-1974.aspx) and [fire-protection guidance](https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx) identify applicable ship fire-protection and means-of-escape requirements. Which requirements apply depends on vessel type and service. A generic geometry oracle is not an alternative to regulatory and professional review.

## Paradigm shifts still ahead

### 1. Placement × module × connectivity co-design

Combine the MARINE-MOD-006 module-choice domains with placement domains, then route electrical, fresh/grey/black water, exhaust, data, HVAC and structural services through explicit feasible graphs. The routing representation must include endpoints, obstacles, bends/lengths, access for maintenance, capacity and route evidence. Do not mark a connection "routed" merely because its endpoint types match.

### 2. Access and evacuation as graph constraints

Turn accessibility and escape into explicit topology / flow models with alternate routes, bottleneck and accessibility metrics. When a qualified evacuation model is available, integrate it before finalizing spatial allocation. Hard regulatory constraints stay outside any weighted objective.

### 3. Robust layout frontiers

Carry uncertainty intervals for mass, centres of gravity, route length, load, clearance and cost. Return robustly feasible, rejected and unresolved candidates distinctly. Do not pretend an assumption-driven frontier is a qualified design recommendation.

### 4. Search for information, not only for designs

For blocked candidates, report which measurement, supplier document, route survey or analysis would best discriminate between the remaining options. Prioritize evidence collection based on expected impact on the decision, while keeping the underlying uncertainty visible.

### 5. Independent verification

Retain a simple exhaustive search path for small spaces and an independent full-validation mode. For any accelerated search, benchmark against exhaustive oracles and use mutation tests that plant overlap, out-of-bounds, invalid rotation, blocked clearance, service overload and false safety claims.

## Verification boundary

Rust build, rustfmt, unit-test execution, example execution, and exact-head hosted checks remain separate qualification gates. Queued, skipped, absent, or stale checks are not passing. This module expands explicit placement search and deterministic screening only; it is not a naval architecture design tool that can certify a vessel.
