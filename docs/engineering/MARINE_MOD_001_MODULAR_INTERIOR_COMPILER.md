# MARINE-MOD-001 — Constraint-carrying modular interiors

## Design thesis

Model the interior as a versioned product platform, not a one-off arrangement of furniture. A vessel has a declared deck coordinate frame and a set of structural and utility service interfaces. A configuration selects versioned modules, places instances in that frame, connects module ports to platform services, and records reserved regions and safety-evidence dispositions.

The configuration manifest is an input to an eventual deterministic validator/compiler. JSON Schema is only the data contract: it cannot establish geometric fit, connector compatibility, loads, egress adequacy, stability, or regulatory compliance.

## What can be customized

The initial catalogue separates function from vessel archetype. Concepts include sleeping pods, wet rooms, compact galleys, convertible lounges/workspaces, storage, medical nooks, and later laboratory, workshop, laundry, crew and accessibility modules. The two fixtures intentionally share a catalogue but use different placements to demonstrate mission-specific configuration rather than two unrelated redraws.

Interfaces are typed and revisioned: electrical power, fresh water, greywater, blackwater, supply/exhaust air, data networking and structural attachment are represented separately. The current LD-INTERFACE-V1-PROVISIONAL profile is a placeholder, not a claimed industry standard. Real work must establish connector dimensions, ratings, isolation, service shutoff, inspection access and supplier compatibility.

## Non-negotiable design constraints

- Unknown mass or CG remains unknown. A design-assumption value is not measured evidence and cannot close a ship-level weight ledger.
- The validator must derive module dimensions in the declared coordinate frame, account for rotation and service-clearance envelopes, and reject out-of-bounds or overlapping placements.
- A connection is not valid merely because its media names match. It must verify direction, connector profile, capacity and required protection/isolation, including cumulative platform-service loads.
- Escape routes, accessible paths, fire and watertight boundaries, emergency equipment, and maintenance clearances are hard constraints. They cannot be traded for cost, comfort or payload in a weighted score.
- A module swap changes mass, CG, load paths, utility demand, noise/thermal performance and the configuration identity. Prior evidence is reused only where its applicability is demonstrated; otherwise it is stale or indeterminate.
- Configurations should be immutable, versioned manifests. A digest should bind the module revisions, placements, interface connections, reserved regions, assumptions and validator version.

## Proposed staged compiler

1. **Contract layer:** schema validation, stable IDs, explicit unit declarations, provenance and unknown-state semantics.
2. **Geometric layer:** 2D/3D envelope, deck boundary, module overlap, service-clearance and reserved-region checks; begin with axis-aligned boxes and then support polygon/mesh geometry.
3. **Interface layer:** port direction/media/profile checks, rated capacity, utility load aggregation, connection routing, isolation and maintainability.
4. **Weight-and-CG layer:** configuration-specific mass ledger; compute a CG only where all required masses and local CGs are available and coordinate transforms are verified. Return indeterminate instead of guessing.
5. **Systems and safety layer:** egress/accessibility, fire and watertight boundaries, structural attachment, stability, sanitary and ventilation review. Computational flags support—not replace—qualified engineering and authority/class review.
6. **Configuration search:** generate alternate valid interiors, then expose Pareto trade-offs among mission utility, payload, serviceability, lifecycle cost, energy and comfort only after hard-gate checks.

## Benchmark plan — targets, not results

Freeze a synthetic test corpus containing valid layouts plus adversarial cases. Preserve each case, expected result, test revision and source hash.

| Benchmark | Proposed target | Measurement rule |
|---|---|---|
| Manifest determinism | 100% reproducibility | Identical inputs and validator version produce identical canonical manifest/digest |
| Data completeness | 100% of placed modules have explicit mass status, ports and provenance | Unknown is allowed only when explicit; never silently impute |
| Bounds and collisions | 100% detection on the frozen negative suite | Include rotations, boundary overrun, module overlap and service-clearance overlap |
| Interface correctness | 100% detection of planted type/direction/profile/capacity errors | Include cumulative overload and missing source/sink services |
| Safety negative controls | Zero false-safe accepts on the frozen adversarial corpus | A single missed escape, access, fire-boundary or emergency-equipment violation fails the gate |
| Weight/CG | 100% agreement with hand-calculated synthetic cases within declared numeric tolerance | Missing inputs must produce indeterminate; compare transformed local CGs and totals |
| Reconfiguration productivity | Measure time and edit count versus a documented manual-redraw baseline | Use identical tasks and score final constraint-valid configurations, not just time to produce a drawing |
| Reuse across archetypes | Report reuse ratio and adapter count | Count reused module revisions separately from bespoke adapters and unresolved incompatibilities |
| Robust optimization | Report Pareto coverage and constraint violations | Compare fixed-grid/random/search baselines on frozen scenarios; never allow hard constraints to be compensated by weighted scores |
| Lifecycle maintainability | Measure replacement steps, isolation time and affected evidence | Simulate component swaps and require an explicit revalidation impact report |

A numerical claim such as “10× faster design” or a target mass tolerance should be set only after a reproducible baseline and the benchmark's system boundaries are defined. The initial target is correctness, traceability and fast iteration without weakening constraints.

## Research context

- Fong, *Practical Applications in CADD for 3-D Machinery Space Design and Modular Outfitting*, TRID: https://trid.trb.org/View/403592. Historical prior art discusses 3-D module generation, service routing, clash checking, weight control and centre of gravity.
- Erikstad, *Design for Modularity: Volume 1 — Optimisation of Ship Design and Operation for Life Cycle* (2019): https://doi.org/10.1007/978-3-030-02810-7_10. Frames modularity as a lifecycle decision and product-platform strategy.
- Strickland, *An evaluation of system modularity and interface standards as a means for continued platform level relevance* (IMDC 2024): https://doi.org/10.59490/imdc.2024.831.
- IMO passenger-ship context: https://www.imo.org/en/ourwork/safety/pages/passengerships.aspx.
- IMO fire protection and escape-route context: https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx.

## Current status and claim ceiling

This change establishes a schema, two synthetic examples and a benchmark protocol. The fixture masses, centres of gravity, module dimensions and utility capacities are invented placeholders. No Rust layout validator, CAD integration, load aggregation, stability result, fire/accessibility assessment or certification is delivered by this schema-only increment. Those are the next implementation gates.


## Executable validator semantics

The Rust module in this implementation slice checks the typed manifest for physical envelope and reserved-region conflicts, blocked service-clearance envelopes, module/revision references, port connection direction/media/profile consistency, declared service capacity, and mass/CG availability. Its receipt digest binds a deterministic serde serialization of the typed input and a separate serialized receipt body. This is an integrity/checking receipt, not a cryptographic signature or source-authenticity claim. The sample capacities and mass properties remain synthetic assumptions; they intentionally keep the overall disposition indeterminate.

## Local replay

From the repository root, after the Rust toolchain is available:

```sh
cargo test -p symthaea-marine-bench modular_interior
cargo run -p symthaea-marine-bench --example modular_interior_validator
```

The example prints the complete receipt for both fixture configurations and exits non-zero if the bookkeeping checks reject either configuration. The expected fixture outcome is still `indeterminate`, since service/weight values are synthetic assumptions, physical routing is unknown, connector profiles are provisional, and safety gates are not assessed. A successful command would demonstrate only that the Rust validator ran on these manifests—not vessel feasibility or approval. These commands have not been executed in the current editing environment.


## MARINE-MOD-003 — configuration-change impact

The companion `modular_interior_impact` module compares exact typed configuration baselines and reports per-artifact changes plus the analysis domains requiring revalidation. A module swap, placement move, service/connection change, reserved-region edit or safety-evidence change cannot silently inherit all evidence from the prior layout.

The analyzer computes canonical JSON digests that ignore ordering differences in arrays while keeping changes to actual item content visible. It binds the from/to configuration digests and report body into a SHA-256 receipt; this is a reproducibility/integrity mechanism, not a digital signature or evidence of source authenticity. Ambiguous duplicate IDs are explicitly marked, never collapsed silently. The dependency map is deliberately conservative and versioned; qualified reviewers must still determine regulatory applicability and the precise impact for each vessel.

From the repository root, when the Rust toolchain is available:

```sh
cargo test -p symthaea-marine-bench modular_interior_impact
cargo run -p symthaea-marine-bench --example modular_interior_impact
```

The benchmark plan should compare analyzer output with a frozen, manually authored change-impact oracle matrix. Proposed target: zero missed revalidation domains in that matrix, 100% deterministic reports, and zero automatic evidence carry-forward when the target configuration differs. These are targets, not achieved measurements. The example fixtures use synthetic mass, CG and service assumptions, so the change report is not a feasibility or safety assessment.


### Frozen change-impact oracle

`fixtures/modular-interiors/change-impact-oracle-v1.json` defines 10 synthetic comparison cases; `modular_interior_impact.rs` consumes it as a table-driven Rust test. The cases cover a no-op, ordering-only changes, moved placements, changed module mass, service/connection edits, reserved-region changes, platform envelope change, provenance/assumption/safety evidence changes, configuration identity/intent changes, and duplicate placement identity.

The oracle asserts required change sections and revalidation domains rather than comparing a brittle full JSON blob. That lets the dependency map evolve with an explicit versioned review while still surfacing missing impact categories. The intended benchmark is zero missed required domains on this frozen oracle, 100% deterministic digests, and zero silent duplicate-ID collapse. At present these assertions are authored—not executed locally.

Additional research:
- Inselmann, Deutschmann & Krause (2025), *Enhancing end-of-life sustainability through modularity and interface design in product development*, studies a Module Interface Graph for separability, reuse and end-of-life outcomes: https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/enhancing-endoflife-sustainability-through-modularity-and-interface-design-in-product-development/15AB7374508B4B9CFD6978127BE03372
- *Concepts of change propagation analysis in engineering design* (published online 2022; journal issue 2023) reviews the core concepts and approaches in engineering change-impact analysis: https://link.springer.com/article/10.1007/s00163-022-00395-y
- Inselmann, Wehrend & Krause (2025), lifecycle-oriented interface standardization across product-family phases: https://doi.org/10.35199/dfx2025.24
- IMO passenger-ship safety and fire protection are hard constraints for applicable vessels; change analysis must defer to the actual vessel's regulatory basis: https://www.imo.org/en/ourwork/safety/pages/passengerships.aspx and https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx

  
## MARINE-MOD-004 — evidence-gated Pareto ranking

The companion `modular_interior_pareto` module implements strict separation between constraint checking and design preference. It accepts candidate configurations only with a validation receipt that verifies against the exact typed configuration. A rejected candidate never ranks; an indeterminate candidate stays review-required and cannot enter the frontier. The objective layer likewise excludes unknown, assumption-only, non-finite or evidence-less values. Objective groups require exact matches for metric ID, unit, direction and comparison basis; there are no implicit unit conversions.

The implementation uses Pareto dominance rather than a weighted scalar score. One design cannot compensate for a violation of a hard gate by being lighter, cheaper or more comfortable. The report identifies eligible candidates, frontier candidates, dominated candidates and the named dominator, then binds a digest over the normalized input manifest and report. This is only as strong as the upstream validation and evidence classification; it is not a vessel approval.

The example intentionally feeds assumption-only mass values from the synthetic fixtures. Expected behavior: zero candidates in the evidence-backed frontier until actual configuration checks and objective evidence meet the declared threshold. That is a fail-closed demonstration, not a failed optimization.

Run when Rust is available:
  
```sh
cargo test -p symthaea-marine-bench modular_interior_pareto
cargo run -p symthaea-marine-bench --example modular_interior_pareto
```

### Benchmark plan (targets, not results)

- 100% agreement with hand-calculated Pareto cases covering minimize/maximize objectives, ties, dominance and trade-offs.
- 100% exclusion of candidates with invalid configuration receipts, hard errors, indeterminate safety/engineering findings, unknown objectives, assumption-only objectives or missing evidence references.
- Zero cross-comparisons across differing units, directions or comparison bases.
- 100% repeatable report digests after candidate input reordering.
- Zero safety false negatives; every candidate with unresolved hard evidence remains outside the frontier.
- Report frontier size and review-required fraction together; a zero frontier can be the correct result when no candidate has sufficient evidence.

### Research

Configure-to-order modular ship-platform optimization is established prior art: it models multiple mission configurations and lifecycle costs rather than selecting one hull/layout in isolation (https://www.sciencedirect.com/science/article/abs/pii/S0029801818309065). The 2024/2025 review of ship arrangement optimization describes cabin, compartment and equipment-placement variables and constraints (https://academic.oup.com/jcde/article/12/1/100/7922870). Digital ship-design surveys cover parametric optimization, simulation-driven design and product-lifecycle workflows (https://doi.org/10.59490/imdc.2024.784). Our differentiator is evidence-gated comparison with explicit incompleteness and independently verified configuration receipts, not Pareto optimization itself.

Safety basis remains vessel- and mission-specific: IMO explains that passenger ships must meet applicable SOLAS/Load Lines requirements and highlights escape and fire protection as key design concerns (https://www.imo.org/en/ourwork/safety/pages/passengerships.aspx; https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx).


## MARINE-MOD-004 — evidence-gated Pareto ranking

The companion `modular_interior_pareto` module separates constraint checks, evidence eligibility, and multi-objective preference. A candidate must have a validation receipt that verifies against the exact typed configuration and a receipt disposition of `candidate_only`. Rejected or indeterminate configurations never enter the Pareto frontier. The objective layer excludes unknown, assumption-only, non-finite or evidence-less values and requires objective evidence references to resolve to non-synthetic provenance declared by the configuration or its placed modules.

Each comparison group requires exact matches for metric IDs, unit strings, minimize/maximize direction and comparison basis. There are no implicit unit conversions and no weighted scalar score that can compensate for a hard-gate failure. The report lists eligible candidates, frontier members, dominated candidates and their dominators, with a digest binding the normalized input manifest and report. This remains a ranking prototype, not a vessel approval; upstream source classification and validation quality are still material limitations.

The example intentionally uses design-assumption mass values from synthetic fixtures. Expected behavior: zero candidates in the evidence-backed frontier until the physical configuration checks and objective evidence meet the threshold. A zero frontier is the correct result when no design meets the evidence gate.

Run when Rust is available:

```sh
cargo test -p symthaea-marine-bench modular_interior_pareto
cargo run -p symthaea-marine-bench --example modular_interior_pareto
```

### Benchmarks (targets, not results)

- 100% agreement with a frozen hand-calculated Pareto oracle for minimizing/maximizing objectives, ties, dominated candidates and trade-offs.
- 100% exclusion of invalid receipts, hard errors, indeterminate safety/engineering findings, unknown objectives, assumption-only objectives and unresolved/synthetic objective references.
- Zero comparisons across different units, directions or comparison bases.
- 100% repeatable report digests when candidate input order changes.
- Zero safety false negatives: any unresolved hard evidence prevents entry to the evidence-backed frontier.
- Report frontier size and review-required fraction together; do not maximize the number of “winners” at the expense of evidence quality.

### Research basis

Configure-to-order modular ship-platform optimization is established prior art: models evaluate multiple mission configurations and lifecycle costs instead of selecting one hull/layout in isolation (https://www.sciencedirect.com/science/article/abs/pii/S0029801818309065). A 2024/2025 review of ship arrangement optimization discusses cabin, compartment and equipment-placement variables and constraints (https://academic.oup.com/jcde/article/12/1/100/7922870). The IMDC digital ship-design review covers parametric optimization, simulation-driven design and product-lifecycle workflows (https://doi.org/10.59490/imdc.2024.784). The differentiator here is evidence-gated comparison, explicit incompleteness and a configuration-bound validation receipt—not Pareto optimization itself.

IMO notes that passenger ships must meet applicable SOLAS and Load Lines requirements and emphasizes fire protection and escape routes as essential design concerns (https://www.imo.org/en/ourwork/safety/pages/passengerships.aspx; https://www.imo.org/en/ourwork/safety/pages/fireprotection-default.aspx). Actual regulatory applicability remains vessel-specific.
