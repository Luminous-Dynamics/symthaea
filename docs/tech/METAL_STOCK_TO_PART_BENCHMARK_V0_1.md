# Metal Stock-to-Part and Secondary-Feedstock Benchmark v0.1

**Owner:** Symthaea  
**Deterministic execution / workcell consumer:** Symtropy  
**Provenance and transformation-event consumer:** Mycelix  
**Regional feedstock/facility context consumer:** Sol-Atlas  
**Evidence class:** synthetic fixtures only

## Research grounding

- NIST's [Additive Manufacturing Benchmark Test Series](https://www.nist.gov/ambench) uses controlled measurement sets and challenge problems to compare model predictions with measured results. Its 2025 cycle included alloy chemistry, microstructure, tensile and fatigue challenges; planning now points to higher-technology-readiness components in a later cycle. This is a model for open challenge design, not proof that this project qualifies metal parts.
- NIST's [Fundamental Measurements for Metal Additive Manufacturing](https://www.nist.gov/programs-projects/fundamental-measurements-metal-additive-manufacturing) emphasizes pre-process feedstock characterization, in-process monitoring, post-process metrology, unambiguous data representation and explicit measurement uncertainty.
- NIST's [Additive Manufacturing of Metals](https://www.nist.gov/programs-projects/additive-manufacturing-metals) frames the scientific path as process → structure/microstructure → property → performance and ties predictive models to curated measurement data.
- Reuse the existing Symthaea owners: [METAL-001 #7981](https://github.com/Luminous-Dynamics/symthaea/issues/7981), [MAT-PSPP-001 #6224](https://github.com/Luminous-Dynamics/symthaea/issues/6224), [ENG-MAT-001 #5074](https://github.com/Luminous-Dynamics/symthaea/issues/5074), [manufacturing process semantics #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686), and [MFG-VALUE #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608). Do not add a second material ontology, scheduler, custody ledger or lifecycle-closure engine.

No NIST measurements or third-party material datasets are embedded in this corpus. External data admission requires source-, version-, license-, attribution- and applicability-specific review.

## Paradigm shift

Do not optimize parts/hour or nominal material yield in isolation. Optimize **quality-conforming, serviceable output over lifecycle resource demand**, subject to hard admissibility conditions.

The evaluator treats these as different facts:

```text
alloy label != resolved material/lot identity
declared grade != measured/qualified stock evidence
planned operation != executed process history
feedstock present != qualified secondary feedstock
simulated dimension != physical inspection
mass counted as scrap != mass available for the same recovery route
one conforming part != capable repeatable production
```

A weighted score cannot compensate for a failed material-identity, process-capability, inspection or conformity gate.

## v0.1 contract

- **Stock:** alloy identifier, lot, material state, identity resolution, feedstock class, contamination state, grade evidence and source record.
- **Qualified secondary feedstock:** custody reference, explicit clear contamination state and requalification reference are required. Mixed or unknown scrap cannot silently satisfy a strict alloy route.
- **Process route:** separate planned and executed operations; bind the input lot and input state to the stock; require process-run evidence; check declared machine capabilities; carry the output material state.
- **Part requirement:** exact target alloy, material state and requirement reference.
- **Mass ledger:** reconcile input mass against finished part, clean offcuts, chips, mixed scrap and residue under the declared tolerance. Keep stream classes separate; claimed clean-offcut recovery cannot exceed the clean-offcut stream.
- **Inspection:** physical measurement source, identified method, measurement record and current calibration state. Simulation cannot replace inspection; out-of-tolerance dimensions fail.
- **Repair/remanufacture:** require an explicit repair-scope reference and fresh final inspection.
- **Receipt:** bind exact corpus bytes (SHA-256) and a 40-character source commit. Unknown JSON fields, malformed scenario identity and duplicate IDs fail closed.

The scope is a synthetic fabrication/machine-shop cell. It does not model an integrated steelworks or detailed process recipes. Advanced casting, forging, joining, heat-treatment physics, additive manufacturing and destructive/non-destructive test models remain separate profiles until their process and measurement contracts are justified.

## Frozen known-answer corpus

The 20 scenarios contain two positive controls and eighteen negative controls:

1. Exact virgin stock → conforming machined part.
2. Mixed scrap offered for exact-grade use.
3. Process/stock lot mismatch.
4. Required heat treatment absent from executed operations.
5. Simulation offered as physical inspection.
6. Stale metrology calibration.
7. Material mass creation.
8. Chips credited as clean offcuts.
9. Required machine capability unavailable.
10. Out-of-tolerance measured part.
11. Custody-traceable, uncontaminated and requalified secondary feedstock.
12. Secondary-feedstock contamination unknown.
13. Missing stock source reference.
14. Missing grade evidence.
15. Final material-state mismatch.
16. Missing process-run evidence.
17. Repair/remanufacture with declared scope and current final inspection.
18. Repair/remanufacture without a declared scope.
19. Missing inspection record.
20. Unknown feedstock class.

A control matches only when both disposition and the complete reason-code set match. An optimizer timeout or an unsupported process route must be handled as blocked/indeterminate by any future planner, never as a successful route.

## Benchmark scorecard

### Gate A — contract correctness

- 100% exact match on the frozen known-answer corpus at the exact source commit.
- 100% of strict-grade identity, process-history, state, capability, calibration, tolerance, mass-balance and scrap-class negative controls blocked with the expected reasons.
- Identical corpus bytes + source commit → byte-stable semantic receipt.
- Independent verifier reconstructs the flow ledger and reason-code dispositions without calling the production evaluator.

### Gate B — decision quality (future evaluator/planner tranche)

Freeze input manifests and a held-out scenario split before tuning. Compare at least:

1. fixed conventional route;
2. material-yield-first routing;
3. repair/remanufacture-first routing where admissible;
4. evidence-aware multi-objective routing with hard quality/capability constraints.

Report separate measures, not one opaque readiness score: conforming parts per hour; first-pass yield and rework; usable-material yield by stream; material and energy per conforming part; tool/consumable cost; cycle/setup/changeover time; inspection coverage and uncertainty; qualified-secondary fraction; imported/opaque dependencies; cost per conforming part; Pareto set; route feasibility; runtime and optimality gap.

For small synthetic instances, compare against exhaustive enumeration as an independent decision oracle: target 100% constraint-feasibility agreement and an optimality gap of 0% when the objective and solution space are fully enumerated. For larger cases, publish feasibility agreement, nondominated alternatives, bounds/gap and runtime rather than claiming global optimality. Those are proposed gates—not measured results.

### Gate C — public-data evaluation

Only after exact data-use terms and provenance are reviewed, evaluate suitable published metrology/model-validation datasets. Separate data used for calibration from held-out assessment. Report errors by quantity and applicable process condition, plus uncertainty/coverage; do not treat a simulation-only match as physical validation. NIST AM Bench provides a strong pattern for measured challenge problems, but v0.1 does not import or reproduce its data.

### Gate D — physical pilot

Require a separately authorized and safety-reviewed facility, exact machine/tool/workholding configuration, qualified procedures, calibrated inspection equipment, source/lot identity, observed energy and material flows, and independent remeasurement. Synthetic or simulated improvements do not count as observed industrial savings.

## Component split

- **Symthaea:** evidence-aware material/process applicability, strict contract validation, baseline/objective comparisons, uncertainty and reports.
- **Symtropy:** deterministic geometry/workcell rehearsal and bounded route execution comparison; synthetic trajectories remain synthetic.
- **Mycelix:** existing stock-lot, custody, work-order, transformation and correction events; an event is not engineering conformity.
- **Sol-Atlas:** read-only regional feedstock, facility, transport and power context with coverage and uncertainty; a regional commodity map is not assay or available inventory.
- **Independent verifier:** reconstruct material flow and expected route disposition from exact inputs, with negative controls and manifest-bound receipts.

## Claim ceiling

A validated v0.1 receipt establishes the declared synthetic software semantics only. It does not qualify any real material, machine, process, article, secondary feedstock, structural use, production capability, cost, energy, emissions or circularity outcome.
