# Road and Rail Engineering Research Plan

**Status:** Early research / data-contract foundation. Not a design method, validated prediction model, product qualification, or authority to specify, build, inspect, or operate transport infrastructure.

## Objective

Build an open, auditable engineering workflow that discovers and evaluates lower-cost, longer-lived, lower-impact road and railway systems. Optimize the *whole asset and its life cycle* rather than ranking material samples by a single predicted property.

The initial stack:

- **Symthaea:** research orchestration, candidate generation, evidence-aware ranking, and materials/property representations.
- **Symtropy:** deterministic system simulation and experiment orchestration. Its current rigid-body engine is not, by itself, a validated pavement finite-element, soil, fracture, or wheel-rail contact solver.
- **Sol-Atlas:** terrain, climate, drainage, spatial constraints, resource access, network criticality, and candidate corridor context.
- **Mycelix:** source provenance, experiment lineage, material-batch evidence, review records, procurement/commons processes, and repair history. Governance votes must not substitute for engineering verification.
- **Licensed civil/railway engineers and accredited laboratories:** own requirements interpretation, test selection, design approval, and field acceptance.

## Important boundary in the current materials model

The existing `symthaea-materials::MaterialAgingModel` produces a hypervector state-similarity score and currently assigns that score to `remaining_strength`. State similarity can be useful representation telemetry; it is not a validated physical measurement of retained tensile strength, fatigue life, fracture resistance, rut resistance, rail wear, or service life.

Therefore:

1. Do not use the existing `remaining_strength` output as a road or rail safety factor, remaining-life estimate, material acceptance criterion, or engineering recommendation.
2. Preserve the HDC/CfC work as a candidate representation and temporal-modeling substrate, while introducing calibrated, property-specific predictors that are evaluated against measured outcomes.
3. Every result must name its target quantity, units, conditioning assumptions, calibration population, uncertainty, and qualification state.
4. If target-specific evidence is missing or outside the training/calibration domain, return **insufficient evidence** rather than a numeric prediction presented as qualified.

For living/self-healing concrete, apply the **claim × property × exposure** evidence grade (E0–E4) and cross-study protocol-comparability rules in [LIVING_CONCRETE_AND_RAIL_SYSTEMS_RESEARCH.md](LIVING_CONCRETE_AND_RAIL_SYSTEMS_RESEARCH.md). Keep crack closure, permeability/ingress recovery, mechanical retention, durability, biological viability and biosafety as separate evidence planes: strength for one endpoint must not silently upgrade another. Technology-maturity labels from reviews are triage aids, not qualification. This shared rubric is a documentation reference only; it does not introduce a parallel evidence schema. The first dataset-based measurement benchmark candidate is tracked in [BIO-CEM-001A2 #7462](https://github.com/Luminous-Dynamics/symthaea/issues/7462); its rights and grouping gates must pass before data processing or model evaluation.

Do not add a second generic observation/provenance ontology for this work. The implementation should reuse canonical owners: SE-SEM-001 (#4861) for physical quantities, units, frames and time; FIELD/FIELD-001A/SE-OBS (#3589/#3633/#3695) for physical observations, calibration and provenance; ENG-EVID-INDEP (#4918) for dependency, common-mode and replication semantics; ENG-EVID-CLAIM (#5771) for evidence-bound claims; and MAT-007/008/013 for materials identity, conditioned measurements and physical-sample lineage. Road/rail-specific work should remain a thin domain profile or adapter unless the transport owner audit demonstrates a concrete semantic gap that existing owners cannot represent losslessly. See the dedicated ownership gates at [BIO-CEM-001 #5751](https://github.com/Luminous-Dynamics/symthaea/issues/5751), [literature benchmark #5753](https://github.com/Luminous-Dynamics/symthaea/issues/5753), [observation-contract preregistration #5767](https://github.com/Luminous-Dynamics/symthaea/issues/5767), and [zero-new-type gate #5768](https://github.com/Luminous-Dynamics/symthaea/issues/5768).

## Current implementation focus: track-geometry forecast benchmark

The execution contract is tracked in [TRANSPORT-RAIL-001A #7452](https://github.com/Luminous-Dynamics/symthaea/issues/7452), under the existing rail/fixed-guideway owner [#6248](https://github.com/Luminous-Dynamics/symthaea/issues/6248). It is a research/data workstream, not a new track-geometry schema or railway dynamics solver. The current FRA RR 26-01 report describes an initial investigation and states that case studies were the next research step; source access to raw track geometry remains a gate, not an assumption. The benchmark therefore separates (a) operational forecasts using only information available at issuance from (b) post-intervention forecasts after maintenance actually occurs, and requires future-time plus segment/route holdouts, negative leakage controls and independently recomputed metrics.

## South African railway data and governance path

The Railway Safety Regulator's [NIIMS NextGen update published 22 June 2026](https://www.rsr.org.za/niims-nextgen-strengthening-railway-safety-through-digital-oversight/) describes an Operator Asset Management Module for asset registers and maintenance plans. Treat it as a potential **authorized data/interoperability lead**, not as public data, an open API, or permission to access operator records. Obtain explicit owner approval, purpose/access terms, and source/version metadata before any dataset admission. Do not replace an operator's asset register or the RSR's compliance/occurrence systems with a Mycelix duplicate.

The same update says Phase 2 of NIIMS NextGen was planned for **1 November 2026** for safety permits, audits and inspections. As of 15 October 2026, that date is future; verify official status rather than presuming those modules are live. Current approved processes remain authoritative until the regulator confirms the transition.

Use the RSR's [rail safety standards index](https://www.rsr.org.za/categories_rt/railway-regulatory-tools/railway-safety-standards/) and [2026 standards-development overview](https://www.rsr.org.za/wp-content/uploads/2026/03/Standards-Development.pdf) to discover applicable sources. Freeze a standard's edition and authority in each project profile only after confirmation with the responsible operator/infrastructure manager and current regulator requirements. FRA forecasts may inform methods, but their datasets, maintenance practices and geography do not establish South African model calibration.

## First research targets

### Roads

Start with conventional and lower-impact alternatives on matched, controlled cases:

- Conventional asphalt baseline versus mixtures containing reclaimed asphalt pavement (RAP), reclaimed concrete aggregate (RCA), or other locally available recycled material.
- Binder/aggregate combinations with rutting, fatigue and low-temperature cracking evaluated separately.
- Granular and stabilized base/subbase materials, compaction, moisture sensitivity, bearing capacity, erosion, and drainage.
- Surface preservation and in-place recycling versus remove-and-rebuild maintenance.
- Climate/traffic-specific layer design and maintenance timing, including heavy-axle exposure, heat, rainfall, flooding, and water ingress.

Do not assume higher recycled content is always better. The candidate must preserve or improve required performance, account for the actual material source and processing, and pass applicable environmental/leaching and construction-quality requirements.

### Railways

Start with asset-level diagnosis and maintainability before proposing novel rail metallurgy:

- Track geometry and alignment observations with measurement method, gauge, track class, and instrument metadata.
- Rail-head wear, rolling-contact fatigue, defect inspection, temperature, axle-load history, and grinding/replacement history.
- Sleeper and fastening condition, ballast fouling/degradation, subgrade settlement, drainage and transition zones.
- Compare conventional ballasted track with suitable slab/ballastless alternatives only for route classes where the whole-life trade-offs can be assessed.
- Evaluate rail steel, welds, coatings, composites, and recycled/by-product materials only against the relevant material and track qualification regime.

Rail operating limits, inspection intervals and safety rules are route/operator/regulator specific. A research model must not infer authorization from geometry observations or simulated operation.

## Shared experimental contract

Every experiment should include:

1. **Question and baseline.** A reproducible target, the current accepted design/process, and a predeclared comparison.
2. **Requirements.** Load spectrum, traffic class, climate envelope, drainage, geometry, construction method, operating constraints and the specific standards/contract clauses selected by the responsible engineer.
3. **Materials provenance.** Source and batch, composition/gradation, supplier or recycling stream, processing, specimen IDs, test protocol/version, conditioning, replicates, raw observations, calibration records and artifact digest.
4. **Property-specific measurements.** For example modulus/stiffness, strength, fatigue/crack response, rutting, wear, skid/friction, water susceptibility, thermal performance and contaminant/leaching results, as applicable.
5. **Uncertainty and domain checks.** Replicate variation, instrument uncertainty, missingness, distribution shifts, out-of-domain inputs and confidence/interval calibration. Never turn a missing measure into a favourable zero.
6. **Independent checks.** The checker must recompute key integrity/metric calculations from frozen input artifacts, not simply rerun the candidate generator.
7. **Negative controls.** Include deliberately corrupted provenance, units, batch IDs, impossible values, out-of-domain scenarios and degraded candidates; they must be rejected or visibly downgraded.
8. **Versioned output.** Code commit, dependency lock, input-manifest digest, dataset split, model/version, standards editions, run receipt, outputs and verifier result.
9. **Decision separation.** The research tool reports alternatives and evidence. The designated professional and asset owner make engineering and procurement decisions.

All physical measurements must carry explicit units. A numeric value without its unit, measurement method, scope and traceable source is not qualified engineering evidence.

## Baselines and metrics

Measure each asset/candidate along independent axes; do not collapse safety into a weighted score:

- **Safety constraints:** mandatory gates from the applicable authority/contract, interpreted by a qualified engineer.
- **Performance:** load capacity, deformation, rutting, cracking/fatigue, wear, track geometry, drainage and serviceability as appropriate to the asset.
- **Life-cycle cost:** material, transport, construction, closures, inspection, planned maintenance, failures, rehabilitation and end-of-life.
- **Environmental performance:** life-cycle emissions/energy, virgin-material demand, water, waste diversion, toxicity/leaching, and repairability.
- **Resilience:** performance under climate extremes, flooding, heat, resource disruption and credible degradation scenarios.
- **Access and equity:** travel time/reliability, accessibility, communities served, disruption burden and whether maintenance reduces service to already underserved places.

Return a Pareto set of feasible trade-offs after mandatory safety and compliance constraints are met. Publish the baseline, units, assumptions, sensitivity analysis and reasons a candidate was excluded. Do not hide trade-offs in one opaque scalar.

## Staged plan and exit criteria

### Phase 0 — Evidence contracts (this change)

- Add typed road-pavement and rail-track observation records in Rust.
- Require stable observation/segment identifiers and at least one provenance record.
- Distinguish literature, synthetic simulation, laboratory test, field measurement and independent review as declared evidence modalities.
- Require a declared field-measurement evidence record for asset-condition observations, so simulation-only output cannot masquerade as a physical observation. This is a data-integrity check, not authentication that the source is truthful.
- Reject non-finite, physically out-of-domain and duplicate-evidence inputs.
- Keep unit-specific fields and explicitly state that structural thresholds are not encoded.
- Test ordinary inputs and adversarial/failure-path records, including a simulation-only negative control.

**Exit:** source-reviewed tests execute on the exact PR head; malformed evidence is rejected, and valid input is described only as *well-formed*, not safe or qualified.

### Phase 1 — Small, auditable benchmark corpus

- Select openly available, reusable, properly licensed datasets and source documents.
- Build a frozen manifest with hashes, units, measurement protocols, locations/time windows where permission allows, missingness and split assignments.
- Use project/site/time-separated holdouts to avoid near-duplicate leakage.
- Start with a narrow problem such as estimating asphalt mixture cracking or pavement condition from named measured covariates, with a simple statistical/engineering baseline.

**Exit:** independent manifest verification, reproducible baseline, documented coverage/limitations, leakage checks, negative controls and uncertainty report. No speculative material recommendation.

### Phase 2 — Calibrated predictors

- Introduce property-specific models; compare against standard empirical/statistical baselines before HDC/CfC-based models.
- Evaluate calibration, error by material/source/region/climate/traffic strata, out-of-domain detection, and prediction intervals.
- Keep candidate generation separate from the verifier and freeze a held-out evaluation set.

**Exit:** predeclared acceptance metrics are met on untouched evaluation data, limitations are documented, and a reviewer can reproduce the result from the pinned artifacts.

### Phase 3 — Coupled system simulation

- Link validated material parameters to an appropriate mechanical/structural solver.
- Couple loads, pavement/track layers, foundation/subgrade, moisture and thermal effects only where the solver's assumptions and numerical validation are documented.
- Benchmark against analytical cases, recognized reference examples, and measured field/lab data.
- Use Symtropy for the simulation tasks it can validly model, or a verified external domain solver where its current rigid-body model is insufficient.

**Exit:** numerical verification and validation evidence are separate and explicit; mesh/time-step/sensitivity effects, conservation or balance checks, model discrepancy and failure modes are reported. A visual simulation alone is not validation.

### Phase 4 — Laboratory and field evaluation

- A qualified lab tests candidate specimens under applicable protocols.
- Pre-register hypotheses, control mixtures, replicates, conditioning and stopping rules.
- Only after laboratory review, consider a bounded, owner-authorized monitored pilot with a conventional fallback and inspection/withdrawal plan.
- Independently record construction QA, sensor calibration, failures, repairs and life-cycle outcomes.

**Exit:** independently reviewed results demonstrate the target improvement with uncertainty, or the candidate is rejected. No full deployment from simulation-only evidence.

## South African deployment considerations

For road use in South Africa, resolve the applicable, current road-authority specifications and contract documents before encoding thresholds. The Department of Transport's COTO page publishes separate sections for earthworks/pavement materials, construction, concrete, asphalt, drainage and quality assurance, and warns that earlier August 2020 versions were no longer applicable. Use the current authoritative edition/revisions and project-specific requirements; do not bake a downloaded threshold into code without versioned provenance.

Rail requirements require separate verification against the applicable operator, infrastructure manager and regulator requirements; road specifications are not a substitute for rail standards.

A pilot should be selected for evidence quality and manageable consequences, not because a model predicts high performance. Prefer an owner-controlled, well-instrumented, low-risk segment where baseline condition, traffic and maintenance history are known.

## Stack integration contracts

- **Symthaea → materials registry:** candidate composition and predicted properties are hypotheses; each predicted quantity links to its own model card, calibration set and uncertainty.
- **Symtropy → evidence:** every run binds solver/version, configuration, mesh/geometry, load cases, boundary conditions, seed/determinism details where relevant, and output digest. Unsupported physics are explicit blockers.
- **Sol-Atlas → context:** geospatial layers must include source, resolution, time/date, coordinate reference system, coverage and uncertainty; missing local data stays missing rather than being silently imputed.
- **Mycelix → provenance/governance:** append-only or revision-linked records preserve who supplied/reviewed/approved evidence, conflicts, licensing and procurement decisions. A vote can authorize a process; it cannot turn an invalid test into a valid one.
- **Independent verifier:** recompute digest shape and content commitments, schema/range checks, units, split membership and report totals independently. Cryptographic integrity proves bytes/lineage, not that the measurement is scientifically true.
- **Human authority:** responsible licensed engineers and asset owners approve specifications, designs, trials and work orders under current rules.

## Research references

1. FHWA, [Asphalt Pavement Program](https://www.fhwa.dot.gov/pavement/asphalt/): performance-based materials selection, mixture design, testing, construction and quality control.
2. FHWA, [Reclaimed Asphalt Pavement](https://www.fhwa.dot.gov/Pavement/recycling/rap/index.cfm): recycled pavement materials must remain cost-effective, environmentally responsible and performant.
3. FHWA, [Aggregates](https://www.fhwa.dot.gov/pavement/aggregates/): aggregate sourcing, regional resource availability, and whole-life performance.
4. FHWA, [Asphalt Binder and Mixtures Laboratory](https://highways.dot.gov/turner-fairbank-highway-research-center/labs/asphalt): characterization of flow, deformation and fracture; laboratory testing for stiffness, strength and ductility.
5. South Africa Department of Transport, [COTO Standard Specifications for Road and Bridge Works](https://www.transport.gov.za/?page_id=1155): use authoritative, current chapter revisions and the applicable project specification rather than assuming a historical edition remains valid.

## Current claim ceiling

This document establishes a research plan only. It does not introduce a new production observation ontology, discover a new road/rail material, predict service life, outperform industry methods, prove compliance, establish cost savings, or authorize construction. Any future code change must be justified against the canonical owner matrix first, then pass exact-head tests before qualification claims.
