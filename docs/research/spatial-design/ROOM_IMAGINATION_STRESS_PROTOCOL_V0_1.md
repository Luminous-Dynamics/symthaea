# Spatial Co-Design Qualification: Counterfactuals, Robustness, and Lifecycle v0.1

**Status:** Research and benchmark proposal; not an implemented capability or a qualified design engine.  
**Tracking issue:** [ROOM-QUAL-002 #7801](https://github.com/Luminous-Dynamics/symthaea/issues/7801)  
**Related experience proposal:** [PR #7785](https://github.com/Luminous-Dynamics/symthaea/pull/7785)  
**Integration home:** [Symtropy SPATIAL-DESIGN-001 #1766](https://github.com/Luminous-Dynamics/symtropy/issues/1766)

## Executive thesis

The field-changing move is from *evaluating generated images* to *qualifying a design process under change*. A plausible picture is cheap evidence. A trustworthy design assistant has to preserve intent through a long interaction, explore genuinely different plans, state what it does not know, reveal objective conflicts, respond correctly to counterfactuals, keep every claim tied to the exact model/evidence revision, and hand decisions to the proper engineering authority.

This proposal adds a 30-case, synthetic, adversarial benchmark. It is intended to reveal failure modes that can escape a demo in which a single prompt yields one attractive render. The cases are fixtures, not a claim of current system performance.

## Paradigm changes

1. **Intent as a living, revisable contract.** Store hard constraints, soft preferences, their source, affected activities/spaces, confidence, unresolved questions, and who is allowed to revise them. Changes must be explicit; a later turn cannot silently erase a hard constraint.
2. **Counterfactual reasoning rather than prompt compliance.** Change one condition at a time and test which parts of the design should change, which invariants should remain, and which conclusions must be recomputed. Each changed consequence should reference its causal input and evidence.
3. **Robustness under uncertainty rather than point-estimate optimism.** Evaluate ranges and scenarios for dimensions, material properties, weather, occupancy schedules, cost and availability. A candidate that only works under one fragile assumption must not be described as robust.
4. **Pareto sets rather than an unaccountable winner.** Preserve separate outcomes and hard feasibility gates. Compare budget, energy, light, privacy, accessibility, repairability, sensory needs, and adaptability without collapsing disagreement into one opaque score.
5. **Active information gathering.** Ask for a measurement or clarification when it could change the decision; otherwise allow lower-fidelity exploration and label the missing evidence. Measure the value of a question, not the number of questions asked.
6. **Whole-life, task-first design.** Qualify journeys and activities, repair access, reversible alterations, manual fallback, future adaptation, and component replacement—not merely a room's visual composition.
7. **Evidence-linked representations.** A plan, render, section, walk-through, simulation output and export must resolve to a precise design revision. Stale analysis, lossy export, absent units, unknown coordinate frames or a missing solver cannot be hidden behind visual polish.
8. **Human agency and plural preferences.** Keep household members' preferences separate where they disagree. Never infer cultural, disability, lifestyle or sensory requirements from demographic proxies. Use post-occupancy learning only with transparent purpose, consent, minimization, retention limits and a right to correct or withdraw.

## Scope and architectural boundary

This is a qualification layer, not a new CAD/BIM engine or standards-verification subsystem.

- **Symthaea:** translate stated intent into editable requirements, propose alternatives, ask decision-relevant questions, explain trade-offs and uncertainty, and abstain when the evidence does not support a claim.
- **Symtropy:** consume exact, versioned geometry and the appropriate facet topology; run deterministic fixtures, geometric predicates and qualified simulation adapters. Geometry does not by itself prove building-code compliance or physical safety.
- **Mycelix:** only through separately qualified integration, maintain consent, provenance, decision rights, approvals and dissent history. Governance evidence is not physical-performance evidence.
- **Sol Atlas:** only through qualified adapters, contribute source-bound site, climate, orientation, terrain and time context. A map or view alone does not make a datum authoritative.
- **Qualified specialists and authorities:** remain responsible for jurisdiction-specific interpretations and approval of structure, fire/life safety, electrical, plumbing, HVAC, accessibility and construction decisions.

Reuse the existing Symthaea room-imagination proposal and Symtropy spatial-design/accessibility/topology work. Do not make duplicate geometry authorities, universal room graphs, or an independent building-code verdict engine.

## Benchmark contract

Machine-readable fixtures are in `ROOM_IMAGINATION_STRESS_BENCHMARK_V0_1.json`. Each case includes a known setup, a controlled intervention, unknowns, required behavior, negative-control failures, and an expected disposition.

Required evaluation properties:

- **Frozen inputs:** each run records benchmark version, case ID, exact input/profile revision, model/solver version, units, spatial frame where applicable, seed and relevant configuration.
- **Independent checker:** the generator cannot certify its own output. The checker validates requirements, evidence references, revision binding, fixture predicates, and declared claim level independently.
- **Adversarial negatives:** delete an evidence reference, alter an artifact revision, change a unit, change a profile, mutate a geometry digest, remove a required model or exceed the stated uncertainty bounds. Each mutation must either be rejected or explicitly downgrade the claim.
- **No leakage:** separate development cases from a future held-out set; do not tune prompts or thresholds against sealed cases. Public cases alone cannot support generalization claims.
- **Deterministic replay:** where the full pipeline is defined as deterministic, replay identical frozen inputs and configuration. For stochastic generation, compare invariant predicates and registered distributional outcomes rather than demanding byte equality.
- **Report failures, not only aggregate scores:** publish every case-level disposition, failure category, skipped reason, unavailability, seed/configuration and artifact hashes. A missing run is not a pass.
- **Layered evidence:** distinguish synthetic contract tests, geometry/provider tests, solver validation, external professional review, and human/use evaluation. A higher layer cannot be claimed because a lower layer passed.

## Proposed acceptance targets — targets, not results

### Blocking gates

| Gate | Proposed target |
|---|---|
| Hard-intent integrity | 100% declared hard constraints preserved across every edit and every admitted variant in the frozen synthetic suite |
| False verification | 0 cases where a render, recommendation, model confidence or missing evidence is represented as verified performance |
| Evidence currentness | 100% rejection or explicit downgrade for seeded stale geometry, stale analysis, wrong profile, wrong unit/frame, and altered artifact reference |
| Unknown handling | 100% explicit unknown/unverified status when a required input/model/reviewer is missing |
| Lifecycle invariants | 100% detection of seeded inaccessible maintenance, prohibited irreversible retrofit, or missing applicable manual-fallback condition |
| Consent/agency | 0 inferred demographic preferences; 0 claims of consensus when fixture dissent remains unresolved |
| Reproducibility | 100% deterministic replay for cases explicitly designated deterministic; stochastic cases report distributional metrics and uncertainty |

### Quality metrics that must remain separate

- **Intent:** hard-constraint violations, requirement/source coverage, corrections needed, constraint survival over edit-sequence length.
- **Causal response:** correct response to intervention, affected/unaffected invariant precision, traceability from change to affected requirement and evidence; account for legitimate coupled consequences.
- **Alternative quality:** geometric/topological diversity after ignoring color/material-only changes, feasible-alternative count, Pareto-set coverage and per-objective regret.
- **Robustness:** feasible scenario fraction, worst-case/quantile outcome for soft objectives, sensitivity to uncertain inputs, false precision and calibration against known synthetic truth.
- **Information gain:** proportion of clarifying questions that can change a decision in the fixture, missed decision-critical questions, unnecessary question burden, and ability to proceed at explicitly lower fidelity.
- **Evidence integrity:** stale artifact detection, source and revision completeness, export round-trip loss, false-verification rate, independent checker agreement.
- **Whole-life function:** task-chain completion, maintenance/service access, reversibility, repair disruption, manual operation, and adaptation cost where each model is explicit.
- **Plural agency:** each person's requirement retention, exposed private preference count (target zero without authorization), unresolved dissent visibility, and consent-state correctness.

Do not reduce these properties into one universal design score. A Pareto hypervolume or regret metric is acceptable only on a fixture with explicit normalized objective bounds and a declared reference point; it never overrides a hard constraint or professional gate.

### Human evaluation gate (future work)

After synthetic and geometry-level gates exist, run a preregistered, task-based co-design study with participants recruited across relevant access and lived-experience cohorts. Report each cohort and distribution—not only the mean—including task completion, time, ability to explain trade-offs and uncertainty, trust calibration, corrections, and comprehension. Recruit and compensate participants appropriately. No human-benefit claim is available from synthetic benchmark success.

## Recommended build order

1. Validate the JSON manifest structurally and ensure all IDs and requirement references are unique/resolvable.
2. Implement a small Rust-first, independent fixture checker with negative controls for stale IDs, missing evidence, incorrect claim level, and inconsistent metadata.
3. Add exact-geometry fixture predicates only against an identified, qualified geometry provider; do not approximate a production adapter in the checker.
4. Implement counterfactual pairs and small known-answer Pareto problems; record per-case reports and content hashes.
5. Integrate one analysis adapter at a time with declared model envelope, units, input provenance, uncertainty and validation corpus.
6. Add human co-design, professional review and post-occupancy feedback only after governance and research protocols are approved.

Do not wait on hosted runners to research, author fixtures, inspect public source and perform local structural checks. Do not report queued workflows as passing. No Rust compiler, production geometry engine, energy simulation, human study or hosted workflow was run as part of this proposal.

## Research basis and why it matters

- Huang et al. (2025), *Performance-Driven Generative Design in Buildings: A Systematic Review*, reviewed 153 core studies and highlighted metric fragmentation, tight toolchain coupling, limited interoperability and the need for validation/reproducibility: https://doi.org/10.3390/buildings15244556
- NIST (2026), *Digital Twins Workshops Summary Report*, identifies interoperability, verification/validation and uncertainty quantification among key digital-twin challenges: https://doi.org/10.6028/NIST.IR.8620
- DOE whole-building energy modeling requires building geometry, construction materials, systems, operating schedules and local weather for physics-based performance estimates: https://www.energy.gov/cmei/buildings/about-building-energy-modeling
- buildingSMART IFC 4.3.2.0 / ISO 16739-1:2024 offers an open, vendor-neutral building/infrastructure information exchange baseline; exchange success still requires checking the actual properties and relationships preserved: https://technical.buildingsmart.org/standards/ifc/ifc-schema-specifications/
- ASHRAE Standard 55-2023 covers thermal environmental conditions through environmental and personal factors; it is not a substitute for evaluating acoustics, air quality or illumination: https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy
- NASA-STD-3001 Volume 2, Version F, dated 14 July 2026, applies human factors, habitability and environmental-health requirements to human-rated spaceflight systems and habitats; off-world work needs its own applicable requirements and models: https://standards.nasa.gov/node/237

These sources guide the proposed workflow and boundaries. They do not certify Symthaea, Symtropy or any generated design.

## Claim ceiling

This artifact proposes benchmark cases and acceptance criteria. It does not claim that the tests have run, that a room model is qualified, or that any design is buildable, accessible, comfortable, safe, code-compliant or professionally approved.


## New research update — uncertainty, calibration, and stability-feasible design (2026-10)

Recent literature strengthens one implementation priority: uncertainty must be tied to *the decision and lifecycle stage*, rather than being a generic confidence badge.

- Malihi (2026), *Towards Uncertainty-Aware Digital Twins in the Built Environment*, frames the model/physical-asset difference as a lifecycle- and level-of-detail-dependent residual envelope. Model calibration should test predicted uncertainty against benchmark residuals and observed behaviour, not merely output an interval: https://doi.org/10.1016/j.dte.2026.100140
- NIST AMS 100-85 (5 October 2026), *On Methods of Uncertainty Quantification in Computational Modeling and Simulation*, recommends risk-based credibility assessment proportional to model influence and decision consequence and discusses when variance propagation is inadequate compared with distributional propagation: https://doi.org/10.6028/NIST.AMS.100-85
- Zhou et al. (published 1 October 2026), *New Bayesian optimisation framework for robust multi-objective design*, explicitly separates expected performance from objective variance and computes a Pareto set only after designs that violate a declared uncertainty/stability tolerance are removed: https://doi.org/10.1016/j.ress.2026.112359

### Consequences for room/building co-design

1. **Every performance claim needs a decision context.** Attach model/version, input-data version, geometric fidelity/level of detail, operating scenarios, time horizon, units, uncertainty representation, calibration/validation domain, residual statistics, exclusions, and the decision tolerance used. A result calibrated for one room or climate is not automatically transportable to another.
2. **Separate uncertainty sources.** Track at least input/measurement uncertainty, numerical/solver uncertainty, model-form uncertainty, and scenario/future-context uncertainty where the chosen method can support the distinction. Do not combine them into one confidence percentage unless the calculation and assumptions are explicit.
3. **Use risk-proportionate evidence.** A moodboard concept can proceed with missing dimensions. A clearance claim needs a relevant geometry source. A daylight or energy claim needs an appropriately scoped model and inputs. A safety-critical or code-related claim requires qualified authority. Raise verification requirements with the consequence of being wrong.
4. **Report stability-feasible Pareto alternatives.** First exclude alternatives that fail hard constraints. Then separately exclude alternatives that exceed explicitly declared robustness/uncertainty tolerances. Show the remaining Pareto set and uncertainty envelopes per scenario; do not silently use a favorable expected value to conceal excessive variation or a known failing scenario.
5. **Calibrate across revisions and lifecycle.** After renovation, material substitution, control-system changes, or post-occupancy observations, compare predictions with measurements and update the model's valid scope only under a documented calibration procedure. Historical prediction accuracy cannot automatically qualify changed inputs or models.

### Additional benchmarks to add after the manifest evaluator

These are targets for later, separately validated model adapters—not properties proven by the current manifest evaluator.

| Metric | Recommended target / reporting rule |
|---|---|
| Calibration coverage | On a held-out, scenario-representative corpus, empirical coverage of a nominal prediction interval should match its stated level within a preregistered statistical tolerance; report sample count and interval |
| Residual-envelope validity | 100% of quantitative outputs state model scope, source-data revision, applicable scenario range, uncertainty method and known exclusions |
| Robust feasibility | 100% of admitted designs pass every declared hard predicate in every required scenario; a failing/unsupported scenario cannot be averaged away |
| Stability-feasible Pareto integrity | 100% of returned options are both hard-feasible and inside the declared objective-variability tolerance; compare Pareto coverage/regret only on a frozen known-answer fixture with explicit bounds |
| Update invalidation | 100% of seeded dependent analyses become stale after a causal input/model/profile change; unrelated results remain current only when the dependency contract proves independence |
| Out-of-domain behaviour | 100% of seeded out-of-scope inputs produce an explicit abstention/downgrade or a reviewed fallback, never an unqualified extrapolation |
| Decision utility | Report whether a proposed clarification changes the feasible set or ranking on the known-answer cases; do not optimize question count alone |

Do not set one universal numeric calibration tolerance across daylight, thermal comfort, acoustic, energy and structural models. Those require model-specific, risk-specific, data-backed tolerances selected before evaluation. Synthetic golden cases can test the checking logic, but only measured benchmark residuals and representative held-out data can inform real calibration claims.


## Interoperability paradigm: requirement contracts, not another proprietary building format

Use **open IFC for exchange and buildingSMART IDS for declared information requirements**, while keeping the spatial-design evidence graph as a separate traceability layer. IDS is a machine-readable standard that can state which objects, classifications, properties, values and units must be delivered and then check whether that information exists in the exchange. That is a strong foundation for the required-data contract, but it does not establish that the geometry itself is accurate, a simulation is valid, a building is buildable, or code compliance has been approved.

- buildingSMART, *Information Delivery Specification (IDS)* — https://technical.buildingsmart.org/Projects/Information-Delivery-Specification-Ids/
- buildingSMART, *IFC specifications; IFC 4.3.2.0 / ISO 16739-1:2024* — https://technical.buildingsmart.org/standards/ifc/ifc-schema-specifications/
- buildingSMART, *IFC Validation Service* — https://info.buildingsmart.org/users/services/validation-service/

### Design-handoff tests to add when a real adapter exists

1. Freeze a versioned information-requirement profile for a particular handoff (for example, a room retrofit or maintenance/service access package); do not write one universal profile for all project stages.
2. Validate the exported IFC against its IFC schema and the matching IDS profile using a recognized validator.
3. Re-import into a different implementation, then compare required entities, relationships, properties, units, classifications and provenance mappings against the original model.
4. For every missing, converted, defaulted or unmapped item, produce a machine-readable loss record. Opening successfully is not equivalence.
5. Keep geometry correctness, performance validation, currentness and professional approval as separate qualification claims.

The current stress case for lossy export tests only the manifest-level detection contract. It does not yet parse or validate IFC/IDS files. Do not claim IDS/IFC compatibility until an actual exporter/validator/round-trip corpus is available.

## Benchmark scale-up: challenge-set discipline over raw case count

A recent 2026 study reports a 679-query, expert-verified benchmark for geometry-intensive BIM/fire-safety reasoning and 84.3% accuracy for its proposed framework. This is a useful comparison point, not a reason to optimize for a larger headline count: the reported accuracy also shows why hard-gate false acceptance and traceable reasoning need to be measured separately. See Xiao et al. (2026), *Automating geometry-intensive compliance checking in BIM: Graph-based semantic reasoning framework*, *Automation in Construction* 189, 107038, https://doi.org/10.1016/j.autcon.2026.107038.

Recommended maturity ladder after the current evaluator is running:

| Level | Corpus and target |
|---|---|
| A — evaluator mechanics | All 47 current oracle assertions pass in the synthetic golden fixture; mutation tests reliably fail for altered observations and invalid evidence references |
| B — exact geometry provider | At least 500 deterministically generated known-answer geometry questions, with 100 held out; 100% on declared synthetic hard predicates, explicit unknown for incomplete provider coverage, and zero renderer/collider-as-authority false accepts |
| C — multi-hop design reasoning | At least 1,000 expert-adjudicated questions linking intent, context, geometry and evidence, with a sealed test split; report accuracy, abstention, calibration, source traceability, and false-accept rate separately |
| D — solver-backed robustness | Per model, publish a held-out corpus and calibration report; every admitted candidate must meet the declared hard predicates across the required scenario set and remain within the predeclared uncertainty/stability envelope |
| E — open handoff | At least 100 representative round-trip fixtures across supported exporters/importers and handoff profiles; 100% detection of seeded required-property loss, unit corruption and identity/reference breakage |
| F — field evidence | Preregistered professional and resident studies, followed by measured post-occupancy comparisons with consent and governance; no synthetic-only claim of occupant benefit |

These are proposed programme targets. Freeze the test split, oracle version, model envelopes, metric definitions and stopping rules *before* tuning against the evaluation set. A target of 100% is appropriate for the present small, deterministic synthetic hard-gate fixtures; real-world probabilistic or expert-adjudicated outcomes need confidence intervals, inter-rater disagreement, and a documented escalation/abstention policy rather than an unsupported universal guarantee.
