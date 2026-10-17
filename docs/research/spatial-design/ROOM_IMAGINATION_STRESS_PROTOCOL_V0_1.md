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
