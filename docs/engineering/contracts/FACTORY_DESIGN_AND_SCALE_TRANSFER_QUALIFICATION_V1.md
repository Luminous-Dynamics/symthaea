# Factory Design and Scale-Transfer Qualification Contract v1

**Status:** design contract; no physical capability or process is qualified by this document.  
**Parent work:** [physical engineering capability research PR #7309](https://github.com/Luminous-Dynamics/symthaea/pull/7309), [CP-03 manufacturing digital thread PR #6390](https://github.com/Luminous-Dynamics/symthaea/pull/6390), [candidate-to-qualified robotics design PR #7423](https://github.com/Luminous-Dynamics/symthaea/pull/7423).

## 1. Decision

Symthaea should act as a **candidate designer and evidence orchestrator**, not as an autonomous authority to build or operate a factory. It can generate requirements, candidate layouts, BOM/routing drafts, simulator inputs, experiment plans, analysis reports and manufacturing packets. Independent verification, qualified people, existing controls and authorized operators retain acceptance, safety and actuation authority.

The governing distinctions are:

`idea != candidate != simulation != measured prototype != qualified process != authorized production`

A candidate is not promoted because a simulation completes, a solver returns a number, or a design is inexpensive. New software artifacts are not physical capability evidence.

This applies to benign community infrastructure—irrigation skids, greenhouse controls, water/energy monitors, tool fixtures and packaging equipment—and future industrial systems. A facility intended to manufacture regulated health products must additionally pass product-, process-, site- and jurisdiction-specific quality and regulatory gates. This contract does not authorize making or releasing medicines.

## 2. “Miniaturize the factory” means different things

Every request to make a factory smaller MUST declare one transformation class:

1. **Modularization:** split capacity into installable cells while retaining the same unit-operation envelope. Shared utilities, safety systems, contamination controls, changeover, staffing and maintenance still need to be accounted for.
2. **Capacity scale-down:** reduce throughput or batch size. This can change residence time, heat/mass transfer, mixing, sensor observability, minimum controllable flow, dead volume, surface-to-volume ratio, measurement uncertainty and product quality. A linear multiplier is not sufficient evidence.
3. **Technology transfer or process intensification:** replace or redesign a unit operation. This is a new process generation that needs its own hazard review, validation and qualification.
4. **Facility footprint reduction:** reduce floor area while retaining support space, utilities, quarantine/storage, waste handling, accessibility, maintenance clearance and safe material/personnel routes.

A smaller drawing that omits support systems is not an equivalent smaller factory. A microfactory is not a shrink-wrapped industrial plant unless its operations, hazards, controls and evidence remain valid in the new envelope.

## 3. Typed artifact boundaries

Reuse the existing engineering catalog, typed physical-model and CP-03 manufacturing contracts rather than defining a parallel ontology.

### `FactoryDesignCandidateV1`

Bind at minimum: stable candidate ID/generation/content digest; requirements snapshot and non-goals; facility boundary, coordinate frame and site assumptions; process-family IDs and process-definition generations (references, not recipe content); equipment, utility, material/waste flow and access envelopes; geometry/BOM/routing and versioned asset references; power/water/thermal/ventilation envelopes with units; hazard/safety requirement references; solver/model versions, fidelity, calibration domain and known limitations; expected performance ranges, uncertainty, resource budgets and cost assumptions; excluded scenarios and invalidation conditions.

Missing dimensions, incompatible units, invalid quantities, unbounded assumptions, stale dependencies or unresolved safety-critical interfaces produce `Unresolved`, not success.

### `ScaleTransferPlanV1`

Bind the source design/process generation to an exact target generation and declare:

- transform class, source/target throughput envelopes and changed components;
- preserved invariants and newly introduced failure modes;
- domain-appropriate similarity criteria, where applicable;
- target operating envelope and excluded regimes;
- calibration data and independent solver evidence required;
- test article, instruments, calibration/currentness, sampling plan and acceptance criteria;
- uncertainty budget and how source-model uncertainty propagates;
- independent verifier profile and required human approvals.

A plan without a defensible physical similarity/transfer argument MUST NOT inherit source-model accuracy or process qualification.

### `QualificationEvidenceEnvelopeV1`

Reference immutable artifacts for each step: input manifest, solver/environment identity, raw output, parser version, output digest, measurement subject, calibration/currentness, uncertainty, independent recomputation and disposition. Hash-shaped strings alone are not verification; verify the exact captured bytes before asserting content integrity.

Maintain the CP-03 distinction among design, plan, execution, as-built, observation, engineering disposition and operational authority.

## 4. Stage gates

| Stage | Permitted claim | Minimum exit evidence |
|---|---|---|
| S0 — requirement | Problem and constraints stated | reviewed requirements, units, site/utility assumptions and no-go conditions |
| S1 — candidate | Candidate is well-formed | deterministic generation, constraint report, geometry/interface checks, BOM/routing identity |
| S2 — virtual | A stated model executed | exact solver/input/output lineage, parser/convergence evidence, independent invariant checker, uncertainty and limits |
| S3 — component bench | A bounded component was tested | article identity, calibrated instruments, raw readings, repeatability, safe test plan and independent comparison |
| S4 — integrated bench cell | Interfaces operate in the tested envelope | electrical/fluid/thermal/software interface checks, interlocks, fault injection, access/maintenance review |
| S5 — pilot | Pilot observed in a frozen campaign | pre-registered normal/adverse cases, quality-owner review, waste/changeover controls where applicable |
| S6 — production | Exact configuration accepted for a declared use | qualified process/site/equipment/personnel, validated methods, current quality/regulatory dispositions, release and rollback/recall procedures |

Use explicit states: `NotEvaluated`, `Blocked`, `Failed`, `QualifiedForDeclaredEnvelope`, `Stale` and `Withdrawn`. One failed required gate cannot be masked by an aggregate score. Any safety/performance-relevant dependency change requires a new generation.

## 5. Physical similarity and conservation checks

The model must state which laws and approximations it actually implements. Where relevant, independently check:

- mass balance with explicit inflow, outflow, accumulation and declared losses;
- energy balance with heat/work flows and stored-energy changes;
- pressure/flow and pump-power constraints for fluid networks;
- thermal boundaries, inertia and sensor placement for thermal systems;
- electrical constraints, parasitic effects and isolation boundaries;
- material/lot changes, timing, actuator limits and sensor failure modes;
- units, frames, tolerance stack-up and measurement uncertainty.

Dimensionless criteria such as Reynolds, Peclet or Biot numbers are mechanism-specific checks, not universal badges. A conservation check is necessary but insufficient for physical fidelity. A target outside calibration domain is extrapolative; numerical convergence is not physical validation.

## 6. Required negative controls

A verifier campaign MUST show that these cases block promotion:

1. Successful dry run presented as external-solver evidence.
2. Missing, stale, mismatched or unverifiable artifact references.
3. Incompatible dimensions/units at a critical interface.
4. Geometry scaled without a physical similarity argument.
5. Throughput reduction that violates minimum controllable flow, sensor resolution or utility limits.
6. Changed feedstock/material/equipment/controller using old qualification.
7. Conservation violation outside explicit numerical tolerance.
8. Failed or uncalibrated measurements replaced with model predictions.
9. Unknown hazard/interlock status treated as pass.
10. Waste, quarantine, access, maintenance or emergency egress omitted.
11. Lower cost/higher predicted yield overriding a hard feasibility failure.
12. Equipment simulation used to infer medicine safety, efficacy, authorization or batch release.

Independently recompute report arithmetic and eligibility from frozen inputs. Preserve negative evidence after model upgrades.

## 7. Health and medicinal-garden integration

Symthaea may help design non-clinical infrastructure: water-sampling logistics, irrigation/energy systems, propagation benches, traceable weighing/labeling tools, packaging-line layout and environmental monitoring. It may help an authorized manufacturer organize an evidence-bound engineering dossier for review.

It MUST NOT generate amateur medicine recipes, extraction/concentration procedures, individualized doses, efficacy claims or advice to substitute a garden plant for an authorized medicine. Botanical identity, contamination testing, clinical evidence, lawful access/benefit-sharing, product authorization, GMP/process validation and batch disposition remain separate expert-owned gates.

For South Africa, check current local requirements with qualified professionals and the responsible authorities. SAHPRA's [licence-to-manufacture guidance](https://www.sahpra.org.za/document/licence-to-manufacture-import-or-export/) and [SA GMP guide](https://www.sahpra.org.za/document/sa-guide-to-good-manufacturing-practice-for-medicines/) are starting references, not a substitute for product/site-specific advice.

## 8. Implementation sequence

1. Add typed candidate and scale-transfer records to current engineering contracts; no actuation and no new solver.
2. Use one benign reference facility (for example an irrigation/monitoring cell) with water/power/heat balances and an independent checker.
3. Validate one safe hardware component with calibrated instrumentation before labelling a model validated.
4. Join components through exact interface/frame/unit contracts; inject sensor loss, stuck actuator, supply interruption and over-limit conditions.
5. Freeze a pilot campaign, raw-artifact retention and success/failure criteria before testing.
6. Leave qualification and release decisions with accountable engineering, quality and regulatory owners.

Never weaken evidence thresholds to obtain green CI. Every assertion names its exact candidate, source head and verifier; queued checks are not passes.

## 9. References

- [Symthaea physical engineering capability research](https://github.com/Luminous-Dynamics/symthaea/pull/7309)
- [CP-03 manufacturing digital thread](https://github.com/Luminous-Dynamics/symthaea/pull/6390)
- [Robotics candidate-to-qualified design loop](https://github.com/Luminous-Dynamics/symthaea/pull/7423)
- [NIST Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing) — validation and quantified uncertainty matter to trustworthy manufacturing twins.
- [SAHPRA licence to manufacture/import/export](https://www.sahpra.org.za/document/licence-to-manufacture-import-or-export/)
- [SA Guide to GMP for Medicines](https://www.sahpra.org.za/document/sa-guide-to-good-manufacturing-practice-for-medicines/)
