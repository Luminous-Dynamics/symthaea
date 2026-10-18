# Manufacturing Evidence and Benchmark Protocol (Draft)

**Status:** Proposed engineering contract; not a certification scheme.  
**Owner:** Symthaea manufacturing workstream  
**Last reviewed:** 2026-10-18  
**Scope:** Metallurgy and material selection, welding, CNC machining, and shipbuilding.  
**Companion:** [Manufacturing Integration Architecture](MANUFACTURING_INTEGRATION.md)

## 1. Purpose

The project should optimize *verified conforming physical outcomes*, not merely plausible simulation output, nominal cycle time, or model confidence.

Every benchmark case must be reproducible and must distinguish:
- measured evidence from simulated or predicted values;
- a calculation from an engineering approval;
- an explicit compatibility record from an unknown pair;
- statistical tolerance analysis from conservative worst-case acceptance;
- a successful software test from validated physical performance.

No benchmark result certifies a material, welding procedure, vessel, component, or manufacturing facility. Applicable standards, classification-society rules, qualified procedures, competent personnel, calibrated equipment, and required independent approvals remain authoritative.

## 2. Required evidence envelope

A result is not eligible for an engineering pass unless the case records the following fields or explicitly marks them unavailable:

| Field | Requirement |
|---|---|
| Case identity | Stable case ID, semantic version, discipline, intended decision and hazard/criticality class |
| Geometry | Part/drawing/model revision and content digest; units and coordinate convention |
| Materials | Exact specification/grade/condition where relevant, supplier/heat/lot identity, certificate references and provenance |
| Process | Process route and revision; machine/tool/fixture identity where relevant; approved procedure and qualification range for joining |
| Environment | Relevant temperature, corrosion/service environment, loading, boundary conditions and operating envelope |
| Metrology | Measured value, units, uncertainty, calibration status and timestamp for each required measurement |
| Rules | Authority, document identifier, revision, applicability date and the specific controlled rule/check identifier |
| Model | Code/model version, configuration digest, input dataset digest and dependency/version manifest |
| Result | Numeric output plus units, uncertainty, assumptions, validity range, evidence references and independent-verification state |
| Decision | `pass`, `fail`, `unverified`, or `not_applicable`; reason codes; reviewer/approval reference when required |

Missing critical evidence must yield `unverified`, not an inferred pass. Conflicting evidence must remain visible and block safety-critical acceptance until resolved. Corrections supersede records; they must not erase prior audit history.

## 3. Benchmark families

### M1 — Material identity and compatibility

**Cases:** Explicitly compatible pairs; explicitly conditional pairs; explicitly incompatible pairs; unlisted pairs; ambiguous grade/temper/heat-treatment designations; dissimilar-metal interfaces; and conflicting or stale material certificates.

**Measures:**
- critical unknown-pair false-accept count;
- incompatible-pair false-accept count;
- correct distinction between compatible, conditional, incompatible and unverified;
- traceability completeness from part to material certificate and heat/lot;
- review burden caused by justified unverified outcomes.

**Hard gate:** Zero critical false accepts in the pinned adversarial set. Unknown is not compatible. Generic family names (for example, “steel”) must not stand in for a required grade or condition when the decision depends on that detail.

A matrix entry only passes this screening check; it does not establish suitability for every temperature, stress, joining process, corrosion environment or service life.

### M2 — Tolerance stack-up and dimensional acceptance

**Cases:** Known analytic stacks; asymmetric sensitivities; RSS-below-limit but worst-case-above-limit cases; limit-boundary values; negative tolerances; NaN/infinity; absent dimensions; and unit/scale mistakes.

**Measures:** Absolute numeric error against analytic fixtures, correct unit conversion, correct invalid-input rejection, and pass/fail confusion matrix.

**Hard gate:** The default `within_spec` result is true only when all relevant numeric inputs are finite and well-formed and the worst-case stack is within the specified limit. RSS is retained as a statistical diagnostic; it cannot independently authorize conservative acceptance. A separately qualified statistical tolerance policy must be explicit, versioned and approved for cases where it is applicable.

### W1 — Welding procedure applicability and quality evidence

**Cases:** In-range and out-of-range procedure qualifications; material/thickness/position/consumable mismatches; high heat input; post-weld heat treatment; missing personnel or equipment qualifications; required NDT missing or premature; repair and re-inspection; revision changes.

**Measures:** Applicability-decision correctness; critical false accepts; completeness of WPS/PQR and material/consumable links; distortion error on held-out measured cases; inspection records correctly tied to weld IDs and revisions.

**Hard gate:** Zero safety-critical false accepts in the adversarial qualification corpus. The application reports missing or out-of-range qualifications as unverified/rejected according to the applicable rule; it does not invent procedure ranges or infer code acceptance from a generic process label.

### C1 — CNC machining and metrology

**Cases:** Tool/fixture collisions, machine travel limits, holder reach, tool wear, thermal drift, chatter indicators, datum/coordinate mistakes, GD&T constraints, probing uncertainty and measurement disagreement.

**Measures:** Collision detection recall on a pinned verified test suite; dimensional prediction MAE/RMSE by feature and material; tool-wear lead time and false-alarm rate; cycle time per conforming part; scrap/rework rate; energy per conforming part.

**Gates:** No missed collision in the defined benchmark suite; report precision/recall as well as false alarms. A simulated toolpath pass is not proof of physical safety: machine kinematics, postprocessor, fixture/tool assembly and machine state need independent checks and a controlled physical prove-out before production use.

### S1 — Shipbuilding and shipyard planning

**Cases:** Plate/block nesting; hull-block assembly and weld sequence; distortion and fit-up; constrained cranes/transport/dry docks; material-lot delays; drawing revisions after work begins; inspection access; rework; repair and dismantling.

**Measures:** Plan feasibility; schedule makespan; critical-path robustness under explicit delay scenarios; crane/workcell conflict count; material yield and scrap; rework hours; inspection access; energy and emissions per accepted assembly.

**Hard gate:** Zero impossible resource overlaps in the validated scheduling model and zero bypasses of required material, welding or inspection gates. An initial hypothesis for scenario testing is 20% lower simulated rework time than a documented baseline without relaxing safety or acceptance constraints; this is a target to test, not an achieved result or universal industry expectation.

## 4. Cross-cutting verification and validation

Use an evidence ladder and label each case with its highest achieved level:

1. **L0 — Schema/property tests:** units, ranges, enum semantics, hashing, missing values and deterministic serialization.
2. **L1 — Analytic fixtures:** hand-computable results and edge cases.
3. **L2 — Reference data:** pinned external datasets or benchmark artifacts with licensing, provenance and train/test split recorded.
4. **L3 — Measured component tests:** calibrated measurements, uncertainty budget, repeats and residual analysis.
5. **L4 — Integrated cell/yard trials:** synchronized machine/inspection/process records, known revisions, fault injection and human-reviewed outcomes.
6. **L5 — Operational evidence:** prospective monitored deployment with documented envelope, safety review, drift monitoring and rollback plan.

A level is not inferred from another. Synthetic data can prove software behavior, not physical accuracy. Validation must hold out complete parts, material lots, weld procedure families, jobs or time windows where leakage would otherwise make results look better than they are.

Every predictive benchmark compares against at least:
- a simple baseline (constant/last-value or empirical rule, as appropriate);
- a conventional domain model when available;
- the proposed model;
- an uncertainty-aware variant where it applies.

Report units, sample count, distribution of errors, worst-case error, baseline-relative effect, uncertainty/coverage where applicable, failures and exclusions. Never publish only the best metric.

## 5. Reproducible benchmark manifest

Each pinned run should include:
- case corpus revision and SHA-256 digest;
- executable/source commit SHA and build configuration;
- dataset/source digests and license/usage notes;
- machine/OS/runtime information when timing is material;
- random seeds for stochastic algorithms;
- exact command or workflow to reproduce the run;
- expected result or reference tolerance;
- observed result, logs, deviations and test status;
- negative-control outcomes and known limitations;
- independent reviewer and review date where required.

A queued workflow is not a passing workflow. An absent receipt is not evidence of success. Any digest, case identity, rules revision or expected-output change must invalidate the prior result and require a new run.

## 6. Cross-repository allocation

- **Symthaea:** physics/engineering calculations, material screening, tolerance analysis, process models, uncertainty and benchmark execution logic.
- **Mycelix:** signed/versioned provenance records, material and procedure evidence, inspection history, authorization and audit trail. Preserve sensitive supplier and personnel data with suitable access control.
- **Symtropy:** geometric/collision simulation, process-cell and shipyard digital twins, fixture/sequence feasibility and calibration against observed outcomes.
- **Sol-Atlas:** geospatially grounded suppliers, mills, machine shops, ports, shipyards, dry docks, energy/water/logistics constraints and repair/recycling capacity.

Cross-repository records should share stable IDs, schema versions, units, source digests and explicit status semantics; they should not duplicate one another's mutable source of truth.

## 7. Proposed engineering targets (not claims of current performance)

| Metric | Initial target |
|---|---|
| Critical safety-gate false accepts on pinned adversarial corpus | 0 |
| Critical records traceable to part and revision | 100% for cases eligible for pass |
| Reproducibility of deterministic benchmark output | Byte-identical normalized manifest/result for identical pinned inputs, where the implementation supports deterministic execution |
| Analytical fixture correctness | 100% within predeclared numeric tolerances |
| Collision suite | 0 missed collisions in the named and versioned benchmark suite |
| Predictive quality | Beat the selected baseline on held-out cases; publish uncertainty and per-case failures, not just an aggregate score |
| CNC capability | Evaluate stable-process (C_{pk}) only where sampling assumptions are justified; use (C_{pk} \geq 1.33) as a candidate production target, not a substitute for drawing/code requirements |
| Shipyard planning | Test the 20% simulated rework-reduction hypothesis against an auditable baseline, with identical constraints and no safety-gate relaxation |
| Drift | Detect a predeclared material/process/data shift and force revalidation before affected predictions regain pass eligibility |

Targets must be tightened or relaxed only through a documented change with rationale, evidence and an owner. A good benchmark must make failure cheap to discover; it must not make pass easy to claim.

## 8. Standards and reference starting points

These references inform the protocol; the project does not reproduce proprietary standard text and does not certify conformity.

- [NIST — Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing): manufacturing digital twins, interoperability, lifecycle integration, verification/validation and quantified uncertainty.
- [NIST — Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed): CAD/CAM/inspection lifecycle testbed, CNC equipment, metrology and MTConnect data.
- [ASME — Verification, Validation and Uncertainty Quantification](https://www.asme.org/codes-standards/vvuq-standards): modeling and simulation credibility concepts.
- [ISO 3834-1:2021](https://www.iso.org/standard/81650.html): selection of welding quality-requirement level. ISO's public page lists this edition as under systematic review; re-check status before freezing rule metadata.
- [IACS UR W11](https://iacs.org.uk/resolutions/unified-requirements/ur-w/ur-w11-rev9-cln): hull structural steels; Revision 10 is listed for entry into force on 1 January 2027.
- [IACS UR W16](https://iacs.org.uk/resolutions/unified-requirements/ur-w/ur-w16-rev3-cln): high-strength steels for welded structures; Revision 4 Corr.1 is listed for entry into force on 1 January 2027.
- [IACS UR W28](https://iacs.org.uk/resolutions/unified-requirements/ur-w/ur-w28-rev2-cln): welding-procedure qualification for hull and marine structures; Revision 3 is listed for entry into force on 1 January 2027.
- [IACS UR W33](https://iacs.org.uk/resolutions/unified-requirements/ur-w/ur-w33-rev1-corr1-cln): NDT of ship-hull steel welds; Revision 2 is listed for entry into force on 1 January 2028.

IACS Unified Requirements are minimum requirements and their incorporation is subject to IACS member-society rules and the relevant implementation terms. The applicable classification society, vessel contract date, flag, jurisdiction and project requirements must be resolved for a real build.

## 9. First executable tranche

1. Correct conservative tolerance acceptance and cover it with adversarial tests.
2. Add explicit unknown material-pair semantics and regression tests.
3. Turn M1 and M2 into deterministic Rust tests before building the larger corpus.
4. Add versioned fixtures for W1/C1/S1, initially with provenance and schema validation, then progressively add reference and physical evidence.
5. Track compile/test/CI receipts against the exact commit SHA; do not interpret queued status as a pass.
