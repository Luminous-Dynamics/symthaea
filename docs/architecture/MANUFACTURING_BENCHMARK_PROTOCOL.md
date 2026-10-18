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
- [NIST — Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing): product-definition standards, conformance testing, interoperability and lifecycle data association.
- [NIST — Monitoring, Diagnostics and Prognostics for Manufacturing Operations](https://www.nist.gov/programs-projects/monitoring-diagnostics-and-prognostics-manufacturing-operations): measurement science and verification/validation for machine/workcell health monitoring and downtime decisions.
- [Urgo et al., *Future-Proof Production Scheduling and Control* (CIRP Annals, 2025)](https://doi.org/10.1016/j.cirp.2025.04.095): robust scheduling and control requirements under disruptions and increasing production complexity.
- [Uddin et al., *Manufacturing Scheduling with Resource and Calendar Constraints* (2026)](https://doi.org/10.3233/ATDE260714): review of resource/calendar-constrained scheduling, including max-plus methods and their limitations for heterogeneous resources and complex decisions. Treat the reviewed methods as candidate components, not a universal solver.
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


## 10. Capacity-aware schedule validation (Symtropy)

The schedule validator must distinguish exclusive assets from explicitly declared pools of interchangeable capacity. A single CNC machine or dry dock is normally exclusive; a declared pool may represent two qualified NDT teams, cranes, or equivalent stations. The pool capacity declaration does not establish job-specific capability, licensing, certification, or interchangeability for a particular task.

### SC2 — Schedule feasibility and resource-pool semantics

**Cases:**
- Capacity-one resources with overlaps and exact back-to-back intervals.
- Capacity-N pools at N simultaneous operations, N+1 operations, multiple operation start/end ties, and nested intervals.
- Missing, duplicate, empty, or zero resource-capacity declarations.
- Resource identifiers used by operations but omitted from the declared capacity inventory.
- Missing/duplicate/self/cyclic predecessors, precedence inversions, invalid interval values and end-time overflow.
- Permutations of the same input to test normalized report determinism.
- Mixed CNC, inspection, welding-bay and crane scenarios with clearly labeled synthetic durations until a licensed and provenance-pinned measured workload is available.

**Hard gates:**
- Zero silent acceptance of a capacity overflow or an undeclared/ambiguous resource capacity in the pinned test suite.
- Back-to-back half-open intervals remain legal.
- A schedule is not marked feasible when any modeled hard constraint fails.
- Diagnostic output is stable for equivalent input records and carries machine-readable violation codes.
- Capability qualification, shift calendars, planned maintenance, setup/changeover, material delivery and process-specific physics must be reported as out of scope until their respective validators are connected; absence of those models must not be narrated as having verified those constraints.

### Scale and efficiency benchmark

Before scale tests, exhaustively compare small interval sets against an independent peak-concurrency oracle, including exact-capacity and half-open boundary cases. Keep those fixtures synthetic and separate from measured workload evidence. Use deterministic generated workloads with declared seeds and resource distributions at 1,000, 10,000 and 100,000 operations. Include low contention, high contention, a single overloaded pool, many small pools, and long nested intervals. Run each configuration in release mode with environment, toolchain, corpus digest and exact source SHA recorded.

Report median and p95 runtime, peak memory if measurable, total event count, capacity-overflow recall against an analytic oracle, false rejection count, and normalized-output digest. Do not use flaky wall-clock thresholds as unit tests. Performance claims require repeated runs and a named hardware/runtime profile. The expected complexity for capacity checking is dominated by sorting at O(n log n), with linear event processing after the sort; the benchmark must confirm practical behavior rather than infer a speedup from asymptotic notation alone. The implementation should use a constant-time-average active set for interval start/end events and avoid re-emitting the full active set for each overflow. A bounded witness may identify the deterministic operation whose start exceeds capacity, while the report states current occupancy and the declared limit; that witness is not a scheduling priority or a job-selection decision.

### Exclusive-resource sweep benchmark

The exclusive-resource validation path should group intervals by resource ID, sort each group's intervals by start time with deterministic tie-breakers, expire intervals whose end is less than or equal to the current start, and compare the current interval only with active intervals in the same resource group. The target complexity is output-sensitive **O(n log n + k)** for *n* valid intervals and *k* conflict pairs emitted. Dense cases can have *k = O(n²)*; the cost of materializing a complete conflict list must not be misrepresented as a sorting bottleneck. Different resource IDs must never be compared against one another. Preserve half-open interval semantics and the existing stable violation contract.

The Symtropy regression fixture with 12,000 operations on distinct resources is a scale-regression test, not a timing receipt; the previous all-pairs approach would examine 71,994,000 operation pairs. The exhaustive small-schedule test separately compares the returned conflict set with a direct pairwise oracle. Neither test proves real shop-floor performance or physical feasibility.

**Reference benchmark method and initial targets (proposed, not measured):**
- Workloads: 1,000, 10,000 and 100,000 operations; include unique-resource sparse schedules, 1,000 resource pools with controlled mixed contention, exact back-to-back boundaries, a single overloaded pool, and a separate bounded dense-overlap case.
- For the 100,000-operation latency target, bound active concurrency to at most 8 per resource so returned conflict output remains bounded. On a named, pinned reference machine in release mode, target p95 validation latency of at most 2 seconds and peak resident memory of at most 512 MiB. These are initial engineering targets to verify, not achieved metrics; review them against a measured baseline before treating them as a gate.
- Run 5 warmups and at least 30 scored repetitions per workload. Record median and p95, peak memory and its measurement method, compiler/toolchain, OS/CPU, build flags, source SHA, input-corpus digest, resource-count distribution, maximum concurrency, conflict-pair count, and normalized result digest.
- For 1k → 10k → 100k sparse/mixed workloads, investigate any runtime growth exceeding 20× per 10× increase in operation count. This is a diagnostic threshold on the pinned benchmark environment, not a wall-clock unit-test assertion and not permission to suppress valid conflicts.
- Dense-overlap tests must validate exact conflict-pair completeness on a smaller declared corpus and report *k* explicitly. Do not demand sub-second runtime for 100k operations if the API is required to materialize billions of conflict pairs; introduce a separately specified summary/witness API if bounded-size diagnostics become a requirement.
- Hard correctness gates: zero mismatch against the independent oracle on the pinned tiny-corpus family; zero silent capacity-overflow or unknown-capacity acceptance; stable normalized output for input permutations; and no regression in half-open back-to-back semantics. Benchmark output must separately label synthetic results and externally measured production evidence.

### External data policy

Use the [NIST Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed) and its [public repository](https://github.com/usnistgov/smstestbed) as candidate sources for CNC/MTConnect and product-lifecycle/inspection evidence. Before importing any subset, record the exact artifact URI, revision or retrieval date, content digest, license/usage conditions, schema, machine/product identity and permitted benchmark role. These resources support CNC and digital-thread studies; they are not, by themselves, a shipyard scheduling ground truth.

Separate all evidence classes in reports:
- **Measured/public reference:** traceable externally obtained data with source digest and usage terms.
- **Synthetic stress fixture:** deliberately generated data that proves validator behavior only.
- **Scenario assumption:** a documented planning assumption, never presented as an observation.
- **Physical qualification:** independently obtained evidence that a real component/process meets its applicable requirement.

The first useful integration target is a small CNC-to-metrology job chain with explicit resource declarations, then a welding/NDT sequence, then a shipyard block/crane scenario. Use the same violation taxonomy and evidence contract across each stage without pretending their physical constraints are interchangeable.
