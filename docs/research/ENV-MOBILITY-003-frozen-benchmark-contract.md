# ENV-MOBILITY-003: Frozen benchmark contract for survey feasibility and repairability

**Status:** Initial synthetic benchmark artifacts. Future independent verifier and production integration are not implemented by this document.  
**Parent:** [PR #7870 — ENV-MOBILITY-001/002](https://github.com/Luminous-Dynamics/symthaea/pull/7870).  
**Existing implementation owners to reuse:** [FACTORY-OPS-001 #7677 / PR #7678](https://github.com/Luminous-Dynamics/symthaea/pull/7678), [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608), [ROB-MFG-001 #5775](https://github.com/Luminous-Dynamics/symthaea/issues/5775), [AIRCAR-001 #7675](https://github.com/Luminous-Dynamics/symthaea/issues/7675), [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686), [ENG-DESIGN-001 #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005), and [FIELD/QIF measurement owners](https://github.com/Luminous-Dynamics/symthaea/issues/3695).

## 1. Why freeze test data before implementation?

The system must not validate itself by repeating its own candidate selector's answer. Freeze input facts and expected dispositions separately. A verifier must derive a disposition from raw facts, not import the proposed mission selector or simply compare a candidate-provided boolean with an expected boolean.

Two subject families are deliberately separate:

1. Mission feasibility and whether a robot is needed at all.
2. Physical repair / component acceptance after maintenance or manufacturing.

A mission-feasibility pass does not establish vehicle qualification. A repairability score does not establish the repaired article's conformity. A feasible production schedule does not qualify a manufacturing process.

## 2. Initial machine-readable artifacts

- [ENV-MOBILITY feasibility corpus v1](fixtures/env-mobility-feasibility-v1.json): 48 deterministic synthetic scenarios across watershed/lake, coast/estuary, agriculture/catchment, disaster assessment, biodiversity/habitat, and remote infrastructure.
- [ENV-MOBILITY repairability corpus v1](fixtures/env-mobility-repairability-v1.json): 28 deterministic synthetic cases across airframe arms, propulsion, wiring harnesses, sensor payloads, surface-vessel modules, underwater pressure interfaces and battery modules. It distinguishes acceptance eligibility, retest, quarantine, serviceability redesign and local-manufacturing closure.

Keep the schemas separate. Do not conflate field-mission authority, part conformity, manufacturing capability, and production release.

The corpus is synthetic and currently has no authoritative physical measurements. No performance claim is supported by its case count.

## 3. Feasibility oracle rule order

For each input, derive the disposition in this order:

1. **Reject known hard violations:** prohibited site/use/data rights; known ecological limit exceedance; out-of-envelope operating conditions; invalid coordinate reference; known insufficient mission reserve; unavailable recovery; incompatible payload; known absence of local failsafe during communications partition; estimated data represented as measured; sensor timeout coerced to a valid zero; or known configuration/evidence mismatch.
2. **Abstain when a required condition is unknown or stale:** permission, operating envelope, required calibration, frame, energy reserve, recovery, payload compatibility, failsafe, offline policy, data rights, source health, configuration, or reference quality.
3. **Prefer the simpler sufficient baseline:** fixed sensors, existing imagery, or a human survey where the explicit synthetic facts state it meets the observation need and no higher-priority hard gate failed.
4. **Admit degraded planning only with declared local resilience:** a communications partition permits store-and-forward planning only if a tested local failsafe and valid offline policy exist.
5. Otherwise, admit the candidate to **the next review stage only**.

Here, `admit` and `admit_degraded` do not authorize physical dispatch, energization, or actuation. Physical operation remains under the applicable controller, operator, environmental, legal and site authority.

## 4. Independent verifier requirements

The first Rust implementation should compose existing owners rather than create another universal robotics, manufacturing, or scheduling engine.

The verifier must:
- parse an exact supported schema version and reject unknown enum strings where they influence a decision;
- reject duplicate case IDs, missing required fields, non-finite quantities where quantities are later added, and schema/version mismatches;
- derive disposition and reason codes from raw facts through its own decision implementation;
- compare derived outputs against the frozen expected labels only after derivation;
- validate that every corpus case is unique and covered by the claimed mission family;
- produce per-case results, corpus SHA-256, verifier version, exact source/configuration identity and a deterministic summary;
- preserve the full failure list, not only aggregate accuracy;
- include mutation tests that intentionally alter high-impact facts and ensure the disposition changes as expected;
- avoid network access and production evaluator imports in the oracle path.

No weighted total score should allow a strong average to offset a hard failure. Hard-constraint false accepts are a blocking failure.

## 5. Required follow-on mutations

The v1 feasibility corpus already covers permission, calibration, operating envelopes, energy, recovery, coordinate reference, configuration mismatch, data-rights and claim-laundering cases. The independent test harness should additionally generate mutations for:

- shared power or clock dependencies falsely described as independent redundancy;
- repeated docking/recovery failures hidden from the service record;
- unknown/stale battery configuration or unrecorded replacement;
- a communications partition with no offline policy or safe local fallback;
- a failed repair followed by an unchanged “passed” acceptance record;
- a fixed-sensor/manual baseline made artificially infeasible by changing assumptions only for the baseline;
- stale mission data presented as current;
- calibration marked current solely because a vendor/status field says so, without the required evidence reference.

The corpus labels and these desired reactions must remain reviewable by a human independent of the production selector.

## 6. Benchmark report format

Publish at least:

- exact corpus and verifier digests;
- exact number of cases and per-disposition counts;
- false accepts, false rejects and abstentions by reason code;
- mutation-by-mutation expected and observed changes;
- deterministic replay confirmation;
- baseline comparisons and limitations;
- explicit evidence class and claim ceiling.

Report both total agreement and the specific high-impact mistakes. A high overall accuracy with one unsafe false accept is a failed gate.

## 7. Next physical reference article

First article should be a simple, non-hazardous, repairable multirotor payload/arm module **or a surface environmental-sensing module**, chosen after the owner audit and supplier/local fabrication check. No new AUV pressure hull or passenger aircraft should be the initial factory reference.

For the exact article, collect:
- editable CAD and drawings with datums/tolerances;
- versioned BOM and substitution rules;
- assembly and repair steps with observed times;
- serial/configuration identity;
- measured dimensions, mass and relevant electrical/interface outputs;
- calibration/test evidence for the payload;
- failure, repair, rework and inspection history;
- exact process/tool/fixture identities and calibration state.

After the workflow is stable, run a small baseline batch and preregister improvements before testing them on a larger batch. Production throughput or first-pass yield should only be claimed after repeatable physical builds and independently reviewed acceptance records.

## 8. Claim ceiling and stop conditions

A passing synthetic verifier demonstrates that the implementation respects the declared fixture contract. It does not show that the constraints are complete, the model matches a physical environment, the sensor is accurate, the repair is durable, the manufacturing process is qualified, or the product is airworthy/seaworthy.

Stop or narrow scope if:
- oracle and candidate share the same production selection function;
- the same test fact can yield inconsistent dispositions without a version change;
- a single known hard-boundary negative control is falsely accepted;
- a repaired/modified article silently inherits evidence invalidated by the change;
- the work duplicates an existing canonical owner rather than supplying a required cross-domain fixture.

## Research and design anchors

- [OSHWA Open Source Hardware Definition](https://oshwa.org/resources/open-source-hardware-definition/): source should be provided in the preferred modification format; standard processes and available components improve reproducibility.
- [NIST Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed): design/manufacturing/inspection integration reference.
- Liu et al., [A review of underwater docking and charging technology for autonomous vehicles](https://doi.org/10.1016/j.oceaneng.2024.117154), *Ocean Engineering* (2024): explains why underwater docking, navigation, charging and recovery require their own validation program.
- Symthaea [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608) is the canonical manufacturing-value benchmark owner; this proposal should bind a domain-specific fixture to it rather than define a competing value engine.
