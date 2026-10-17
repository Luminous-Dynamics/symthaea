# Open Mobility — Task-Conditioned Architecture Pareto Benchmark v0.1

**Status:** research protocol draft  
**Tracking issue:** [MOBILITY-ASSIST-008 / #7880](https://github.com/Luminous-Dynamics/symthaea/issues/7880)  
**Reference implementation:** draft branch for [architecture Pareto evaluator](https://github.com/Luminous-Dynamics/symthaea/blob/feat/mobility-architecture-pareto/crates/core/symthaea-core/src/architecture_pareto.rs)  
**Scenario corpus:** [architecture-pareto-scenarios-v1.json](./fixtures/architecture-pareto-scenarios-v1.json)  
**Claim ceiling:** synthetic comparison and report-only software evidence. No physical, clinical, regulatory or hardware-readiness claim.

## 1. Why compare whole architectures?

Peak torque, actuator mass, battery energy density and structural stiffness are useful component descriptors, but none alone tells whether the *system* is better for a wearer doing a specific task. Mass carried on the body is not equivalent to off-board mass; distal mass is not equivalent to trunk mass; remote actuation can reduce worn mass while adding tether restrictions and off-board equipment. Stiffness, active degrees of freedom, energy, comfort, setup and repair complexity also interact.

Research basis:

- A 2026 system-level lightweight-design review argues for co-design across structure, stiffness, distal mass, actuation placement, dynamic behaviour and human compatibility, rather than isolated component mass reduction. It is a focused narrative review, not a universal quantitative standard: https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1835857/full
- A 2026 scoping review of active lower-limb exoskeletons/exosuits notes that off-board actuation via Bowden cables can reduce worn mass but creates a tethered design; it also notes that use time and donning/doffing are critically important but sparsely reported: https://link.springer.com/article/10.1186/s12984-026-01955-2
- ISO 18646-6:2026 defines lower-limb wearable-robot performance tests using an anthropomorphic test-dummy robot. It includes powered and non-powered designs but excludes lower-limb systems driven by biosignals such as EMG. This protocol complements mechanical qualification; it does not replace that standard: https://www.iso.org/standard/88497.html

## 2. Paradigm shifts

### 2.1 Component optimization → system-level co-design

The comparison unit is a task-conditioned wearable system: worn structure, off-board hardware, transmission/tether, power source and relevant service components. A lighter wearable is not automatically lighter overall. Report both wearer-borne and total-system mass, and publish where the mass sits on the body.

### 2.2 One weighted score → non-dominated trade-off frontier

Do not pick universal weights for task completion, wearer-borne mass, off-board mass, distal mass, energy, setup, movement restriction, comfort and serviceability. A weighted sum can hide the fact that one user values comfort while another needs higher task success. Instead, show all Pareto-optimal candidates and the precise dimensions on which a dominated candidate loses.

This means the "winner" is often a small *set* of candidates with real trade-offs. User preferences and profile constraints can be applied later, explicitly, rather than covertly baked into a universal score.

### 2.3 Range claims → task-conditioned energy evidence

Energy per task is meaningful only under a declared task protocol, speed, payload, grade, temperature and task mix. Report Wh/task and Wh/hour; include reserve prediction error and low-energy transitions in later stages. Do not infer a reliable range from battery percentage alone.

### 2.4 Mechanical performance → mechanical + interaction + maintenance evidence

Retain human outcomes, machine performance and human–robot interaction as separate layers. Comfort and movement-restriction metrics require declared measurement methods. Service component count is descriptive, not an optimization direction: one integrated module may have fewer parts but be slower or costlier to replace than a modular assembly. Later bench work must measure actual tools, skill, time, calibration, part availability and whether a repaired unit returns to service correctly.

### 2.5 Learned optimizer → transparent evidence pipeline

Learned systems may propose candidate configurations, but they should not change evidence levels or choose a scalar score that makes the answer look simpler than the trade-off really is. Every candidate remains bound to the same task, wearer profile, environment, protocol revision and declared evidence level.

## 3. Required comparison context

A candidate set is valid only when all candidates are compared under the same:

- benchmark scenario and task identity;
- wearer/device profile and environment profile;
- task protocol revision;
- declared evidence level (SyntheticReplay, Simulation, Bench, HardwareInLoop, or HumanStudy);
- operating conditions: speed, payload, grade and ambient temperature.

The evaluator accepts one common comparison request to make cross-context ranking impossible by construction. This does not authenticate the caller's claims or calibrate the measurements.

## 4. Metric dictionary

| Dimension | Required field | Unit / direction |
|---|---|---|
| Task outcome | Task success rate | Proportion [0, 1], higher is better |
| Wearer burden | Wearer-borne mass | kg, lower is better |
| Whole system | Total-system mass, including off-board equipment | kg, lower is better |
| Off-board dependency | Off-board mass | kg, lower is better, reported separately |
| Mass distribution | Worn mass on torso, pelvis, proximal limb, distal limb, and other wearer locations | kg per segment, lower mass per segment is an objective dimension, trade-offs visible |
| Energy | Energy per declared task | Wh/task, lower is better under identical conditions |
| Setup | Time to declared ready state | seconds, lower is better |
| Mobility | Movement-restriction index | Declared normalized scale [0, 1], lower is better |
| Comfort | Declared comfort score and measurement method | Normalized [0, 1], higher is better; self-report and proxy must be distinguished |
| Tether burden | Tether interference events | Events/task, lower is better; no tether can be a structural zero but missing data is never zero |
| Serviceability | Repair/replacement time and service-component count | minutes lower is better; component count is descriptive only, because fewer parts do not always mean easier repair |

Mass distribution must reconcile with wearer-borne mass within a declared tolerance. Total-system mass must not understate wearer plus off-board mass. These consistency checks catch denominator games; they cannot verify the mass measurement itself.

## 5. Pareto rules

For each candidate, the software evaluates all declared metric directions. Candidate A dominates candidate B only if A is no worse on every required dimension and strictly better on at least one dimension, within the declared numeric tolerance.

- Keep every non-dominated candidate, including those optimized for different priorities.
- Report each dominated candidate, every candidate that dominates it, and the dimensions explaining that dominance.
- Do not average dimensions, invent a universal weight vector, or call the Pareto set a product ranking.
- Strictly required numeric fields must be present and finite. Missing values are not filled with zero.
- Invalid IDs, duplicate candidates, non-finite values, invalid normalized scores, inconsistent mass distributions and inconsistent whole-system mass prevent ranking.
- The report echoes the declared evidence level; it never converts synthetic or simulation output into bench/human evidence.

## 6. Reference implementation status

Draft [PR #7882](https://github.com/Luminous-Dynamics/symthaea/pull/7882) adds a report-only Rust evaluator in `symthaea-core::architecture_pareto`. It validates mass denominators/distribution, rejects missing required numeric fields, reports each candidate's Pareto status and dominators, and has no scalar aggregate score. The v1 synthetic fixture has five illustrative candidates: four deliberately non-dominated architectures with different trade-offs and one dominated overbuilt candidate.

Ten unit tests and three public-API integration tests are authored. They have **not been executed** in this environment; test counts are not passes. The fixture is synthetic, not a measurement campaign or product recommendation.

## 7. Proposed benchmark gates

These are software/research gates, not hardware or clinical standards.

1. **Fair comparison:** every candidate in a report shares the same task/profile/environment/protocol/evidence context by construction.
2. **No hidden mass:** publish wearer-borne mass, segment distribution, off-board mass and total-system mass together; inconsistencies are rejected.
3. **Missingness integrity:** every required metric is either explicitly supplied and validated or the candidate fails deserialization/validation. Missing is never interpreted as zero.
4. **Correct frontier:** on the v1 synthetic corpus, exactly the predeclared non-dominated candidates remain on the frontier; dominated candidates include correct dominating IDs and dimension explanations.
5. **Deterministic replay:** same candidate order, request and schema produce identical report structure and serialized bytes.
6. **No scalar score:** the report contains no composite "best architecture" score.
7. **Positive/negative controls:** at least one true trade-off pair remains non-dominated, while a deliberately overbuilt candidate is dominated under the corpus metrics.
8. **Evidence honesty:** synthetic values remain tagged SyntheticReplay; no physical output, clinical benefit, safety or energy-saving claim is inferred.

For the starter corpus, candidate values are deliberately illustrative. They must not be copied as engineering targets or treated as measured properties of any real product.

## 8. Execution sequence

1. Review the metric directions and operating-condition schema.
2. Validate the strict JSON fixture and the deterministic Rust Pareto evaluator.
3. Add property-based sweeps for malformed floats, mass-balance overflow, tie cases and dimension-by-dimension dominance.
4. Add measurement-system guidance before any physical architecture candidates are scored.
5. Evaluate physical designs only after independent mass, electrical-energy, interface and task-outcome measurements are available with calibration and uncertainty records.
6. Keep human-use and clinical outcomes separate and pursue them only under appropriate ethics, privacy and regulatory processes.

## 9. Non-claims

This protocol does not establish that a passive, soft, rigid, remote-actuation or hybrid architecture is universally superior. It does not claim that synthetic mass/energy values are representative of real devices, and it does not validate torque, pressure, braking, emergency release or human-carrying flight. The goal is a reproducible method that makes trade-offs and missing evidence visible rather than producing a deceptively simple score.
