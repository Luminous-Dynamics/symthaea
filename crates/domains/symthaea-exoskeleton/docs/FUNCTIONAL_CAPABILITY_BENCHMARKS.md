# Functional Capability Benchmark and Cybernetics R&D Contract

**Status:** research design proposal; not a validated medical-device specification  
**Scope:** Symthaea exoskeleton and the Luminous Dynamics cybernetics toolchain  
**Safety status:** simulation/research only; no authorization for physical actuation or human trials

## 1. Why this work exists

The next milestone should not be another isolated gait controller or a larger simulated skeleton. It should be a reproducible way to answer:

> Does the device measurably improve a person's ability to perform meaningful activities, with less effort and burden, while preserving control, comfort, safety, privacy and independence?

The programme should evaluate the wearer, device and human-device interaction together. A device that increases actuator output but worsens wearer effort, discomfort, balance, cognitive burden or practical usability is not an improvement.

This proposal extends the existing exoskeleton evidence-gate, adaptive-assistance, biomechanical-envelope and trace/replay work. It does not replace those controls.

## 2. Existing implementation: findings to reconcile first

The public repository already contains useful foundations:

- `crates/domains/symthaea-exoskeleton/src/` contains the six-channel research simulator, intent-estimation path, embodiment, training and full-frame research simulator.
- `symthaea-exoskeleton-hal-series-72/` contains design evidence for fail-closed startup, independent fault handling, mechanical envelopes, wearer-specific assistance limits, shadow-only adaptive assistance, degraded-mode orchestration and control-trace replay.
- The newer safety documents explicitly state that software gates do not certify a wearable device or authorize physical actuation.

Before these paths are represented as one coherent platform, resolve the following apparent inconsistencies:

1. **Φ and physical authority:** the top-level `full_frame.rs` describes `set_consciousness(phi)` as changing motor gain/stiffness; `FullFrameCallback::modulate_force` scales a supplied force vector and `friction_multiplier` changes friction as a function of motor gain. The series-72 evidence gate says Φ must never scale gravity, contacts, friction or human forces. The safety-gated rule must be canonical: Φ is not a validated actuator-safety signal and must not directly increase or shape physical authority.
2. **Full-frame force accounting:** audit whether all declared human and exoskeleton joint torques are actually mapped into the articulated plant, rather than merely generated as telemetry or represented in comments. Add conservation/reconstruction tests for each force and torque contribution.
3. **Energy accounting:** the top-level simple simulator currently decrements battery state by a fixed per-step amount. It should be reconciled with the evidence requirement to account for actuator mechanical work, electrical/drive losses and idle consumption. A proxy must be labelled as a proxy.
4. **Canonical implementation:** candidate/series snapshots must not be mistaken for the current canonical crate. Identify the authoritative implementation, its exact revision, and which gates are implemented, structurally checked, CI-verified, HIL-verified or externally validated.

These are audit findings, not claims that the newer candidate gates have been independently re-run in this proposal.

## 3. Target system architecture

Separate fast physical safety from adaptive cognition. The cognitive system may request a bounded assistance intent; it must not be the final authority over a motor.

### Layer 0 — actuator-local protection

A physically independent safety path owns the emergency stop, normally-safe power state, watchdog, current/torque/speed/temperature limits, joint travel limits, battery protection and actuator-local interlocks. It must remain effective if Symthaea, the operating system, network, sensor-fusion process or Mycelix service fails.

### Layer 1 — deterministic control

A bounded-latency controller performs the validated low-level impedance/admittance or torque-control loop. It enforces passivity/energy constraints where applicable, torque slew, joint limits, power limits, sensor freshness and a single most-restrictive degraded mode.

### Layer 2 — Symthaea intent and task assistance

Symthaea fuses non-invasive measurements and user commands to estimate movement intent and task context. It proposes a task-level assistance request, confidence and expiry—not unconstrained motor torques. Adaptation is shadow-only until an independently reviewed candidate passes the complete validation path.

Candidate sensing modalities: joint encoders, IMUs, foot-pressure/contact sensing, load cells, EMG and mechanomyography. Ultrasound is a research option. Invasive neural interfaces are explicitly not a prerequisite for useful first products.

### Layer 3 — Symtropy physical twin

Symtropy supplies the deterministic plant, contact and articulated-body simulation, sensitivity analysis, actuator/thermal/energy models, adversarial disturbance campaigns and hardware-in-the-loop test fixtures. Model discrepancy must be measured against benchtop observations; simulation outputs are not human-subject evidence.

### Layer 4 — Sol-Atlas geometry and configuration

Sol-Atlas owns the spatial representation: wearer/device geometry, joint-axis alignment, link dimensions, range-of-motion envelopes, collision clearance, environment geometry and fit/configuration revisions. A geometric model is not a clinical fit approval.

### Layer 5 — Mycelix provenance and consent

Mycelix can anchor device identity, component and material provenance, calibration revision, signed test receipts, maintenance, software/model versions, consent and authorized roles. It must not become a remote actuation authority. Wearer stop remains local and immediate; a network partition must not prevent a stop.

## 4. Functional benchmark suite

Benchmark distinct activities instead of optimizing only one idealized walking cycle. Keep the first software campaign deterministic and synthetic; add physical and human evidence only through separately approved stages.

| Task family | Scenarios | Primary measurements |
|---|---|---|
| Level mobility | start, stop, steady walking, cadence changes | speed, step timing, symmetry, joint work, intent error |
| Transitions | sit-to-stand, stand-to-sit, kneel/return where appropriate | completion, time, stability margin, peak interaction torque |
| Terrain | ramps, stairs, curbs, uneven ground, surface transitions | task success, foot clearance, balance margin, mode-transition errors |
| Perturbations | push/pull disturbances, trips, slips, unexpected stops | recovery response, unintended assistance, stop latency, residual authority |
| Daily activities | turning, reaching, carrying a light object, doorway negotiation | task completion, interference, user effort, collision/clearance |
| Degraded operation | sensor dropout/drift, clock fault, stale packets, actuator loss, low battery, thermal warning | fault detection, authority reduction, latching, recovery procedure |
| Long-duration use | fatigue, fit migration, repeated don/doff, battery depletion | drift, discomfort, interface pressure, energy/distance, service burden |

### Outcome domains

**Human function:** task completion and independence; 10-Meter Walk Test, 6-Minute Walk Test, Timed Up and Go and Five-Times Sit-to-Stand where clinically appropriate and selected by the study team.

**Physiology and effort:** metabolic rate measured with appropriate instrumentation in human studies; normalized EMG and joint kinetics as complementary measures. Simulator-derived metabolic estimates must never be reported as measured metabolic cost.

**Human-device interaction:** interface pressure and shear, alignment error, relative migration, range-of-motion restriction, interaction torque, discomfort, perceived agency, confidence and cognitive workload.

**Machine performance:** mass, actuator bandwidth, command latency and jitter, torque-tracking error, energy per distance/task, runtime, thermal rise, acoustic noise, donning/doffing time, adjustment count, maintainability and component cost.

**Safety and reliability:** emergency-stop response, residual torque after stop, watchdog response, passivity/energy-budget violations, sensor disagreement handling, false intent activations, missed intent transitions, falls/near-falls, component faults and recovery time. Report each safety violation individually; never average it away inside a composite score.

## 5. Experimental protocol and comparison discipline

For every claim:

1. Freeze the device revision, wearer profile, calibration, firmware, controller, model, scenario, environment, objective weights and analysis method.
2. Compare against at least: unassisted condition; a fixed, non-adaptive controller; and the strongest relevant published or commercial baseline that can be tested fairly.
3. Use matched task conditions and, when human research is approved, appropriately designed randomized/crossover comparisons with participant-level results.
4. Publish confidence intervals, sample size, exclusions, missing data, adverse events, subgroup results and every failed scenario.
5. Keep synthetic, benchtop, HIL, healthy-volunteer, patient and field evidence as distinct evidence classes.
6. Preregister any composite objective. A change to objective weights creates a new benchmark version.
7. Do not claim “beats SOTA” from a simulator-only result, a single participant, a non-matched comparison or a result without independent replication.

### Release stages

- **B0 — software invariants:** deterministic replay, property/fuzz testing, finite-value checks, sequence/timestamp validity, exact applied-command capture, fail-closed startup and zero-authority stop invariants.
- **B1 — plant credibility:** dimensional/unit checks, torque and energy reconstruction, parameter sensitivity, model comparison against reference data, and explicit model-discrepancy reporting.
- **B2 — bench/HIL:** calibrated sensors and actuators, electrical/thermal/mechanical limits, watchdog/stop fault injection, communication faults, power interruption and actuator failure. Independent hardware safety remains authoritative.
- **B3 — ergonomics/usability:** fit, pressure, migration, donning/doffing, task interference and user-reported agency, only under the appropriate ethics and safety approvals.
- **B4 — clinical or occupational validation:** approved study protocol, qualified clinicians/ergonomists, prespecified outcomes, participant safeguards and applicable regulatory oversight.

No stage implies approval for the next one.

## 6. Proposed controller research: task-level, user-governed assistance

Test a hierarchy rather than one opaque policy:

- A task recognizer estimates walking, transition, stair, ramp, turn, rest and unknown states.
- A user-specific baseline captures comfortable kinematics and effort without presuming that one gait pattern is universally correct.
- A bounded controller proposes assistance profiles for each task and joint.
- A safety supervisor intersects that proposal with the mechanical, wearer, calibration, thermal, energy and timing envelopes.
- Loss of confidence or disagreement reduces authority immediately. Increases require stable evidence, bounded increments and explicit acknowledgement.
- The wearer can reduce or stop assistance at any time; no learned policy can override that input.

Research hypotheses to compare experimentally:

1. **Task-aware assistance** will outperform a single gait-cycle policy on transitions and mixed terrain.
2. **Hybrid soft/rigid transmission** can improve comfort and alignment tolerance without sacrificing required support.
3. **Adaptive impedance** can improve assistance only when its shadow-evaluated objective includes wearer effort, tracking, discomfort, stability and actuator effort.
4. **Personalized geometry and calibration** can reduce fit error and interface loads.
5. **Multimodal intent sensing** can reduce false activation and cognitive burden versus any single sensor modality.
6. **Local repairable modularity** can improve availability and lifetime cost in settings where imported specialist service is difficult.

Each is a hypothesis to test, not a claimed result.

## 7. Initial R&D portfolio and priority

| Priority | Capability | Why it matters | Initial route |
|---|---|---|---|
| P0 | Multi-activity lower-limb assistive exosuit | Mobility, transfers, terrain and independence | Extend the existing exosuit and build B0–B2 evidence |
| P1 | Upper-limb/shoulder assistance | Reaching, lifting and activities of daily living; potential occupational use | Soft support with bounded elbow/shoulder assistance |
| P1 | Dexterous low-cost prosthetic hand with sensory feedback | Restores useful manipulation and makes touch/force observable | Non-invasive EMG control plus skin-based vibrotactile feedback first |
| P1 | Hand rehabilitation orthosis | Repetitive, measurable therapy and home practice | Tendon-driven soft hand exoskeleton with clinician-supervised profiles |
| P2 | Balance and fall-prevention wearable | Fall risk, confidence and community mobility | IMU-based detection and user-controlled haptic cueing before powered intervention |
| P2 | Adaptive orthosis/FES interface | Supports selected neurological and musculoskeletal rehabilitation needs | Research with clinical partners; separate electrical and clinical safety case |
| P3 | Invasive BCI/neural implants | Potential sensory and motor restoration, but high clinical, surgical and regulatory burden | Defer until non-invasive products and translational partnerships mature |

The first product programme should optimize for a meaningful capability floor, affordability, repairability and user choice—not superhuman strength.

## 8. Paradigm shifts worth pursuing

1. **From joint torque to task capability:** optimize a person's real activity, not a motor's output.
2. **From device-centric to wearer-governed:** assistance is negotiated, bounded and revocable, not imposed.
3. **From one-size-fits-all to a verified personal twin:** geometry, calibration and assistance are versioned per device and wearer.
4. **From opaque online learning to evidence-gated adaptation:** test candidates in shadow, replay and HIL before any controlled release.
5. **From prototype to serviceable infrastructure:** publish repair instructions, spare-part interfaces, provenance and lifecycle cost.
6. **From isolated device to interoperable assistive ecosystem:** wheelchair, orthosis, prosthesis, exosuit and communication aids should exchange consented, minimal data through explicit interfaces.
7. **From performance-only to life outcomes:** independence, fatigue, comfort, participation, ownership, cost and service availability count as first-class requirements.

## 9. Standards and regulatory workstream

The applicable path depends on intended use and jurisdiction. Do not assume that a recreational/industrial wearable and a medical rehabilitation exoskeleton follow the same route.

Starting references:

- [ISO 14971:2019 — Medical device risk management](https://www.iso.org/standard/72704.html), reviewed and confirmed current in 2025.
- [ASTM F3474-25 — Functional ergonomic parameters and test metrics for exoskeletons](https://store.astm.org/f3474-25.html).
- [FDA 21 CFR 890.3480 — Powered lower-extremity exoskeleton](https://www.ecfr.gov/current/title-21/section-890.3480). FDA identifies this as a Class II prescription-device category for medical-purpose powered lower-extremity exoskeletons.
- [FDA product classification PHL](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpcd/classification.cfm?ID=4210), which lists IEC 80601-2-78 and ISO 10328 among recognized consensus standards.
- [SAHPRA classification guideline SAHPGL-MD-04, Version 5](https://www.sahpra.org.za/document/guideline-for-classification-of-medical-devices-and-ivds/) for South African market planning.

A regulatory specialist must establish the actual classification and conformity route for each intended use before claims, trials or commercialization.

## 10. Immediate work sequence (independent of CI runners)

1. Reconcile the top-level full-frame model with the series-72 safety contract; prevent Φ from changing physical forces or friction.
2. Audit torque injection and energy accounting; add dimensional/unit and reconstruction tests.
3. Extract one authoritative safety-gated controller path and identify copied candidate snapshots as historical evidence, not active implementation.
4. Implement B0 task/scenario fixtures and a machine-readable benchmark manifest with frozen inputs and replayable outputs.
5. Add B1 plant credibility tests and comparison harnesses; keep all synthetic metrics explicitly labelled.
6. Define the benchtop and HIL instrumentation/BOM before ordering or connecting actuators.
7. Recruit assistive-technology users, rehabilitation clinicians, biomechanists and manufacturing partners to set meaningful outcomes and review prototypes before any human trial.

## 11. Evidence record schema

Every benchmark run should carry:

- repository, commit and build identity;
- model, controller and firmware revisions;
- device and wearer-profile identifiers (pseudonymous);
- calibration identity and validity interval;
- sensor/actuator configuration and units;
- scenario ID, seed, environmental assumptions and frozen parameters;
- raw-data references with consent and retention policy;
- measured results with uncertainty and missingness;
- faults, interventions, stop events and safety-envelope outcomes;
- trace digest, replay result and independent reviewer disposition;
- evidence class: synthetic, simulation, bench, HIL, human factors, clinical or field.

Do not put raw biometric streams or high-dimensional cognitive state into default public evidence bundles. Publish the minimum necessary derived evidence, with explicit consent and access controls.

## 12. Success criterion

The programme succeeds when an independently reviewable device revision demonstrates a prespecified, meaningful improvement in a real task for a defined user group, without unacceptable deterioration in safety, comfort, agency, cognitive burden, reliability or cost—and the result can be reproduced from the exact evidence bundle.

A higher Φ, a stronger motor, a larger model or a successful CI run is not, by itself, evidence of improved human capability.
