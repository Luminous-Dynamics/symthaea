# Open Mobility — Practical Wearability and Lifecycle Benchmark v0.1

**Status:** research protocol draft  
**Tracking issue:** [MOBILITY-ASSIST-006 / #7850](https://github.com/Luminous-Dynamics/symthaea/issues/7850)  
**Related work:** [#7766 disability-first mobility benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7766), [#7774 fitness-preserving assistance](https://github.com/Luminous-Dynamics/symthaea/issues/7774), [#7784 dual-lane benchmark harness](https://github.com/Luminous-Dynamics/symthaea/issues/7784), [#7832 intent-probability calibration](https://github.com/Luminous-Dynamics/symthaea/issues/7832)  
**Claim ceiling:** synthetic/replay evidence only; no clinical efficacy, hardware-safety, regulatory-compliance, or flight-readiness claim.

## 1. Why this benchmark exists

A wearable robot that performs well after a technician fits it in a lab may still fail its purpose if the user cannot don it, align it, transfer into it, tolerate it, charge it, clean it, or recover from a fault without another person. The product boundary should therefore include the full wearer workflow, not just the actuator and gait controller.

Recent evidence motivates that shift:

- A 2026 study examined setup and takedown of four occupational exoskeletons with 29 participants and analyzed how procedure complexity, part count, and usability problems affect deployment: https://pubmed.ncbi.nlm.nih.gov/42048400/
- A 2026 passive-exoskeleton field trial with 17 healthcare professionals reported ease of use but variable comfort and movement restriction during clinical work: https://doi.org/10.1016/j.apergo.2026.104746
- A 2026 home-use study involving seven people with paraplegia after extended use identified caregiver availability as a dominant factor in potential abandonment: https://www.nature.com/articles/s41394-026-00739-z
- A 2026 review of human–robot interface mechanics highlights pressure/shear, interface migration and joint-axis misalignment as concerns for comfort and tissue injury: https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1733173/full
- A home/community study of the Myosuit identified donning difficulty as a barrier to independence and frequent use: https://pmc.ncbi.nlm.nih.gov/articles/PMC9714034/
- A randomized clinical trial involving 161 veterans with paralysis found low use and no significant patient-reported mental/physical health differences for wheelchair plus exoskeleton versus wheelchair alone over four months; the authors noted that safe modifications reducing companion requirements may improve personal use: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2823146

These findings motivate what to measure. Small or population-specific studies do not establish universal thresholds.

## 2. Paradigm shifts

### 2.1 Fit is a dynamic state, not a one-time measurement

Static fit at startup is insufficient. Limb posture, soft-tissue movement, clothing, straps, fatigue, sweat and repeated motion can change the physical interface. The system should maintain a *fit-confidence report* based on actual measured or manually verified evidence and distinguish it from model inference.

The report may include device-to-body alignment quality, interface migration, fastening state, sensor consistency and the validity window of the last fitting check. Uncertain fit must not authorize more power or resistance. This report is advisory; it must not pretend software can infer skin integrity or safe pressure without suitable sensing and validation.

### 2.2 Treat the body interface as a critical subsystem

Cuffs, straps, braces, textiles and load paths transmit actuator forces to the wearer. Optimize comfort and mechanics together. Joint self-alignment and compliant attachment can reduce the consequences of anatomical variation, but add mass, complexity and new wear parts. Evaluate the complete Pareto trade-off rather than simply maximizing stiffness or minimum mass.

Where pressure/shear is instrumented, retain spatial distribution, peak values, duration and location—not only a whole-session average. No universal pressure/shear limit is asserted here. Establish task- and population-appropriate limits with biomechanical, clinical and regulatory experts before physical testing.

### 2.3 Independence is an end-to-end capability

Model the entire journey: device retrieval, transfer or seating, donning, fit confirmation, startup, mode selection, task completion, recovery, doffing, cleaning, charging and storage. Support needs vary. Report wearer-only and assisted workflows separately; a user who needs help should not be characterized as failing.

### 2.4 Degraded operation must be designed, not improvised

Battery decline, fit drift, sensor disagreement, communication loss, reset and actuator faults need an explicit state machine and independently reviewed platform response. The policy layer may report a fault or request support, but may not substitute for a hardware-level risk control. A reconnected cognitive model must not silently restore a stale movement or resistance request.

### 2.5 Serviceability is mobility

An unavailable fastener, proprietary connector, undocumented calibration or difficult-to-replace liner can make an otherwise capable system unusable. Measure fault isolation, module change, required tool/skill, re-calibration, return-to-service and consumable availability. Repairability and support burden belong in the benchmark beside gait performance.

### 2.6 Optimize real task completion, not just lab range

A device should be tested on declared in-scope environments: doorway transitions, sit/stand or transfer steps, turns, reaching while seated, clothing transitions, surface changes and task interruptions. Each platform profile must state which environments it has evidence for. Unknown or out-of-scope environments are not silently treated as supported.

### 2.7 Keep products physically distinct

Share typed intent, evidence, policy and receipt contracts where useful across rehabilitation, personal mobility, industrial load support and space/EVA systems. Keep their mechanical architectures, failure analyses, environment assumptions and qualification programs separate. This protocol does not qualify flight or life-support hardware.

## 3. Wearer lifecycle map

Every benchmark profile should map these stages to explicit preconditions, critical steps, completion criteria, time budget and support needs.

1. **Prepare:** retrieve the device, confirm power and required consumables, identify the intended profile.
2. **Transfer/position:** make the device accessible from the intended starting posture and transfer method.
3. **Don:** engage and fasten all applicable attachment points; the user must be able to distinguish incomplete from complete fastening.
4. **Fit check:** verify compatible size/profile, fit/alignment evidence, strap/latch state and sensor agreement.
5. **Start/mode select:** show the active profile and any limitation; do not inherit a prior session's training resistance.
6. **Perform task:** log outcomes, interruptions, user-reported comfort, fit changes and assistance events.
7. **Degrade/recover:** exercise loss-of-sensor, low-energy, communication, reset, increased-support, stop and restart scenarios.
8. **Doff:** disengage without requiring an unsafe posture or a hidden software sequence; test profile-appropriate independent and assisted workflows.
9. **Maintain:** clean, inspect, replace approved wear parts, charge or swap the battery, isolate a simulated fault and verify re-calibration.
10. **Store/restart:** safe storage and restart without stale configuration or residual action requests.

A successful software state transition is not proof that a physical latch, stop, brake or safe state occurred. On real hardware, these endpoints require independent sensing and engineering validation.

## 4. Scenario families

| Family | Seeded cases | Negative controls |
|---|---|---|
| FIT-STATIC | Fit profile chosen for body/garment dimensions; manually verified alignment | Incompatible profile; fit not confirmed; contradictory latch/fit evidence |
| FIT-DYNAMIC | Repeated movement with interface migration and changed posture | Drift exceeds declared profile tolerance; fit evidence becomes stale |
| INTERFACE-LOAD | Synthetic pressure/shear maps over time; strap tension and load-path traces | Hotspot hidden by average; sensor dropout; peak duration omitted |
| DON-DOFF | Wearer-only, seated, one-hand where relevant, assisted and caregiver-unavailable workflow | Skipped critical step, ambiguous closure, unreachable release |
| TRANSITIONS | Sit/stand, reach, doorway, turn, permitted uneven surface and task change | Unsupported terrain, blocked transition, route or posture constraint violated |
| ENERGY | Task-conditioned battery draw, reserve estimation, warning and swap | Sudden depletion, underestimated load, stale battery estimate |
| FAULT-RECOVERY | Sensor disagreement, stale data, connection loss, reset and restart | Stale request restored, failure masked, unsafe default or missing receipt |
| SERVICE | Cleaning, liner/cuff replacement, connector cycle, battery/module service | Wrong part, undocumented tool, calibration lost, fault cleared without verification |
| AGENCY | Mode switch, increased assistance, stop, recovery and cancellation | Mode changed without user action; stop is swallowed; training settings leak into mobility |
| LONG-SESSION | Fit, comfort, thermal and battery observations over simulated/approved duration | Session average hides a transient threshold violation or repeated complaint |

Use deterministic synthetic and replay traces first. Human-participant protocols require appropriate ethical review, accessible consent, domain expertise and regulatory review as applicable. Physical tests require a platform-specific safety plan.

The starter [scenario corpus v1](./fixtures/wearability-lifecycle-scenarios-v1.json) contains 12 named positive and negative replay cases for verified fit, incomplete fastening, dynamic fit drift, missing/conflicted interface evidence, caregiver unavailability, doff requests, restart-boundary configuration, task-energy reserve, calibration invalidation after service, timeouts and missing measurements. The fixture states expected software/report dispositions, not physical actions. Add cases only with stable identifiers and preserve positive controls.

## 5. Metric dictionary

Keep dimensions separate. Every report must publish definitions, units, numerator/denominator, aggregation, exclusions and uncertainty method.

| Dimension | Metric |
|---|---|
| Donning/doffing | Completion rate; median/p90 time; step count; critical-step omissions; prompts; reattempts; user-only versus assisted completion |
| Transfer/setup | Success rate and time by initial posture, mobility context and support profile; unsafe or abandoned transfers |
| Fit quality | Alignment error where measured; fit-check failure; interface migration over time; fit adjustment frequency |
| Interface mechanics | Spatial pressure/shear distribution, peak and exposure duration, load-path repeatability, actuator output versus transmitted interface load |
| Comfort | User-reported discomfort/pain, localized hotspots, perceived restriction, thermal comfort, skin/interface observations and reasons to stop |
| Mobility | Task success, completion time, stability/perturbation response, trajectory or posture error, compensatory motion where reliably measured |
| Autonomy/agency | User overrides, unwanted initiations, confirmations, cancellations, recovery after misunderstood intent, ALI if distributions are justified, plus subjective agency separately |
| Energy | Wh/hour, Wh/task, task-conditioned range distribution, reserve prediction error, charging/swap time, low-energy warnings and safe transition outcomes |
| Reliability | Failures per exposure hour or task; false alarms; missed fault detection; availability; recovery success/time; configuration drift |
| Serviceability | Diagnosis success, time to isolate, tool/part count, repair duration, recalibration needs and return-to-service verification |
| Independence | Workflow completion without a helper where the profile claims self-service; caregiver time; unavailable-caregiver impact; abandonment reasons |
| Lifecycle cost | Consumables, service hours, battery replacement, downtime, repair travel, training and total cost per successful use |
| Evidence quality | Expected versus present traces, measurements and receipts; source/config digest; unmeasured fields and exclusions |

## 6. Candidate target gates

These targets are proposed research gates, not established standards. Freeze them per profile before testing.

1. **Critical-step integrity:** zero seeded bypasses of required fit, latch, stop or release checks in the declared software/replay corpus. Preserve positive controls to avoid passing by rejecting all cases.
2. **Don/doff usability:** test an initial hypothesis of at least 90% first-attempt workflow completion inside a predeclared time budget without skipping critical steps. Stratify by profile and support requirements; never pool a group into a misleading overall pass.
3. **Independent-use claim:** if a product profile claims self-service, every required lifecycle stage must have a separately scored wearer-only scenario. Caregiver support remains a valid pathway but is reported separately.
4. **Emergency disengagement:** zero misses in the seeded software/harness cases; later physical release time, required force and endpoint sensing must have hardware-specific acceptance criteria approved by the responsible experts.
5. **Fit/comfort:** no invented universal safe pressure threshold. Instrument the interface, publish distributions and transient events, and set physical limits before tests using relevant expertise and standards.
6. **Dynamic fit:** zero silent promotion to a higher-assistance/resistance envelope when fit evidence is stale, invalid or conflicted; all transitions and causes appear in the receipt.
7. **Power budgeting:** report range prediction error and reserve across declared task/load/environment profiles. Never claim range from state-of-charge alone; the low-energy transition must be exercised under seeded degraded conditions.
8. **Serviceability:** for a nominated non-safety-critical field-replaceable module, test a provisional goal of replacement within ten minutes using documented tools, with configuration/calibration lineage preserved. This is a design target, not a universal industry threshold.
9. **Reliability:** publish failures per task/hour and confidence bounds; include all aborted and repaired runs. Do not treat absence of observed failures in a tiny test set as proof of zero field risk.
10. **Adoption:** report use frequency, reason for non-use, discomfort, independence, caregiver dependency and maintenance burden in any ethically approved field evaluation. Task performance alone is not adoption.
11. **Evidence traceability:** every result binds exact source/build/profile/corpus revision, configuration, measurement calibration, environment, timing semantics, denominators, exclusions and evidence level.

## 7. Architecture implications for Symthaea

The cognition layer may infer intent, summarize evidence and propose adjustments. It must not directly authorize physical force, resolve safety conflicts in its own favor, or authenticate its own measurements.

**Current reference implementation:** draft [PR #7865](https://github.com/Luminous-Dynamics/symthaea/pull/7865) adds a report-only Rust evaluator with typed lifecycle stages and evidence states, deterministic dispositions, and 15 unit tests plus three public-API integration tests against 18 versioned scenarios. Its additional task gates reject an explicitly unsupported task, restricted space, or incompatible hygiene workflow, and request review when thermal discomfort is reported, duty-cycle limits are reached, or compatibility is unknown. Tests have not been executed; the count describes authored tests, not test results.

The evaluator is deliberately a report, not physical control authority. It cannot determine tissue-safe pressure, prove a physical latch, validate battery range, or execute emergency doffing. Its declared inputs remain caller assertions unless separate validated measurement evidence is attached.

**Further implementation work:**

- Extend lifecycle receipts with source/timestamp, session identity, per-finding reason codes and calibration provenance.
- Add a benchmark manifest that distinguishes caller assertion, sensor measurement, independent observation and operator annotation.
- Expand deterministic replay cases for task transitions, real-world confined-space restrictions, cleaning/hygiene compatibility, repeated don/doff cycles and comfort over duration.
- Reuse existing embodiment/mobility contracts where possible rather than adding a second physical command API.

Reuse existing embodiment and mobility contracts where possible. Do not add a second physical command API. Keep new types report-only until ownership and integration points are reviewed.

## 8. Experiment sequence

1. Protocol review: define which platform profile and wearer workflows are actually in scope.
2. Synthetic fixture schema and deterministic validator.
3. Replay harness with a simple baseline before learned optimization.
4. Interface/fit and workflow sweeps using declared body/profile parameters.
5. Battery/task profile and fault-injection sweep.
6. Measurement-system study: confirm that pressure, shear, migration, range and energy measures are calibrated and repeatable.
7. Bench/HIL under an approved platform-specific plan.
8. Human-use and longitudinal adoption studies under appropriate ethical, clinical and regulatory oversight.

## 9. Non-claims

This protocol does not establish physical safety, clinical efficacy, tissue-safe pressure thresholds, regulatory compliance, universal independent donning, fitness benefits, or support for all terrains/users. It does not qualify aerial mobility or EVA. A deterministic receipt makes evidence traceable; it does not make an unmeasured event true.

The intended outcome is a wearable system people can put on, trust, comfortably use, maintain and remove—while retaining control and choosing the level of help that serves their lives.
