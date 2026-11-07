# Task-Oriented Seakeeping for the Liveaboard — Protocol v0.1

**Status:** research / benchmark protocol proposal. No hydrodynamic prediction, comfort threshold or vessel approval is claimed.  
**Parent:** [MARINE-LIVEABOARD-001 #10926](https://github.com/Luminous-Dynamics/symthaea/issues/10926)  
**Workstream:** [MARINE-HABITABILITY-SEAKEEPING-001 #10966](https://github.com/Luminous-Dynamics/symthaea/issues/10966)

## 1. Research thesis

Evaluate seakeeping as a property of the **vessel–occupant–task–location–exposure system**, not as one hull-level scalar.

A design can have an acceptable global RMS value while a berth, galley, companionway, helm or workshop remains unsuitable in a particular operating condition. A good result in the saloon cannot compensate for a failed critical task elsewhere.

The evaluation chain is:

\`mission → task → posture → exact occupied location → exposure → local motion → task outcome\`

This work consumes appropriate validated seakeeping outputs and physical observations. It does not implement another hydrodynamic solver or infer task performance directly from unvalidated global motion numbers.

## 2. Research anchors and state of practice

1. **Zu et al., first published online 27 February 2026**, propose translating vessel functional requirements into motion-critical tasks, then selecting task-specific seakeeping criteria. The paper treats motion-induced interruptions (MII), motion-induced fatigue (MIF), and motion-sickness incidence (MSI) as distinct outcomes. Its case studies are workboats with crew tasks, so it provides a method—not ready-made liveaboard thresholds. https://journals.sagepub.com/doi/10.1177/14750902251412460
2. **Mauro et al., Ocean Engineering 342 (30 December 2025), 122825**, describe a concept-stage motion-induced comfort model for small cruise ships. Their study uses a 25-ship hull database, predicts heave/pitch transfer functions in head seas and reports verification on an external ship. This is a stronger model-based comparison anchor than a single-hull anecdote, but cruise ship results do not automatically transfer to small yachts, multihulls or liveaboard interiors. https://doi.org/10.1016/j.oceaneng.2025.122825
3. **ISO 2631-1:1997 + Amd 1:2010** remains the published whole-body vibration standard as checked 7 November 2026; the ISO record says it was confirmed in 2021 and is expected to be replaced by ISO/FDIS 2631-1. It gives measurement principles and informative guidance for health, comfort, perception and motion sickness. It is not a complete ship habitability method or a substitute for a task-specific risk assessment. https://www.iso.org/standard/7612.html
4. The 2026 DIS preview for ISO 2631-1 distinguishes the frequency domains 0.5–80 Hz for health/comfort/perception and 0.1–0.5 Hz for motion sickness. This is a draft preview and must not be represented as the content of a yet-unpublished replacement edition. https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso%3A2631%3A-1%3Adis%3Aed-3%3Av1%3Aen

## 3. What is measured

### Vessel-level response
Capture six-degree-of-freedom motion and the reference frame:

- surge, sway, heave;
- roll, pitch, yaw;
- position, heading, speed, loading, draft and trim;
- encountered wave spectrum, direction, period/frequency, wind, current and water depth.

### Local occupied-point response
At each exact berth, seat, standing position, workstation or deck location, capture:

- local translational acceleration in vessel-fixed axes;
- angular velocity and angular acceleration;
- sensor orientation, mounting method and distance/offset from the declared point;
- frequency-weighted whole-body measures and dose measures only where the selected method is applicable;
- exposure start/end, missing samples, sample gap and sensor calibration evidence.

Vessel motion is not identical to local occupant exposure. Coordinates, transfer path, posture, support surface and occupied location must be bound to the exact design revision.

### Task outcomes
Do not equate a motion metric with a human outcome. Keep separate:

- MII: slips, loss of balance, tipping, involuntary bracing/regrip, object displacement, interruptions and recovery time;
- MIF: fatigue or degraded task performance only where an adequately supported method is available;
- MSI: motion-sickness outcome or validated predicted incidence within the model's actual population, frequency, posture and exposure domain;
- task effectiveness: completion time, errors, interrupted time and successful completion under the declared condition;
- occupant report: comfort, nausea, sleep disturbance and perceived safety, with sample size, protocol and uncertainty.

Self-report is useful evidence, but it is not a calibrated accelerometer or proof of structural safety.

## 4. First liveaboard task map

The machine-readable task matrix covers nine task families: sleep/rest, eating/social, cooking, personal care, circulation, work/learning, helm/navigation, workshop/maintenance and deck/emergency tasks.

Every task record must declare:

- mission profile and operating condition;
- exact design / interior revision;
- vessel architecture and exact location identity;
- posture and support / handhold assumptions;
- duration, frequency and exposure window;
- task criticality and consequences of interruption;
- which motion-induced effects are relevant;
- chosen evaluation method, validation domain and threshold source;
- evidence status and unknowns.

For multihulls, do not collapse the port/starboard demi-hulls, bridge deck, cross-deck and central accommodation into one location. For monohulls, preserve fore/aft/centreline and vertical deck level. For trimarans, preserve main hull and ama locations.

## 5. Criteria register and threshold discipline

The criteria register records published and literature-derived examples with the context in which they were reported. Examples include:

- MII/min risk bands reported in the 2026 task-oriented paper (0.1 possible, 0.5 probable, 1.5 serious, 3 severe and 5 extreme), adapted from earlier work for particular crew activities;
- context-specific motion-sickness and fatigue/physical-task criteria summarised from Nordforsk, MARIN and ISO-related literature in the 2026 paper;
- pleasure-vessel / cruise-ship criteria collated in Mauro et al. (2025), such as MSI, motion-induced response, vertical/lateral acceleration, roll and vibration-dose criteria.

These numbers are not one interchangeable scale. They refer to different populations, tasks, vessels, exposure windows and methods. The register marks them as **reference context only**. No LOS-18 acceptance limit is set until a qualified human-factors / naval-architecture review selects a method and applicability basis.

## 6. Evaluation and comparison ladder

Compare four approaches using the same frozen hull, loading, sea states and occupied-point roster:

1. **Global scalar baseline:** a declared RMS / peak response metric, to demonstrate what it misses.
2. **Standard-method baseline:** applicable frequency-weighted whole-body / motion-sickness measurement, using the current published edition and correct exposure domain.
3. **Task-oriented evaluation:** task, posture, location and exposure specific MII / MIF / MSI / task-effectiveness criteria.
4. **Validated reference:** compatible model-test, sea-trial or external-ship data with exact geometry, condition and measurement provenance.

Report by observable, location, task and condition:

- coverage of required tasks/locations;
- task-operability time windows;
- MII/min or task-interruption rate where the chosen method is valid;
- MSI only where a validated MSI method applies;
- task completion time, error rate, recovery time and handhold/support dependence;
- human-reported outcome and sampling uncertainty;
- model bias/error and prediction-interval coverage;
- worst-location and worst-condition results;
- false-operable, out-of-domain, unknown and missing-evidence dispositions.

Do not pool heave acceleration, roll angle, MSI, task completion and subjective comfort into one weighted score. Keep safety-critical emergency and deck tasks as independent hard gates.

## 7. Benchmark targets — research goals, not results

- 100% of declared motion-critical tasks bind an exact design revision, location, posture, exposure and operating condition.
- 100% of required sensor channels and trial dispositions remain accounted for; no silent omission of failed trials or difficult locations.
- Zero tasks labelled operable when required evidence is stale, missing, outside support or below a reviewed hard threshold.
- 100% disjoint development and held-out validation groups, grouped by independent hull / model-test / sea-trial rather than time samples.
- Report bias, MAE/RMSE where meaningful, empirical interval coverage and worst-stratum error separately for each observable and location.
- Reproduce an adopted MSI/MII reference method on its validated domain before proposing a numerical target for LOS-18.
- No universal comfort MAE target until compatible reference data and human-factor criteria are selected.
- Zero emergency-egress or safety-critical deck-task clearances inferred from a comfort score.

## 8. Falsification suite

- Global RMS passes but the galley or companionway has repeated motion interruptions: location/task remains failed.
- Saloon is acceptable but a berth produces high exposure under a validated MSI method: berth task fails independently.
- Candidate passes one speed/heading but misses another frozen sea-state stratum: envelope incomplete.
- Pooled prediction error is low but worst-location error is high: disclose conditional failure.
- Same hull or model-test group appears in both development and holdout under different labels: leakage.
- Missing sensor calibration, orientation, time alignment, location or exposure window: affected result indeterminate.
- Acclimatization assumptions are altered after labels are observed: comparison invalid.
- Weighted score hides failed emergency deck / egress task: reject the aggregate pass.

## 9. Ownership boundaries

- Symthaea marine physics / validated solver owners supply applicable vessel response, uncertainty and validation artifacts.
- Symthaea task benchmark owns the task/location/effectiveness contract and evaluation policy.
- Symtropy supports geometric occupancy, access, reach, furniture/service clearances and scenario visualization; its rigid-body engine is not a substitute for validated seakeeping analysis.
- Qualified naval architects and human-factors specialists review methods and thresholds.
- Physical human-participant studies require appropriate ethics review, informed consent, safe protocols and privacy handling.
- Mycelix may preserve approved provenance/inspection records; it does not turn a self-report or signed record into scientific truth.

## 10. Claim ceiling

A schema pass establishes only that the task/evidence contract is internally consistent. A model-validation pass establishes performance only on the declared validation domain. Neither establishes habitability, comfort for all occupants, health outcomes, fatigue prediction, injury risk, seaworthiness, stability, structural adequacy, legal compliance, class/flag approval or authority to operate.
