# Display Metrology Reproducibility and Decision-Capability Protocol

**Protocol:** ODRH-MET-001 v0.1.0  
**Parent:** [ODRH-001 #8030](https://github.com/Luminous-Dynamics/symthaea/issues/8030)  
**Research program:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)  
**Manufacturing route owner:** [VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012)  
**Status:** proposed protocol; no physical measurements or calibrated-instrument claims

## 1. Core shift: benchmark the measurement system before benchmarking a display

A measurement result is useful only when its instrument, setup, procedure, uncertainty and decision threshold allow the result to distinguish the candidate from the baseline. A benchmark can be repeatable and still be wrong; two instruments can agree while sharing a calibration or geometry bias. Therefore this protocol treats **measurement capability and reproducibility as first-class outputs**, not footnotes.

Use the latest applicable International Committee for Display Metrology (ICDM) Information Display Measurements Standard (IDMS) procedures as the starting point for display-specific measurements. The ICDM site lists IDMS v1.3 (2025) and indicates that v1.4 is targeted for 2028. This project does not reproduce or replace IDMS: for each applicable measurand, record the exact method/edition used, any adaptation, and the reason the procedure is applicable. If no relevant published procedure can be applied, state that gap rather than invent a standard method.

Reference: [ICDM — IDMS and current editions](https://www.sid.org/Standards/ICDM/ICDM).

## 2. Freeze a measurand contract before collecting comparison data

Each metric must declare:

- physical quantity / measurand and unit; the capability profile for every instrument used by that observation must declare the same measurand and unit, not merely a matching label;
- exact method identifier and method edition, or public method reference;
- DUT model, hardware revision and firmware where available;
- video input/host, resolution, refresh, color encoding/range, bit depth, SDR/HDR mode, display OSD and dynamic-feature states;
- exact stimulus/pattern digest, window/patch geometry, average picture level, sequence, dwell/warm-up and settling conditions;
- measurement point, ROI, screen distance, view angle, orientation and surrounding/ambient conditions;
- instrument model/serial, spectral or temporal response needed for that measurand, configuration and calibration identity/expiry;
- sample timing, count, integration/detection algorithm and invalid-sample policy;
- uncertainty budget, confidence/coverage convention, instrument range and expected minimum engineering effect;
- raw-artifact retention path/hash, analysis version, exclusions/aborts, and independent-verifier identity.

For each instrument used by a measured observation, the run record must bind the exact calibration artifact plus its digest, and a separate measurement-capability profile artifact plus its digest. The latter documents operating range, spectral/temporal response or sampling limits, and the measurands for which the instrument is appropriate. A missing, unresolved, role-mismatched, or digest-mismatched artifact prevents the measurement record from validating. A capability profile marked UNASSESSED or INADEQUATE cannot support a measured observation; a declared-but-unreviewed profile may support an exploratory, unqualified measurement but not be promoted to independent qualification.

These bindings make the evidence addressable and mutation-detectable; they do **not** by themselves establish metrological traceability or fitness for purpose. NIST notes that traceability of a measurement result requires a documented, unbroken chain of calibrations, with each contributing uncertainty, and that calibration alone does not establish fitness for a particular decision. [NIST Policy on Metrological Traceability](https://www.nist.gov/calibrations/traceability).

For an exploratory measured record, a declared-but-unreviewed capability profile may be recorded, but its limitations must remain visible and it cannot support an independently reviewed observation. To mark a result INDEPENDENTLY_REVIEWED, every instrument used for that observation must have both a calibration artifact and a capability-profile artifact linked by exact digest, and the capability profile itself must be marked independently reviewed. A successful software check validates these internal links and status relationships only; it does not verify the truth of the source document, reviewer independence, calibration quality, or actual sensor fitness.

A report must distinguish actual input state from requested state. For example, a requested HDR mode does not establish that the signal negotiated or the display operated in HDR.

### Timestamp and calibration-date syntax

The run-record schema labels acquisition/review values as RFC 3339 date-times and calibration expiry as a date. Under JSON Schema Draft 2020-12, `format` is annotation-only by default unless format-assertion behavior is explicitly enabled. Therefore `open-display-platform-verify` also performs an application-level syntax check: impossible Gregorian dates, malformed time/offset fields, missing timezone, invalid fractional-second syntax, and impossible calibration expiry dates must reject. Calibration expiry and acquisition-date comparisons are applied only to syntactically valid values.

This is syntax and internal-consistency validation, not proof that a device was calibrated, that the calibration certificate is genuine, or that its stated validity period is appropriate. RFC 3339 permits a seconds field of `60` for leap seconds; this verifier checks that field's syntax but does not maintain an authoritative leap-second history. A future extension that needs that level of assurance must bind an authoritative time/leap-second policy.

Reference: [JSON Schema Draft 2020-12 Validation — format annotation and assertion behavior](https://json-schema.org/draft/2020-12/json-schema-validation).

## 3. Decision capability before candidate comparison

Freeze the smallest engineering difference that would alter a decision: MDE (minimum decision effect). Build an uncertainty budget for the measurement system and the declared operating conditions.

Proposed instrument-capability gate:

**U95_total ≤ MDE / 3**

where U95_total is the expanded uncertainty at the project's frozen coverage convention, including relevant instrument, geometry, repeatability, drift, signal/setup and analysis terms. The factor of three is a project planning criterion, **not a universal display-metrology standard**. It must be justified against the intended decision and revised if the effect/budget cannot be supported.

If this capability gate cannot be met, report INSTRUMENT_CAPABILITY_INADEQUATE or NOT_EVALUATED; do not soften the quality floor, suppress the uncertainty, or claim the candidate passed.

### Minimum uncertainty-budget record

For the declared measurand, separate the relevant contributors instead of assigning an unsupported global error percentage. Typical terms include reference-instrument calibration, sensor repeatability, remount/geometry variation, spectral/temporal response mismatch, sampling/integration, drift/temperature, DUT variation, and analysis/reducer sensitivity. Include only terms relevant to the measurand, but justify omissions.

If a first-order uncertainty model is appropriate, document sensitivity coefficients and the covariance terms:

`u_c² = Σᵢ(cᵢuᵢ)² + 2Σᵢ<ⱼ Cov(cᵢXᵢ, cⱼXⱼ)`

Then document the method that maps combined standard uncertainty `u_c` to the expanded interval `U = k u_c`, including its coverage convention, coverage factor, distribution/degrees-of-freedom assumptions, and any nonlinear/Monte Carlo treatment. Do not assume all contributions are independent merely because they come from different rows in a spreadsheet. If no defensible combined-uncertainty calculation exists yet, the decision-capability result remains NOT_EVALUATED.

The formula and `U95_total <= MDE/3` are a local gate design; they do not certify an instrument, establish traceability, or replace an applicable IDMS method.

## 4. Repetition and reproducibility design

Separate these sources of variation and record which ones have actually been tested:

1. consecutive measurements in one run;
2. independently restarted runs;
3. remounting/repositioning the sensor and/or DUT;
4. different operators;
5. different sessions/days;
6. different instruments of the same declared class;
7. different sites/labs.

Do not imply multi-operator or cross-lab reproducibility from repeat readings made without changing the setup. Randomize or balance measurement order when drift or warming could bias candidate-vs-baseline comparison; freeze the policy before observing outcomes. Retain every valid reading, failure, abort and deviation.

Report raw observations, count, mean/median as justified, dispersion, intervals, per-stratum results and missingness. Predeclare whether mean, percentile, worst-case, or a model-based estimate decides each engineering gate. Do not select a reducer after seeing which candidate it favors.

For two independent estimates with compatible measurands and the same uncertainty convention, a simple diagnostic comparison is:

**abs(x_A − x_B) ≤ sqrt(U_A² + U_B²)**

when covariance is negligible. If common calibration/reference or other covariance exists, include the covariance term or do not present the simple independent comparison. Passing pairwise agreement is not proof of absolute accuracy: a shared systematic error can make two instruments agree.

### 4A. Control measurements and drift

The executable conservative screening model is documented in [ODRH_DRIFT_SCREENING_ALGORITHM.md](ODRH_DRIFT_SCREENING_ALGORITHM.md) and implemented in `tools/open-display-platform-verify/src/drift_control.rs`. Its output is a control-screening status only; it is deliberately not a physical qualification or a drift-correction model.


A repeatability study can look stable while the instrument drifts slowly enough to bias a candidate-versus-baseline comparison. When a suitable stable control/reference stimulus or artifact exists, define an interleaved schedule before data collection (for example, start/end of each measurement block and after a remount or relevant reconfiguration). Record the control identity, measurand/unit, expected range, calibration/provenance, geometry and uncertainty. Do not call an ordinary DUT remeasurement a traceable reference standard.

If no suitable control is available, mark drift surveillance NOT_EVALUATED. Control results monitor the measurement chain; they do not prove the display itself is stable or accurate.

Freeze the control guard band from the measurand, MDE and uncertainty budget. There is no universal percent threshold for every optical, timing or power channel. Preregister control cadence, test order/randomization, any blind duplicates, and the action on an excursion. Retain the exact order or randomization seed and every failed control.

If a control exceeds the guard band, mark the affected interval NOT_EVALUATED according to the frozen policy and reacquire or investigate. Do not silently interpolate, normalize away or retroactively correct the drift unless that correction model and uncertainty were independently validated and preregistered. Preserve raw traces and deviations.

Synthetic validation cases should cover no drift, drift within the guard band, drift just beyond it, gradual drift crossing between controls, a single outlier, a remount shift, and a control/reference mismatch. Report false-alarm and missed-drift behavior. Passing synthetic cases does not prove that a physical control artifact is stable.

## 4A. Control-schedule coverage

A fixed guard band is not enough if controls are too sparse. Each experimental run must preregister a maximum allowed **planned** control gap, including run-start-to-first-control and last-control-to-run-end. A schedule that exceeds that cap is invalid before comparing DUT results. A missing, invalid or out-of-band control remains explicit and quarantines affected observations under the declared screening algorithm.

The maximum gap is a bound on the schedule, not a claim about drift-detection power. Pick it only after selecting the measurand, reference artifact, expected environmental dynamics, minimum decision effect, and feasible sampling cost. Test false-alarm and missed-excursion performance using synthetic and physical repeat studies before using the algorithm for qualification. Sparse in-band endpoints cannot prove stability between them.

## 5. Pattern/context robustness

A display result must not silently depend on a single conveniently chosen patch. Reuse the existing [pattern robustness and uncertainty protocol](DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md) and frozen conventional-display workload.

Record per-pattern/context outcomes for all predeclared strata relevant to the measurand: patch size and APL, background/preceding pattern, dwell, local dimming behavior, OLED/other temporal response, window position, screen angle and image mode where relevant. Include the required standards-aligned pattern and any diagnostic patterns as different, explicitly identified methods. Custom patterns may characterize a failure mode; they do not silently replace standards-defined stimuli.

The result includes the full distribution and the worst predeclared protected stratum. An average cannot cancel a quality-floor violation in a particular mode or viewing zone.

## 6. Optical / temporal / energy limits

### Optical measurements

- A camera may support geometry, relative comparisons or temporal event detection only to the extent established by its model and calibration.
- Absolute luminance, chromaticity or color difference requires an instrument and calibration/model suitable for that measurand, plus declared geometry and conditions.
- Distinguish the instrument's calibration status from the fixture's geometric repeatability.
- Unsupported quantities are NOT_EVALUATED.

### Temporal measurements

Record input event, application response, render/present timestamps, output timing, optical onset/threshold rule, sensor sample rate, acquisition clock and uncertainty. Distinguish render time from scanout and end-to-end input-to-photon latency. Preserve waveform/edge samples so another reducer can test threshold and filtering sensitivity.

### Energy measurements

Use the existing [visual utility-per-joule protocol](VISUAL_UTILITY_PER_JOULE_PROTOCOL.md). Freeze AC-wall vs DC-rail accounting boundary; do not compare unlike boundaries without a justified model. Preserve raw samples, sampling timestamps, missing/gap handling, integration formula, baseline/task denominator, calibration, idle/background subtraction policy, workload, and uncertainty. Report energy per successful task alongside task success, latency and required quality floors.

## 6A. Conservative field-of-view / uniform-patch fit screen

ODRH now includes an optional geometry screen in `tools/open-display-platform-verify/src/geometry.rs`. Its synthetic input is [display-measurement-geometry-synthetic-v0.1.json](display-measurement-geometry-synthetic-v0.1.json), constrained by [display-measurement-geometry.schema.json](display-measurement-geometry.schema.json).

The model estimates an ideal rectangular field-of-view footprint on a planar target:

`W = 2 d tan(theta_h / 2)`, `H = 2 d tan(theta_v / 2)`

where angular inputs are in radians for evaluation. It also calculates first-order standard-uncertainty estimates from distance and angular uncertainty. For the fit decision, it takes the upper coverage-factor bounds for distance and each angle, recomputes the nonlinear footprint, and adds the centering uncertainty and required edge guard on both sides. The resulting required uniform-patch extent must fit inside the declared patch dimensions in both axes. The report also returns remaining patch margin in millimetres and patch-utilization fractions for width and height, so a just-passing setup can be distinguished from one with generous clearance. Utilization above 1.0 indicates that the guarded footprint exceeds the declared uniform patch.

This is an ideal-geometry screen and an independent cross-check for fixture setup planning; it is **not** a calibrated lens/distortion model, a detector angular-response model, a photometric measurement, or proof that a given instrument's specified FOV describes its effective measurement cone. The coverage factor is an input convention, not a universal guarantee of statistical coverage. Real instrument response, pupil/lens geometry, alignment, target flatness and instrument-specific measurement instructions must still be assessed.

Synthetic known-answer and fail-closed tests cover tangent geometry, conservative bounds, inadequate uniform-patch dimensions, invalid FOV/uncertainty, schema closure, and refusal to promote physical validation. To evaluate the fixture question, use:

```text
cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
  --geometry-record docs/engineering/display-measurement-geometry-synthetic-v0.1.json \
  --geometry-schema docs/engineering/display-measurement-geometry.schema.json
```

The sample result is synthetic only; a future device-specific input must bind exact instrument identity, FOV source/evidence, distance measurement, uncertainties, patch definition, and setup revision before it can guide physical comparisons.

## 7. Benchmark gates

### G0 — protocol integrity

- 100% of reported metrics bind exact method edition, DUT state, pattern/sequence identity, instrument/calibration identity, units, setup geometry, raw artifact references and analysis revision.
- 100% of unsupported mandatory metrics explicitly remain NOT_EVALUATED.
- Zero silent drops of failed, aborted, excluded or out-of-tolerance observations.
- An independent checker recomputes deterministic arithmetic outputs for all frozen known-answer cases.
- Mutation tests reject stale calibration, mode/pattern mismatch, incomplete samples, wrong units, missing raw data, control schedules exceeding a preregistered maximum gap, and false pass claims.

### G1 — decision capability

- For every confirmatory metric, report whether U95_total ≤ MDE/3 holds under the declared project convention.
- If it fails, block candidate ranking on that metric until the instrument/setup is improved or the study question is honestly narrowed.
- Do not use the same candidate-derived optimizer/reducer as its sole expected-value oracle.

### G2 — repeatability / reproducibility

- Same-configuration deterministic report fields are byte-identical where determinism is promised.
- Within-run, restart, remount, inter-operator and inter-site performance are separately reported; untested levels remain NOT_EVALUATED.
- Cross-instrument estimates are compared under compatible measurands and uncertainty; unexplained disagreement stays unresolved rather than being averaged away.
- No universal percent agreement threshold is asserted before the measurand, instrument capability and effect threshold are frozen.

### G3 — useful engineering decisions

- A candidate cannot pass by lower cost, latency or energy if any predeclared optical, temporal, electrical, thermal, reliability or safety floor fails.
- The primary comparison is a quality-constrained Pareto set, with measurement uncertainty shown on each metric.
- Proposed improvements are supported by predeclared confidence/uncertainty bounds against a matched baseline; exploratory and confirmatory data remain distinguishable.

## 8. Failure cases

1. Stale/wrong calibration or wrong instrument configuration -> affected metric fails closed.
2. Pattern, mode, APL, dwell, geometry or ambient conditions mismatch -> no unqualified direct comparison.
3. Raw trace/sample loss exceeds protocol limits -> reject the affected metric.
4. A camera result is marketed as calibrated photometry without a suitable calibration/model -> reject the claim.
5. Two instruments agree but share a calibration or method bias -> agreement is not independent accuracy evidence.
6. Combined uncertainty is bigger than the decision margin -> no ranking on that metric.
7. Failed/aborted reads are omitted from denominators or summarized away -> invalidate the report.
8. A standards-aligned result uses a custom method with unreported deviations -> claim not supported.
9. CI or synthetic-test success is presented as optical, safety or human-perception validation -> reject.

## 9. Relationship to other work

- [OPEN_DISPLAY_REFERENCE_HARDWARE.md](OPEN_DISPLAY_REFERENCE_HARDWARE.md): fixture and component-open boundary.
- [OPEN_DISPLAY_HARDWARE_REUSE_CANDIDATES.md](OPEN_DISPLAY_HARDWARE_REUSE_CANDIDATES.md): reusable optical latency hardware and instrument candidates.
- [DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md](DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md): per-pattern context and conservative uncertainty handling.
- [VISUAL_UTILITY_PER_JOULE_PROTOCOL.md](VISUAL_UTILITY_PER_JOULE_PROTOCOL.md): task success and energy accounting.
- [display manufacturing route protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/vis-tech-001-program-2026-10-17/docs/engineering/DISPLAY_MANUFACTURING_ROUTE_PROTOCOL.md): manufacturing evidence, yield/repair/cost identities and Pareto gate.
- [Mycelix VIS-EVID-001 #5157](https://github.com/Luminous-Dynamics/mycelix/issues/5157): evidence provenance; a receipt does not itself certify the physical measurement.

All proposed gates are unqualified until instruments, procedures, baselines and measurements have been independently reviewed.
