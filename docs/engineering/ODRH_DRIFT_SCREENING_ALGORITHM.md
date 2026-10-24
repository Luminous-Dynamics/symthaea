# ODRH-DRIFT-001 — Conservative control-excursion screening core

**Issue:** [ODRH-DRIFT-001 #8110](https://github.com/Luminous-Dynamics/symthaea/issues/8110)  
**Implementation:** `tools/open-display-platform-verify/src/drift_control.rs`  
**State:** software reference algorithm with synthetic tests; not a physical drift study

## Why this is a paradigm shift

A fixture cannot be trusted just because consecutive readings look stable. Slow thermal drift, sensor movement, reconfiguration and reference changes can make a candidate look better or worse than a baseline. The experiment therefore needs **control evidence throughout the run**, alongside the DUT observations.

The initial Rust core is intentionally conservative: it does not fit a drift curve, shift results back into a guard band, or infer the cause of an excursion. It records what happened at scheduled controls and marks the affected window as NOT_EVALUATED. This prevents software from hiding a physical problem it has not modeled.

## Algorithm v0.1

The caller must freeze a run interval, nominal control value, absolute guard half-width, a positive maximum allowed gap, ordered control slots and DUT observation times before using the report.

For a nominal value (c) and half-width (g ge 0), the accepted interval is:

[
[c-g, c+g]
]

Boundary values are accepted inclusively. Each scheduled control gets one of four dispositions:

- `IN_BAND`: a finite observed value lies inside the inclusive guard interval;
- `OUT_OF_BAND`: a finite observed value lies outside the interval;
- `MISSING`: the scheduled control slot has no reading;
- `INVALID`: a supplied value is non-finite.

For each missing, out-of-band or invalid slot, the affected time window starts at the preceding acceptable control or run start and ends at the next acceptable control or run end, inclusively. A DUT observation in that window is marked `NOT_EVALUATED_CONTROL_EXCURSION`; overlapping windows retain all responsible control IDs. This is conservative screening, not an estimate of the instant drift began.

An observation outside all such windows is labelled `NO_OUT_OF_BAND_CONTROL_DETECTED`, **not PASS**. A sparse schedule can miss drift between anchors; the algorithm does not establish accuracy, stability, traceability, calibration validity, or fitness for a display-metrology decision.

## Fail-closed behavior and known limits

The checker rejects non-finite or reversed run bounds, non-finite nominal/guard parameters, negative guard width, empty or duplicate IDs, non-monotonic control times, out-of-run times, duplicate DUT IDs, and a schedule without controls. Missing controls do not disappear from denominators: they become an explicit bad control slot and quarantine the interval.

The plan now declares a maximum allowed control-schedule gap. The verifier computes the largest gap including run-start-to-first-control, between consecutive scheduled slots, and last-control-to-run-end; it rejects a schedule that exceeds the declared cap. This detects an under-specified plan but does not establish that the chosen cadence is physically adequate. The algorithm does not select the control cadence, establish guard limits from historical distribution, estimate false-alarm/missed-drift rates, infer the unobserved drift trajectory between anchors, or provide a validated drift-correction model. Those require a chosen instrument/measurand, suitable reference artifact, repeated observations, and an uncertainty model.

## Synthetic benchmark matrix

| Scenario | Expected behavior |
|---|---|
| Reading exactly at either guard boundary | `IN_BAND` |
| Reading immediately beyond either guard boundary | `OUT_OF_BAND` |
| Missing scheduled control | `MISSING`; quarantine bounded by surrounding acceptable controls or run edges |
| Non-finite observed reading | `INVALID`; same conservative quarantine |
| Bad first control | Quarantine begins at run start |
| Bad last control | Quarantine extends to run end |
| Two bad controls separated by one good control | Preserve both control identities and each affected interval |
| Duplicate control ID or unordered schedule | Reject plan |
| A run-edge or inter-control schedule gap exceeds the predeclared cap | Reject plan before screening observations |
| Observation outside run interval | Reject plan |
| No out-of-band control detected | Explicitly do not claim physical qualification |

Required follow-up benchmarks should measure false-alarm and missed-excursion behavior only after the control noise distribution, smallest decision effect, measurement cadence and chosen guard-band design are frozen. Do not choose those thresholds based on which display candidate wins.

## Relationship to metrology literature

The [NIST Measurement Process Characterization handbook](https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc453.htm) treats instrument drift as a slow change in response and discusses check standards/control-chart monitoring; its [instrument-drift bias discussion](https://www.itl.nist.gov/div898/handbook/mpc/section3/mpc3312.htm) notes that particular drift-elimination designs rely on assumptions such as linear drift and equally spaced observations. This ODRH v0.1 algorithm intentionally makes neither of those assumptions: it quarantines instead of correcting. It is not a replacement for an applicable statistical control-chart or IDMS procedure.

The [ICDM Information Display Measurements Standard](https://www.sid.org/Standards/ICDM/ICDM) provides display measurement procedures but does not set universal performance/compliance thresholds. A selected measurement method and project-specific acceptance criteria still need to be recorded separately.

## Reproducing the synthetic example

The exact synthetic plan is [display-drift-control-plan-v0.1.json](display-drift-control-plan-v0.1.json), governed by [display-drift-control-plan.schema.json](display-drift-control-plan.schema.json). It caps the planned control gap at 40 seconds and its synthetic schedule's maximum edge/inter-slot gap is 40 seconds. Run:

    cargo run --manifest-path tools/open-display-platform-verify/Cargo.toml -- \
      --drift-plan docs/engineering/display-drift-control-plan-v0.1.json \
      --drift-schema docs/engineering/display-drift-control-plan.schema.json

The scenario has acceptable controls at 10 and 80 elapsed seconds, and an out-of-band control at 50. Its 60-second DUT observation is quarantined; observations at 5 and 90 seconds are not bracketed by the failure interval. These are synthetic known-answer values, not a recommendation for control cadence or an empirical drift rate.

The focused workflow retains the run-specific CAD/receipt bundle separately; its optional drift output is in the standard verifier JSON report and has this same synthetic-only claim ceiling.

## Test status boundary

The Rust unit tests use synthetic values and verify the deterministic behavior of this screening function. They do not show that a real sensor drifts, that a given cadence will detect real drift, that the guard half-width is appropriate, or that any physical display measurement is accurate. No physical control artifact, instrument or display was selected for this software-only increment.
