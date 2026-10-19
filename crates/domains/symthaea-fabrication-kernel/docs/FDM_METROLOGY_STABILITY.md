# FDM metrology stability monitoring benchmark

Status: synthetic research benchmark; not a calibrated instrument monitor.

## Why this is a separate benchmark

Spatial calibration and multi-fidelity sampling ask where to measure and which measurement method to buy. This module asks whether the measuring system remains trustworthy across a sequence of observations. A high-confidence number acquired from a drifting gauge is not high-confidence evidence merely because the last calibration record is present.

NIST's measurement-process characterization guidance explicitly covers repeatability, reproducibility, stability, bias, resolution, linearity and drift. Its uncertainty guidance cautions that a gauge study can cease to represent the working measurement process when that process changes. NIST also describes check standards and control charts as tools for monitoring short- and long-term behavior:

- Gauge R&R / measurement-system characterization: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc4.htm
- Measurement uncertainty from gauge studies: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc46.htm
- Control of artifact calibration: https://itl.nist.gov/div898/handbook/mpc/section3/mpc35.htm
- Gauge drift and control-chart techniques: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc453.htm
- ISO/ASTM 52902:2023 geometric capability artefacts: https://www.iso.org/standard/79683.html

These sources motivate the protocol; they do not validate the synthetic noise model, thresholds or a printer-specific monitoring procedure.

## Paradigm shift: from “calibrated” to “currently in control”

The benchmark treats metrology trust as a time-bounded claim, not a permanent boolean attached to an instrument. A system must observe independent check standards over time, detect evidence that its assumptions no longer hold, and stop treating subsequent results as trustworthy until requalification occurs.

The first implementation is deliberately conservative: once a policy quarantines a stream, quarantine remains active for the rest of that trace. It does not auto-clear after quiet readings or issue a recalibration certificate. A future version should introduce an explicit recovery protocol with independent standards, recalibration evidence and a new trust epoch.

## Frozen synthetic protocol

- Three stable nominal check standards (10, 20 and 30 mm) are interleaved to expose range-dependent bias rather than checking only one zero point.
- Each arm contains 48 sequential observations.
- The declared baseline standard uncertainty is 0.006 mm. This is an arbitrary synthetic parameter, not a claimed instrument specification.
- A deterministic seeded noise function permits exact replay.
- The policy sees only nominal/check-standard identity, observed value, residual, declared uncertainty and history. It does not receive the stress-case label or future residuals.
- The evaluator applies six hidden profiles: stable process, abrupt bias shift, linear bias drift, gain change across the range, variance inflation and a single transient outlier.
- Five policies are compared: no-monitoring negative control, Shewhart point-limit rule, EWMA, two-sided CUSUM, and a hybrid persistence guard that warns on a single point outlier before quarantining persistent anomalies.
- Evidence remains explicitly marked synthetic. Nothing in this module reads an instrument, controls a printer, generates G-code or changes a production release decision.

The profiles and thresholds are test fixtures, not universal models of gauge behavior. In particular, a bounded deterministic noise source is not a substitute for estimating a real measurement distribution. The synthetic baseline assumes each interleaved reference standard has zero residual before a stress is injected; a physical monitor must instead use a validated baseline for each standard/method/configuration and its traceable value and uncertainty.

## Metrics that matter

| Metric | Why it is reported separately |
|---|---|
| False quarantine on stable stream | Measures needless withdrawal of usable evidence |
| First alert and first quarantine | Separates an early warning from a hard trust withdrawal |
| Premature quarantine before a known change | Exposes false alarms inside the pre-change stable prefix |
| Detection delay from injected change | Measures how long an out-of-control stream remains admissible |
| Trusted readings after change | Exposes the dangerous case where stale trust persists |
| Warning / quarantined / trusted counts | Makes dispositions auditable rather than reducing them to one score |
| Maximum and mean absolute residual | Keeps tail excursions visible alongside averages |
| Transient-outlier quarantine | Quantifies the cost of treating one extreme reading as persistent drift |
| Exact trace, seed and SHA-256 digest | Supports deterministic independent replay and tamper rejection |

Averaging is not allowed to hide a critical trust failure: a strategy with lower mean residual but a high post-change trusted fraction is not a better monitor.

## Acceptance targets for the benchmark harness

These are project targets, not published industry norms or results:

1. 100% exact report replay and rejection of tampered observations, decisions, metrics and digest.
2. The no-monitoring negative control must remain trusted during injected changes and therefore expose nonzero unsafe-trust fraction.
3. On the frozen seed panel, monitored policies should not quarantine the stable profile; if a false quarantine appears under broader randomized seeds, report it rather than relaxing the thresholds after seeing the result.
4. Persistent offset, gradual drift, range-dependent bias and variance inflation must be included in the report, with detection-delay and residual-trust counts given per profile.
5. The hybrid policy should distinguish a single point warning from persistent evidence; it must not silently erase the warning or call an unresolved stream trusted after quarantine.
6. Run multiple preregistered seeds and report distributions, not just the best seed. Freeze the scenario panel, thresholds, seed list and metrics before comparing any tuning change.
7. Keep false-quarantine and unsafe-trust metrics as separate guardrails. A composite average score may not trade away trustworthy evidence to gain convenience.

## Interpretation and known limitations

- A stable synthetic stream passing only demonstrates deterministic benchmark semantics.
- This does not estimate false-alarm probability at an industrial rate, guarantee detection of every drift, or certify a Gauge R&R study.
- The current gain-change fixture is a simplified range-dependent bias; it does not model complex geometry, orientation, material, temperature, operator or machine interactions.
- Variance inflation is represented through a simple noise-amplitude change. A physical study must estimate tails, dependence, heteroscedasticity and environmental covariates from measured data.
- Check standards themselves are assumed stable and traceable. Physical deployment needs evidence for artifact stability, calibration chain, reference values and uncertainty.
- The stream is one declared measurement process. Results do not transfer automatically across instruments, operators, firmware, measurement setups or material/geometry families.
- A real quarantine must define its affected evidence scope: at minimum, all measurements after the last accepted control point that could be affected by the drift must be reviewed. This harness reports sequential disposition but does not yet maintain a production evidence ledger or retroactively invalidate part records.
- No autonomous recalibration, correction-field update, G-code change, production release, or safety authority is introduced.

## Physical validation sequence

1. Define the actual measurement method, instrument identity, reference standards, calibration traceability and controlled work environment.
2. Preregister a stability study with multiple check standards spanning the range, repeat structure, time schedule, environment, operators/configurations where applicable and acceptance limits.
3. Estimate measurement uncertainty and control-chart parameters from a separate baseline period; hold out later sequences for prospective detection testing.
4. Compare stable periods, known calibration perturbations (only under a safe laboratory procedure), range dependence and real drift using independent reference measurements.
5. Determine the affected evidence window and recovery/reauthorization protocol with the quality owner before using detections to change a production disposition.

The next research slice should join the temporal monitor to the multi-fidelity spatial benchmark, so method selection and repeat count reflect current instrument health. It must not permit a cheap screen to override a calibration-stability failure.
