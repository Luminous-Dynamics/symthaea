# Display Pattern Robustness and Uncertainty Protocol

**Protocol:** ODRH-PAT-001 v0.1.0  
**Parent:** [ODRH-001 #8030](https://github.com/Luminous-Dynamics/symthaea/issues/8030)  
**Measurement contract:** [DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md)  
**Status:** proposed; no physical measurements, panel qualification, or calibration claim

## 1. Objective

A display should not appear superior because a benchmark measures only one convenient patch, window size, background, viewing angle, or dynamic mode. This protocol makes **pattern/context robustness and uncertainty explicit**, while preserving separate measurands rather than blending incompatible properties into one score.

Use the applicable method in the current [Information Display Measurements Standard (IDMS)](https://www.sid.org/Standards/ICDM/ICDM) when one directly covers the measurand. IDMS v1.3 was released in May 2025; record exact method/edition and any adaptations. Do not reproduce or claim to replace the standard. A custom workload is diagnostic unless its relation to the chosen standard method has been independently established.

## 2. Freeze the comparison domain before measurement

For each measurement campaign, preregister:
- device under test, hardware/firmware revision and operating modes;
- signal source, resolution, refresh rate, color encoding/range/bit depth and SDR/HDR state;
- exact pattern corpus revision, file or generator digest, sequence order and dwell/settle conditions;
- patch/window size, average picture level (APL), location, surrounding image and immediately preceding pattern;
- display OSD/dynamic features, brightness/contrast settings and warm-up policy;
- test environment, measurement geometry, instrument/calibration identities and raw-data capture;
- protected strata, exclusion rules, the primary summary statistic and uncertainty/reporting convention.

Record the **actual negotiated/output mode and observed configuration**, not merely the requested configuration. Any comparison whose settings differ outside the preregistered tolerance is unmatched until rerun or the deviation is modeled and disclosed.

## 3. Minimum workload lanes

The corpus must distinguish at least these diagnostic dimensions. These are workload lanes, not a replacement for any prescribed standard stimulus:

1. **Static level / APL:** dark, mid-level and bright fields/windows, with APL and patch size recorded. Treat OLED power limiting, LCD local dimming and other content-dependent behavior as device-specific hypotheses to test.
2. **Spatial distribution:** center and predeclared distributed locations; record ROI, screen coordinates, measurement geometry and any required view angle.
3. **Color and tone:** only where the selected instrument supports the measurand; bind stimulus values, signal range, transfer function and instrument spectral/photometric capability.
4. **Temporal response:** deterministic transitions, refresh/scan timing or optical latency patterns with retained waveform samples, declared onset/threshold rule and timing uncertainty.
5. **Dynamic workload:** predeclared moving imagery or representative task footage for content-dependent power and temporal behavior. Record source digest, playback frame pacing and drops.
6. **Repeat and remount:** selected controls and key patterns repeated after restart and remount/repositioning to expose setup sensitivity.
7. **Failure/edge cases:** near range/threshold limits, pattern/context changes, missing samples and unsupported modes. Keep failures visible; do not drop them from the corpus.

Do not force every instrument to report every lane. Unsupported or uncalibrated metrics are `NOT_EVALUATED`.

## 4. Uncertainty and stratified reporting

For every reported metric:
- define the measurand and unit;
- list relevant instrument, geometry, repeatability, environment/drift, stimulus/setup, sampling and analysis uncertainty terms;
- identify common references/covariance when comparing tools or sessions;
- bind raw samples/artifacts, calibration status, method, configuration and reducer version;
- report sample count, dispersion and uncertainty/coverage convention alongside a predeclared summary.

Show per-stratum results and the worst protected stratum. A global average must not conceal a failure in one required mode, patch class, location or view. Avoid treating non-independent repeated frames/samples as independent trials. A large number of frames from one unchanged setup does not prove cross-run or cross-operator repeatability.

For decision-making, follow the project-level capability check in the [metrology reproducibility protocol](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md): a metric whose expanded uncertainty is too large relative to the preregistered minimum decision effect is ineligible for ranking. `U95_total <= MDE/3` is an ODRH planning hypothesis, not a universal standards requirement.

## 5. Comparison and benchmark gates

### G0 — corpus integrity
- 100% of workload cases have stable IDs, exact content/generator digest, lane/context labels, required configuration and method status.
- Mutations with unknown cases, missing digests, duplicate IDs or incompatible output modes fail closed.
- All required raw observations and failed/aborted/excluded cases remain discoverable.

### G1 — context robustness
- Compare candidates under a matched preregistered workload.
- Report outcomes per lane, protected stratum and the worst protected stratum; do not aggregate away floor violations.
- Mode/context-dependent metrics remain separate dimensions when a single physical model is not justified.

### G2 — uncertainty capability
- Every confirmatory metric reports whether its measurement uncertainty meets the frozen decision-capability gate.
- An unsupported metric or failed capability gate cannot become a candidate win, and it cannot be silently converted to zero error.

### G3 — independent reproducibility
- An independent reducer reproduces deterministic transformations from retained raw artifacts.
- Repeat-run, remount, inter-operator, inter-instrument and cross-site evidence are reported separately; untested levels remain `NOT_EVALUATED`.
- No universal tolerance percentage is adopted until the measurand, instrument class, uncertainty budget and engineering effect are frozen.

## 6. Fail-closed cases

1. Requested HDR or color mode is mistaken for confirmed output mode.
2. Pattern/window/APL/dwell/OSD differs between candidate and baseline without an explicit matched-protocol explanation.
3. View angle, ROI, distance or instrument placement differs outside the declared setup envelope.
4. Camera-derived values are presented as absolute photometry/colorimetry without a suitable validated calibration/model.
5. A pooled average hides a violation in a protected stratum.
6. Missing, failed or inconvenient measurements are omitted without a recorded disposition.
7. Uncertainty exceeds the minimum decision effect but the result is still ranked.
8. Synthetic workload or CI validation is described as human-perception, optical or physical performance validation.

## 7. References and integration

- [ICDM / IDMS](https://www.sid.org/Standards/ICDM/ICDM): current display measurement procedures and methodology.
- [NIST/SEMATECH Measurement Process Characterization handbook](https://www.itl.nist.gov/div898/handbook/mpc/mpc.htm): repeatability, reproducibility, bias, stability and uncertainty context.
- [IEC 62087-7:2018](https://webstore.iec.ch/en/publication/27420): applicable mains-powered computer-monitor power-method reference; it covers static and moving picture scenarios within its scope.
- [ODRH metrology protocol](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md): measurement identity, uncertainty and decision capability.
- [Visual utility-per-joule protocol](VISUAL_UTILITY_PER_JOULE_PROTOCOL.md): task success and energy accounting.
- [ODRH drift screening algorithm](ODRH_DRIFT_SCREENING_ALGORITHM.md): conservative control screening, subject to its maximum planned control-gap constraint.

This protocol remains proposed until a specific display, workload and suitable instrument setup are selected, measured and independently reviewed.
