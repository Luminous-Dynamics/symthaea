# Display Pattern Robustness and Measurement Uncertainty

**Status:** proposed research protocol; no display is qualified by this document  
**Tracking issue:** [VIS-METRO-001 #7894](https://github.com/Luminous-Dynamics/symthaea/issues/7894)  
**Parent program:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)  
**Related work:** [VIS-BASE-001 #7853](https://github.com/Luminous-Dynamics/symthaea/issues/7853), [VIS-ENERGY-001 #7878](https://github.com/Luminous-Dynamics/symthaea/issues/7878)

## 1. Why pattern robustness matters

A display measurement is conditional on a stimulus and its context—not merely on a panel model or a single pattern label. Pattern size, average picture level (APL), color distribution, background, spatial context, preceding images and temporal timing can change the optical response.

Bergquist's *Contesting Patterns*, first published 16 September 2026, explains these dependencies and shows how a reported color-gamut estimate can differ when the pattern method changes (81.0% versus 87.8% BT.2020 coverage in one example). The result is not that one pattern is universally correct; it is that a measurement must state its pattern contract and its claim must not generalize beyond tested contexts without evidence.

Reference: https://doi.org/10.1002/msid.70096

The Society for Information Display's International Committee for Display Metrology publishes the Information Display Measurements Standard (IDMS); the cited current release is v1.3 (2025). For the applicable display/metric, pin that edition and the matching IEC method rather than inventing a competing pattern definition. Reference: https://www.sid.org/Standards/ICDM/ICDM

## 2. Pattern contract

Every optical or temporal measurement should bind, as applicable:

- immutable image/stimulus bytes or reproducible generator parameters and a cryptographic digest;
- input color encoding, primaries/white point, transfer function, bit depth and metadata;
- patch geometry, size, spatial position, display region/ROI and background;
- APL and channel/color distribution, including the pattern family/standard identifier;
- display mode, local-dimming/backlight policy, refresh rate, HDR mode and any content-adaptive setting;
- stimulus order, preceding stimulus, washout interval, dwell/capture duration and frame cadence;
- ambient illumination, viewing geometry, instrument field of view/measurement angle, calibration and stray-light controls;
- standard edition, deviations and why a pattern is applicable to the claim;
- raw capture/trace digest, missing samples or frames, and uncertainty method.

Use the same display mode for all related attributes unless the protocol explicitly defines a mode comparison. A test-mode change is a new stratum, not a silent convenience. For sequential measurements, preceding-pattern effects and washout must be controlled or deliberately varied and reported.

## 3. Bounded initial pattern ensemble

Do not add a second generic corpus. Map this protocol to existing IDs in `visual-display-workload-corpus-v0.1.json`:

| Existing case | Pattern/context dimension | Required reporting |
|---|---|---|
| `VIS-DISP-PHYSICAL-PEAK-015` | Small peak-luminance stimulus | Pattern/patch size, APL, duration, clipping and device mode |
| `VIS-DISP-PHYSICAL-FLASH-016` | Full-screen HDR stimulus | Full-screen loading, flash duration, thermal/start state |
| `VIS-DISP-ACTIVE-DIMMING-019` | Checkerboard / local-dimming context | Contrast, halo/stray-light context and neighboring bright regions |
| `VIS-DISP-COLOR-VOLUME-020` | Color distribution / gamut | Channel combinations, test-pattern family and colorimetric geometry |
| `VIS-DISP-EOTF-021` | Signal-level response | Input values, pattern APL, time sequence and measured luminance |
| `VIS-DISP-SUBTITLE-FLICKER-022` | Moving/temporal context | Background, subtitle region, switching cadence and temporal capture |
| `VIS-DISP-POWER-026` | Power stimulus and task content | Applicable standardized power stimulus and separate task-realistic workload |
| `VIS-DISP-COLOR-GRADIENT-028` | Smooth gradients / banding | Gradient direction, bit depth, spatial context and display processing |
| `VIS-DISP-STALE-PROFILE-030` | Deliberately mismatched profile | Must reject or mark NOT EVALUATED |
| `VIS-DISP-RAW-EVIDENCE-031` | Missing/corrupt stimulus or capture | Must block qualification |

For an initial cross-pattern claim, predeclare at least five pattern/context conditions relevant to that claim. A reasonable conventional-display starting set may combine the applicable standardized method, a multicolour/context pattern, a varied-APL condition, a representative task image/sequence, and an explicit preceding-stimulus/washout condition. Do not treat this suggested set as an industry standard, and do not substitute custom stimuli for a required standardized test.

## 4. Report sensitivity, not only an average

Report separately:

1. every pattern's metric result and evidence status;
2. within-pattern repeated-run variation;
3. cross-pattern minimum, maximum and spread;
4. the worst predeclared pattern/context stratum;
5. failed, missing, excluded and invalid measurements with their causes;
6. the pattern subset that the final claim actually covers;
7. device-mode and calibration changes across the run;
8. uncertainty and covariance assumptions.

The mean alone is insufficient. Do not remove an inconvenient test pattern after seeing the results. If the metric response is intentionally context-dependent (for example, display power as a function of APL), publish that response surface rather than declaring all variation an instrument error.

## 5. Energy claim uncertainty

The task-conditioned energy protocol defines nominal energy per successful task. This protocol adds a conservative interval for the relative saving when baseline and treatment energy are represented by valid bounded intervals:

`B ∈ [B_{low}, B_{high}]`, `T ∈ [T_{low}, T_{high}]`

`saving = 100 × (1 - T/B)`

A conservative interval-arithmetic envelope is:

`saving_{low} = 100 × (1 - T_{high}/B_{low})`

`saving_{high} = 100 × (1 - T_{low}/B_{high})`

This requires `B_low > 0`. Reject reversed, negative or non-finite energy intervals. The result is **not** a confidence interval or probability statement; it is an outer interval under the declared component bounds. It makes no independence assumption and may be wider than a covariance-aware estimate.

For the initial 10% energy target, the lower bound must be at least 10% before the energy target can pass. If the nominal saving is 10% but its uncertainty interval crosses below 10%, the outcome is inconclusive for that gate—not a pass. Task success, p95/p99 latency and optical floors remain separate mandatory gates.

Follow recognized measurement uncertainty practice: use repeat-run data to estimate repeatability where suitable, include instrument calibration/accuracy and other significant components, and preserve covariance when supported by evidence. NIST TN 1297 describes combined standard uncertainty and the law of propagation of uncertainty: https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-5-combined-standard-uncertainty

If an interval is intended as an approximately 95% coverage interval, document how it was constructed and do not conflate it with a hard engineering bound. Monte Carlo propagation is acceptable only with declared input distributions, dependency assumptions, seed, implementation/version and convergence checks. Do not label arbitrary endpoints as probabilistic.

## 6. Fail-closed controls

Include at least these negatives in the test/verification plan:

- stimulus digest mismatch or incorrect color/transfer metadata;
- measurement taken in a different display mode than the declared protocol;
- missing or altered preceding stimulus/washout;
- deleted pattern row or changed corpus version;
- one pattern silently omitted after observing a poor result;
- stale/absent calibration or unsupported instrument field of view;
- absent, corrupted or time-misaligned raw capture/power samples;
- malformed interval, non-finite input or baseline interval including zero;
- nominal energy saving passes 10%, but the conservative lower bound does not;
- uncertainty widened or exclusions changed after outcomes were inspected.

Missing mandatory data yields NOT EVALUATED / UNQUALIFIED. A validator should not convert unknown evidence to a favorable default.

## 7. Independent verification and evidence boundary

The verifier should recalculate pattern digests and energy ratios without importing the candidate strategy's own decision logic. Bind exact source, binary/configuration, corpus, pattern, display-mode and calibration identities into the existing Mycelix visual-evidence profile. A content hash verifies byte identity relative to that hash; it does not verify the correctness of the physical measurement or calibration.

This document defines a measurement protocol and a bounded arithmetic reference. It contains no qualified display result, certification, measured energy improvement, or human-perception claim. Physical claims require the relevant calibrated equipment, named device/mode, raw measurements, uncertainty analysis and independent review.
