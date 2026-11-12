# HTS Fusion Magnet Benchmark — REBCO conductor-to-coil translation

**Programme:** HTS-FUSION-MAGNET-001  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11736  
**Date:** 2026-11-12  
**Status:** Source register and benchmark design  
**Claim ceiling:** No Luminous reproduction, magnet design qualification, high-field operation, quench test, or hardware authorization is claimed.

## 1. Core thesis

The benchmark target is not a record critical temperature, tape critical current, or headline kiloampere rating. It is a **manufacturable, protectable, whole-magnet service envelope**.

The relevant translation chain is:

`material → tape → cable → joint → winding → coil → protection → cryogenic plant`

Each stage changes the useful operating envelope. Material performance can be lost through anisotropy, joints, cabling, winding strain, AC losses, mechanical loads, protection delays, and cryogenic parasitic power.

## 2. Source cases

The machine-readable case set is in `data/research/hts_fusion_magnet_reference_cases_v0_1.json`. It contains six distinct reference cases and twelve adversarial controls.

### 2.1 LASSO coil quench-detection demonstrations

Bykovskiy et al., *Demonstration of temperature-based quench detection on high-current insulated ReBCO coils made of LASSO cables*, Superconductor Science and Technology, published 4 September 2026.

- DOI: https://doi.org/10.1088/1361-6668/aea2da
- Official repository: https://infoscience.epfl.ch/entities/publication/3b2eb43f-1977-479e-9137-5a6e04177eb4
- Methods: twisted-pair superconducting-wire detectors (SQD), shielded thermocouple chains (TCC), fibre-optic sensing (FOS).
- Demonstrated sample sizes include a 20 m, 12-tape LASSO racetrack coil in SULTAN and a 50 m, 15-tape LASSO double-pancake coil in JORDI.

The paper is important because it examines integration at increasing conductor length and stored energy, not just a sensor coupon. However, exact per-event detection latency distributions, false-positive/negative rates and raw signals need extraction before quantitative ranking. Its sample demonstrations do not qualify a complete fusion magnet.

### 2.2 Microwave sensor-free quench detection

Rong et al., *Microwave-based, sensor-free rapid quench detection for REBCO-based magnets and power transmission lines*, Superconductor Science and Technology, published 29 June 2026.

- DOI: https://doi.org/10.1088/1361-6668/ae77b2
- Publisher record: https://iopscience.iop.org/article/10.1088/1361-6668/ae77b2

This is a new comparator family, not just another temperature sensor. The exact method, experimental/simulation division, measured outputs, test envelope, and false-alarm/detection performance must be extracted from the full article before it is ranked.

### 2.3 REBCO SQD and stabilizer protection modelling

Zgour et al., *Quench Protection in Insulated REBCO Conductors: Design Optimization and Fast Detection via REBCO SQD*, IEEE Transactions on Applied Superconductivity, 2026.

- DOI: https://doi.org/10.1109/TASC.2026.3659105
- Uses one-dimensional THEA modelling to compare copper stabilizer optimization with an electrically isolated but thermally coupled REBCO SQD.
- The article reports modelled hotspot/protection behaviour and earlier SQD transition under selected assumptions.

This is a numerical reference. A modelled hotspot result cannot be promoted to a measured hotspot or a protection qualification.

### 2.4 80 kA fusion-relevant CICC demonstration

Jin et al., *Performance of the First 80 kA HTS CICC for High-Field Application in Future Fusion Reactors*, Engineering, 2025.

- DOI: https://doi.org/10.1016/j.eng.2025.05.015
- Source reports 288 REBCO tapes and an 80 kA transport-current operating point at 10.85 T and 4.5 K under cyclic Lorentz loading.
- The paper also discusses a 46.5 kA / 20 T scaled design point.

**Comparison rule:** retain 80 kA at 10.85 T / 4.5 K as the measured source operating point and 46.5 kA at 20 T as a separate scaled/design projection. Do not merge them into one observed result.

### 2.5 HFRC six-around-one full-size CICC

*Manufacturing and performance test of a 21.59 kA @77 K, self-field full-size REBCO-HFRC cable-in-conduit conductor*, Fusion Engineering and Design, February 2025.

- DOI: https://doi.org/10.1016/j.fusengdes.2024.114775
- Reports 21.59 kA at 77 K in self-field and 36.70 nΩ total resistance, with separately reported upper/lower joint resistance.

This is a useful conductor manufacturing and joint reference, but it is not directly comparable to a 20 T / 4.5 K fusion operating point.

### 2.6 Material-to-engineering translation review

*From critical temperature to engineering reality: A quantitative framework for assessing the translation gap in superconducting materials*, October 2026.

- DOI: https://doi.org/10.1016/j.nxmate.2026.103138
- The review synthesizes REBCO routine piece-length, yield and production-readiness estimates and introduces an application-specific translation score using in-field margin, engineering current density, cryogenic access and mechanical/manufacturing robustness.
- It reports 0.3–2 km routine piece lengths, estimated 80–90% yield, and a review-assigned production readiness around TRL 7. These are review-synthesized estimates, not a guarantee for each vendor or product generation.
- Its REBCO high-field score at 20 K is 64 under the review's chosen scale. The scoring weights and intended-use class must be reproduced before using it in any design ranking.

## 3. Benchmark architecture

### Stage A — matched conductor identity

Every record binds:
- conductor/tape manufacturer and exact product generation where known;
- architecture, tape count, length, cabling and reinforcement;
- joint and termination identity;
- field magnitude and orientation;
- temperature, current and current definition;
- mechanical preload, strain and load history;
- test facility and configuration;
- source and evidence class;
- calibration, uncertainty, raw artifacts and rights.

### Stage B — detection tournament

Compare five sensing classes:
1. voltage taps;
2. SQD;
3. TCC;
4. FOS / fibre Bragg gratings;
5. microwave sensor-free detection.

Run comparisons only on matched event identity, time basis and configuration. Report:
- event-to-detection latency distribution;
- detection signal-to-noise and threshold margin;
- false-positive rate and missed-event rate;
- spatial coverage and blind zones;
- electromagnetic interference sensitivity;
- clock uncertainty, sensor dropout, drift and lead faults;
- nuisance-trip consequences and integration burden.

No detector is ranked using a different quench definition or a different instrumentation window.

### Stage C — conductor and coil physics

Metrics include:
- `Ic(B,T,theta)` and n-value/transition;
- current-sharing temperature and operating margin;
- strain and mechanical margin;
- AC losses under declared ramp/cycle;
- joint resistance and local heat load;
- electrothermal hotspot response;
- normal-zone propagation and detection delay;
- cryogenic heat load and refrigeration input.

### Stage D — manufacturing and lifetime

Track:
- length yield and defect mapping;
- cable/joint reproducibility;
- winding and reinforcement process window;
- damage through fabrication and cycling;
- repairability and inspection access;
- replacement time and availability;
- cost per qualified conductor/coil service.

## 4. Research paradigm shifts

**From tape-level performance to the complete conductor-to-coil chain.** A tape's self-field critical current is not an operating rating for a high-field cable or coil.

**From sensor selection to protection co-design.** Sensor, conductor, stabilizer, winding, thermal coupling and protection response must be designed together.

**From single detection latency to detection quality.** A very fast false trip or a missed local hot spot is not an improvement.

**From peak current/field to lifetime magnet service.** AC loss, cryogenic input, mechanical margin, joints, repair and availability belong in the optimization.

**From material prestige to application-conditioned translation.** Compare candidates at the intended field, temperature, field orientation, stress, current density, manufacturing length and cryogenic constraints.

## 5. Prospective benchmark targets

These are goals, not results:

- Reproduce each source-specific `Ic(B,T,theta)` / conductor reference within source uncertainty before extrapolating.
- Reproduce LASSO quench-event observables on the exact 20 m and 50 m cases, then assess held-out events.
- Compare all five detector classes on matched events; report latency, false alarms, misses and coverage jointly.
- Reproduce THEA model cases and independently compare outputs with physical traces; preserve model discrepancy.
- Match conductor AC loss, joint heat load and current-sharing behaviour at source-defined conditions.
- Optimize Pareto trade-offs among field/current/temperature margin, protection, AC loss, cryogenic power, manufacturing yield and lifecycle cost.
- Require a no-extrapolation disposition outside source-qualified field/temperature/angle/mechanical envelopes.

No scalar score may hide a failed protection gate, invalid material envelope, unqualified conductor identity or missing cryogenic load.

## 6. Negative controls

The JSON manifest freezes twelve controls, including:
- self-field 77 K result promoted to 20 T / 4.5 K;
- scaled design current/field promoted to a measured test;
- current compared without tape count, cable type or field angle;
- Tc used as a proxy for high-field current;
- different event definitions/clocks used in detector latency comparison;
- simulated THEA temperature reported as a physical measurement;
- microwave detection paper title treated as protection qualification;
- missing sensor interpreted as no quench;
- AC loss, joint heating or cryogenic power omitted;
- conductor prototype treated as a complete qualified magnet.

## 7. Recommended next execution order

1. Admit paper versions, raw time-series and data rights for LASSO and microwave detection.
2. Extract source-specific detection traces and form matched event IDs.
3. Reconstruct the 80 kA CICC measured point separately from its 20 T scaled target.
4. Reconstruct the 77 K self-field HFRC test without cross-promoting it to fusion conditions.
5. Reproduce the STGI review's scale and weighting, then test ranking sensitivity.
6. Identify the smallest exact THEA or other specialist-solver case only after input and reference closure.

## 8. Claim ceiling

This is a source register and benchmark specification. It is not a superconducting coil design, numerical reproduction, measured performance result, high-field qualification, quench protection certification, cryogenic operating procedure or authorization to energize hardware.
