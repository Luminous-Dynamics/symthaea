# EDLC Sequential Reserve-Service Benchmark v0.1

**As of:** 2026-11-13  
**Status:** synthetic contract and replay implementation authored; Rust execution, hosted CI, physical validation and independent replication pending.

## 1. Why a sequence benchmark is needed

A storage device is not a reliable reserve merely because it can deliver one pulse after standby. An application may call it twice, or repeatedly, before it has recovered. The first call can pass while the second fails because of incomplete redistribution, voltage sag, thermal accumulation, changed ESR, an external recharge dependency, or installed-system auxiliary draw.

The benchmark therefore treats reserve capability as a history-dependent trajectory:

`initial standby → call 1 → recovery → call 2 → recovery → … → call N`

The qualification unit is a **complete device sequence**. The sequence passes only when each ordered event and each applicable inter-event recovery gate passes. Event-level averages are diagnostic; they cannot compensate for a failed call.

This is deliberately distinct from:
- **EDLC standby-service audit** (#11908): trace-level open-circuit / regulated-hold standby and first post-standby discharge.
- **Mission-readiness audit** (#11953): first-call service across dwell / environment strata and frozen event-arrival weights.
- **Sequential reserve audit** (#11978): repeated calls, inter-call recovery, and false-ready outcomes.

## 2. Measurement contract

### Frozen before evaluation

A physical campaign must preregister:
- unique sequence profile ID and exact ordered event IDs;
- the actual load waveform/profile ID for every call;
- initial standby dwell, per-call idle/recovery windows, operating temperature and voltage history;
- device boundary (cell, complete module/device, or installed system);
- all service, recovery, thermal and energy acceptance thresholds;
- open-circuit versus regulated-voltage-hold mode;
- event order, readiness-probe protocol, analysis plan, censoring rules and sample-size target.

Event-arrival distributions must be derived from application logs, service contracts or explicitly labelled engineering assumptions. Do not invent a universal arrival distribution.

### At each call

Record synchronized terminal voltage/current/time through the exact application waveform. Derive:
- delivered energy at the application terminals (Wh);
- minimum sustained power over the declared interval (W), not a peak sample;
- minimum application voltage / cutoff and voltage sag;
- response or wake latency;
- temperature rise and maximum temperature;
- waveform identity, start/end times and measurement uncertainty.

Include converter and protection losses at the installed-system boundary where applicable. If the application includes a converter, a capacitor-only result cannot be advertised as delivered system performance.

### Between calls

Measure:
- time to restored readiness;
- readiness-probe result and exact probe profile;
- source-side replenishment energy;
- auxiliary energy used during the idle/recovery interval.

A voltage rebound is not, by itself, proof that usable energy or power has recovered. The readiness probe must measure an application-relevant capability. Its own energy, disturbance, latency and impact on the next event must be characterized; use a matched control or an identical probe on all devices. Do not silently let the probe consume reserve without accounting for it.

A false-ready event occurs when the readiness probe reports ready but the subsequent actual application event fails its service contract.

### Mode separation

1. **Open circuit:** no external energy input. Any measured replenishment contradicts the protocol and invalidates the record.
2. **Regulated voltage hold:** report energy supplied by the holding source at the source boundary; do not count top-up as retained reserve.
3. **Installed system:** include BMS, balancing, converter, sensors, communications and contactor/relay quiescent loads.

Do not pool these modes into one result.

## 3. Scoring

For event (k) on device (i):

`event_pass(i,k) = service_pass(i,k) AND recovery_pass(i,k)`

The service gate is conjunctive across delivered Wh, minimum sustained W, response latency, minimum terminal voltage and temperature rise. The recovery gate is conjunctive across readiness-probe pass, time-to-readiness deadline, available idle window, auxiliary-energy budget and source-replenishment budget.

For a fully measured sequence:

`sequence_pass(i) = AND_k event_pass(i,k)`

Report the sequence pass fraction over unique valid device sequences. Failed devices count as failures. Invalid and censored sequences remain visible, with reasons, and are not silently classified as passes. Repeated events from one physical serial are correlated within-device observations, not independent devices.

Report at minimum:
1. complete-sequence pass fraction;
2. event-position service, recovery and joint-gate pass fractions;
3. per-device ordered failure trace and earliest failing event;
4. time-to-restored-readiness distribution and deadline misses;
5. false-ready count/rate;
6. delivered Wh and minimum sustained W by event position;
7. external replenishment and auxiliary-energy burden;
8. device/lot/lab variance, sample counts and uncertainty.

The initial Rust audit currently reports descriptive synthetic counts; it does **not yet** provide physical-data ingestion, confidence bounds, device-health models or lifetime extrapolation.

## 4. Synthetic known-answer case

The fixture contains three synthetic devices and a frozen three-call profile:
- first call after 168 h open-circuit standby;
- second call after 6 h;
- third call after 24 h;
- minimum delivered energy 0.8 Wh, minimum sustained power 1 W, maximum response latency 0.8 s, minimum application voltage 1 V, maximum temperature rise 10 °C;
- maximum restoration time 4 h and declared auxiliary / replenishment budgets.

Expected replay:
- event 1 service: 3/3;
- event 2 service: 3/3, but one recovery is late (5 h > 4 h);
- event 3 service: 2/3; one readiness probe passes but the real load fails;
- complete sequence: 1/3;
- false-ready events: 1.

These are deliberately invented values to exercise fail-closed arithmetic. They are not product data, probabilities or a claim about an EDLC.

Run from the workspace root:

```sh
cargo run -p xtask -- edlc-sequential-reserve-audit
cargo test -p xtask edlc_sequential_reserve
```

The command and 14 authored unit tests are committed to the research branch. Neither command is claimed to have run in this environment.

## 5. Literature and industry position

IEC 62391-2:2025 is the current IEC sectional specification for power-application EDLCs. It selects test/measurement and quality-assessment procedures from IEC 62391-1; its published leakage-current and maintain-voltage procedures are useful baselines, but they do not on their own establish the ability to meet a repeated application mission after standby. Use the controlled normative document and the applicable detail specification for lab work. See the [IEC publication page](https://webstore.iec.ch/en/publication/82213).

Manufacturer specifications demonstrate that industry already addresses important pieces of reliability:
- Eaton's XLR-51 datasheet specifies nominal leakage after a 72 h charge-and-hold condition, a high-temperature life condition and a separate cycle test. These are valuable manufacturer-defined endpoints, not a public repeated-call, mission-profile delivered-energy dataset. See the [Eaton XLR-51 datasheet](https://www.eaton.com/content/dam/eaton/products/electronic-components/resources/data-sheet/eaton-xlr-51-supercapacitor-module-data-sheet.pdf).
- Skeleton's 54 V / 188 F SkelMod datasheet lists 76.14 Wh maximum stored energy, 3.0 mΩ 10 ms ESR, 3.4 mΩ 1 s ESR, 10-year uncharged room-temperature storage life and projected cycle-life conditions. These values have different test boundaries and do not demonstrate repeated-call readiness after charged standby. See the [Skeleton SkelMod 54 V / 188 F datasheet](https://www.skeletontech.com/hubfs/01-DS-211229-SKELMOD-51V177F-1D.pdf?hsLang=en) for the currently published 51 V / 54 V module family data; retain the exact variant and test conditions.

A 2026 experimental study tested a 20-cell series pack of 2.7 V / 100 F cells (54 V nominal) and reports that, in its 25 °C open-circuit test, pack voltage fell from about 54 V to 33 V over 24 h. In its load-rate experiments, effective capacitance fell from 99% of nominal at 1 A to 77% at 10 A constant current and to 60% at 200 W constant power. A two-branch equivalent-circuit model reduced voltage-prediction RMSE by 49–83% across the cases evaluated, at the cost of additional model parameters. This is an important matched-method comparator for load-rate and model-order sensitivity, but it is one study and one device setup—not a universal derating factor, transferable prediction error, or market-wide SOTA value. See [Shijas and Dash (2026)](https://doi.org/10.1080/19397038.2026.2720202).

A February 2026 [operando-characterization review](https://doi.org/10.1016/j.ccr.2025.217268) surveys in-situ/operando electrochemical, optical, magnetic, gravimetric and neutron characterisation and identifies challenges in standardisation, spatiotemporal resolution, multimodal integration and industrial translation. The September 2026 [self-discharge review](https://doi.org/10.1002/tcr.70230) distinguishes ohmic leakage, parasitic Faradaic reactions and charge redistribution, and reports no new dataset. The implication is to combine orthogonal diagnostics with application-level trace qualification rather than infer mechanism from voltage decay alone.

**Competitive differentiation:** do not claim that Luminous invented capacitor reliability, lifetime testing or reliability-by-design. The opportunity is to make the qualification *executable and evidence-linked*: immutable device/lot identity, raw trace digests, exact waveform identity, calibration and fixture provenance, application-boundary energy accounting, fail-closed event replay, statistical reliability bounds and independent reproducibility.

## 6. Benchmark targets

These are proposed engineering targets, not established standards or results:

| Gate | Proposed target | Why it matters |
|---|---|---|
| G0 software contract | Strict schema, deterministic replay, negative controls, no hidden pass state | Data-contract integrity |
| G1 reference | Reproduce a commercial reference module's declared test endpoints, then run the same frozen sequential profile | Establish matched baseline |
| G2 prototype | Improve mission-weighted delivered Wh per installed kg **or** reduce required installed mass while meeting every power, voltage, latency, thermal and safety gate | Compete on useful service, not headline capacitance |
| G3 reliability | One-sided 95% lower confidence bound meets the application's declared sequence-reliability requirement, stratified by lot and relevant conditions | Lower-tail reliability |
| G4 recovery | False-ready rate below the application limit; recovery-time distribution and repeated-call failure modes reported | Prevent unavailable reserve being labelled ready |
| G5 independent replication | Same frozen profile across at least three labs, with instrument/fixture calibration and device/lot traceability | External credibility |
| G6 production | Process capability, yield, cost, repairability, recycling and field-outage evidence | Commercial readiness |

For zero observed failures, the exact one-sided 95% binomial lower bound is `p_L = 0.05^(1/n)`. The minimum independent-device sequence count for a target reliability floor `p_0` is `n_min = ceil(ln(0.05) / ln(p_0))`. These are mathematical lower bounds under an ideal independent, identically distributed Bernoulli model, not a substitute for power analysis. Repeated events are not independent devices; lot/lab clustering, multiple strata, censoring and nonzero failures require an appropriate design and analysis. A multi-stratum programme should not reuse one nominal 95% per-stratum interval and imply simultaneous 95% coverage.

## 7. What would beat current practice

The defensible path is not one giant composite score. It is a matched, falsifiable test programme:

1. **Capability surface:** (E_{usable}(t_{idle},T,V_{hold},P_{load},history)), with the complete delivered-energy and power envelope rather than nominal (C) or open-circuit (V).
2. **Sequence reliability:** probability that a unique physical device satisfies every call and recovery gate for a preregistered profile.
3. **Recovery truth:** readiness-probe calibration against the subsequent real load; quantify false-ready and false-not-ready rates.
4. **Mechanism-aware diagnostics:** pair electrical traces with ESR/EIS, leakage, temperature and appropriate post-mortem/operando characterization. Mechanism labels must abstain when observations are non-identifying.
5. **Ageing-aware forecasts:** train on whole-device and whole-lot splits; predict held-out sequence outcomes and remaining useful mission life. Report calibration, coverage, false-pass cost and abstention—not only (R^2).
6. **Installed-system accounting:** count converters, control electronics, balancing, communications, enclosure and thermal management at the same boundary as the commercial comparator.
7. **Evidence receipts:** immutable protocol, serial, lot, waveform, raw trace digest, instrument, fixture, calibration, analysis version, failures and deviations.

Only after matching the application and boundary should a stretch target such as 20% more mission-delivered Wh per installed kg be considered. It must be achieved without violating any hard service, life, safety, cost or maintainability threshold; it is a proposal, not a current claim.

## 8. Physical data required before claims

The next physical evidence contract should require:
- manufacturer / build / chemistry / lot / serial and immutable device configuration;
- exact raw voltage-current-time traces for every call and readiness probe, with cryptographic digest;
- terminal and source-side channels with synchronization, range, bandwidth, sample rate, uncertainty and calibration certificate;
- fixture ID, wiring/contact resistance, ambient and device temperatures;
- waveform/profile file and version, firmware/controller versions, initial voltage and charge protocol;
- complete mass/volume boundary and auxiliary loads;
- deviations, shutdowns, failures, censoring and missingness;
- test operator, independent reviewer, laboratory and safety approval.

No physical test, partner access or independent replication is asserted by this software increment.
