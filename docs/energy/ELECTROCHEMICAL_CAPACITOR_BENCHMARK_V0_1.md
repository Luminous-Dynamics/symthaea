# Electrochemical capacitor device benchmark v0.1

Status: research and measurement contract. This document is not a synthesis recipe, device design release, safety certification, commercial claim or permission to operate laboratory equipment.

Parent: [CAPACITOR-DEVICE-DISCOVERY-001 #11841](https://github.com/Luminous-Dynamics/symthaea/issues/11841).


### Raw-data reproducibility and trace-evidence receipt — 14 November 2026

Two additional research contracts now extend the benchmark without confusing published material datasets with physical module evidence:

- [EDLC-OPEN-DATA-REPRO-001 #12004](https://github.com/Luminous-Dynamics/symthaea/issues/12004) and [EDLC_EXTERNAL_RAW_DATA_REPRODUCTION_V0_1.md](EDLC_EXTERNAL_RAW_DATA_REPRODUCTION_V0_1.md) define rights-aware reproduction of public CV/GCD/EIS/ageing data. The DataverseNO dataset DOI 10.18710/F4NFMJ (version 1.0, CC0 1.0) lists 26 public files. Three source file IDs and repository MD5 values are registered; local raw-byte download and SHA-256 verification remain pending. The Zenodo TiO2-ALD flexible-supercapacitor dataset is a 10,000-cycle ageing candidate, but its explicit licence was not established and redistribution remains blocked.
- [EDLC-TRACE-EVIDENCE-RECEIPT-001 #12005](https://github.com/Luminous-Dynamics/symthaea/issues/12005) adds a strict receipt schema, nine-artifact synthetic package, and Rust auditor. The CLI is `edlc-trace-evidence-audit`; **20 unit tests are authored**. It checks relative-path confinement, byte length, SHA-256, CSV structure, device/instrument/calibration/protocol/environment cross-links and open-circuit source-current semantics. See [EDLC_TRACE_EVIDENCE_RECEIPT_V0_1.md](EDLC_TRACE_EVIDENCE_RECEIPT_V0_1.md).

Independent runner-free verification has re-fetched all nine positive receipt artifacts plus two negative/control files from the branch. Their byte lengths and all eleven SHA-256 digests match the manifest/test constants. This is reproducible byte-integrity evidence only. It does not authenticate the creator, physical device, calibration certificate or lab procedure. No external research CSV has been downloaded or reproduced in this increment, and no physical device evidence has been admitted.


## Purpose and claim boundary

Qualify complete electrochemical capacitor devices and modules from immutable material, electrode, electrolyte, process, assembly and test identities. Begin with an EDLC reference. Keep aqueous zinc-ion hybrid supercapacitors as a separate, higher-risk research lane.

This v0.1 is a data/evidence contract. Synthetic fixtures can test software semantics only. Vendor data is a vendor claim; literature data is a literature comparator. Neither establishes Luminous device performance.

## Device taxonomy — do not pool

- EDLC: predominantly non-Faradaic electric double-layer storage.
- Pseudocapacitor: surface/near-surface Faradaic storage; state the mechanism and material.
- Asymmetric / hybrid capacitor: distinct electrode mechanisms; identify each electrode.
- Zinc-ion hybrid supercapacitor (ZIHSC): zinc-based Faradaic electrode paired with a capacitive/pseudocapacitive electrode.
- Lithium-ion capacitor: lithium-insertion electrode paired with a capacitive electrode; do not relabel it as EDLC or ZIHSC.
- Dielectric capacitor: field-polarization storage; do not compare its pulse-energy metric directly with electrochemical cycling metrics.

## Canonical device identity

Every device revision must bind:
1. chemistry and storage-mechanism class;
2. positive and negative electrode composition, mass, loading, area, thickness, density, porosity and batch;
3. electrolyte identity, composition, amount, water content where relevant and batch;
4. separator identity, thickness, mass and batch;
5. current collectors, conductive additives, binder and tabs;
6. geometry, seals, casing, terminals and device volume;
7. process recipe, equipment, operator, timestamps, atmosphere and process deviations;
8. unique physical device serial and parent material-lot IDs;
9. test fixture, instrument, calibration, protocol revision and raw-data digest.

Missing identity is UNKNOWN; never infer it from a chemistry name.

## Mandatory boundaries

Report all applicable boundaries separately:
- Active-material mass / volume (screening only).
- Electrode mass / volume (includes current collector only if explicitly stated).
- Complete device mass / volume (both electrodes, electrolyte, separator, current collectors, binder/additives, tabs, seal and casing).
- Module mass / volume (balancing, monitoring and thermal/electrical hardware included).
- Installed system mass / volume (converters, enclosure, cooling, protection and wiring included).

Never convert an active-material value into a device or module value without measured component masses and a traceable bill of materials.

## Mandatory performance vector

Each value carries units, boundary, voltage window, current/power profile, temperature, measurement method, uncertainty, source evidence and test time.

| Metric | Minimum definition |
|---|---|
| Capacitance | F and F/g with boundary; specify measurement method/frequency/current |
| Usable energy | Integrate measured discharge power over declared voltage cutoffs; report ESR-related loss |
| Specific / volumetric energy | Wh/kg and Wh/L at electrode, device, module and system boundaries where available |
| Power | W/kg and W/L; distinguish peak, pulse duration, continuous and thermal limits |
| ESR / impedance | DC ESR plus EIS spectrum, amplitude, frequency, temperature and state |
| Efficiency | Coulombic and energy efficiency over a defined cycle |
| Leakage / self-discharge | Current and voltage decay versus time, temperature and initial voltage |
| Degradation | Capacitance loss, ESR growth, leakage growth and failure mode versus cycle count and cumulative throughput |
| Lifetime | Cycle protocol, depth/voltage window, rest periods, temperature and end-of-life criterion |
| Calendar aging | Time, voltage, temperature and periodic diagnostic protocol |
| Standby service | Measured usable Wh/W delivered after a declared open-circuit or voltage-hold interval and representative load; include converter/balancer standby loss at system boundary |
| Thermal behavior | Temperature rise under stated pulse/continuous load; sensor location and uncertainty |
| Safety | Defined abuse/short-circuit/overvoltage/thermal test, facility, procedure, result and failure observations |
| Manufacturing | First-pass yield, accepted yield, scrap, rework, batch variability, material utilisation, process energy and cost |

Do not state “cycle life” without the voltage window, current profile, temperature, rest, end-of-life threshold and treatment of failed/invalid devices.


## Standby energy availability and voltage-hold reliability

Child experiment: [EDLC-STANDBY-SERVICE-001 #11908](https://github.com/Luminous-Dynamics/symthaea/issues/11908). The primary metric is measured usable energy delivered into a declared load after a specified standby history:

E_available(t_idle, T, V_hold, prior_history)

This is distinct from nominal capacitance, open-circuit voltage retention, and cycle-life retention. Report the complete charge/hold/discharge trace; time to minimum usable voltage; leakage current versus time; open-circuit self-discharge; ESR/EIS evolution; recovery/refresh energy; and lower-tail device/lab/lot distribution. At the installed-system boundary, include converter, balancing, monitoring and protection standby draw.

IEC 62391-2:2025 provides a leakage-current baseline for power-application EDLCs. Its publicly searchable excerpt specifies 20 °C ± 2 °C, rated voltage applied through a protective resistor no greater than 1 000 Ω, at least 12 h pre-discharge unless otherwise agreed, and 24 h electrification unless otherwise agreed; the applicable detail specification sets the leakage-current requirement. The same sectional specification separately includes a maintain-voltage test (5.11) referencing IEC 62391-1:2022, 6.4: at least 12 h pre-discharge, rated-voltage charging for 8 h, and a 24 h retention-voltage measurement unless otherwise agreed. Preserve both standardized procedures as comparators, but do not treat their endpoints as multi-month standby-life or post-standby usable-energy qualification. Use the controlled standard and manufacturer detail specification for physical testing; do not use search excerpts as a substitute for the normative document.

A synthetic arithmetic fixture is available at fixtures/energy/electrochemical-capacitor/standby_service_known_answer_v0_1.json. It validates trapezoidal integration of leakage charge and terminal V×I discharge energy only. It is not a physical device, a self-discharge model or lifetime evidence.

### Standby modes, service availability and fair comparisons

Separate three histories in the data model and leaderboard: open-circuit (unpowered) storage; regulated/constant-voltage hold with measured source replenishment; and installed-system standby with converters, balancing, BMS/sensing, communication, contactors and thermal controls. A constant-voltage hold can inject energy from an external source, so source-side V/I and integrated hold energy must be reported. It must not be credited as energy retained from the initial charge.

The primary endpoint is terminal energy delivered into the frozen representative load after standby, measured as \\(E_{\mathrm{load}}=\\int V_{\mathrm{term}}(t) I_{\mathrm{load}}(t)\\,dt\\) over the declared service window and voltage cutoffs. At installed-system boundary, report energy delivered to the application load and separately meter auxiliary energy during standby and conversion/conditioning losses. Compare against a matched initial reference on the same device boundary, voltage window, load waveform and temperature. Also report usable peak/continuous power, start latency, time-to-cutoff, recovery/refresh energy and temperature rise.

Do not collapse service into a single mean. Define a device-level service pass against frozen application requirements (minimum Wh, minimum W, allowed voltage sag/cutoff and maximum response time). Report pass fraction, lower-tail quantiles and a predeclared one-sided confidence bound by design, lot and laboratory; sample sizes are chosen from a pilot variance estimate and power analysis. A repeated read or cycle on one specimen is not another independent device.

A public vendor benchmark illustrates endpoint differences: Eaton's XLR-51 188 F module datasheet reports 8 mA nominal leakage at 20 °C after 72 h charge and hold, alongside separate 1 500 h elevated-temperature life and 1 000 000-cycle specifications. Skeleton's current 54 V / 188 F datasheet lists 10-year uncharged storage life at room temperature and separate cycle-life conditions. These are vendor-defined endpoints, not directly comparable post-standby delivered-energy measurements. Preserve each device's conditions and do not infer the missing outcome.

### Prospective standby interval grid

A useful exploratory dwell grid is 24 h, 72 h, 168 h (7 d), 720 h (30 d) and 2,160 h (90 d), with 6–12 month follow-up only where the target application needs it. These are programme design points, not IEC requirements. The 24 h point permits comparison with standardized time scales; Eaton's vendor-reported 72 h charge-and-hold leakage endpoint provides context for the second; week/month/quarter points measure application service beyond short laboratory checks. Begin with a variance and instrumentation pilot and select the confirmatory specimen count by power analysis. Keep each standby mode separate and test all conditions within rated and facility-approved limits.

Do not automatically re-use one specimen for every dwell: discharge/recharge and reconditioning change its prior history. Use randomized independent dwell cohorts for clean comparisons, or preregister a repeated-measures design whose history is explicitly represented. The application service pass is evaluated per physical serial against frozen Wh, W, voltage-sag/cutoff and response-time constraints; report pass fractions and one-sided lower confidence bounds by design and lot. Preserve stopped, invalid, failed and right-censored outcomes.

### Mechanism-identifiability research frontier

A 2026 review distinguishes three major self-discharge contributions: ohmic leakage, parasitic Faradaic reactions and charge redistribution. Voltage decay alone may not identify which mechanism dominates. Treat mechanism attribution as a hypothesis tested by orthogonal observations, not as a label inferred from a single fitted curve. Reuse the existing leakage, ESR/ESL and frequency-resolved impedance work rather than duplicating it; add matched open-circuit and regulated-hold histories, current transients, temperature/initial-voltage/history strata within rated limits, and post-standby load delivery. Fit competing models on training devices/lots and assess mechanism discrimination and service prediction on held-out lots; report non-identifiability and abstain where the observations cannot distinguish mechanisms.

### Mission-weighted standby-to-service readiness

Child experiment: [EDLC-MISSION-READINESS-001 #11953](https://github.com/Luminous-Dynamics/symthaea/issues/11953). A standby reserve is useful only if it can satisfy the actual load when called. This section qualifies the transition from an idle state to the application event, rather than a voltage or energy snapshot alone.

A mission stratum is a frozen tuple of standby mode, dwell duration, temperature and voltage history, prior conditioning, event waveform, service duration and application acceptance requirements. Keep open-circuit standby and regulated-voltage-hold in separate profiles; the latter records external source replenishment and is not autonomous reserve. The physical boundary (device, module or installed system) is separately declared. Event-arrival weights must originate from application logs, contractual mission requirements, or an explicitly labelled engineering assumption. They sum to one and are frozen before evaluation; there is no universal event-arrival distribution.

For each unique physical device, a service event passes only if all of the following hold on that same device:
- application-terminal delivered energy meets the declared Wh requirement;
- minimum sustained power over the load waveform meets the declared W requirement (peak sampled power is insufficient);
- start / wake / converter response latency meets the declared maximum;
- minimum application-terminal voltage remains above cutoff and voltage sag stays within bounds;
- temperature rise remains within the application limit;
- standby auxiliary energy remains within its budget;
- external replenishment energy remains within the declared allowance.

These are conjunctive requirements. Excess Wh cannot compensate for insufficient sustained W, excessive response latency, voltage collapse or thermal failure. Record the actual synchronized voltage/current/time and load waveform. At installed-system boundary meter the application terminals, standby auxiliary draw and conversion losses. During regulated hold meter source-side V/I and integrated hold energy; do not count source replenishment as initial stored energy.

For stratum j with frozen event weight w_j and observed service-pass fraction p_j, report the descriptive mission-weighted point estimate R_mission = Σ w_j p_j. Always publish every per-stratum result, the worst stratum and lower-tail delivered energy alongside the weighted figure. Report per-stratum one-sided Wilson lower confidence bounds; these are not a simultaneous confidence bound for the weighted total. A joint bound requires an explicitly justified multiplicity method. Invalid/censored outcomes remain visible and are not silently converted to passes.

Do not reuse the same device across different dwell cohorts unless a preregistered repeated-measures design explicitly models the complete prior history. Otherwise use randomized independent physical devices per stratum. Repeated cycles and reads are not independent specimens. Select confirmatory sample sizes from a pilot variance estimate and power analysis.

A synthetic known-answer scorecard is available at `fixtures/energy/electrochemical-capacitor/mission_readiness_known_answer_v0_1.json`. It uses invented 24 h / 7 d / 30 d weights of 0.2 / 0.3 / 0.5, 15 synthetic serials and a distinct `load_profile_id` per stratum to exercise conjunction and weighted arithmetic; its expected 0.54 mission pass rate is not a physical estimate or application probability. The shape contract is defined in `schemas/energy/edlc_mission_readiness_v0_1.schema.json`. The existing `xtask` exposes `edlc-mission-readiness-audit` and 14 authored negative-control tests. It checks schema structure, profile weight provenance, weight sum, load-profile identity, unique serials across dwell strata, no replenishment during open-circuit standby, all pass criteria, per-stratum rates and one-sided Wilson lower bounds. The audit is authored but has not been compiled or executed in this environment. The synthetic scorecard contains summary values only; it does not contain a physical application waveform, raw terminal trace, instrument record or calibration receipt.

## Explicit mass and volume denominator contract

Every record carries measured `mass_kg`, measured `volume_l`, a denominator basis, and explicit included/excluded component lists. A `complete` denominator must include positive and negative electrodes, electrolyte, separator, current collectors, binder/additives, tabs, seals and case, with both mass and volume present. Module records additionally include busbars, module enclosure and monitoring/balancing; installed-system records additionally include power conversion, thermal management, protection, wiring, system enclosure and controls. Vendor modules with only a reported mass remain partial until the complete denominator is known. The denominator basis must match the exact declared boundary (complete-device weighed, module measured, or installed-system measured). Evaluation eligibility requires measured energy-density and power-density metrics at that exact same boundary—not a metric transferred upward or downward—plus raw-data and evidence-package digests and a unique physical device serial. Dielectric capacitors are retained as a separate taxonomy entry but cannot be promoted by this electrochemical-device v0.1 audit; they need their own device and standards contract. Synthetic in-memory test records exercise this rule but are not physical observations.

## ZIHSC-specific evidence

In addition to the common vector, report:
- zinc utilization and remaining zinc inventory;
- N/P capacity ratio and its measurement basis;
- zinc areal capacity, current density and plating/stripping efficiency;
- electrolyte-to-capacity ratio and electrolyte mass/volume;
- cathode active loading, areal capacity, porosity and thickness;
- anode corrosion / mass loss, gas evolution, morphology and parasitic reaction evidence;
- self-discharge during defined charged storage;
- complete-device inactive mass/volume and full-cell energy efficiency.

The 2026 literature warns that >100 Wh/kg reports may be cathode-mass-normalized and low-loading; do not use these as full-device comparators. A November 2026 aqueous zinc-battery review reports a representative active-material value of 119.8 Wh/kg versus 33.6 Wh/kg at whole-cell boundary. This is a boundary warning, not a ZIHSC comparator.

## External comparators

### Practical high-loading ZIHC comparator (July 2026)

Wu et al., *Advanced Science*, first published 27 July 2026, deliberately studied the kinetics–loading trade-off in zinc-ion hybrid capacitors. Raising carbon cathode loading from 2 to 10 mg cm⁻² impaired high-rate retention: at 20 A g⁻¹, reported capacity retention fell from 43.8% to 17.0%. The authors addressed both transport channels by combining CNT conductive networks with directional freeze-dried, low-tortuosity long-range pore channels. Their pouch cell used 10 mg cm⁻² carbon loading and N/P ≈16.4, delivering 38.3 mAh g⁻¹ at 10 A g⁻¹ over 3,000 cycles, with 99.1% Coulombic efficiency. The reported device-level energy density was 2.82 Wh kg⁻¹ based on a 0.746 g total device weight.

This is a useful practical comparator precisely because the device-level energy is much lower than many active-material-normalized headlines. Preserve the authors' own device mass basis and distinguish their coin-cell screening, electrode metrics and pouch-cell result. Do not treat 2.82 Wh kg⁻¹ as a universal commercial benchmark or claim direct parity without reproducing the same protocol and denominator.

**Research implication:** our first ZIHC ablation should compare (i) conventional thick porous carbon, (ii) conductive-network-only, (iii) long-range pore architecture-only, and (iv) combined electron/ion transport design, at matched loading, electrode balance, electrolyte quantity, cell format and test protocol. This is an ablation of architecture and transport, not a race for a novel chemical formula.

Source: Wu et al., DOI [10.1002/advs.76823](https://doi.org/10.1002/advs.76823); [full text](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.76823).


### Additional commercial comparator — Skeleton SkelMod 54V / 188F

Skeleton's 5 June 2026 datasheet specifies a 54 V / 188 F SkelMod module with 76.14 Wh maximum stored energy, 16 kg typical mass, 13.5 L volume, 3.0 mΩ rated DC 10 ms ESR, 3.4 mΩ rated DC 1 s ESR and 162.6 kW power calculated from 1 s ESR. The module includes an integrated supercapacitor management system, cell balancing, four temperature sensors and CAN communication, with natural cooling. The datasheet lists IP65 protection with the IP covers installed; its notes state that the IP covers are separately supplied.

Dividing vendor-stated maximum stored energy by vendor module mass gives approximately **4.76 Wh/kg**; dividing by the vendor-stated module volume gives **5.64 Wh/L**. These are maximum-stored-energy proxies at module boundary, not independently measured usable energy or installed-system metrics. Preserve rated voltage, voltage window, test conditions and module BOM disclosure. The datasheet's cycle-life rows specify 51 V and 48 V variants/voltage windows; do not transfer those cycle-life figures to the 54 V variant.

The commercial landscape therefore already includes high-power modules with integrated balancing, monitoring, communications and environmental protection. Our potential wedge is not "a capacitor with low ESR" alone. It must be a device or module that improves a declared application service—e.g. power transient support per installed mass, cost, heat, lifetime or local manufacturing constraint—against these incumbent modules.

Source: [Skeleton SkelMod 51V / 54V188F data sheet](https://www.skeletontech.com/hubfs/01-DS-211229-SKELMOD-51V177F-1D.pdf?hsLang=en).

### Commercial EDLC module

Eaton XLR-48 public datasheet reports:
- maximum working voltage: 48.6 V;
- capacitance: 166 F;
- stored energy: 54 Wh;
- maximum initial ESR: 5 mΩ;
- a 1,000,000-cycle specification with stated capacitance / ESR change limits and vendor test conditions.

Use only as a **module-level commercial reference**. Obtain the complete datasheet and preserve its test method, temperature, voltage and lifetime conditions. Do not infer a cell-level energy density from the module record.

### Standards map

- IEC 62391-1:2022: generic EDLC terms, inspection, measurement and quality assessment.
- IEC 62391-2:2025: sectional specification for power-application EDLCs.
- UL 810A Second Edition (published 3 February 2026): electrochemical capacitor product requirements for its stated equipment scope.
- Battery standards such as IEC 62619 and grid-system standards such as IEC 62933-5-2 may apply to different subsystems/products. They are not interchangeable with capacitor standards.

Standards applicability is jurisdiction-, application-, chemistry- and product-boundary-specific. Testing to a standard is not a certification decision.

## Initial physical experiment plan

### A. Establish a reference

1. Select a commercially available EDLC module and a laboratory EDLC control with complete manufacturer or fabrication metadata.
2. Freeze a measurement protocol and voltage/temperature/current windows before collecting comparative data.
3. Calibrate mass, volume, voltage, current, time and temperature measurement chains; retain calibration records.
4. Measure capacitance, usable energy, ESR/EIS, leakage/self-discharge, efficiency, pulse/continuous power and thermal rise.
5. Record every device including failures, invalid runs and outliers with declared disposition.

### B. Carbon EDLC screen

A research partner selects safe, established carbon electrode variants for a controlled comparative screen. Potential factors include pore architecture / surface chemistry, electrode loading / thickness, binder/conductive additive fraction and electrolyte family. Use a preregistered fractional-factorial or optimal DOE, batch-blocked randomisation and matched device geometry. Do not change multiple uncontrolled variables between the comparator and candidate.

The first question is not “which carbon has the highest BET surface area?” It is “which reproducible electrode/process combination delivers better complete-device energy, ESR, power, leakage and lifetime under the declared duty cycle?”

### C. ZIHSC screen

Run only after a facility and qualified supervisor confirm chemical, electrical and waste controls. Keep this lane separate from EDLC.
- Compare a declared zinc-based electrode and capacitive/pseudocapacitive counter-electrode using matched complete-device boundaries.
- Record zinc excess/utilization, N/P ratio, electrolyte quantity, loading and all inactive mass.
- Evaluate corrosion, gas evolution, self-discharge and failure modes alongside energy/power.
- Do not promote a coin-cell result to pouch or module performance.
- Keep material synthesis and hazardous wet-chemistry steps under partner laboratory procedures and authorization.

### D. Hybrid system demonstration

Use commercially qualified battery and EDLC modules at low voltage with current limiting, fusing, enclosure, thermal monitoring and independent review. Compare battery-only and battery-plus-capacitor configurations under the same load trace.

Measure:
- peak and RMS battery current;
- battery voltage sag and temperature;
- capacitor energy throughput and ESR heating;
- converter round-trip loss and standby draw;
- total system mass, volume and cost;
- battery stress / degradation only over a sufficiently long matched test.

The capacitor is useful only when system-level benefit exceeds the additional mass, losses, complexity and cost.

### Model-informed long-range ZIHC envelope (not a physical SOTA result)

Wu et al., *Advanced Science* (6 November 2024), analyse four coupled parameters in a representative ZIHC cell model: carbon loading (m_c), carbon specific capacity (Q_{g,c}), N/P and electrolyte-to-carbon ratio E/C. Their model indicates that simultaneous ranges such as (m_c=10–20\;\mathrm{mg\,cm^{-2}}), (Q_{g,c}>100\;\mathrm{mAh\,g^{-1}}), N/P <20 and E/C <5 are needed to approach >30 Wh kg⁻¹ cell energy density.

**Critical boundary caveat:** the authors state that their model excludes stainless-steel casing, springs and current-collecting plates. Therefore >30 Wh kg⁻¹ is a model-derived cell-boundary design envelope, not a measured complete packaged-device result. We must add all excluded components and measured volume before comparing it with complete-device data.

This yields a useful two-step target: first reproduce the 2026 measured pouch-cell comparator (10 mg cm⁻², N/P ≈16.4, 2.82 Wh kg⁻¹ based on 0.746 g total device weight); then explore the four-parameter envelope, but report the full packaged-device metric independently. Source: [Wu et al., DOI 10.1002/advs.202408997](https://doi.org/10.1002/advs.202408997).


## Matched SOTA targets for ZIHC

- **Parity experiment:** reproduce the Wu et al. 2026 pouch-cell result on the same reported basis: 10 mg cm⁻² carbon cathode loading, N/P ≈16.4, 10 A g⁻¹, 3,000 cycles, specific capacity and Coulombic efficiency, with complete device mass accounting and a disclosed 0.746 g denominator comparator.
- **Stretch experiment:** improve device-level usable Wh/kg or Wh/L by at least 20% versus a matched reproduction while keeping cathode loading at or above 10 mg cm⁻², N/P no worse than 16.4, and 3,000-cycle capacity retention / energy efficiency no worse than the matched baseline. This is an internal research target, not an achieved result.
- **Robustness experiment:** demonstrate the same improvement on at least three independent material lots, with each device tracked separately and uncertainty intervals reported. Only later, after the cross-lab minimum and power analysis, claim independent-lab reproducibility.

The 20% target is deliberately subordinate to matched test conditions. It cannot be met by reducing the inactive-mass denominator, lowering the loading, increasing zinc excess, increasing electrolyte quantity or changing the voltage window without separately disclosing and benchmarking those changes.

## Benchmark tournament

Compare:
- commercial EDLC baseline;
- conventional carbon EDLC;
- pseudocapacitive control (separate mechanism);
- ZIHSC candidates;
- battery-only system;
- battery + capacitor system.

For experiment selection compare random, space-filling, conventional DOE, uncertainty sampling, diversity sampling and decision-focused selection under identical frozen experiment/time/material/compute budgets.

Primary comparison is a mission-conditioned Pareto frontier. A new design must improve at least one predeclared mission objective without violating hard constraints or predeclared non-inferiority margins for safety, lifetime, efficiency and manufacturability. Choose thresholds after baseline variance and mission acceptance requirements are known; do not retrofit targets after observing results.

## Reproducibility

- Separate device replication from repeated cycling.
- Block randomization by independent material batch and fabrication date.
- Record fixture, instrument and calibration uncertainty.
- Use independent physical serial numbers.
- Preserve right-censored lifetime observations, stopped tests, invalid tests and failures.
- A cross-laboratory variance analysis is not eligible until at least three independent labs each test at least three unique physical devices under a common protocol. Actual sample count must be selected by power analysis.
- Report common-protocol, lab-native and deviation experiments separately.

## Software contract acceptance

- Schema rejects missing device class, boundary, physical device identity or protocol identity.
- Active-material-only metrics cannot satisfy a complete-device metric requirement.
- Vendor/literature/synthetic evidence cannot satisfy a physical-measurement or independent-lab hard gate.
- Missing or unknown data remains UNKNOWN; it is not imputed into a pass.
- Invalid, failed and censored observations remain represented.
- Device, material lot, process, instrument, calibration and source identities are immutable/versioned.
- JSON key order and input record order do not change canonical audit results.
- Negative controls include wrong boundary, mixed device classes, missing inactive mass, impossible unit/boundary joins, duplicate physical serials, uncalibrated instrument, expired/missing protocol, fabricated physical evidence, and false promotion of vendor claims.

## Local audit command

The workspace `xtask` exposes a dependency-free domain audit using the existing `serde_json` dependency:

```sh
cargo run -p xtask -- electrochemical-capacitor-audit
cargo test -p xtask electrochemical_capacitor
cargo run -p xtask -- edlc-standby-service-audit
cargo test -p xtask edlc_standby_service
cargo run -p xtask -- edlc-mission-readiness-audit
cargo test -p xtask edlc_mission_readiness
```

The existing capacitor command checks schema/fixture structure, canonical identifiers, device and metric categories, units/voltage-window sanity, evidence-class/status alignment, digest syntax, duplicate physical serials, source-register uniqueness and fail-closed promotion rules. The standby command enforces its synthetic trace contract, recomputes trapezoidal leakage charge and terminal V×I discharge energy, checks peak sampled power and rejects unit, chronology, sign, expected-value or claim-ceiling violations. The mission-readiness command checks event-weight provenance and normalization, unique serials across strata, operating-mode separation, conjunctive Wh/W/latency/voltage/thermal/auxiliary-energy requirements, descriptive weighted pass rates and per-stratum one-sided Wilson lower bounds. These audits are not complete JSON Schema evaluators, physical instrument authenticators or lifetime models. Rust compilation, formatting, tests and hosted CI remain pending until a Rust toolchain/runner is available.

## Claim ceiling

A schema/fixture/CLI pass establishes software semantics only. No capacitance, energy density, power, cycle life, safety, process yield, scale-up, commercial viability or SOTA superiority is established until physical devices are made and independently measured.

## Sequential reserve-service extension (2026-11-13)

A first-call standby result does not prove repeated-call readiness. The new [EDLC-SEQUENTIAL-RESERVE-001 contract](EDLC_SEQUENTIAL_RESERVE_BENCHMARK_V0_1.md) qualifies the complete ordered trajectory: initial standby, first call, inter-call recovery, later calls, and a non-compensatory sequence pass. It explicitly detects late recovery and false-ready outcomes. The synthetic fixture expects 1/3 complete sequences to pass and one false-ready event. Later calls now require an exact readiness-probe profile ID, identical across compared sequences; these are known-answer arithmetic only.

The Rust audit is available as `cargo run -p xtask -- edlc-sequential-reserve-audit`. The module and 16 tests are authored but remain uncompiled and unexecuted in this environment. Physical ingestion, calibrated traces, statistical bounds, lifetime prediction and independent lab replication remain future gates.

New literature comparators registered: Shijas and Dash (2026), a module-level study with dynamic load transitions and rate-sensitive effective capacitance; and Yuan et al. (2026), an operando-characterization review that highlights reproducibility and standardization gaps. Neither is treated as a universal EDLC benchmark or Luminous result.
