# Electrochemical capacitor device benchmark v0.1

Status: research and measurement contract. This document is not a synthesis recipe, device design release, safety certification, commercial claim or permission to operate laboratory equipment.

Parent: [CAPACITOR-DEVICE-DISCOVERY-001 #11841](https://github.com/Luminous-Dynamics/symthaea/issues/11841).

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
| Thermal behavior | Temperature rise under stated pulse/continuous load; sensor location and uncertainty |
| Safety | Defined abuse/short-circuit/overvoltage/thermal test, facility, procedure, result and failure observations |
| Manufacturing | First-pass yield, accepted yield, scrap, rework, batch variability, material utilisation, process energy and cost |

Do not state “cycle life” without the voltage window, current profile, temperature, rest, end-of-life threshold and treatment of failed/invalid devices.

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

## Claim ceiling

A schema/fixture/CLI pass establishes software semantics only. No capacitance, energy density, power, cycle life, safety, process yield, scale-up, commercial viability or SOTA superiority is established until physical devices are made and independently measured.