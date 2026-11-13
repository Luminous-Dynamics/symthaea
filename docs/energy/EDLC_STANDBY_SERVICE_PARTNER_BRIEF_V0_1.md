# Partner brief: EDLC standby service and voltage-hold reliability

**Version:** 0.1  
**Date:** 13 November 2026  
**Status:** research measurement proposal. No partner, facility access, specimen, test authorization, or physical result is claimed.

## Purpose

Determine how much useful electrical service a fully identified EDLC device or module can deliver after a declared standby or voltage-hold interval. The central output is measured usable energy and power after standby—not open-circuit voltage retention alone.

Target applications include emergency/ride-through reserve, intermittent renewable harvesting, remote monitoring, autonomous sensors and industrial DC-link/control reserve.

## Why it matters

Cycle-life tests do not fully characterize standby service. A capacitor may exhibit high capacitance retention while losing useful energy through leakage/self-discharge, suffering ESR growth, or failing to maintain the required voltage under load. A 24-hour leakage-current test is a useful standardized baseline but does not establish multi-week or multi-month standby life.

IEC 62391-2:2025 provides the power-application EDLC reference. Its publicly searchable leakage-current procedure specifies 20 °C ± 2 °C, rated voltage through a protective resistor no greater than 1 000 Ω, at least 12 hours discharge before the test unless otherwise agreed, and 24-hour electrification unless otherwise agreed; the applicable detail specification sets the leakage-current requirement. The same sectional specification separately includes a maintain-voltage test (5.11) referencing IEC 62391-1:2022, 6.4: at least 12 hours pre-discharge, rated-voltage charging for 8 hours, then a 24-hour retention-voltage measurement unless otherwise agreed. These are useful standardized quality-assessment procedures, but neither substitutes for measuring usable energy and power delivered into a representative load after the declared standby history. The host laboratory must use the controlled normative document and the applicable manufacturer's specification.

Source: https://webstore.iec.ch/en/publication/82213

## Proposed measurement chain

1. Identify exact device/module serial, design revision, materials/process lots, voltage/current ratings and complete component boundary.
2. Establish instrument/fixture identity and calibration; record temperature and environmental conditions.
3. Measure initial capacitance, ESR/EIS, leakage, and usable discharge energy/power under a frozen profile.
4. Run the applicable standardized leakage measurement under the controlled standard and manufacturer detail specification.
5. Run separate open-circuit self-discharge and controlled voltage-hold arms for declared use cases; do not treat these as interchangeable.
6. After each declared standby interval, discharge into the frozen representative load and measure terminal V/I/time, delivered Wh, W, voltage sag, thermal behavior and cutoff time.
7. Re-measure capacitance, ESR/EIS, leakage and physical/package condition.
8. Retain failures, invalid, stopped and censored observations with reasons.

Longer standby windows (for example, a 168-hour diagnostic) are exploratory and must be compatible with the rated device envelope, manufacturer restrictions and facility review. Never exceed rated voltage, temperature or approved test limits to accelerate aging.

## Primary outputs

- Usable Wh delivered after each standby interval.
- Usable W and time to minimum useful voltage under the actual load.
- Standby energy retention relative to a matched initial reference.
- Charge/hold/discharge service efficiency and recovery/refresh energy.
- Leakage current versus time and applied voltage.
- Open-circuit voltage/self-discharge versus time.
- ESR and complex impedance evolution.
- Capacitance, thermal behavior, gas/swelling, seal/terminal/package condition and failure mode.
- Device, module and installed-system mass/volume at explicitly separate boundaries.
- Device/lot/lab variation, lower-tail confidence bounds, censoring and competing failure modes.
- Cost per delivered standby Wh/event and maintenance/replacement burden where data permit.

Open-circuit voltage alone does not establish delivered energy. The discharge trace must include simultaneous terminal voltage and current over the declared load window.

## Standby-mode separation and source-energy accounting

Do not pool these operating histories:

- **Open-circuit standby:** isolate the device from external charging; measure voltage decay and later deliverable energy under the frozen load. This is the autonomous-reserve case.
- **Regulated/constant-voltage hold:** a source actively maintains voltage and may replenish lost charge. Record source-side V/I and integrated hold energy, current-vs-time, control duty and all balancing/monitoring draw. Do not count energy supplied during the hold as energy retained from the initial charge.
- **Installed-system standby:** include converter quiescent draw, balancing, BMS/sensing, communications, contactors and thermal management at the declared system boundary; report external auxiliary supply separately where used.

For every interval, pair the post-standby load discharge with a matched initial reference using the same voltage window and load profile. Report delivered Wh and W, cutoff/start latency, hold-source energy, auxiliary energy, recovery/refresh energy, and a service-pass outcome against application limits. The service-pass fraction and a predeclared one-sided confidence bound are more decision-relevant than the mean alone; sample counts must follow a variance/power pilot, with unique physical devices and lots—not repeated cycles—forming the independent units.

A current manufacturer comparator also demonstrates why conditions must stay attached to each number: Eaton's XLR-51 188 F module datasheet reports 8 mA nominal leakage at 20 °C after 72 h charge and hold. That is a vendor leakage specification, not post-standby delivered Wh and not unpowered calendar standby. Skeleton's 54 V / 188 F datasheet reports uncharged storage life but does not provide a directly comparable charged-standby delivered-energy result. Do not rank devices by these unlike endpoints.

## Comparators

Use traceable commercial EDLC modules such as Eaton XLR-48 and Skeleton SkelMod 54V/188F, keeping vendor specifications separate from independent laboratory measurements. Use a traceable laboratory EDLC control for device-level comparisons. Compare only at matched device/module/system boundaries and matched voltage window, load, temperature and history.

Vendor-stated cycle-life or leakage values remain manufacturer claims unless independently tested.

## Reliability analysis

Use a pilot to characterize variance, then determine confirmatory sample size by power analysis. Begin with at least three independent fabrication lots for initial process repeatability, but do not treat that minimum as a sufficient sample count.

For prognostics, compare:
- no-model baseline;
- constant leakage model;
- empirical exponential self-discharge model;
- richer statistical/mechanistic models only where the data support them.

Hold out complete designs/lots and at least one temperature or standby stratum. Report prediction interval coverage, calibration, worst-group error, false-pass rate at the application service threshold and abstention. Do not extrapolate beyond the validated time/temperature/voltage domain.

## Partner capability requested

- Calibrated EDLC cycler and leakage-current measurement chain.
- Controlled voltage-hold / calendar-ageing capability.
- Simultaneous terminal voltage/current logging and low-current resolution appropriate to device class.
- Environmental temperature monitoring and, where applicable, humidity control.
- ESR/EIS characterization and physical inspection.
- Documented instrument/calibration records and raw trace export.
- Competent electrical safety review, rated-limit controls, protective paths, enclosure, fusing, thermal monitoring and emergency response.

Facility capability must be confirmed directly. This brief does not imply a partnership or access.

## Luminous contribution

Subject to written agreement:
- frozen use-case and matched-comparison protocol;
- immutable device/lot/fixture/calibration/raw-data identities;
- synthetic arithmetic fixtures and independent trace integration;
- uncertainty, held-out validation and negative controls;
- data ingestion and reliability analysis in Symthaea/SAM, with Symtropy supporting later system-level standby service analysis.

No AI prediction or software audit substitutes for calibrated physical testing.

## Data and IP terms

Before physical work, agree:
1. background and foreground IP;
2. access and reuse rights for raw traces, images, metadata and derived results;
3. data retention, independent reanalysis and publication terms;
4. attribution, confidentiality and embargo;
5. retention and disclosure of negative, invalid and failed observations.

## Safety and claim boundary

This document authorizes no purchase, wiring, charge, test, equipment operation or product release. Physical work is under the partner facility's approved procedures, manufacturer's operating limits and competent-person authority.

A leakage test or arithmetic fixture proves no long-term standby life, safety, product qualification or commercial superiority.

## Public tracking

- EDLC reference parent: https://github.com/Luminous-Dynamics/symthaea/issues/11852
- Focused standby experiment: https://github.com/Luminous-Dynamics/symthaea/issues/11908
- Capacitor discovery parent: https://github.com/Luminous-Dynamics/symthaea/issues/11841
- Benchmark draft: https://github.com/Luminous-Dynamics/symthaea/pull/11844
