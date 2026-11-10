# Electric aircraft benchmark plan (research targets, not achieved results)

Baseline date: 2026-11-10. Public specifications and regulatory sources change; every benchmark run should pin the source revision, aircraft configuration, mission, environmental conditions, and software/model commit.

## Industry reference points

- **Pipistrel Velis Electro:** EASA type-certified two-seat electric trainer. Its manufacturer publishes 57.6 kW maximum takeoff power, 35 kW cruise condition, 600 kg MTOW, 172 kg payload, 15:1 best-glide ratio, and up to 50 minutes endurance plus VFR reserve. It uses two liquid-cooled batteries (approximately 70 kg each) in a 24.8 kWh, 345 VDC system. A normal 35–95% charge takes up to 1 h 20 min; 30–100% can take up to 2 h depending on ambient temperature and battery age. Pipistrel publishes an initial 500-hour battery limitation and 2,000-hour motor TBO. These are configuration-specific operating limits, not universal battery-life figures. This is the clearest certified light-electric baseline, not a directly comparable cargo or regional aircraft. Source: https://www.pipistrel.si/products/general-aviation/velis-electro/
- **BETA ALIA CTOL CX300 / A250:** BETA's 2025 Form 10-K describes the CX300 as a fixed-wing electric utility aircraft intended to carry six people or 200 cubic feet of cargo plus two crew on missions up to approximately 215 nautical miles. The certification-intent CTOL carries 225 kWh rated onboard energy in five nominally identical 45 kWh packs; BETA says the architecture allows the aircraft to be configured with four or five packs according to range/payload target. BETA also reported a 336-nautical-mile proof-of-concept flight on one charge; that should be kept separate from the certification-intent mission, payload and reserve conditions. Its Q3 2026 release (2026-11-03) reports more than 250,000 nautical miles flown across its fleet, 168 open/interoperable charging sites, launches of six of seven FAA eIPP selections, CX300 market-survey authorization, and an A250 Special Airworthiness Certificate for customer market-survey flights. These are important operational and integration milestones, **not** a CX300/A250 type certificate. The Q3 release reports 1,040 firm and option orders and >$4B backlog; options/backlog are not deliveries or certified operations. BETA's current motor page lists H500B values of 640 kW maximum, 534 kW takeoff, 300 kW maximum continuous, 97.5% ISA efficiency, 6.0 kW/kg maximum and 4.2 kW/kg continuous power density, and 75 kg installed mass including inverters. Those published power, density and mass figures do not reconcile arithmetically under one common mass boundary; do not derive or compare specific power until BETA clarifies the configuration and denominators. Treat all manufacturer disclosures as company-reported until independently measured. Sources: https://investors.beta.team/financials/annual-reports/content/0001628280-26-015838/bta-20251231.htm ; https://www.sec.gov/Archives/edgar/data/1784570/000162828026069120/q32026earningsrelease.htm ; https://beta.team/motor
- **Heart Aerospace ES-30:** the manufacturer's current programme page describes a 30-seat hybrid-electric aircraft with 200 km all-electric range, 800 km hybrid range, 30-minute charging, and a 2031 type-certification target. These are programme targets, not certified performance. Source: https://www.heartaerospace.com/es-30
- **Safran ENGINeUS 100 B1:** EASA announced the first type certificate under SC E-19 for this electric engine on 2025-02-04. This is a useful propulsion-certification reference; an engine certificate is not aircraft certification. Source: https://www.easa.europa.eu/en/newsroom-and-events/press-releases/how-easa-certified-safrans-engineus-100-electric-engine
- **NASA electrified-aircraft propulsion research:** NASA's programme includes the 1.4 MW partially superconducting High-Efficiency Megawatt Motor (HEMM), integrated powertrain architecture work, and the HEATherR thermal-management concept. A NASA HEMM electromagnetic redesign paper describes original design objectives above 16 kW/kg electromagnetic specific power and above 98% efficiency; those are motor-design objectives at a specific electromagnetic/motor boundary, not demonstrated installed propulsion-system performance including cryocooler, inverter, cooling, containment, wiring and mounts. NASA's HEMM material also describes cryogenic rotor cooling and ongoing test work. HEATherR investigates power-system loss reduction by a factor of four and local outer-mold-line cooling to reduce coolant-loop mass, cooling power and drag. NEAT enables full-size/multi-MW powertrain ground tests under simulated altitude up to 60,000 ft. These are research architectures and test capabilities, not evidence that a complete aircraft system has achieved the targets. NASA's 2026 powertrain-health digital-twin work uses nominal testbed data to update a nonlinear model and a Kalman filter to identify/isolate off-nominal component behaviour. A 2024 NASA parametric battery study argues for replacing fixed specific-energy/specific-power black boxes with voltage/current/SOC-dependent battery models. NASA's integrated-powertrain risk analysis identifies battery performance shortfall and the mass/volume of thermal-runaway protection as major barriers. Sources: https://www.nasa.gov/mission/eap/ ; https://www.nasa.gov/eap-technology/electric-machines/hemm/ ; https://ntrs.nasa.gov/citations/20205004521 ; https://www.nasa.gov/eap-technology/heather/ ; https://www.nasa.gov/eap-labs-and-testbeds/neat/ ; https://ntrs.nasa.gov/citations/20260004592 ; https://ntrs.nasa.gov/citations/20240007599 ; https://ntrs.nasa.gov/api/citations/20250006289/downloads/Integrated%20Powertrains%20Electrified%20Aircraft.pdf

## New energy-storage frontier (2026)

- ARPA-E / DOE's JOULES-1K Phase 2 targets at least 1,000 Wh/kg and 1,000 Wh/L at **end-of-life, net energy-storage-system level** for selected projects serving drones, robots, eVTOLs and related applications. This is a funded R&D target, not a commercially qualified manned-aircraft battery and not a suitable assumed aircraft design input. Some projects target cell-level values >1,000 Wh/kg while separately targeting lower module/system figures; preserve the exact measurement boundary. Source: https://www.energy.gov/nepa/articles/cx-270980-jumpstart-opportunities-unleash-leadership-energy-storage-1k-energy-storage and https://arpa-e.energy.gov/sites/default/files/2026-01/JOULES-1K_Phase%202_project_descriptions_Final.pdf
- NASA's thermal-fluid module work evaluates liquid cooling of an integrated battery module at 0.5C–2.5C discharge. Its significance for Luminous is methodological: test module, cooling loop, pressure drop, flow and temperatures together, then reconcile the added cooling mass and power against usable energy/power. Source: https://ntrs.nasa.gov/citations/20260008248

## Regulatory and market context

- FAA AC 20-184A was issued 2026-09-18 and provides an acceptable means (not the only means) of compliance for rechargeable lithium battery installation, operation, and maintenance in covered aircraft categories: https://www.faa.gov/airports/resources/advisory_circulars/index.cfm/go/document.information/documentID/1045700
- GAO-26-107816 (2026-05-27) reports that, as of March 2026, FAA had not issued a type certificate for a manned electric aircraft and was evaluating electric designs case-by-case: https://www.gao.gov/products/gao-26-107816
- EASA SC E-19 defines special conditions for electric/hybrid propulsion certification: https://www.easa.europa.eu/en/special-condition-sc-e-19-electric-hybrid-propulsion-system
- SACAA's airworthiness page explains that a South African Type Acceptance Certificate (TAC) accepts a foreign type certificate, and that the foreign type-certificate holder applies; a TAC is not a substitute for an original type-certification programme for a new Luminous aircraft: https://www.caa.co.za/industry-information/airworthiness/

## Operational-life and infrastructure benchmarks

The industry comparison must include more than energy density. Pipistrel's published 35–95% charging time, 30–100% charging time, liquid cooling, non-swappable battery arrangement, initial 500-hour battery limitation, and 2,000-hour motor TBO establish useful operational reference points. BETA's 10-K estimates typical battery replacement after 12–24 months depending on customer usage and reports a five-pack modular architecture, in-house destructive battery testing, and a charger network. Those are manufacturer statements, not independently comparable fleet data.

For every Luminous aircraft concept, collect and compare: battery calendar age and equivalent full cycles at retirement; energy throughput to retirement; usable energy at defined SOC/temperature/age; replacement labour and pack cost; charger power and grid connection; charging turnaround including thermal conditioning; dispatch availability; maintenance hours; and payload-kilometres per pack-life. Do not compare calendar-life or cycle-life without matching depth of discharge, C-rate, temperature, storage conditions, and end-of-life criterion.

## Benchmark tiers

### Tier 0 — analytical/software correctness

- Dimensional checks and unit tests for every equation.
- Monotonicity: more mass, higher speed at fixed L/D, lower chain efficiency, lower SOH, lower initial SOC, or higher protected reserve must not improve the energy margin.
- Adversarial inputs: NaN, infinity, negative values, duplicate IDs, zero/invalid efficiencies, impossible voltage/power combinations.
- Deterministic replay: identical manifest + inputs + model version must reproduce identical outputs.
- Independent oracle: cross-check calculations in a separately authored spreadsheet or numerical script; do not share implementation code.

### Tier 1 — subsystem correlation

- Motor/inverter efficiency-map prediction versus dynamometer measurements.
- Propeller thrust/torque/power maps versus static and wind-tunnel/rig testing.
- Pack voltage/current/resistance/capacity/thermal prediction versus controlled discharge and thermal-chamber data across SOC, temperature, current rate, charge/discharge direction, SOH and rest time.
- BMS sensor and estimator residuals across SOC, temperature, ageing, and charge/discharge rates.
- Power-distribution fault injection: open circuit, short circuit, sensor failure, contactor failure, insulation fault, and common-cause supply loss.

### Tier 2 — aircraft ground and flight-test correlation

- Aerodynamic polar and drag prediction against validated CFD, wind tunnel, and flight-test identification.
- Mission energy: predicted versus measured Wh by phase, with bias, RMSE, and calibrated prediction intervals.
- Takeoff distance, climb gradient/rate, cruise power, glide performance, and landing approach envelope.
- Thermal excursions under worst-case ambient, climb, turnaround, and degraded-cooling conditions.
- Dispatch reliability, turnaround time, maintenance hours, and cost per payload-kilometre.

### Tier 3 — safety and certification evidence

- FHA/PSSA/SSA traceability, FMEA/FMECA, fault-tree analysis, common-cause analysis, and zonal safety assessment.
- Battery abuse/propagation/containment tests and aircraft-level installation evidence.
- Software/hardware assurance and verification evidence appropriate to the agreed certification basis.
- Environmental, EMC, lightning, HIRF, vibration, icing, bird-strike, crashworthiness, and continued-airworthiness evidence as applicable to the aircraft category and design.
- Independent audits and regulatory engagement; no model score substitutes for certification.

## Comparison rules

1. Compare identical payload, route length, altitude, airspeed, weather, reserve policy, battery state-of-health, and operational assumptions.
2. Separate measured, certified, manufacturer-claimed, simulated, and aspirational values.
3. Normalize by payload-kilometre and include battery mass, reserve, charging losses, degradation, maintenance, and ground infrastructure.
4. Report confidence intervals and failed/incomplete runs; never select only favourable seeds or scenarios.
5. Freeze all source data and manifests by hash before comparing versions.

Use [METRIC_BOUNDARIES.md](METRIC_BOUNDARIES.md) for mandatory numerator/denominator, configuration, operating-point, evidence-class and uncertainty definitions. A comparison with mismatched or unresolved boundaries is `Incomplete`, not a valid SOTA comparison.

## Proposed stretch goals

These are internal research goals, not industry-standard thresholds or demonstrated capabilities:

- Mission-energy prediction: bias within ±3% and RMSE below 5% on a preregistered validated test set before using the model for detailed design trades.
- Critical component electrical/thermal model residuals: publish calibrated prediction intervals with at least 95% empirical coverage on an independent holdout set.
- Design exploration: demonstrate at least 20% reduction in an explicitly defined lifecycle-cost or energy-per-payload-kilometre objective against a frozen baseline, with no degradation in required safety margins, payload, mission, or dispatch constraints.
- Configuration/evidence completeness: 100% of safety-critical components linked to approved configuration, supplier/material traceability, inspection, test, and maintenance records before release review.

A stretch goal is not an acceptance criterion until its measurement protocol and baseline are preregistered and independently reviewed.

Do not adopt a generic motor kW/kg target from a different mass boundary. NASA's HEMM >16 kW/kg electromagnetic design objective is a higher motor-design reference but excludes the complete installed-system boundary needed for an aircraft comparison. BETA's H500B lists 6.0 kW/kg maximum and 4.2 kW/kg continuous, alongside 640 kW maximum, 300 kW continuous and 75 kg installed including inverters; because those stated values do not reconcile exactly to one denominator, preserve the published values and mark the specific-power comparison unresolved. First establish a common motor-only and motor+inverter+cooling+mounts boundary, at both peak and thermally steady continuous duty, before setting a Luminous numerical target.
