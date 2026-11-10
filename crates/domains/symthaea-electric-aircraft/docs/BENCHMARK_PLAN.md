# Electric aircraft benchmark plan (research targets, not achieved results)

Baseline date: 2026-11-10. Public specifications and regulatory sources change; every benchmark run should pin the source revision, aircraft configuration, mission, environmental conditions, and software/model commit.

## Industry reference points

- **Pipistrel Velis Electro:** EASA type-certified two-seat electric trainer. Its manufacturer publishes 57.6 kW maximum takeoff power, 35 kW cruise condition, 600 kg MTOW, 172 kg payload, 15:1 best-glide ratio, and up to 50 minutes endurance plus VFR reserve. It uses two liquid-cooled batteries (approximately 70 kg each) in a 24.8 kWh, 345 VDC system. A normal 35–95% charge takes up to 1 h 20 min; 30–100% can take up to 2 h depending on ambient temperature and battery age. Pipistrel publishes an initial 500-hour battery limitation and 2,000-hour motor TBO. These are configuration-specific operating limits, not universal battery-life figures. This is the clearest certified light-electric baseline, not a directly comparable cargo or regional aircraft. Source: https://www.pipistrel.si/products/general-aviation/velis-electro/
- **BETA ALIA CTOL CX300:** BETA's 2025 Form 10-K describes a fixed-wing electric utility aircraft with a manufacturer-stated mission of approximately 215 nautical miles and 225 kWh onboard rated energy. A November 2026 IEEE Spectrum report describes five liquid-cooled packs of roughly 45 kWh each, about 255 kg each: approximately 176 Wh/kg on rated pack energy, with the caveat that not all rated energy is usable in flight. The same article reports a 75 kg, 640 kW H500A motor (about 8.5 kW/kg) and 97.5% efficiency from internal testing; treat this as a reported test figure, not independently verified mission-average efficiency. It reports the CX300's expected FAA certification in 2027 and says the earlier first-half-2026 motor-certification target passed without a public announcement. BETA's SEC-filed Q2 2026 update said CX300 requirements definition had closed with FAA acceptance of the compliance requirements and H500A formal FAA testing was underway at that reporting date. IEEE also describes BETA's shared CX300/A250 component platform and charging network, an important ecosystem strategy. Treat flight demonstrations, design claims, internal tests, certification milestones, and certified capabilities as distinct evidence classes. Sources: https://investors.beta.team/financials/annual-reports/content/0001628280-26-015838/bta-20251231.htm ; https://www.sec.gov/Archives/edgar/data/1784570/000162828026055933/a2026q2earningsrelease.htm ; https://spectrum.ieee.org/beta-technologies-electric-aircraft
- **Heart Aerospace ES-30:** the manufacturer's current programme page describes a 30-seat hybrid-electric aircraft with 200 km all-electric range, 800 km hybrid range, 30-minute charging, and a 2031 type-certification target. These are programme targets, not certified performance. Source: https://www.heartaerospace.com/es-30
- **Safran ENGINeUS 100 B1:** EASA announced the first type certificate under SC E-19 for this electric engine on 2025-02-04. This is a useful propulsion-certification reference; an engine certificate is not aircraft certification. Source: https://www.easa.europa.eu/en/newsroom-and-events/press-releases/how-easa-certified-safrans-engineus-100-electric-engine
- **NASA electrified-aircraft propulsion research:** use NASA reference architectures and integrated-powertrain studies to define subsystem and mission-level comparison cases. An August 2026 NASA Advanced Supercomputing seminar describes OpenMDAO/pyCycle multidisciplinary optimization for conventional, turboelectric and turbogenerator architectures across off-design conditions. It emphasizes that electrical conversion losses must be weighed against aircraft-level gains from distributed propulsion and integration. NASA also identifies battery performance shortfall and the mass/volume needed for thermal-runaway protection as major integrated-powertrain risks. Sources: https://www.nas.nasa.gov/pubs/ams/2026/08-06-26.html ; https://ntrs.nasa.gov/api/citations/20250006289/downloads/Integrated%20Powertrains%20Electrified%20Aircraft.pdf

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

## Proposed stretch goals

These are internal research goals, not industry-standard thresholds or demonstrated capabilities:

- Mission-energy prediction: bias within ±3% and RMSE below 5% on a preregistered validated test set before using the model for detailed design trades.
- Critical component electrical/thermal model residuals: publish calibrated prediction intervals with at least 95% empirical coverage on an independent holdout set.
- Design exploration: demonstrate at least 20% reduction in an explicitly defined lifecycle-cost or energy-per-payload-kilometre objective against a frozen baseline, with no degradation in required safety margins, payload, mission, or dispatch constraints.
- Configuration/evidence completeness: 100% of safety-critical components linked to approved configuration, supplier/material traceability, inspection, test, and maintenance records before release review.

A stretch goal is not an acceptance criterion until its measurement protocol and baseline are preregistered and independently reviewed.
