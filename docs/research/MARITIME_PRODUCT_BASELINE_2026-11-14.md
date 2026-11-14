# Marine product benchmark baseline — 2026-11-14

Status: research baseline / source discovery only  
Owners: Symthaea PWC-CO-DESIGN-001 #12095; Symthaea TUG-CO-DESIGN-001 #12096  
Related validation substrate: #8082, #8083, #8092, #8102, #8109, #8130, #8138  
This document is not a design approval, physical test report, class submission, certification, or seaworthiness determination.

## 1. Why this baseline exists

The programme now has two product families:

- **PWC-SIT:** 2–3 rider sit-down personal watercraft, comparing petrol and battery-electric architectures under matched missions.
- **HARBOUR-TUG:** harbour-assist tug in a provisional 60–70 metric-tonne bollard-pull reference class, with harbour-assist and escort missions treated separately.

The benchmark must compare useful service under declared, matched conditions—not isolated catalogue values. Manufacturer specifications, public procurement statements, analytical estimates, CFD results, model tests and full-scale measurements must remain distinct evidence classes.

A URL or published specification is a source lead, not proof that two products have equivalent capabilities or that a performance claim has been independently reproduced.

## 2. Personal watercraft: public comparator anchors

| Candidate / source | Publicly stated values | Evidence class | Important caveats |
|---|---|---|---|
| 2026 Kawasaki Jet Ski Ultra 310LX | 1,498 cc supercharged engine; 160 mm single-stage axial-flow jet pump; 3-blade impeller; 3 riders; 496 lb load capacity; 21.1 US gal fuel capacity; 996.7 lb estimated dry weight; 1,137.8 lb curb weight | Manufacturer specification | Estimated dry and curb weights are distinct; curb-weight definition includes fluids and a >90% fuel tank. Model-specific US-market specification. |
| Yamaha GP SVHO | 1,812 cc supercharged 4-cylinder; 160 mm axial-flow pump; 344 kg dry weight; 70 L fuel; 1–3 riders; 3.35 m length | Manufacturer specification | Exact model-year / market revision must be pinned before a scored comparison. Dry-weight basis must be checked against the other candidates. |
| Taiga Orca WX3 | 120 kW; 371 kg; 3 riders; up to 2 hours stop-and-go play time; 1 h 20 min at 30 km/h; fast charge under 30 minutes | Manufacturer-published test claim / specification | The manufacturer expressly says range, weight, speed, acceleration and battery life vary with environment, weather, regulations and final build. The quoted duration is not a standardised matched test. |

Primary source links:

- Kawasaki 2026 Ultra 310LX: https://www.kawasaki.com/en-us/watercraft/jet-ski/supercharged-sit-down/jet-ski-ultra-310/2026-jet-ski-ultra-310lx
- Yamaha GP SVHO specifications: https://global.yamaha-motor.com/business/waverunner/products/gpsvho/spec.html
- Taiga Orca WX3: https://www.taigamotors.com/en/products/orca-wx3

### 2.1 Comparability audit

These figures are not a valid efficiency or performance ranking. In particular:

- Kawasaki publishes both estimated dry and curb weight; Yamaha and Taiga present their own weight definitions. Do not rank weights until definitions and included equipment are reconciled.
- Petrol fuel capacity and electric battery energy are not interchangeable metrics.
- Published session duration is not equivalent to measured distance, energy per rider-nautical-mile or time-on-task under the same course and rider payload.
- Engine displacement and motor peak power do not establish installed propulsor efficiency.
- Seat count is not a complete payload/trim definition.
- Model year, market configuration, firmware, hull/impeller revision and optional equipment matter.

No missing power, battery-capacity, acceleration, top-speed, price or noise values may be filled from inference. Keep them Unknown until sourced from an exact product configuration or measured.

### 2.2 Provisional PWC mission suite

Freeze mission and conditions before candidate results are inspected. At minimum, maintain separate cases for:

1. Solo rider / low payload and low-speed handling.
2. Rated multi-rider payload / mixed recreational use.
3. Repeated acceleration bursts, turns and idle/low-speed sections.
4. Declared cruise profile and distance/range.
5. Rough-water and wave-encounter response within the safe test envelope.
6. Launch, recovery, reboarding and post-ejection recovery tasks.
7. Tow-sport use only when tow-point loads, stability and recovery requirements are specified.

Each trial manifest must identify actual rider mass and distribution, cargo, water temperature/density, current, wind, waves, depth, course, speed/throttle trace, start/end energy state, instrumentation and uncertainty.

Compare petrol and electric architectures on at least two separately reported boundaries:

- **Onboard energy consumed:** fuel mass/volume with declared heating-value assumptions versus measured battery/DC energy, including the exact measurement point.
- **Lifecycle / upstream impact:** electricity generation mix, fuel supply, battery manufacture/replacement, maintenance and end-of-life assumptions.

Never combine these into one unlabeled energy number.

## 3. Tugboats: public comparator anchors

| Candidate / source | Publicly stated values | Evidence class | Important caveats |
|---|---|---|---|
| TNPA new tug deliveries (June 2024) | 60-tonne bollard pull; first four of seven delivered for Durban and East London | Port-authority statement | Procurement / delivery reference, not a standardized public test dataset. Exact tug model configuration, test procedure and mission records are still needed. |
| Damen RSD Tug 2513 Electric | 70 t bollard pull astern; 24.73 m LOA; 13.13 m beam; 6.50 m max draft; 12 kn; product sheet lists maximum battery capacity 2,782 kWh; product page says fully charged in 2 hours | Manufacturer specification | Bollard-pull direction is explicitly astern in the product sheet; do not relabel as ahead pull. Verify charging-power / installation conditions. Do not interpret charge time as mission endurance. |
| Damen RSD Tug 2513 diesel | 80 t bollard pull; 24.73 m LOA; 13.13 m beam; 6.2 m max draft; 12.6 kn | Manufacturer specification | Same nominal platform dimensions do not mean equal displacement, propulsion, draft, pull test or mission capability. |
| Crowley eWolf | 6.2 MWh main propulsion battery; 2 × 2,100 kW electric motors; 12 kn; 70 short-ton bollard pull (estimated in cited specification) | Project / manufacturer specification | 70 short tons is approximately 63.5 metric tonnes. The cited value is marked estimated; do not label it a 70 metric-tonne tug or independent measured result. |

Primary source links:

- TNPA 12 June 2024 media statement: https://www.transnetnationalportsauthority.net/Corporate%20Affairs/Press%20Releases/Media%20Statement%20-%20TNPA%20welcomes%20delivery%20of%20four%20new%20tugboats.pdf
- Damen RSD-E 2513 Electric: https://www.damen.com/vessels/tugs/electric-tugs/rsd-tug-2513-electric
- Damen RSD-E 2513 Electric product sheet (states 70 t bollard pull astern and maximum 2,782 kWh battery capacity): https://media.damen.com/image/upload/v1655475611/catalogue/tugs/electic-tugs/rsd-e-2513/product-sheet-rsd-tug-2513-electric.pdf
- Damen RSD 2513: https://www.damen.com/vessels/tugs/rsd-tugs/rsd-tug-2513
- Crowley eWolf specification sheet: https://www.marinelog.com/wp-content/uploads/2021/07/THS-08240-eWolf.pdf

### 3.1 Comparability audit

The 60 t TNPA procurement/delivery baseline is directly relevant to South African port requirements. Damen's 70 t electric and 80 t diesel products provide a useful manufacturer-published comparison. Crowley's eWolf illustrates a large battery-electric harbour tug, but uses short tons and an estimated pull figure.

These are not yet matched benchmarks. Before a score is admissible, freeze:

- exact tug design and as-tested configuration;
- bollard-pull direction (ahead / astern), continuous / overload definition and test-window method;
- propulsor / azimuth / VSP configuration, loading, draft, trim, displacement and auxiliaries;
- line geometry, line angle, load cell, calibration, sample rate and force stabilization;
- site depth, quay proximity, current, wind, wave and water density;
- propulsion and auxiliary power boundary, fuel/shore electricity and battery state;
- escort task, assisted ship, relative geometry, towline angle, speed and transition test conditions;
- class/flag/test-procedure and evidence status.

### 3.2 Tug mission suite

Do not conflate:

- **Static bollard pull** — a static force measurement under a declared test procedure.
- **Escort performance** — full-scale EPN vector (F, B, V, C) when supported by the applicable class-society trial; CFD EPN has a different evidence status and omits C.
- **Harbour-assist mission service** — completed assist jobs, transit, standby, assist duration, force/time history, recovery, energy, reserve and availability.
- **Coastal towing / salvage** — separate operational and structural load envelopes.

A high bollard-pull result does not prove superior escort capability. A high static pull does not prove the best energy per completed port-assist job.

## 4. Standards and benchmark authorities to pin

### Personal watercraft

- ISO 13590:2022 covers PWC identification, permanently installed petrol fuel systems, electrical systems, steering, ventilation, hull structure/flotation, stability, mooring/towing, flooding, off-throttle steering and owner's manual. Its published scope excludes outboard-powered PWC and jet-powered surfboards.
- Official scope: https://www.iso.org/standard/76684.html
- ISO Online Browsing Platform change summary: https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso%3A13590%3Aed-3%3Av1%3Aen

A separate applicability profile is required for jet-powered surfboards / jetboards and for hydrofoils. Do not claim ISO 13590 covers them.

### Resistance and propulsion CFD

- ITTC Recommended Procedure 7.5-03-02-02, Revision 03 (2024), is the benchmark-database guidance for CFD validation for resistance and propulsion.
- Official PDF: https://ittc.info/media/11956/75-03-02-02.pdf
- ITTC benchmark repository: https://www.ittc.info/benchmark-repository/

Use the relevant benchmark family and exact experimental reference conditions. Resistance/propulsion, manoeuvring and cavitation/noise are different validation families. A solution passing one family is not globally validated.

### Tug performance

- Lloyd's Register EPN definitions distinguish F (maximum steering force), B (maximum braking force), V (speed at which F/B are determined) and C (transition time). Trial EPN is full-scale testing; CFD EPN is model-derived and omits C.
- Reference: https://www.imorules.com/GUID-6DED09FF-329A-47D9-A734-53115D867928.html
- MARIN Bollard Pull Joint Industry Project: https://www.marin.nl/en/jips/bollard-pull

MARIN identifies test-environment, water-depth, quay geometry, line length/angle, wind/current/waves and load-cell factors as affecting bollard-pull measurement. Obtain and review the JIP procedure before adopting it. A public web description is not a substitute for the actual methodology.

### Spatial / operational marine evidence

IHO's current in-force table (updated 29 October 2026) lists S-100 Edition 5.2.1; S-98 Edition 3.0.0 and S-164 Edition 2.0.0 entered into force 12 November 2026. Relevant product editions include S-102 Bathymetric Surface 3.0.0, S-104 Water Level 2.0.0, S-111 Surface Currents 2.0.0, S-127 Marine Traffic Management 2.0.0, S-129 Under Keel Clearance Management 2.0.0, and S-131 Marine Harbour Infrastructure 2.0.0.

- Current in-force table: https://iho.int/en/s-100-standards-in-force-0
- Standards catalogue: https://iho.int/standards-and-specifications

Pin each exact product specification and dependency edition. Never apply a blanket "latest S-100" substitution to every product.

## 5. Benchmark design: what would constitute being better?

### 5.1 PWC — non-compensatory scorecard

Report at least:

- net onboard energy per rider-nautical-mile and per rider-hour;
- useful session duration and usable range for a declared speed/throttle trace;
- acceleration and speed response under declared payload;
- turning / manoeuvrability and low-speed control;
- ride acceleration / comfort and rough-water operating envelope;
- launch/recovery, reboarding and off-throttle control;
- intake/impeller/nozzle performance, cavitation/ventilation and debris tolerance;
- noise / underwater radiated noise under a declared method;
- scheduled service labour, unscheduled downtime, corrosion and parts cost;
- acquisition and lifecycle cost, with transparent assumptions;
- environmental and regulatory constraints.

### 5.2 Tug — non-compensatory scorecard

Report at least:

- continuous and overload bollard pull, direction and test conditions;
- EPN F/B/V/C where a compatible full-scale trial exists;
- energy per completed harbour-assist job and daily job mix;
- job completion rate, reserve margin and dispatch availability;
- charging/fuelling constraints, shore connection and grid compatibility;
- propulsion / auxiliary power and transient response;
- stability, towing gear, line loads and degraded modes;
- maintenance labour, repair time, component life and lifecycle cost;
- underwater radiated noise, local emissions and lifecycle emissions;
- shipyard buildability, inspection access, spare-parts and retrofit burden.

### 5.3 Candidate stretch targets (research goals, not achievements or standards)

**PWC**
- Once a matched baseline is measured, explore >=10% lower net onboard energy per declared mixed-use mission, with uncertainty intervals excluding a negligible change before claiming improvement.
- Preserve rated payload, acceleration, handling and safety margins; no Pareto regression hidden behind one aggregate score.
- Explore >=20% more useful mission time at a declared cruise profile as a separate stretch case, not an assumed product promise.
- Explore >=25% lower scheduled service labour on a matched maintenance task list.

**Tug**
- First meet the selected class / procurement bollard-pull requirement on a class-accepted procedure.
- Preserve or improve full EPN F/B/V/C for a matched escort mission; do not trade away C or a force component silently.
- Once matched harbour job profiles and metrology are frozen, explore >=10% lower net energy per completed assist job without reducing job completion, reserve, availability or safety.
- Set numeric availability / repair-time / maintenance goals only after baseline fleet data are admitted.

**Integrity gates (both product families)**
- 100% source/configuration/measurement-boundary traceability for every reported result.
- Zero silent extrapolation outside the validated model domain.
- Zero synthetic / analytical / CFD evidence promoted to physical measurement.
- All failures, missing data, censored runs and out-of-domain results remain in the denominator and audit output.
- Zero false-safe decisions in pre-registered safety, stability, electrical, propulsion-loss and emergency-recovery controls.
- Every reported improvement includes uncertainty, matched conditions, adverse cases and counterexamples.

Do not set numeric targets for top speed, EPN, MTBF, availability, charging throughput or cost before the reference class, duty cycle, measurement protocol and incumbent data are frozen.

## 6. Paradigm shifts to test

1. **Mission contract instead of catalogue comparison.** Compare completed useful service under an identical, preregistered task definition.
2. **Installed-system co-design.** Optimise hull + appendages + propulsor + powertrain + control + operating policy as a system. Charge all added actuator, pump, cooling, mass and payload penalties to the total.
3. **Uncertainty-aware design search.** Use cheap analytical models for screening, high-fidelity models for discriminating candidates, and physical tests for release claims. Compare methods by total cost including failed attempts and required final validation.
4. **Causal physical A/B campaigns.** Freeze mission, instrumentation, conditions and analysis before tests; randomize/counterbalance trial order where practical; retain failed/incomplete trials.
5. **Repairability as performance.** Track service hours, time-to-repair, parts availability, corrosion, configuration drift and lifecycle cost—not just peak output.
6. **As-built twin, not CAD twin.** Digital-twin identity must bind actual materials, serial components, processes, firmware, inspection, commissioning and repair history.
7. **Port / energy co-design for tugs.** A battery-electric tug must be designed with the actual dispatch profile, simultaneous charging demand, shore grid, reserve and port infrastructure.
8. **Ecological performance as an engineering dimension.** Acoustic exposure, local emissions, fouling/antifouling impacts and sensitive-site restrictions must be measured or explicitly marked unknown.
9. **Operator-assist before autonomy.** Use bounded decision support and situational awareness; do not grant a model authority to navigate, tow or actuate a vessel.

PWC waterjet mechanical components and tug azimuth/VSP propulsion should not be forced into a single hardware platform. Reuse where justified: electrical architecture, safety practices, controls/diagnostics, materials/process knowledge, instrumentation, data contracts, supplier qualification and lifecycle evidence.

## 7. Work ownership and immediate next sequence

| Owner | Existing work to reuse | Product-specific follow-on |
|---|---|---|
| Symthaea | #8082 hydrodynamics; #8092/#8102/#8109 matched energy/traces; #8130 envelope; #8138 fidelity-cost; #11676 acoustic round robin | #12095 PWC and #12096 tug product co-design |
| Symtropy | #2048 shipyard DFM; #3798 tug operations; #3865 manoeuvring; #3911 ITTC/SIMMAN validation; #4406 vessel/port digital thread | #5179 bollard-pull / escort-trial evidence |
| Mycelix | #3708/#3709 Mobility Commons; #3084/#3085/#3095 manufacturing lifecycle; #5968 maritime integration | #6154 marine configuration, inspection and repair lineage |
| Sol-Atlas | #115 maritime atlas; #654 source-reversible projection | #809 metocean and test-site context |

Recommended sequence:

1. Admit exact public comparator sources and their terms / revision states.
2. Freeze one PWC-SIT mixed-use mission profile and one separate tug harbour-assist mission profile.
3. Freeze a separate escort profile; do not treat it as the same task as static pull.
4. Establish the current benchmark and independent analytical known-answer / negative-control set.
5. Run only the model classes whose validity domain is explicitly declared.
6. Select the smallest decisive physical experiment, not the largest simulation campaign.
7. Bind test results to exact as-built configuration and site conditions.
8. Compare against the incumbent with uncertainty and adverse cases.
9. Requalify after any material component, process, firmware, hull or propulsion change.

## 8. Claim ceiling and next decisions

This baseline supports competitor discovery and test planning only. It does not establish that a candidate hull is better, a solver is validated, a vessel is safe, or a design meets class / statutory requirements.

The next design decisions that genuinely require project-owner input are:
- target PWC market / use profile (performance sport, family recreation, rental fleet, rescue/utility);
- target tug mission mix and port constraints (harbour assist only vs escort / coastal towing);
- intended build region and regulatory markets;
- cost / payload / range / maintenance trade-off priorities.

Until those are selected, preserve multiple mission scenarios and do not optimize a single guessed product.


## 9. SOTA research refresh — 2026-11-14

This section adds recent research anchors to the product baseline. Published results remain study-specific: record the vessel/propulsor class, plant architecture, duty cycle, measurement boundary and evidence tier before comparing them with this programme.

### 9.1 PWC / installed waterjet dynamics

**MIT electric-PWC integration proof of concept (May 2025).** Hudspeth's MIT thesis documents conversion of a 1997 Yamaha Wave Venture 760 using a 72 V, 10.4 Ah battery pack, waterproof housing and a 5 kW motor coupled to the existing impeller shaft. The reported check is a dry test monitoring battery/motor health; it demonstrates an integration path, not an on-water range, power, reliability or product-performance result. The nominal battery energy implied by 72 V × 10.4 Ah is about 0.749 kWh before usable-SOC limits and conversion losses; at a continuous 5 kW draw that is about 9 minutes idealized. This arithmetic is a unit sanity check, not an estimate of actual prototype runtime because real load and battery voltage profile are not specified by the abstract. Rights status is In Copyright / Educational Use Permitted; do not copy thesis figures or data into commercial materials without permission. Source: https://dspace.mit.edu/entities/publication/3fb93265-740c-4252-8535-70916fa8b9e6

**Waterjet dynamics during manoeuvring (May 2026).** Wang et al. investigate a fully coupled 6-DOF / rotating-waterjet CFD model for an amphibious vehicle. The abstract reports suction-induced attitude changes, vortex-associated performance loss during turning, transient pump operating-point shifts during zig-zag manoeuvres, and left/right inflow / jet-performance asymmetry. This is a high-value hypothesis source for installed PWC waterjet testing, but the vehicle class, geometry, operating points and validation domain differ; no numerical transfer to PWC is warranted. DOI: https://doi.org/10.1016/j.oceaneng.2026.125403

**Low-speed waterjet pump loss decomposition (15 January 2026).** Fei et al. study a waterjet pump device at low navigational speeds and report that jet velocity is a major influence on propulsion efficiency and energy dissipation in their test/model domain. PWC planing speeds, intake immersion, hull acceleration and turning conditions may be materially different; this is a mechanism/reference source, not a PWC calibration dataset. DOI: https://doi.org/10.1016/j.oceaneng.2025.123706

**Implication for PWC research:** the next differentiating benchmark should test the installed intake–pump–hull–nozzle system through acceleration, planing transition, both-direction turns, steering reversals and wave/current conditions. Record pump RPM/torque, thrust or a defensible force proxy, power, intake pressure/flow, hull six-DOF response, cavitation/ventilation indicators and rider-relevant acceleration. A steady open-water pump map alone is not enough.

### 9.2 Tug hybrid powertrain / energy-management SOTA

**TD3 tug EMS (9 February 2026).** Li, Long and Tian evaluate a methanol range-extended series-hybrid port tug. Their paper reports TD3 methanol-consumption reductions of 18.5% versus rule-based control, 10.2% versus ECMS and 7.3% versus DDPG, with a reported gap below 2.5% from dynamic programming. Their HIL setup uses a 1 ms model step, average communication/processing latency around 3 ms (maximum <=5 ms), and reports less than 1.8% deviation from pure simulation. This is a strong named-method comparator and an HIL result; it is not a full-scale vessel sea-trial or fleet-wide savings claim. The authors state that actual-vessel uncertainties such as complex flow, wind/waves, mooring loads, degradation and sensor noise remain for future work. The paper says data are in the article and further inquiries should go to the corresponding author. Source: https://www.mdpi.com/1996-1073/19/4/905

**Methanol range-extender capacity matching (2 March 2026).** Li, Tian and Long report a battery-dominant series-hybrid design with 200 kWh / 600 kW LFP, a 250 kW methanol APU and 400/600 kW PMSM propulsion-motor ratings. They report 50.8% equivalent fuel saving versus a conventional diesel baseline over one typical simulated one-hour duty cycle, with 65% pure-electric time and 62% braking-energy recovery. Critically, the authors say the work uses a specific duty cycle and idealized component models, and calls for additional real operational data and HIL validation. Underlying data are available on request. Treat the 50.8% as a simulation result for that exact study, not a field result or a generic target. Source: https://www.mdpi.com/2075-1702/14/3/274

**Implication for tug research:** a credible new result cannot be merely “our RL beats a weak rule base.” Reproduce or match TD3, rule-based, ECMS, DDPG, MPC and an offline DP oracle under identical plant maps and duty cycles. Then seek advantage in held-out vessel / port / job blocks, forecast-error and equipment-degradation robustness, hard-constraint compliance, worst-case execution time, lifecycle cost and port charging feasibility. Dynamic programming is a non-deployable clairvoyant reference, not a controller.

### 9.3 Propeller and acoustic research: relevant mainly to tug / conventional-propeller classes

- Gaggero et al. (published 30 July 2025) report a propeller optimized for underwater radiated-noise reduction on a small passenger boat, with numerical and model-scale work followed by a full-scale refit and test that confirmed the study-specific noise reduction. This is a valuable example of a simulation -> model test -> retrofit -> full-scale validation chain; it is not direct evidence for a PWC waterjet. DOI: https://doi.org/10.1016/j.oceaneng.2025.121412
- Gaggero and Serani (January 2026) report physics-informed parametric dimensionality reduction for marine propeller shape optimization. The method embeds pressure distributions and performance observables into the reduced design space and uses BEM / RANS comparisons. It is a relevant methodology comparator for conventional tug / vessel propellers, not a waterjet pump validation result. DOI: https://doi.org/10.1016/j.apor.2026.104932
- ITTC's Cavitation and Noise Round Robin page (updated 2 July 2026) specifies a 31 January 2027 results deadline; the no-rudder test is mandatory and the rudder test optional. The dataset is agreement-controlled; a downloadable geometry or catalogue listing is not proof of permission to reuse. Source: https://ittc.info/benchmark-repository/round-robin-test-cavitation-and-noise/

### 9.4 How we can outperform responsibly

We should not claim "better than SOTA" until we beat named baselines on equivalent conditions. The strongest opportunities are:

1. **PWC:** demonstrate better *installed system* net energy and handling over a frozen mixed-use course, including turning/transient losses, at matched rider payload and water conditions. Compare petrol and electric systems using explicit onboard-energy and lifecycle-energy boundaries.
2. **Tugs:** compete against tuned RB, ECMS, DDPG, TD3 and deployable MPC/robust-MPC baselines; include a DP oracle only as a clairvoyant bound. Report fuel/electricity, SOC reserve, battery throughput/degradation, transient thrust tracking, job completion, availability, emissions, compute/deadline performance and robustness.
3. **Both:** turn design changes into an as-built configuration generation; trace materials, processes, serialized components, inspections, repairs and requalification. A digital twin is useful only when tied to the exact physical configuration.
4. **Both:** compare complete experiment cost, including failed runs and required high-fidelity / physical validation. A design selected cheaply but requiring more expensive validation is not necessarily the better research strategy.

All numeric literature values above are external reference results with their own model, cycle, assumptions and validation boundaries. They are not Symthaea achievements or expected outcomes.
