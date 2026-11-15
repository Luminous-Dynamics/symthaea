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


## 10. Waterjet-specific cavitation test method — ITTC correction

The resistance/propulsion CFD benchmark guidance cited above is not a substitute for the ITTC's waterjet-specific cavitation procedure. Add the following procedure to the PWC evidence plan:

- **ITTC 7.5-02-03-03.8, Revision 01 (effective 2024): Modelling the Behaviour of Cavitation in Waterjets.** Official procedure: https://www.ittc.info/media/11826/75-02-03-038.pdf
- It distinguishes three complementary test families:
  1. **Self-propulsion / towing-tank tests:** measure hull–waterjet interaction, flow rate, thrust and effective system power. The procedure explicitly cautions that ordinary self-propulsion tests are generally not appropriate to evaluate cavitation because facilities may not achieve depressurization/cavitation similarity, and model pumps may be surrogates rather than geometrically scaled production pumps.
  2. **Waterjet-system tests:** test the actual inlet and pump without the full ship hull. Representative incoming boundary-layer flow should be considered; define the operating cases from ship speed (including accelerating cases), loading and inlet velocity ratio (IVR), and vary the cavitation parameters. Record the boundary-layer and operating conditions.
  3. **Pump-loop tests:** isolate pump hydraulic behaviour and cavitation performance without the hull/intake distortion. Measure the actual pump inflow, shaft rate, static and total pressure, flow, torque and thrust as applicable. The procedure describes a 3% reduction in total-head rise as a common cavitation-breakdown criterion.
- Keep cavitation inception, performance breakdown and erosion separate. The procedure says quantitative erosion prediction for complex waterjet geometries remains immature; it does not itself provide a complete underwater-radiated-noise qualification method.
- Model-scale transfer is not automatic. The procedure identifies Reynolds number, geometry/roughness, manufacturing tolerances, turbulence, water quality, nucleation and incoming-boundary-layer similarity as relevant scale effects. A pump-only result cannot validate installed hull–intake behaviour, and a towing-tank propulsion result cannot be promoted to cavitation validation without the appropriate pressure/similarity and viewing method.

**PWC application boundary:** these procedures are directly relevant to waterjet test-method design, but do not supply PWC-specific calibration data or prove that a particular facility can accommodate the required pump, pressure, optical or full-scale instrumentation. Adapt the method to the PWC operating envelope and exact hardware; retain an explicit applicability review.

**Discriminating experimental sequence:**

1. Freeze the exact PWC-SIT geometry, loading, speed/acceleration schedule and pump operating map.
2. Measure baseline installed resistance/trim/attitude and thrust/power over straight-line and turning cases.
3. Use a waterjet-system rig with representative inlet boundary-layer profiles and IVR sweeps to isolate inlet distortion.
4. Use a pump-loop/cavitation facility for cavitation inception and head/thrust breakdown where the facility can reproduce the required pressure and scaling conditions.
5. Correlate the decomposed measurements; do not tune the model and claim validation on the same held-out case.
6. Carry model/full-scale mismatch as an uncertainty term until a defensible scale-effect argument and full-scale observations exist.

This resolves an important test-design ambiguity: one experiment cannot simultaneously isolate pump physics, installed hull interaction and full-scale cavitation.

### 10.1 South African physical-test partner leads

These are facility/research leads for capability discussions, not confirmed partners or proof that a specific test can be run there.

- **Stellenbosch University, Towing Tank:** the university describes a 90 m tank, 4.2 m breadth and 2 m depth, maximum trolley operating velocity of 7 m/s (8.2 m/s absolute maximum), and instrumentation for trolley velocity, model drag and bow/stern trim. Research team capabilities include CFD, turbomachinery, shape optimisation, measurement, digital twins and underwater acoustics. Facility page: https://www.su.ac.za/en/faculties/engineering/departments/mechanical-mechatronic-engineering/research/naval-architecture-design/facilities-and-experiments ; research chair: https://www.su.ac.za/en/faculties/engineering/departments/mechanical-mechatronic-engineering/research/naval-architecture-design
- **CSIR Coastal Engineering and Port Infrastructure, Stellenbosch:** publicly describes physical/numerical modelling of port layout and ship manoeuvring, marine surveys and a large hydraulic modelling facility. Capability page: https://wwwprod.csir.co.za/coastal-engineering-and-port-infrastructure

Before selecting a facility, request its capability statement for: waterjet/pump installation, depressurization and cavitation number range, water quality control, optical access/PIV/LDV, boundary-layer conditioning, load/torque/pressure metrology, sample rates/time synchronization, model dimensions, uncertainty procedures, staff availability, cost, IP/data terms and safety/ethics permissions. A towing tank and a cavitation tunnel are complementary, not interchangeable.

## 11. Tug EMS: deeper audit of the 2026 TD3 comparator

The TD3 study is a strong benchmark anchor, but a more exact reading narrows what its results establish.

**Primary reference:** Li, Long and Tian, *Research on Energy Management Optimization for Hybrid-Powered Port Tugboat Systems Based on a Dual-Delay Deep Deterministic Policy Gradient Algorithm*, Energies 19(4), 905, published 9 February 2026. https://www.mdpi.com/1996-1073/19/4/905

Reported evidence and conditions:

- The primary benchmark is a 3,600 s, seven-phase port duty cycle: departure, approach, push, standby, tow, return and berth standby. Its model includes a peak required power above 2,600 kW.
- The headline algorithm comparison uses the same simulation platform and duty cycle, with initial SOC 0.6 and initial battery temperature 25 °C. TD3 reports 18.5% lower methanol use than rule-based, 10.2% lower than ECMS and 7.3% lower than DDPG; its reported gap to the offline DP reference is below 2.5%.
- The HIL evaluation uses that same typical duty cycle and initial SOC. The reported upper/lower-level communication and processing delay averages about 3 ms and does not exceed 5 ms; HIL methanol consumption is reported as 0.6% above pure simulation and overall difference as below 1.8%. This is meaningful controller execution evidence, but it is not a vessel trial or validation of real hydrodynamic disturbances.
- The paper's generalization test applies trained controllers to one unseen sequence with higher peak loads and initial SOC 0.5. This is useful, but it is not a broad holdout across ports, vessels, seasons, crews, weather, job classes or long-horizon fleet operation.
- Battery constraints include an SOC range, current limits and temperature bounds. High-rate discharge and deep cycling are penalized as an indirect aging mitigation; this is not equivalent to a validated calendar/cycle-life or replacement-cost model.
- The paper reports subsystem validation evidence, including engine-map and battery tests, but its system-level duty-cycle/HIL results remain tied to the declared simulation and hardware setup. Data availability says data are within the article, with additional inquiries to the corresponding author; do not assume an openly downloadable raw operational-fleet dataset.

**Benchmark challenge for our programme:** reproduce the stated TD3 comparison as closely as the disclosed methods permit, then go beyond it using preregistered group holdouts and more demanding deployment tests:

1. Hold out complete vessel/port/job blocks and, where data allow, season or operating regime; never split adjacent samples from one mission across train and test.
2. Test forecast uncertainty and decision-time leakage, not just a single altered duty cycle.
3. Perturb plant maps and operating conditions: battery temperature/resistance/capacity, motor/inverter efficiency, APU ramp/minimum-load/start behaviour, auxiliary loads, sea/current/wind loads and sensor bias/dropout.
4. Inject equipment faults and degraded availability: battery derating, generator unavailable, propulsion-unit derating, shore-power interruption, delayed/stale messages and reserve-critical emergency work.
5. Compare tuned rule-based, ECMS, DDPG, TD3, deployable MPC/robust MPC and a clearly labelled offline DP oracle under the same frozen plant, constraints and energy boundary.
6. Report constraint violations and safe-fallback activation separately from energy savings. Include worst-case/maximum execution time, deadline misses, memory/compute load, action saturation and recovery behaviour.
7. Use a degradation model with independently identified parameters and uncertainty; do not equate throughput penalties with battery-life validation.
8. Score completed assist jobs, force/power response, reserve, dispatch availability and lifecycle resource cost. A controller that saves fuel by failing or postponing difficult jobs is not an improvement.

**Proposed stretch objective:** find a robust, interpretable MPC-plus-learning supervisory controller that matches TD3 energy performance within a preregistered non-inferiority margin while improving held-out mission regret, constraint/fault outcomes, worst-case compute or degradation-adjusted cost. This is a research target, not a demonstrated result. Do not assume the hybrid controller will win before the matched experiment.

## 12. Standards and digital-thread status note

ISO confirms **ISO 23247-5:2026** (Digital thread for digital twin) was published in June 2026. It describes connectivity, management and maintenance of manufacturing digital twins across design, planning, production and testing. Official source: https://www.iso.org/standard/87425.html

ISO 23247-6:2026 was independently verified as published on 14 July 2026. It addresses digital-twin composition and identifies integrated, unified and federated compositions, with guidance for interoperation among different parties. Official ISO record: https://www.iso.org/standard/87426.html. Part 5 addresses the manufacturing digital thread across product lifecycle activities. These standards guide manufacturing digital-thread/composition design; they do not certify marine design, hydrodynamic model validity, seaworthiness or vessel safety.


## 13. October 2026 tug EMS SOTA: event-triggered control and hybrid storage

**New comparator:** Gao, Wu and Tian, *Energy management strategy for hybrid tugboats based on operating condition recognition and event-triggered mechanism*, Ocean Engineering, DOI https://doi.org/10.1016/j.oceaneng.2026.127569, published 15 October 2026. Bibliographic record: https://trid.trb.org/View/2763148

The study proposes a diesel–battery–supercapacitor hybrid architecture and a hierarchical event-triggered EMS:

- Weighted K-means recognizes four operating conditions: standby, sailing, manoeuvring and working, using load statistics, engine imbalance/correlation and spectral entropy.
- Supercapacitors buffer high-frequency power demand; the abstract reports battery-power peaks falling from approximately 1,800 kW to 1,100 kW during rapid load fluctuations.
- The event-triggered controller updates parameters only when its trigger conditions require them.
- Reported equivalent fuel consumption is 415.1 kg versus 447.2 kg for the rule-based REMS comparator (a 32.1 kg / approximately 7.18% difference for the study's evaluated sequence).
- The paper reports 225 controller updates, 97.0% fewer than REMS and GA, and 0.0372 s per-step computation time.
- It also reports average diesel-engine load rate increasing from 49.9% to 53.7%, with terminal battery SOC near 45%; the abstract reports a 40.1% reduction in battery-power standard deviation versus GA.

**Evidence boundary:** these are study-specific simulation results using representative tug load data and the authors' diesel–battery–supercapacitor topology. They do not establish full-scale operational fuel savings, battery life, class-approved control, or superiority over the methanol TD3 study. Do not pool the 7.18% result with the TD3 paper's 18.5% / 10.2% / 7.3% figures: architectures, controllers, duty cycles and study protocols differ.

### Updated competitive experiment

Retain the February 2026 TD3 reproduction as one comparator arm and add the October 2026 event-triggered hierarchical controller as another. Compare, on identical admitted plant maps and mission blocks:

1. tuned rule-based and engine load-following baselines;
2. ECMS and adaptive-ECMS;
3. DDPG and TD3;
4. deployable MPC / robust MPC;
5. event-triggered hierarchical EMS with battery-only storage;
6. event-triggered hierarchical EMS with battery + supercapacitor;
7. offline dynamic programming as a clairvoyant reference on short, fully known cycles.

Separate the controller contribution from the storage-architecture contribution. Use a factorial comparison where possible: same controller across battery-only and battery-supercapacitor plants, and same plant across controller families. Charge supercapacitor mass, converter losses, thermal management, capex, maintenance and replacement to lifecycle results.

Additional metrics: battery peak/RMS current and power, high-frequency energy share, supercapacitor throughput/thermal state, DC bus ripple/sag and recovery where qualified, engine starts/ramp/low-load hours, trigger count and false/missed triggers, CPU time and worst-case execution time, job completion, reserve, final SOC, degradation uncertainty and lifecycle cost.

The paradigm opportunity is **control update policy co-designed with energy-storage physics**: a controller may obtain better response and lower compute burden by separating slow energy scheduling from fast transient buffering. This is a hypothesis to test, not an assumed winner.


## 14. South African propulsion-shaft observability opportunity

Stellenbosch University's published facilities include a rotating machinery test rig that applies repeatable, directly measured impacts to a simplified propeller disk while collecting indirect shaft-line signals, and the S.A. Agulhas II full-scale laboratory with shaft shear/thrust, shaft RPM, temperature, accelerometer and strain-gauge instrumentation. Sources:
- https://www.su.ac.za/en/faculties/engineering/departments/mechanical-mechatronic-engineering/research/naval-architecture-design/facilities
- https://www.su.ac.za/en/faculties/engineering/departments/mechanical-mechatronic-engineering/research/naval-architecture-design

This suggests a separate benchmark for reconstructing propulsion-shaft transient loads from indirect measurements, with a directly measured laboratory excitation reference and a full-scale context dataset where access/rights permit. It is not a PWC waterjet test, does not establish tug-transfer validity, and does not assume facility access.

New research owner: [MARINE-SHAFT-OBSERVABILITY-001 #12187](https://github.com/Luminous-Dynamics/symthaea/issues/12187). It composes the existing marine trial-campaign binding (#11064), shipboard power-transient (#11028), marine structural-health (#11089), tug product (#12096), tug EMS (#12136), reference-data admission (#8146) and Symtropy tug trial-evidence (#5179) owners. The new issue owns only the indirect shaft-load/virtual-sensor benchmark and its observability/abstention contract.

Candidate value for the tug programme: measured/estimated shaft torque and transient load may improve propulsion-load accounting, reserve assessment and fault-context interpretation. The estimate is advisory only; it cannot authorize physical actuation or replace direct protection/instrumentation. Model transfer among the Stellenbosch rig, S.A. Agulhas II, tug shafting and waterjet drives requires separate configuration and domain evidence.

## 15. South African test-facility collaboration: capability before commitment

Current public capability evidence supports two distinct leads:

- Stellenbosch University: a 90 m × 4.2 m × 2 m towing tank, with drag, trim and trolley-speed measurement; a rotating machinery rig; and a sensor-rich full-scale research vessel. These assets have different test purposes and should not be treated as one universal marine lab. Facility details: https://www.su.ac.za/en/faculties/engineering/departments/mechanical-mechatronic-engineering/research/naval-architecture-design/facilities
- CSIR Coastal Engineering and Port Infrastructure: physical modelling of ship motions/navigation, stability, force measurement and wave conditions; metocean data collection and port studies. Public capabilities: https://www.csir.co.za/what-we-do/mobility-and-logistics/infrastructure-engineering/coastal-engineering-and-port

No public source reviewed here confirms a dedicated waterjet cavitation tunnel at either facility. The immediate action is a capability inquiry, not booking or partner announcement. Request pump fit, cavitation pressure range, optical access, inlet boundary-layer conditioning, torque/pressure/flow metrology, data rights, uncertainty procedures, cost and available dates before selecting a facility.

The work should choose the **smallest discriminating experiment**:
- towing tank for installed hull resistance, trim and force interaction where a suitable waterjet model can be instrumented;
- dedicated waterjet-system rig for inlet distortion and IVR sweeps;
- pump loop/cavitation tunnel for pump cavitation inception and breakdown;
- rotating machinery rig for shaft transient-load observability;
- full-scale craft/vessel only for claims that require full-scale evidence.


## 16. Competitor correction and stronger Damen electric-tug reference — Aqua Pollux

A currently indexed Damen product sheet is for the delivered **RSD Tug 2513 Electric “Aqua Pollux”**, yard 514003, delivered October 2024. It reports:
- classification notation: MACH Escort Tug; bollard pull 70 t; maximum steering force 73 t; maximum braking force 96 t; maximum escort speed 10 kn; unrestricted navigation;
- performance line: 70 t bollard pull and 12.6 kn speed, specified in full-electric (battery-powered) mode;
- 2 × Ramme TW 1400r_L electric main motors; 3,700 bkW at 500 rpm;
- maximum battery capacity 2,782 kWh;
- 2 × Caterpillar C32TA generator sets, 1,175 kVA each, described in the sheet as backup-generator engines;
- Damen Triton monitoring dashboard/cloud listed in the nautical/communications equipment.

Official product sheet: https://medialibrary.damen.com/m/17d9e60bcdcd2257/original/product-sheet-damen-rsd-tug-2513-electric-dms-aqua-pollux.pdf

**Critical interpretation:** this is a manufacturer product sheet/classification statement, not the raw class trial report. The values 73 t steering, 96 t braking and 10 kn escort speed are useful (F,B,V)-type comparator anchors, but the sheet does not report EPN transition time (C). Do not label it a complete measured full-scale EPN ((F,B,V,C)) result. Obtain the class/sea-trial report, measurement basis and uncertainty before scoring it as a full EPN comparator.

**Source-reconciliation note:** an older product-sheet version in the research history described 70 t bollard pull astern, whereas this currently indexed Aqua Pollux sheet gives a 70 t bollard-pull line and the escort notation above without a direction on that line. Keep source revision, vessel identity, pull direction and measurement basis in the incomparability ledger; do not silently overwrite one edition with another. Confirm the exact configuration and test report with Damen.

Combine this with Damen's 2 February 2021 operational description (manufacturer statement, not an independently audited duty-cycle distribution): at least two average-harbour berthing/unberthing operations on battery alone, 70 t push/pull for at least 30 minutes during those operations, two-hour shore recharge, four independent battery packs and an expected battery-system life of approximately 30,000 cycles. Source: https://www.damen.com/insights-center/articles/developing-zero-emissions-harbour-towage-capabilities

**Competitive implication:** the benchmark should separately evaluate static pull, escort (F/B/V), transition time (C), complete harbour jobs, high-pull duration, reserve, charge turnaround, redundancy, actual availability and battery lifecycle. The manufacturer sheet's battery capacity and headline pull do not prove that a different tug architecture can complete the same job sequence.


## 17. Hull biofouling, condition drift and clean-hull performance retention — SOTA update

### 17.1 New physics-guided incremental-learning comparator

Lang et al., *Physics-guided machine learning for ship biofouling assessment in support of maritime decarbonization*, Transportation Research Part D, Volume 156, July 2026, Article 105364, DOI https://doi.org/10.1016/j.trd.2026.105364.

The study combines a physics-informed warm-up baseline representing ideal/unfouled calm-water vessel states with incremental model updates on sliding temporal windows. It evaluates degradation trends and post-cleaning recovery against conventional retraining and ISO 19030. The authors report better capture of gradual degradation and rapid recovery in their study context.

**Limits:** the authors state they do not have permission to share the data. This is a method comparator, not an openly reproducible operational dataset or a model that can be transferred to PWC waterjets or harbour-tug azimuth/VSP propulsion without an independent domain-validity study. Do not infer its performance metrics or data availability beyond the published record.

Source: https://www.sciencedirect.com/science/article/pii/S1361920926001574

### 17.2 Separate observed drift from fouling attribution

A rise in normalized fuel/shaft power or a speed loss is not uniquely identifiable as biofouling. It may also result from wave/current/weather, temperature/density, payload/draft/trim, speed profile, route, propulsion degradation, intake blockage, debris, impeller/nozzle/propeller damage, cavitation/ventilation, engine/motor/inverter/gearbox efficiency, battery SOH, control changes or sensor/calibration drift.

Report independent stages:
1. Observed performance residual at a declared reference condition.
2. Candidate cause(s), with uncertainty and competing explanations.
3. Independent hull-zone / niche-area / coating inspection evidence.
4. Biosecurity and cleaning authorization/capture/waste disposition.
5. Post-intervention matched performance response.
6. Human/operator/authority disposition.

A performance model may trigger inspection, but must not convert performance drift into confirmed biological fouling without inspection or other admitted reference evidence. Incremental model updates must be time-causal: no future measurements, post-cleaning traces or inspection labels may leak into prior online predictions. Preserve the original reference model and historical predictions; each update creates a versioned successor.

### 17.3 New and corrected standards map

- **ISO 19030-1:2016** (confirmed current in 2022): same-vessel hull/propeller performance measurement principles and indicators for conventional fixed-pitch propeller ships. It is not automatically a conformance procedure for PWC waterjets or tug azimuth/VSP configurations; any extension requires a declared scope and validation. Official record: https://www.iso.org/standard/63774.html
- **ISO 6319:2026** (published March 2026): plans, conducts and documents in-water cleaning of external submerged hull/niche surfaces, across biofilm, microfouling and macrofouling, with or without capture. It does not cover internal piping. Official record: https://www.iso.org/standard/85997.html
- **ISO 20679:2025** (published January 2025): independent performance testing of in-water cleaning systems, including efficacy/safety and captured/processed/disposed debris. It does not set universal cleaning acceptance criteria; these remain for competent authorities. Official record: https://www.iso.org/standard/86393.html
- **IMO MEPC.378(80)** (2023) and **MEPC.1/Circ.918** (29 April 2025): biofouling management and in-water cleaning guidance. The latter expressly recognizes coating damage and release of harmful waste / invasive species as risks. Sources: https://www.imo.org/en/ourwork/environment/pages/biofouling.aspx and https://wwwcdn.imo.org/localresources/en/OurWork/Environment/Documents/Biofouling%20pages/MEPC.1-Circ.918%20-%20Guidance%20On%20In-Water%20Cleaning%20Of%20Ships%27%20Biofouling%20%28Secretariat%29.pdf

Do not collapse performance trend measurement, cleaning execution/documentation and cleaning-system efficacy testing into one standard-conformance label.

### 17.4 South African engagement window

The IMO reports that a Regional In-Water Cleaning Technology Demonstration and Workshop took place in Cape Town on **5–8 October 2026**. It involved government agencies from 12 countries, industry, South African technology providers and experts. The National Department of Transport hosted with SAMSA and TNPA as implementing agencies. Live demonstrations by licensed South African providers covered in-water cleaning; SAMSA's sector report describes inspection, cleaning, capture, filtration, verification and documentation.

Sources:
- IMO (8 October 2026): https://www.imo.org/en/mediacentre/pages/whatsnew-2511.aspx
- SAMSA sector report (21 October 2026): https://blog.samsa.org.za/2026/10/21/samsa-participates-in-test-biofouling-regional-in-water-cleaning-technology-demonstration-and-workshop/

This is a concrete local stakeholder/technology landscape to research, not an endorsement, invitation, data licence or cleaning permit for Luminous. The follow-up should map provider capabilities and local requirements before any proposed trial.

### 17.5 Extended benchmark panel — both product families, different methods

**PWC-SIT:** fouling/debris/vegetation at hull and waterjet intake; distinguish hull resistance, intake blockage, ventilation/cavitation, pump/impeller/nozzle damage and motor/controller drift. Use a PWC-specific same-craft reference-condition method rather than claiming ISO 19030 compliance. Keep cleaning and biosecurity evidence separate.

**HARBOUR-TUG:** distinguish hull/propeller/azimuth/VSP fouling from propulsion machinery, shaftline, battery/generator, loading and dispatch changes. Report energy per completed job and availability with configuration/condition normalization. Cleaning cannot be credited with recovery unless matched pre/post cases and uncertainty support it.

**Shared metrics:** detection delay, drift-estimation residual, interval coverage, false fouling attribution, inspection false-negative and niche-area miss rates, inspected-area coverage, independently tested cleaning efficacy, captured fraction/waste disposition, coating damage, off-hire/service time, energy recovery, lifecycle cost and environmental/biosecurity disposition. Do not use a single composite score that allows an energy benefit to compensate for unauthorized cleaning or uncontrolled release.

**Proposed integrity targets (not results):**
- 100% exact vessel/configuration/coating/propulsor/condition fingerprint and calibration lineage for every performance observation.
- Zero fouling confirmation from power/speed residual alone.
- Zero future-data leakage into reference normalization or incremental updates.
- Zero silent overwrite of reference baselines or historical predictions.
- Zero unapproved cleaning, undocumented waste/capture or unsupported species claims.
- All unknown/partial hull zones, niche areas and sensor coverage gaps preserved as unknown, not zero.
- Performance recovery claims require independent matched-condition observations and uncertainty intervals excluding the pre-registered negligible-effect region.

Linked owner work: Symthaea #11608 (biofouling/cleaning/biosecurity), #11612 (hull/acoustic/propulsion evidence join), #12095/#12135 (PWC), #12096/#12136 (tug). The biofouling auditor/schema/fixture already exist on the marine WAPS branch; this product baseline reuses them rather than creating a duplicate implementation.
