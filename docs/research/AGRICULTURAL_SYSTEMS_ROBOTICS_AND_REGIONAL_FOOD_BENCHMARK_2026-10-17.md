# Agricultural Systems, Farm Robotics, and Regional Food Resilience
## Research and benchmark contract — 2026-10-17

**Status:** proposal / research contract. Not an implementation qualification, vendor endorsement, agronomic recommendation, or authorization to operate machinery.

**Purpose:** connect existing agricultural work across Symthaea, Symtropy, Mycelix, and Sol-Atlas without creating duplicate simulation, food-cooperative, geospatial, or evidence stacks. Establish a testable platform-selection method for different farm types, integrated greenhouse/earth-house/solar systems, new farming techniques, and regional food logistics.

## 1. Design thesis: optimize the food system, not the robot

The unit of optimization is not a robot, hectare, crop, or solar panel in isolation. It is a locally bounded food-and-resource service under uncertain weather, water, labor, market, infrastructure, and supply conditions.

Every candidate plan should state:
- the service requested (for example, weed a specific field block, monitor irrigation, harvest a crop, move produce, or reduce post-harvest loss);
- the actual farm/site profile, operating envelope, and crop/system;
- evidence and uncertainty for soil, water, climate, light, terrain, crop stage, labor, energy, access, maintenance and market assumptions;
- feasible alternatives and a relevant conventional baseline;
- whole-life cost and repair/support requirements;
- operational hazards, human oversight, failure behavior, and explicit decision/actuation authority;
- outcome metrics and the conditions under which the design should be rejected.

Do not use a composite score to hide hard constraints. Food safety, worker safety, water availability/quality, legal requirements, physical access and crop survival are constraints—not compensable preferences.

## 2. Paradigm shifts to test

1. **Crop/task-first, not robot-first.** Select the least complex platform that reliably completes the whole task. For some farms, an improved implement, sensor node, or shared tractor attachment will beat an autonomous robot.
2. **One modular base, many bounded implements—where justified.** Standardize power, mounting, telemetry, tool identity, calibration, field maps and safe-stop behavior. Do not assume one chassis can perform every crop task economically.
3. **From autonomy as a binary to graded autonomy.** Begin with logging and advisory operation, then supervised execution in bounded conditions, and only later consider narrowly authorized automation. Missing, stale, contradictory or out-of-distribution evidence must cause abstention or safe stop, not confident guessing.
4. **From demo videos to season-level service metrics.** Report hectares or crop units completed, quality, downtime, supervision hours, repair burden, misses, false actions, weather/terrain limits and cost. Do not infer production readiness from simulation or a short demonstration.
5. **From centralized fleets to repairable, cooperative access.** Evaluate shared ownership, pay-per-task and local maintenance networks alongside outright purchase. Include spare parts, consumables, training, connectivity and local repair capacity in cost.
6. **From yield-only to joint resource productivity.** Co-optimize saleable food, nutrition, water, energy, land, labor, soil condition and resilience. State trade-offs rather than burying them in one “sustainability” number.
7. **From closed-loop claims to measured closure.** Recovered water, nutrients, energy or biomass count as recycled only when flow measurements and losses are recorded. A process sketch is not proof of a closed loop.
8. **From average-year planning to distributional robustness.** Test heatwaves, drought, dust, low light, sensor drift, power and connectivity loss, road disruption, input shortages, crop disease and synchronized regional harvest failure.
9. **From farm optimization to regional coordination.** Production plans should consider downstream storage, processing, transport, market demand, producer bargaining power and local availability without making a central planner the owner of all decisions.
10. **From model output to evidence-linked engineering.** Preserve the distinction between source observations, model inferences, candidate plans, human approvals, authorized actions and measured effects.

## 3. Farm-robotics platform matrix

These are candidate classes and examples for comparative evaluation, not universally “best” robots. Commercial claims below are vendor or project reports unless explicitly described as a peer-reviewed study. Local price, availability, service, import, spares, warranties, regulations and total cost in Southern Africa remain to be checked before purchase.

| Farming context / task | Best initial platform class | Candidate examples / evidence | Why it fits | Key caveats and selection gates |
|---|---|---|---|---|
| Smallholder diversified vegetables; scouting, row following, localized weeding/seeding | Low-speed, modular wheeled UGV with interchangeable tools; human-supervised first | Sowbot / Feldfreund DevKit ROS 2 is an open, containerized stack with RTK-GNSS and ESP32 hardware control; multi-row mission following is documented in Gazebo | Open interfaces, approachable field-robotics research, modular hardware, opportunity to build and repair locally | The project is under development. Simulation validation is not field qualification. Measure traction, row variability, rain/dust ingress, tool accuracy, safety and repair time on local plots. |
| Research and crop-specific implement development | Small modular development base | farm-ng Amiga is marketed as a configurable agricultural robot/developer platform | Useful for fast integration of sensor payloads, tools, teleoperation and crop-specific workflows | Commercial price, duty/import, after-sales service, payload/runtime in the proposed configuration and South African parts lead time must be verified. Compare against locally fabricated hardware. |
| Large or medium field vegetables / pasture; spot treatment | Existing tractor + vision-guided precision implement, where tractor and support infrastructure already exist | Ecorobotix ARA is a tractor-mounted ultra-high-precision spot sprayer; the vendor reports a 6 × 6 cm spray footprint and up to 95% plant-protection-product savings depending on weed density | Uses existing tractor fleet; can avoid cost of a new autonomous mobile base; measurable input-use comparison | Vendor claims are not independent guarantees. Crop algorithms, label constraints, calibration, drift, chemical handling, local registration, crop damage and actual input reduction require local validation. |
| Early, low-density weed management on small plots | Small semi-autonomous spot-treatment prototype; compare mechanical removal before considering laser actuation | The 2026 LiteWeed paper reports a roughly CAD $500 prototype, Raspberry Pi 5 edge inference, and 96–97% removal in its reported Canadian/Tanzanian trials | Illustrates a possible resource-constrained design rather than requiring industrial-scale machinery | Early research, not a proven durable product. Laser systems need engineered enclosure, interlocks, eye/fire safety, crop-damage analysis and legal review. Initial replication should be effects-disabled / non-laser or use certified test controls. |
| Cereals and broad-acre row crops | Conventional tractor or existing implement with guidance/section control; evaluate autonomy only against this baseline | Commercial precision implements and guided tractors form the baseline; avoid assuming a bespoke small UGV can outperform them at large scale | High area coverage, existing service ecosystem, compatibility with established operations | Soil compaction, fuel/energy, field geometry, operator cost, service support and total cost per treated hectare must be measured. |
| Orchards and vineyards | Narrow, low-centre-of-gravity UGV or tractor-mounted sensors for scouting, spraying support and canopy inspection | Select platform against row width, slope, canopy GNSS occlusion, trellis/branches and turning radius | Repeated rows enable topological navigation; sensors can inspect locations conventional overhead imagery misses | Test GNSS-degraded navigation, rollover risk, spraying drift, root/soil compaction and narrow-row recovery. Do not import a flat-field assumption. |
| Greenhouses, polytunnels and glasshouses | Start with fixed sensors / irrigation and environment controls; use rails/gantries where the structure is purpose-built, mobile base where retrofit flexibility matters | Agri-OpenCore reports reusable ROS 2 components, safety/navigation/fleet work, and simulation for strawberry, tomato and packhouse environments; commercial partners have conducted harvesting/navigation trials | Controlled geometry enables repeatable sensing and may simplify localization; fixed infrastructure can reduce navigation complexity | A harvest demo is not production economics. Test saleable fruit per hour, damage, occlusion, human exception handling, washdown, disease hygiene, energy and full-season availability. Human-in-the-loop / teleoperation may be better initially. |
| Plant nurseries / propagation | Bench/rail gantry, cart robot, or simple fixed automation | Choose by tray standardization, bench geometry, repetitive operations and need to avoid leaf/root damage | Highly structured environments can make simple automation economically attractive | Validate tray/cultivar variation and cleaning requirements. A mobile outdoor rover may be unnecessary complexity. |
| Livestock operations | Task-specific barn/fence/inspection robot and fixed animal-monitoring sensors, not a generic crop rover | Compare established milking, feeding, manure and monitoring automation against manual baselines | Different mobility, sensing, hygiene and welfare needs than crop rows | Animal welfare, biosecurity, entrapment, emergency access and human override are non-negotiable. Validate separately from crop robotics. |
| Mapping, crop stress, canopy and field scouting | UAV or satellite imagery paired with calibrated ground truth; ground rover for close inspection | RGB, multispectral and thermal payloads answer different questions and are not interchangeable ground truth | Efficient coverage and repeat surveys; useful input to Sol-Atlas and Symthaea | Drone rules, weather, image resolution, georegistration, privacy, sensor calibration and ground-truth design constrain conclusions. Imagery alone cannot establish soil chemistry, food safety or yield. |
| Packing, storage, and farm-to-market movement | Fixed sensors, conveyors, pallet/tote handling, or indoor AGV on defined routes | Start with lot records and temperature monitoring; automate physical movement only when throughput supports it | Repeatable paths can simplify safety and reduce handling delays | Cold-chain evidence requires calibrated time series and exception handling. A working warehouse robot does not establish food safety or freshness. |

### Provisional platform strategy

For a research-first pilot, assess these in this order:

1. **Baseline first:** map tasks, costs and existing tools; add low-risk sensor logging and supervised data collection.
2. **Open modular UGV candidate:** evaluate the Sowbot/Feldfreund ROS 2 stack and local-build option against a locally serviceable prototype and a commercially supported dev kit. No purchase decision until total-cost and local-support evidence exists.
3. **Use existing implements where they win:** for large-area work, compare tractor-based precision systems rather than forcing a mobile robot into every workflow.
4. **Delay high-risk or immature automation:** chemical application, laser actuation, autonomous heavy machinery, unsupervised harvesting and livestock interaction require task-specific safety cases and validated operating domains.

The recommendation should be allowed to return **“no robot”** when sensors, better work planning, a mechanical implement, irrigation repair, storage or shared conventional equipment offers better risk-adjusted value.

## 4. Shared robotics interface and authority boundary

A candidate cross-stack contract should keep these artifacts distinct:

- **SiteProfile:** geospatial, crop, terrain, weather, water, access and environmental constraints.
- **TaskSpecification:** crop/task, target area, deadline, quality threshold, prohibited outcomes and conventional baseline.
- **RobotProfile:** chassis, tools, envelope, payload, power, sensing, localization assumptions, maintenance and safety functions.
- **ObservationRecord:** value, unit, method, timestamp, source, calibration, uncertainty and data-quality status.
- **AnalysisArtifact:** prediction, diagnostic, sensitivity result or candidate plan with model/version, assumptions and evidence links.
- **ActionProposal:** suggested action and predicted effects; this is not permission to execute.
- **Authorization:** an explicit human or separately governed control service grants a narrow scope, duration and limits.
- **EffectReceipt:** what the actuator actually did, with timestamp and device identity.
- **OutcomeRecord:** independently observed crop/resource/safety/economic result linked to the preregistered plan.

Required boundary:

    model prediction != measured observation
    analysis artifact != authorization
    authorization != successful actuation
    actuation receipt != agronomic success
    harvest quantity != food safety or nutritional adequacy
    inventory / provenance record != physical quality qualification

The initial cross-stack agricultural profiles remain **read-only and effects-disabled**. This document creates no pesticide/nutrient dosing, irrigation, machinery, procurement, allocation, or harvest authority. Any later physical-control work needs a separate bounded protocol, interlocks, fail-safe behavior, audit trail, trained operators, and independent safety review.

## 5. Benchmark suite

Freeze metrics and acceptance thresholds before examining test results. Report distributions and failure cases; do not compress all outcomes into a single score.

### A. Perception and decision quality

- Per-class precision, recall, F1 / average precision, calibration and abstention rate for crop/weed/fruit/disease/obstacle detection.
- False treatment, missed treatment, crop-damage and non-target disturbance rates, separately by crop stage, target size, lighting, dust, wind, shadows, species and site.
- Prediction interval coverage and forecast error; compare against naive seasonal and agronomist/standard-tool baselines where appropriate.
- Leave-one-farm, leave-one-season and where feasible leave-one-region-out validation. No random image split may leak near-identical frames from the same plot into train and test.
- Explicit stale, missing, contradictory, out-of-distribution and miscalibrated-sensor tests; expected response must include abstention or bounded safe fallback.

### B. Mobility, manipulation and operational robustness

- Task completion and intervention rate per hectare / row / hour.
- Navigation failure, crop strike, tool placement error, obstacle detection, localization drift, recovery time and safety-stop latency.
- Coverage, productive minutes versus total elapsed minutes, battery/energy use, cleaning, charging, setup and human supervision hours.
- Performance under wet/dry soil, slopes, ruts, dust, thermal stress, low light, rain exposure, vegetation occlusion, GNSS outage and connectivity loss.
- Reliability by repeated runs and field-days; record repair events, maintenance minutes, spare part cost and failed missions. A simulation result must never be relabeled as a field result.

### C. Agronomy and resource outcomes

- Marketable (not merely total) yield per area, quality grade, damage/rejection rate and yield stability across seasons.
- Water withdrawal and water consumption per kg of marketable product, with system boundary and losses stated.
- Energy consumed and generated; kWh per kg; pump/cooling/storage loads; irrigation and nutrient recovery mass balances.
- Crop protection products per hectare and per kg output; efficacy and crop injury versus a conventional control. Chemical reductions matter only if efficacy, crop safety and applicable label/regulatory constraints are maintained.
- Soil compaction, infiltration, erosion, soil carbon with repeatable methods, nutrient losses and non-target biodiversity.
- For integrated solar-food sites, calculate a crop-and-electricity land-equivalent ratio against separately produced food and electricity. Include seasonal yield, shading, maintenance access, embodied materials and displaced habitat; LER above 1 is not proof of universal environmental benefit.

### D. Economics and adoption

- Full cost per completed task, treated hectare and kg of marketable crop, including capital amortization, labor, operator training, integration, internet, repairs, consumables, insurance, downtime and spares.
- Payback and net present value across low/base/high utilization, power/fuel prices, crop prices and failure scenarios.
- Supervision hours per productive hour; time to train a new operator; task abandonment and farmer-reported usefulness.
- Local serviceability: proportion of repairs possible locally, spare-parts lead time, tool-change time, mean time to repair, import dependence and vendor lock-in.
- Compare purchase, lease, cooperative/shared-asset, pay-per-task and conventional/manual alternatives.

### E. Regional food network

- Forecast error for crop-by-location-by-harvest-window supply and demand, with uncertainty and revision history.
- Fill rate, delivery reliability, transit/handling loss, cold-chain excursions, stockout duration, delivered cost and producer netback.
- Share of final retail/contract value retained by the producer; time to payment, contract fulfillment and small-producer participation.
- Robustness to drought, road/fuel disruptions, border delay, cold-store failure and simultaneous supplier shortages.
- Concentration and bargaining effects: never improve aggregate efficiency by hiding increased dependency, exclusion, or reduced local access.

## 6. Stage-gated target-setting

These are proposed evaluation goals to test, not claimed present capabilities or guaranteed improvements.

**Gate 0 — evidence readiness**
- 100% of benchmark cases declare source, unit, site/time boundary, evidence status and intended claim.
- Every reported metric is reproducible from versioned inputs and a documented evaluator.
- 100% of actuation-sensitive tests include a negative control demonstrating that a model output alone cannot actuate equipment.

**Gate 1 — simulation and replay**
- Deterministic replay of defined synthetic scenarios; independently test missing/stale/conflicting data and unsafe proposals.
- All hard-constraint violations in the curated adversarial suite are rejected. Any known false acceptance blocks the relevant claim.
- Separate metrics for navigation, perception, tool performance, agronomic inference and economic model error.

**Gate 2 — shadow mode on site**
- Observe and recommend without control authority. Compare predictions to calibrated sensors and human-labeled outcomes across representative field conditions.
- Establish site-specific baseline distributions before approving a numeric performance target.
- Log all abstentions, corrections, near misses, sensor losses and human overrides.

**Gate 3 — supervised, bounded trial**
- Preregister an acceptable crop-damage limit, task success threshold, stop condition, minimum independent observations, operator coverage and failure budget before a trial.
- Compare with randomized or matched control plots where feasible. Include enough time to observe agronomic effects, not merely robot task completion.
- No claim of lower inputs, higher yield, safety or savings until the corresponding outcome is measured against a baseline.

**Gate 4 — multi-site operational qualification**
- Independent replication on distinct sites and seasons; explicit limits on crops, machine configurations, weather and operator skill.
- Economics must include maintenance and downtime over a meaningful operating period.
- Any major software, tool, crop or site shift triggers defined requalification rather than inheriting an unbounded prior pass.

Initial numerical thresholds should be selected after a baseline feasibility run and recorded before holdout evaluation. Use safety-critical zero-tolerance rules for unauthorized actuation and violations of explicit hard constraints; use statistical confidence intervals and severity-weighted reporting for biological and operational failure rates.

## 7. Cross-project ownership (extend existing work)

This RFC composes existing owners rather than replacing them:

- **Symthaea:** #5967 AGRI-MFG infrastructure chain; #5999/#6000 bounded evidence contract and independent corpus qualification; #6213 hydroponic analysis profile. Own cognition, evidence semantics, prediction quality and agribot task intelligence—not unbounded physical authority.
- **Symtropy:** #1661 AGRI-VOI measurement selection; #1649 source-pinned seasonal food portfolio; transport and physical-resource simulation. Own deterministic scenario simulation, counterfactuals, sensitivity and risk/economic trade-offs.
- **Mycelix:** #5001 neighborhood producer cooperative; #4900/#4901 logistics commons architecture/reference model. Own participant-owned offers, commitments, provenance, custody, delivery, shared capacity and market transparency—not agricultural truth or physical quality.
- **Sol-Atlas:** #107 Johannesburg/Highveld food-resilience model profile; site-suitability work. Own geospatial evidence, site constraints, climate/resource layers and explicit spatial provenance.
- **Human operators and domain experts:** own crop selection, local operating rules, safety, approval of physical interventions, and interpretation of field evidence.

Cross-stack hypothesis: integration should beat independently optimized components on measured service outcomes while preserving local control, independent verification and failure containment. Benchmark both composed and uncomposed arms to isolate incremental value.

## 8. First research tranche

1. Inventory existing Agribot, hydroponics, site-suitability, food cooperative and logistics primitives; record gaps rather than implement a second stack.
2. Select one reference task with low hazard and clear ground truth: irrigation/environment observation or scouting on a defined vegetable plot. Keep output effects-disabled.
3. Compare three platform strategies: fixed sensors / manual work, existing implement + instrumentation, and modular UGV. Include a no-robot baseline.
4. Build a frozen synthetic scenario corpus for navigation/observation/supply disruptions, then add permitted, rights-cleared field data with source-file hashes and annotation provenance.
5. Reuse Agri-OpenCore / ROS 2 components where license and integration fit; keep Rust domain contracts and validators independent of a particular vendor or middleware.
6. Add greenhouse/earth-integrated and agrivoltaic candidate designs as simulation profiles with separate crop, energy, water, structural, safety, cost and environmental evidence.
7. Link the physical production profile to Mycelix's cooperative/logistics fixtures and Sol-Atlas site conditions; evaluate regional disruption scenarios.
8. Only propose a field trial after simulator, calibration, operator and safety prerequisites are documented.

## 9. Current research references

1. Agri-OpenCore, “Agri-OpenCore completes: an open core for agri-robotics” (15 Oct 2026): https://agri-opencore.org/2026/10/agri-opencore-completes/
2. Agri-OpenCore, project outcomes and field trials: https://agri-opencore.org/outcomes/
3. Sowbot / Feldfreund open ROS 2 reference stack (development project): https://github.com/Agroecology-Lab/feldfreund_devkit_ros
4. Kumar et al., “Modular 4WD agricultural robot for cutting, collection, and precision seeding: design and simulation-based evaluation,” *Scientific Reports* (17 Mar 2026): https://www.nature.com/articles/s41598-026-44388-6
5. “LiteWeed: An affordable edge-AI robotic platform for laser weed management in resource-constrained environments,” *Smart Agricultural Technology* (2026), DOI: https://doi.org/10.1016/j.atech.2026.102340
6. “Autonomous navigation in agricultural systems: Techniques, challenges, and future directions,” *Smart Agricultural Technology* (Aug 2026), DOI: https://doi.org/10.1016/j.atech.2026.102119
7. “A Systematic Review of 59 Field Robots for Agricultural Tasks: Applications, Trends, and Future Directions,” *Agronomy* (13 Sep 2025): https://www.mdpi.com/2073-4395/15/9/2185
8. Ecorobotix ARA product/technical claims (vendor source; must be independently tested): https://ecorobotix.com/en-us/crop-care/ara-field-sprayer/
9. farm-ng Amiga (vendor source; price, regional support and configured capability require separate verification): https://farm-ng.com/
10. “Agrivoltaic Systems: Systematic Review and Meta-Analysis Relating to Energy, Soil, and Crop Parameters,” *Annual Review of Environment and Resources* (2026): https://www.annualreviews.org/content/journals/10.1146/annurev-environ-121224-073319
11. FAO Regional Office for Africa, “Taking action for Africa’s agrifood systems amid global uncertainty” (7 Apr 2026): https://www.fao.org/africa/news-stories/news-detail/taking-action-for-africa-s-agrifood-systems-amid-global-uncertainty/en

Source status is not equivalent to independent replication. Vendor claims stay vendor claims; simulation studies stay simulation results; no local field-performance or South African procurement result is asserted here.

## 10. Standards-informed autonomy, safety, and evaluation refinement

A current review published 20 April 2026 organizes agricultural robotics along task, platform, autonomy-stack, and evaluation-setting axes. It distinguishes laboratory, greenhouse, open-field single-season, and open-field multi-season/multi-site evidence, and notes that uptime, failure recovery, near-miss events, and human-proximity triggers are not consistently reported. Therefore results must be stratified by setting and environmental domain; a higher detector score in one dataset is not a field-readiness claim.

Reference:
- Almtireen and Ryalat, *Robotics in Precision Agriculture: Task-, Platform-, and Evaluation-Oriented Review* (2026): https://www.mdpi.com/2218-6581/15/4/81
- Polanía et al., *Control System for the Navigation of the Agricultural Robots: A Review* (2026): https://onlinelibrary.wiley.com/doi/10.1002/rob.70199

### Safety standards as design inputs, not badges

Use ISO 18497-1:2024 for agricultural autonomous-machinery design principles, vocabulary, safe-work information and residual-risk disclosures, and ISO 18497-4:2024 for verification and validation principles:
- https://www.iso.org/standard/82684.html
- https://www.iso.org/standard/82688.html

These standards do not, by themselves, establish legal compliance or certify any machine. The project should express a task- and configuration-specific Operating Domain (ODD) and Safety Case: robot/tool/software identity, allowed tasks, allowed environment, sensor freshness and calibration, stopping and recovery behavior, human override, foreseeable misuse, evidence references, residual-risk owner, and change-triggered requalification. South African rules and field-specific safety must be reviewed by qualified specialists before real operation.

A model must not receive an ambient ability to drive, irrigate, dose, spray, seed, harvest, or route physical assets. Recommendations, authorizations, device effect receipts, and agronomic outcome observations remain separate types. Unknown, stale, conflicting, failed, and not-tested are different states.

A dedicated follow-on safety work item is open: [AGRI-ROBOTICS-SAFETY-001 #7818](https://github.com/Luminous-Dynamics/symthaea/issues/7818). The first implementation target remains effects-disabled contract and corpus validation.

### Benchmark expansion informed by the reviews

For every robot-task pairing, preregister task-specific measures and record the test setting:
- **Navigation:** cross-track and heading error, route completion, intervention rate, localization-loss recovery, and performance under canopy, dust, uneven soil, slope, and GNSS denial where relevant.
- **Manipulation/harvest:** successful picks or tool placements, quality/damage/rejection, cycle time, throughput, occlusion, and exception-handling rate.
- **Operational reliability:** uptime, failed missions, mean time to recover/repair, charging, cleaning, setup, tool change, maintenance and human-supervision hours.
- **Safety:** stop response, obstacle/human-proximity triggers, near misses, crop/soil/property damage, failed interlocks, and every out-of-domain operation attempt.
- **Economics:** full cost per completed service unit and per marketable kilogram, not only the robot purchase price.
- **External validity:** hold out complete farms, seasons, and regions where feasible; never let neighboring video frames from the same plot leak into both train and test sets.

A 5 cm navigation tracking error has been discussed as an evaluation metric in a 2026 navigation review; treat it only as a candidate metric for tasks where that tolerance matters. The engineering tolerance must be justified by crop/implement geometry, target size, crop damage, and hazard analysis, not adopted as a universal pass threshold.

### Agrivoltaics: test integration rather than presume it

The 2026 Annual Review meta-analysis includes 66 studies and reports substantial heterogeneity across module geometries and crops. Higher mean land-equivalent ratios in some crop groups do not establish that an arbitrary layout is superior. Each local design should jointly measure:
- saleable crop yield and quality;
- AC electricity delivered at the system boundary, including curtailment/storage losses if claimed;
- land-equivalent ratio with explicit separate-system baselines;
- irrigation withdrawal/consumption and microclimate response;
- robot/tractor turning, clearance, headland, shade and maintenance access;
- lifecycle material and ecological effects, capital and operating costs;
- multi-season crop stability and downside outcomes.

Reference: Barros et al., *Agrivoltaic Systems: Systematic Review and Meta-Analysis Relating to Energy, Soil, and Crop Parameters* (2026), https://www.annualreviews.org/content/journals/10.1146/annurev-environ-121224-073319

The intended paradigm shift is a falsifiable co-design process: the best combined system must outperform relevant separate or conventional alternatives on declared objectives and constraints at a particular site. A visually integrated greenhouse, an autonomy demo, an LER above 1, or a vendor-reported input saving is not sufficient evidence by itself.

## 10. Explicit nonclaims

This document does not establish crop/yield benefits, disease diagnosis, food safety, nutrient adequacy, regulatory compliance, safe machinery operation, commercial viability, positive ROI, improved farmer bargaining power, land-equivalent superiority, or any other real-world outcome. These require task- and site-specific evidence and independent validation.
