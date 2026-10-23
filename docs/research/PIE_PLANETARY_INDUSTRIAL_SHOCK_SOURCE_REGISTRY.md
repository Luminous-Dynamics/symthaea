# PIE Planetary Industrial Shock Source Registry

Date: 2026-09-12

## Purpose

Provide source-first Moon/Mars industrial shock families for PIE-009E/009F without inventing event probabilities, universal derating factors, or site-independent outage durations.

A source can establish that a hazard exists and may quantify an exact observed/tested configuration. It does **not** automatically become a universal campaign parameter.

## Registry rule

Every shock source record should preserve:

- stable hazard/source ID;
- body and site/environment scope;
- source URL / document ID / release date;
- direct observation, test, model, or operational-procedure class;
- exact phenomenon actually supported;
- affected industrial subsystem(s);
- spatial/temporal scope;
- allowed translation into PIE scenario parameters;
- forbidden inferences;
- superseding / errata lineage.

---

# Moon shock families

## M-SHOCK-ILLUMINATION-SOUTH-POLE

**Primary source:** NASA, *Challenging Conditions at the Lunar South Pole*  
https://www.nasa.gov/reference/moonbase-environment/

**Source character:** environment/architecture reference based on lunar south-pole geometry and terrain.

**Supports:**
- low solar elevation and terrain-driven moving shadows;
- location-dependent extended darkness;
- close coupling between illumination and thermal environment;
- need to account for site topography when planning solar power and long-duration operation.

**Potential PIE translation:**
- site-specific external-power reduction or interruption scenario;
- higher heater/survival-power demand during cold/dark periods;
- site-dependent shock duration derived from an evidence-bearing illumination product.

**Does not establish:**
- one universal lunar darkness duration;
- one universal solar capacity factor;
- a probability of outage.

**Useful companion source:** NASA SVS south-pole illumination products, including 2026 two-hour-cadence visualization products derived from LOLA terrain:  
https://svs.gsfc.nasa.gov/5027

---

## M-SHOCK-THERMAL-SOUTH-POLE

**Primary sources:**
- NASA Moon Base environment reference: https://www.nasa.gov/reference/moonbase-environment/
- NASA NTRS LEMS south-pole thermal-vacuum test planning, document 20260005985: https://ntrs.nasa.gov/citations/20260005985

**Supports:**
- extreme site-dependent cold and repeated transitions between thermal regimes;
- direct coupling between low-temperature survival and electrical power demand;
- long-duration hardware must treat operating temperature and survival temperature separately.

**Potential PIE translation:**
- service-load increase for thermal survival;
- machine availability reduction when declared operating envelopes are exceeded;
- explicit recovery delay after thermal survival mode.

**Does not establish:**
- a universal component failure rate;
- one thermal profile for every south-pole site;
- automatic permanent damage after every cold event.

---

## M-SHOCK-DUST-CONTAMINATION

**Primary sources:**
- NASA Lunar Surface Technology — Dust Mitigation: https://www.nasa.gov/lunar-surface-technology/
- NASA 2025 Blue Ghost EDS flight demonstration: https://www.nasa.gov/image-article/nasas-dust-shield-successfully-repels-lunar-regolith-on-moon/
- NASA dust-mitigation technology catalog: https://www.nasa.gov/dust-mitigation/
- NASA NTRS strategies toward lunar dust adhesion mitigation, document 20250000649: https://ntrs.nasa.gov/citations/20250000649

**Supports:**
- lunar dust is abrasive and electrostatic;
- contamination can affect solar panels, radiators, optics, seals, joints, tools, suits, and other exposed hardware;
- active/passive mitigation is an explicit technology area;
- EDS has demonstrated lunar-surface dust removal on specific exposed surfaces.

**Potential PIE translation:**
- declared solar/radiator/sensor/mechanism derating scenario tied to a source/test record;
- maintenance burden or cleaning-energy demand;
- common-mode exposure groups for hardware sharing the same dust environment/mitigation weakness.

**Does not establish:**
- one universal dust accumulation rate;
- one universal equipment-life penalty;
- guaranteed mitigation efficiency for every surface;
- permission to erase the dust hazard because one mitigation technology exists.

---

## M-SHOCK-DUST-POWER-DEGRADATION

**Primary source:** NASA TechPort, Dust Mitigation for Flexible Solar Arrays, project 116295  
https://techport.nasa.gov/projects/116295

**Supports:**
- dust adhesion on charged lunar solar arrays is a recognized long-duration power-generation concern;
- solar-array dust mitigation is an active engineering problem.

**Potential PIE translation:**
- source-bounded solar-generation derating or maintenance scenario;
- alternate architecture comparison between more reserve/storage and stronger mitigation.

**Does not establish:**
- a universal percentage power loss per unit time;
- a failure probability for lunar solar arrays.

---

# Mars shock families

## R-SHOCK-DUST-STORM-SOLAR

**Primary sources:**
- NASA/JPL historical rover dust-storm operations: https://www.jpl.nasa.gov/news/nasa-mars-rovers-braving-severe-dust-storms/
- NASA Mars dust-cycle overview: https://www.nasa.gov/general/dust-cycle/
- NASA NTRS 2018 global dust storm review: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20190027303.pdf

**Supports:**
- airborne Martian dust can greatly reduce direct sunlight and therefore solar-electric production in exact observed cases;
- regional/global dust storms can alter atmospheric thermal structure and persist over operationally significant periods;
- severe dust events can force systems into low-power operational modes.

**Potential PIE translation:**
- declared external solar-power reduction scenario;
- explicit survival-mode service shedding;
- heater/survival-energy stress coupled to power loss;
- recovery period after a storm rather than instant nominal restoration.

**Does not establish:**
- a universal dust-storm probability;
- a universal percentage derating for all Mars sites/architectures;
- that nuclear/non-solar systems experience the same power shock.

---

## R-SHOCK-SURFACE-DUST-ACCUMULATION

**Primary sources:**
- NASA/JPL rover experience with panel dust accumulation and cleaning: https://www.jpl.nasa.gov/news/mars-exploration-rover-status-report-rovers-resume-driving/
- NASA/JPL InSight dust-cleaning discussion: https://www.jpl.nasa.gov/news/for-insight-dust-cleanings-will-yield-new-science/

**Supports:**
- settled dust can reduce solar-panel performance;
- wind/dust-devil cleaning can sometimes improve output;
- cleaning behavior is site/event dependent and inconsistent.

**Potential PIE translation:**
- surface-dust derating with explicit maintenance or cleaning recovery;
- scenario branch with or without a declared cleaning event.

**Does not establish:**
- guaranteed natural dust-devil cleaning;
- a universal secular dust-deposition rate;
- permission to count future cleaning as reserve power.

---

## R-SHOCK-SOLAR-CONJUNCTION-COMMAND

**Primary sources:**
- NASA/JPL conjunction operations overview: https://www.jpl.nasa.gov/news/whats-mars-solar-conjunction-and-why-does-it-matter/
- NASA 2023 Mars fleet conjunction operations: https://www.nasa.gov/solar-system/planets/mars/nasas-mars-fleet-will-still-conduct-science-while-lying-low/
- NASA 2025/2026 MAVEN conjunction note: https://science.nasa.gov/blogs/maven/2025/12/23/nasa-works-maven-spacecraft-issue-ahead-of-solar-conjunction/

**Supports:**
- Mars solar conjunction periodically disrupts/restricts Earth-Mars commanding;
- missions prepare autonomous/limited-operation plans for the conjunction window;
- Earth command availability and local system operation are separable.

**Potential PIE translation:**
- autonomy stress scenario with Earth command/support unavailable for a declared interval;
- prohibition on emergency Earth intervention during the interval;
- delayed diagnostic/recovery support.

**Does not establish:**
- loss of local power or local communications by itself;
- one universal outage duration for every conjunction architecture;
- inability of a sufficiently autonomous local industrial system to continue operating.

---

## R-SHOCK-DUST-ELECTRICAL-DISCHARGE

**Primary source:** NASA/JPL, 2025 Perseverance observations of electrical discharges in Mars dust devils/storms  
https://www.jpl.nasa.gov/news/nasa-rover-detects-electric-sparks-in-mars-dust-devils-storms/

**Supports:**
- electrical discharges in Martian dust events have now been directly observed by Perseverance instrumentation.

**Potential PIE translation:**
- hazard-source record that can motivate later electrical/EMI/material-effects testing.

**Does not establish:**
- a quantified electronics failure probability;
- a universal discharge severity for industrial sites;
- any default derating until subsystem-level evidence exists.

---

# Cross-body operational shocks

The following are legitimate PIE scenario classes but require project/site-specific evidence rather than one planetary default:

- local grid segment loss;
- storage-system outage;
- transport corridor interruption;
- resource-site closure or excavation outage;
- common-mode software/controller failure;
- critical spare exhaustion;
- maintenance-capacity loss;
- missed Earth resupply;
- receiver/launcher/elevator/tug outage where the industrial chain depends on that mode.

These should connect to PIE-009E as declared scenario inputs and to PIE-009F as mandatory or non-mandatory survival cases.

# Translation contract into PIE-009E

A source-backed shock record should never serialize directly into a probability. Instead it should provide one or more explicit scenario parameters such as:

- external-power envelope;
- service-capacity factor;
- common-mode affected equipment group;
- transport-open/closed state;
- scheduled import missed/delayed state;
- maintenance burden;
- repair delay;
- heater/survival-power increment;
- site/resource availability factor.

The campaign must retain the source/evidence record that justified each parameter or clearly label it as a synthetic stress-test assumption.

# Promotion rule

`hazard source -> scoped shock record -> synthetic/evidence-bounded scenario -> PIE-009E campaign -> PIE-009F survival gate + robust Pareto -> PIE-009C decision sensitivity`

A mitigation technology may reduce a shock parameter only when its own evidence scope is bound. It must not delete the parent hazard record.

Tracks #1799, #1788, #1793, #1702, #1647 and master #1604.
