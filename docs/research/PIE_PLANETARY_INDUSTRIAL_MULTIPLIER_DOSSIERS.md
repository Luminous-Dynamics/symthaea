# PIE Planetary Industrial-Multiplier Dossiers

Date: 2026-09-12

## Purpose

This document sits above the source-first process matrix in #1644. It does **not** rank Moon/Mars process families with one score and does not convert demonstrations into plant defaults.

Instead, each dossier asks a more useful industrial question:

> If this process works at a useful scale, which additional capabilities does it unlock, which local feedback loops become possible, and which Earth-supplied blockers still prevent reproductive closure?

The dossiers are intended for PIE-004/005 registry design and later PIE-009 seed campaigns.

## Multiplier vector

Each candidate should eventually report a vector rather than a scalar score:

- evidence maturity and exact scope;
- local-resource dependence and site sensitivity;
- coproduct breadth;
- power-generation / power-distribution feedback;
- civil-construction enablement;
- machine-part / maintenance enablement;
- transport / propellant enablement;
- chemistry / biology enablement;
- circularity / waste-heat potential;
- Earth-import blocker sensitivity;
- scale-up uncertainty;
- material-grade / purity uncertainty;
- equipment-lifecycle / maintenance uncertainty.

A candidate may dominate in one dimension and lose in another. PIE should preserve those alternatives.

---

# Moon dossiers

## M-L1 — Regolith -> MRE -> oxygen + metals -> local power / manufacturing

### Evidence anchors

1. NASA TechPort MRE project, updated 2026-07-31:
   https://techport.nasa.gov/projects/116413

   The project describes molten regolith electrolysis as a Moon/Mars process that melts oxide regolith and electrochemically separates oxygen and metals. The project description states that iron, silicon, aluminum and other species separate according to breakdown voltage. Treat this as process-family and project-scope evidence, not a universal yield/purity law.

2. NASA Kennedy MRE vacuum demonstration summary:
   https://www.nasa.gov/centers-and-facilities/kennedy/nasa-kennedy-breathes-life-into-moon-soil-testing/

   NASA reports oxygen extraction from lunar-regolith simulant using a MRE reactor in a vacuum chamber. This is relevant terrestrial/vacuum demonstration evidence, not lunar field qualification.

3. NASA NTRS system model of a lunar MRE plant:
   https://ntrs.nasa.gov/citations/20240013999

   Useful because it explicitly treats an MRE plant as a chain from excavation through product storage rather than an isolated reactor.

4. NASA TechPort ISRUPower / Blue Alchemist project, updated 2026-07-17:
   https://techport.nasa.gov/projects/146991

   The project describes an integrated chain using lunar-regolith simulants to produce silicon solar cells, aluminum wire, oxygen, iron and slag in lunar environmental conditions, and reports TRL-6 maturation for the integrated project. PIE must preserve that as project-reported scope and not silently promote every sub-process to independently qualified hardware.

### Potential multiplier loop

`regolith -> oxygen + Fe/Si/Al-bearing products -> wire / photovoltaic hardware / structures -> increased local electrical capacity -> increased regolith processing`

This is one of the strongest candidate positive-feedback loops because the process family may couple a bulk local feedstock directly to both commodity production and expansion of the utility that powers further production.

### Downstream capabilities potentially enabled

- oxygen for life support and propulsion feedstock;
- iron-rich structural or manufacturing feedstocks after suitable refining/qualification;
- silicon-bearing feedstock for glass / photovoltaic chains after suitable purification;
- aluminum-bearing feedstock for conductors / structures after suitable purification;
- slag/ceramic-type coproduct routes if characterized and useful;
- local power generation and transmission if the Blue Alchemist-style downstream chain proves scalable and maintainable.

### Critical unresolved blockers

- excavation, comminution, grading and continuous reactor feeding;
- electrode / crucible lifetime and locally replaceable refractory materials;
- product separation and post-refining requirements;
- actual Fe/Si/Al product grade and consistency for downstream manufacture;
- electrical specific energy and peak-power requirements across scale;
- thermal rejection / recuperation and long-duration furnace cycling;
- vacuum seals, insulation, high-temperature sensors, power electronics and controls;
- dust exposure, maintenance interval, autonomous cleanout and reactor rebuild;
- whether locally produced silicon and aluminum can meet the exact requirements of the photovoltaic / conductor process chain without high-leverage imported dopants, coatings, contacts or electronics.

### Decision-critical experiments

PIE-009C should test whether uncertainty in any of the following changes the preferred seed architecture:

1. sustained specific energy per kg of processed feed;
2. electrode/refractory replacement interval;
3. product-grade distribution, not only bulk mass yield;
4. continuous-feed uptime and maintenance burden;
5. fraction of the solar-power manufacturing chain that remains import-dependent;
6. thermal recovery effectiveness at realistic duty cycle.

### Non-claims

This dossier does not establish lunar production rate, plant lifetime, commercial economics, unrestricted scale-up, or complete local manufacture of photovoltaic systems.

---

## M-L2 — Regolith -> civil construction

### Evidence anchors

NASA TechPort MMPACT, updated 2026-08-10:
https://techport.nasa.gov/projects/116319

NASA describes MMPACT as developing use of lunar in-situ materials for large infrastructure such as habitats, berms, landing pads, blast shields, walkways, foundations/floors, storage facilities and roads.

NASA NTRS project overview:
https://ntrs.nasa.gov/citations/20205007535

### Potential multiplier loop

`raw / minimally processed regolith -> civil infrastructure -> safer landing / transport / equipment siting -> lower dust and infrastructure losses -> higher industrial availability`

The multiplier is different from MRE: this route may require much less purification, so it could create useful infrastructure before high-grade metallurgy is mature.

### Potentially enabled capabilities

- landing/launch pads and blast protection;
- roads / prepared industrial surfaces;
- berms and radiation/dust shielding;
- foundations and equipment pads;
- storage / civil structures;
- protection of higher-value imported machines.

### Critical unresolved blockers

- site-specific feedstock variability;
- energy per unit of finished structure;
- wear life of excavation, deposition, microwave/thermal and finishing equipment;
- structural-property distributions under lunar thermal cycling/vacuum;
- repairability and crack/damage inspection;
- suitability distinctions between unpressurized civil works and pressure-bearing structures;
- dust created during construction itself.

### Decision-critical experiments

- construction energy and throughput under representative lunar feedstocks;
- long-duration equipment wear and replacement-part demand;
- structural degradation under thermal cycling;
- whether prepared roads/pads materially improve logistics reliability enough to dominate alternative surface-transport infrastructure.

---

## M-L3 — Polar volatiles -> water -> H2/O2 / chemistry

### Evidence anchors

NASA ISRU overview:
https://www.nasa.gov/overview-in-situ-resource-utilization/

NASA TechPort water-extraction portfolio:
https://techport.nasa.gov/projects/93846

Spatial occurrence evidence must remain separate under PIE-P0-E / #1702; an orbital or remote-sensing resource signature is not a recoverable reserve.

### Potential multiplier loop

`site-specific icy material -> water -> life support + electrolysis -> H2/O2 -> propulsion / reduction chemistry / process feedstock -> expanded transport and industrial reach`

### Potentially enabled capabilities

- potable/process water after appropriate treatment;
- oxygen and hydrogen via electrolysis;
- hydrogen-reduction metallurgy routes;
- propulsion feedstocks;
- thermal storage / shielding applications;
- biological / agricultural water supply.

### Critical unresolved blockers

- actual site abundance, depth, spatial variability and accessibility;
- acquisition/recovery efficiency;
- volatile loss during excavation and transfer;
- contamination and purification burden;
- power and heat needed for extraction;
- cold-trap / thermal-management integration;
- transport from resource site to processing/use site;
- cryogenic/storage boiloff and handling where relevant.

### Decision-critical experiments

The highest-value measurements are likely site- and architecture-dependent. PIE should prioritize those that change whether a water-first industrial seed beats a regolith/metals-first seed, including recoverable water per excavation effort, thermal duty, contaminant burden, and transport cost to the industrial hub.

---

## M-L4 — Beneficiation + reduction/refining -> iron/steel -> machine parts

### Evidence anchors

NASA TechPort MMOST:
https://techport.nasa.gov/projects/102905

The project describes a chain combining particle-size sorting, electromagnetic beneficiation, hydrogen reduction, electrolysis and melt refining to produce metallic iron/steel and oxygen.

NASA TechPort CRUMBLE:
https://techport.nasa.gov/projects/158666

The project investigates lunar-regolith milling/comminution, an important upstream operation whose wear and throughput can bottleneck beneficiation.

### Potential multiplier loop

`regolith -> beneficiated Fe-bearing feed -> metallic iron / steel -> structures / gears / shafts / housings / tooling -> mining and process-equipment maintenance -> more resource acquisition`

### Critical unresolved blockers

- beneficiation gain vs energy/wear cost;
- reductant/hydrogen closure and recycle losses;
- alloy chemistry and availability of required alloying elements;
- heat treatment and controlled microstructure;
- machining/casting/additive-manufacturing quality;
- bearings, lubricants, seals, sensors, electronics and cutting tools that may remain imported;
- metrology and nondestructive inspection.

### Decision-critical experiments

PIE should distinguish **bulk local steel** from **machine-grade replacement capability**. The important experiments are those that show whether locally produced material can meet the actual grade, tolerances and fatigue/wear requirements of high-leverage replacement parts.

---

# Mars dossiers

## M-R1 — Atmospheric CO2 -> oxygen

### Evidence anchors

NASA MOXIE mission result:
https://www.nasa.gov/solar-system/nasas-oxygen-generating-experiment-moxie-completes-mars-mission/

NASA reports 16 Mars operating runs, 122 g total O2, and a best reported operating point of 12 g/h at 98% purity or better.

NASA TechPort project description:
https://techport.nasa.gov/projects/116291

The TechPort description contains project/model scale language including a 20 g/h, 99.6% oxygen description. PIE must **not merge that project description with the achieved mission operating points**. Planned/design capability and achieved flight evidence are separate evidence records.

### Potential multiplier loop

`Mars atmosphere -> O2 -> life support / oxidizer -> larger surface and ascent logistics capability`

This process has unusually strong environmental evidence because the core process operated on Mars, but the industrial multiplier still depends on scale-up, storage/liquefaction, power, compressor durability and maintenance.

### Critical unresolved blockers

- industrial-scale atmospheric intake/compression;
- power supply and thermal management;
- long-duration stack/compressor life;
- dust handling and filtration;
- liquefaction/storage if used for propulsion;
- redundancy, serviceability and spare-stack requirements;
- scale-up law from the demonstrated unit to plant scale.

### Decision-critical experiments

- compressor/stack lifetime at high duty cycle;
- plant-scale specific energy and peak-power profile;
- oxygen conditioning, liquefaction and storage burden;
- maintenance/spare mass per tonne of delivered O2.

---

## M-R2 — Water + CO2 -> H2/O2 -> methane + recycled water

### Evidence anchors

NASA Mars water extraction portfolio:
https://techport.nasa.gov/projects/93846

NASA Advanced Mars Water Acquisition System (AMWAS):
https://techport.nasa.gov/projects/93346

NASA proton-conducting-ceramic methane/oxygen research:
https://www.nasa.gov/directorates/stmd/space-tech-research-grants/producing-methane-and-oxygen-on-mars-using-proton-conducting-ceramics/

### Candidate chain

`water acquisition -> purification -> electrolysis -> H2 + O2`

`Mars CO2 + H2 -> Sabatier / methanation -> CH4 + H2O`

`recovered H2O -> electrolysis / process recycle`

### Multiplier potential

- ascent/surface methane-oxygen propellant;
- oxygen/life-support integration;
- hydrogen and methane as broader chemical feedstocks;
- water recycle reducing virgin acquisition demand;
- potential coupling with carbon/biological chemistry.

### Critical unresolved blockers

- accessible water site and recovery rate;
- contaminant/perchlorate handling and purifier life;
- hydrogen leakage/loss inventory;
- integrated compressor/reactor/electrolyzer duty;
- catalyst life and replacement;
- heat integration;
- methane/oxygen storage and transfer;
- whether alternative oxygen routes change the optimal methane/oxygen balance.

### Decision-critical experiments

The key uncertainty is not merely Sabatier conversion efficiency. PIE should test the whole closed chain and identify whether architecture choice is controlled by water acquisition, hydrogen loss, catalyst life, compressor duty, power, or storage.

---

## M-R3 — Regolith -> enriched ore / steel + water / heat recovery

### Evidence anchor

NASA TechPort `Regolith to Steel Powder, Oxygen & Water with Small Equipment`, updated 2026-01-22:
https://techport.nasa.gov/projects/93728

The project describes raw Martian regolith beneficiation/enrichment, water liberation, iron-oxide reduction using H2/CO mixtures, metal purification, steel-powder production, and heat recycling.

### Potential multiplier loop

`regolith -> enriched iron feed + recovered water -> metal/steel powder -> structures / replacement parts -> mining/process equipment -> more regolith processing`

with a secondary thermal loop:

`hot spent material / reactor heat -> feed preheat / water liberation -> reduced utility burden`

### Critical unresolved blockers

- feedstock/site composition variability;
- beneficiation selectivity and wear;
- reducing-gas closure and recycle;
- steel chemistry/grade and powder quality;
- thermal-recovery hardware complexity;
- machining/additive post-processing and heat treatment;
- high-leverage imported bearings, controls, seals and tooling.

### Decision-critical experiments

The highest-value question is whether this chain can produce **qualified machine-replacement feedstock**, not merely metallic product. PIE should focus on composition, microstructure, fatigue/wear and downstream manufacturing yield.

---

## M-R4 — Mars biomanufacturing as an industrial branch

### Evidence anchors

NASA CUBES page, updated 2026:
https://www.nasa.gov/directorates/stmd/space-tech-research-grants/the-center-for-the-utilization-of-biological-engineering-in-space-cubes/

NASA TechPort CUBES-II:
https://techport.nasa.gov/projects/118324

NASA describes research toward integrated multi-organism systems using Mars-like resources to support fuel, materials, food, pharmaceuticals, media production and related functions.

### Potential multiplier loop

`CO2 / water / regolith-derived nutrients / waste -> biological media -> food / polymers / pharmaceuticals / specialty chemicals -> crew health + manufacturing -> larger sustained population / industrial workforce`

Biology may be especially valuable for low-mass, high-complexity products that are inefficient to reproduce through a miniature terrestrial petrochemical/pharmaceutical industry.

### Critical unresolved blockers

- nutrient closure and trace-element supply;
- contamination control and genetic/biological stability;
- long-duration productivity;
- illumination/power/thermal burden;
- water quality and perchlorate remediation;
- sterilization and reactor replacement parts;
- downstream purification;
- coupling biological waste streams to non-biological industry.

### Decision-critical experiments

- which products show the largest Earth-import displacement per kg of bioreactor infrastructure;
- stability of yields under realistic recycled media;
- contamination recovery without large imported consumable burden;
- local production of media components and reactor consumables;
- whether biological products unlock otherwise critical dependency cliffs.

---

# Cross-body observations

## 1. Early industry should probably be grade-tiered

PIE should not assume all local material must reach aerospace/electronic grade before it is useful.

Likely tiers include:

1. bulk civil material;
2. industrial structural material;
3. machine-grade material;
4. electrical / optical / solar-grade material;
5. electronic / pharmaceutical / high-purity specialty material.

Processes can create large value at lower tiers while higher-grade feedstocks remain imported.

## 2. Coproducts matter

A process with one modest output may be less valuable than a process with several independently useful products **if** those coproducts can actually be separated and qualified.

PIE must therefore track coproduct grade, disposition, storage and downstream demand. A coproduct is not automatically useful merely because mass leaves the reactor.

## 3. Power feedback is a special multiplier

A process chain that can locally expand electrical generation or transmission can have superlinear system value because power is an upstream dependency of nearly every later process.

That makes the Blue-Alchemist-style lunar chain strategically interesting, while simultaneously making its unresolved imported components and lifecycle assumptions especially decision-critical.

## 4. Machine replacement is more important than local bulk mass

A settlement can achieve high local tonnage while remaining fragile if imported bearings, controllers, seals, catalysts, refractory components, sensors or power electronics stop the machines that make the bulk material.

Every dossier should therefore eventually connect to PIE-007/008 dependency closure and PIE-009D dynamic campaign semantics.

## 5. Thermal networks deserve equal treatment with electrical networks

MRE, reduction, water extraction, Sabatier chemistry, drying, construction and biological systems all create or consume heat at different temperature levels.

PIE-002 must keep temperature-qualified heat reusable only where thermodynamically compatible. Heat recovery should be represented as a finite allocatable flow, not a generic efficiency bonus.

---

# Promotion path for a multiplier claim

A candidate industrial multiplier should not be promoted from this document directly into a settlement claim.

Required path:

`source record`
-> `body/site/resource occurrence evidence`
-> `recoverable/deliverable feedstock`
-> `PIE-001 bulk mass closure`
-> `PIE-002 energy/power/thermal closure`
-> `PIE-003 composition/grade/constituent closure`
-> `PIE-006 inventory/circularity`
-> `PIE-007/008 dependency/reproductive closure`
-> `PIE-009D temporal campaign`
-> `PIE-009A Pareto comparison`
-> `PIE-009C decision-sensitivity / next experiment`

At every stage, campaign confidence is bounded by the weakest evidence-bearing link.

## Immediate research priority

Before an exhaustive registry, build one carefully bounded multiplier chain per body:

### Moon reference multiplier

`regolith -> MRE or competing oxygen/metals route -> one graded metal product -> wire or structural component -> explicit local power/maintenance contribution`

### Mars reference multiplier

`CO2 + site-specific water -> O2/H2 -> CH4 + recycled water -> explicit propulsion/chemistry contribution`

Then inject realistic blockers rather than hiding them: controller, catalyst, bearing, refractory/electrode, seal, purifier, sensor, compressor, power electronics, cryogenic hardware or other imported dependencies.

The objective is not to prove self-sufficiency. It is to discover the **smallest imported set that enables the largest resilient local industrial capability**.

Tracks #1758, #1644, #1607, #1608, #1702, #1713, #1752 and master #1604.
