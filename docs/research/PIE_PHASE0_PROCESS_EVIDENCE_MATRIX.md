# PIE Phase-0 Moon/Mars process evidence matrix

Date: 2026-09-11

## Purpose

Seed PIE-004/005 with **source-first process families** before any numeric process registry is treated as operational truth.

This document is a research index, not a plant database. It intentionally separates:

- flight or relevant-environment demonstration;
- integrated terrestrial / analog demonstration;
- laboratory measurement;
- published model;
- developer/vendor projection;
- unresolved resource/site assumptions.

Numeric throughput, yield, purity, power, mass, or TRL values must remain attached to the exact source/evidence record that supports them. They must not become universal defaults for a process family.

## Evidence-class rule

PIE should not impose a single total ordering across unlike evidence origins. `VendorProjection`, `LiteratureModel`, `LabMeasured`, `RelevantEnvironmentMeasured`, `IntegratedDemonstration`, and `Qualified` describe different evidence character. Promotion requires declared scope, not enum order.

---

# Moon seed matrix

| Family | Initial evidence source | Conservative PIE seed class | What the source supports | What it does **not** establish |
|---|---|---|---|---|
| Regolith beneficiation — magnetic + electrostatic | NASA TechPort LuSTR: https://techport.nasa.gov/projects/158644 | `RelevantEnvironmentMeasured` for the reported parabolic-flight separation scope; ground metrics remain separately tagged | Integrated magnetic separation + electrostatic sieving; NASA reports 2025 lunar-gravity parabolic testing and earlier ground enrichment/size-separation results | Full-scale lunar mining plant, long-duration dust reliability, universal ore upgrade, production economics |
| Regolith milling / comminution | NASA TechPort CRUMBLE: https://techport.nasa.gov/projects/158666 | `LabMeasured` / development pending exact result lineage | Milling/beneficiation research for lunar regolith simulants under low-gravity/vacuum-relevant conditions | Industrial throughput, wear life, lunar qualification |
| General regolith/ice handling + water capture | NASA TechPort FLEET: https://techport.nasa.gov/projects/116395 | Mixed `LabMeasured` + model/component-development records | Excavation, transport, reactor feeding, product capture are explicit ISRU gaps; project developed hardware concepts/experiments for these gaps | End-to-end water or oxygen plant closure |
| Molten regolith electrolysis (MRE) | NASA TechPort: https://techport.nasa.gov/projects/116413 | `IntegratedDemonstration` only for exact demonstrated terrestrial/vacuum scope; otherwise development evidence | Raw oxide regolith can be melted/electrolyzed to separate oxygen and metals; project is explicitly Moon/Mars applicable | Lunar production rate, lifetime, product grade, settlement-scale economics |
| Lunar construction from regolith | NASA NTRS MMPACT: https://ntrs.nasa.gov/citations/20205007535 and NASA lunar-surface portfolio: https://www.nasa.gov/lunar-surface-technology/ | `IntegratedDevelopment` / measured sub-process records as individually sourced | Autonomous construction, regolith feedstocks, microwave/sintering routes for pads, roads, berms, shelters and related civil works | That one construction recipe works at every lunar site or meets pressure-structure requirements |
| Oxygen + iron/steel via beneficiation/reduction/refining | NASA TechPort MMOST: https://techport.nasa.gov/projects/102905 | `VendorProjection` / developer demonstration lineage until specific measurements are bound | Integrated concept combining size sorting, electromagnetic beneficiation, hydrogen reduction, electrolysis and melt refining | Qualified lunar steel chemistry, industrial throughput, closed hydrogen/reductant logistics |
| Ionic-liquid electrochemical metals + oxygen | NASA TechPort: https://techport.nasa.gov/projects/125599 | `VendorProjection` / development evidence | Candidate lower-temperature electrochemical route for lunar metals/oxygen | Claimed purity/rate/energy as universal or qualified performance |
| Polar water/volatile extraction | NASA TechPort: https://techport.nasa.gov/projects/154566 | `VendorProjection` / development evidence; resource occurrence handled separately | Candidate extraction of water plus volatile species from icy-regolith analogs | That listed volatiles are recoverable at a particular site or economic concentration |
| Ice-mining thermal integration | NASA TechPort thermal-management project: https://techport.nasa.gov/projects/113546 | `Model` / development evidence unless exact test records are attached | Useful process coupling: heat source -> sublimation/extraction, cold environment -> volatile capture | Settlement-scale water production or site abundance |
| Locally manufactured solar power | NASA TechPort ISRUPower / Blue Alchemist: https://techport.nasa.gov/projects/146991 and NASA lunar-surface portfolio | `IntegratedDemonstration` for the stated developer/NASA Tipping Point scope | NASA reports an integrated system producing silicon solar cells, aluminum wire, oxygen, iron and slag from regolith simulant in lunar environmental conditions, with TRL-6 maturation reported by TechPort | Independent qualification of every process step, indefinite lunar lifetime, or unrestricted scale-up |
| Feed/removal for continuous oxygen extraction | NASA TechPort FaRROE: https://techport.nasa.gov/projects/125516 | `VendorProjection` / development evidence | Continuous regolith feed/removal and reactor sealing are explicit integration problems | Proven long-duration lunar reactor operation |

## Moon priority multiplier chains

PIE should initially evaluate, without assuming any is optimal:

1. `regolith -> civil construction`;
2. `regolith -> beneficiation -> oxygen + metals`;
3. `regolith -> silicon/aluminum/iron -> local solar power -> more processing capacity`;
4. `icy regolith -> water -> H2/O2 -> life support / chemical / transport feedstocks`;
5. `metals -> wire / structures / tools -> maintenance and equipment replacement`.

---

# Mars seed matrix

| Family | Initial evidence source | Conservative PIE seed class | What the source supports | What it does **not** establish |
|---|---|---|---|---|
| Atmospheric CO2 -> oxygen | NASA MOXIE mission result: https://www.nasa.gov/solar-system/nasas-oxygen-generating-experiment-moxie-completes-mars-mission/ | `IntegratedDemonstration` / **Mars flight evidence** | MOXIE operated on Mars 16 times; NASA reports 122 g total O2, up to 12 g/h, and >=98% purity at its best operating point | Direct scale-up to crew/ascent plant, lifetime, liquefaction/storage system, economics |
| Mars soil/ice -> water | NASA TechPort water-extraction portfolio: https://techport.nasa.gov/projects/93846 | Mixed model/lab/relevant-environment evidence by sub-project | Multiple acquisition architectures for granular soil, hydrated material and subsurface ice; explicit mass/power/volume technology-gap work | One universal Mars water abundance or extraction rate |
| Hot-CO2 water extraction | NASA TechPort MRWE / AMWAS: https://techport.nasa.gov/projects/8856 and https://techport.nasa.gov/projects/93346 | `VendorProjection` + development/test records | Recirculating heated CO2 as a candidate heat-transfer/extraction medium; purification train concepts | Site-independent water concentration, flight qualification, industrial duty cycle |
| Deep-ice water extraction | NASA TechPort RedWater: https://techport.nasa.gov/projects/116361 | `RelevantEnvironmentMeasured` only for the explicitly reported component/integration scope | Combined coiled-tubing + RodWell-style deep-ice extraction concept with TRL maturation work | Universal Mars site feasibility or settlement water supply |
| CO2 + H2O -> CH4 + O2 | NASA research: https://www.nasa.gov/directorates/stmd/space-tech-research-grants/producing-methane-and-oxygen-on-mars-using-proton-conducting-ceramics/ | `LabMeasured` / model/development records | Integrated electrolysis + Sabatier/protonic-ceramic research path from local CO2/water toward methane/oxygen | Flight demonstration or full-scale propellant plant |
| Integrated Mars resource-processing architecture | NASA TechPort MARCO POLO APM: https://techport.nasa.gov/projects/16846 | `IntegratedDemonstration` only for exact Mars-analog subsystem scope | Modular architecture joining atmosphere capture, soil/water processing, electrolysis, Sabatier chemistry and power | Operational Mars plant or qualified settlement production |
| Regolith -> enriched ore -> steel powder + O2 + water | NASA TechPort: https://techport.nasa.gov/projects/93728 | `VendorProjection` / development evidence | Candidate chain covering ore enrichment, reduction, purification, steel-powder production and heat/water recovery | Demonstrated Mars steel plant, structural-alloy qualification, industrial throughput |
| Molten regolith electrolysis on Mars feedstocks | NASA TechPort MRE: https://techport.nasa.gov/projects/116413 | Same exact test evidence as MRE; Mars applicability remains a separate scope claim | Process family is explicitly intended for Moon or Mars oxide regolith | Mars-specific feedstock performance without corresponding evidence |
| Mars biomanufacturing | NASA TechPort CUBES-II: https://techport.nasa.gov/projects/118324 and NASA CUBES page | `IntegratedDevelopment` / lab and Mars-like-condition evidence records by experiment | Integrated multi-organism biomanufacturing research for fuel, materials, food, pharmaceuticals and in-situ media; perchlorate remediation is within the program scope | Flight-proven closed biomanufacturing economy, stable decades-long yields, contamination-free autonomous operation |
| Thermal recovery in ISRU | NASA TechPort heat-recovery work: https://techport.nasa.gov/projects/8879 | `VendorProjection` / lab-development evidence | Waste heat from spent regolith may be recoverable to preheat fresh feed; demonstrates why PIE-002 temperature-aware heat accounting matters | Universal recovery fraction or equipment lifetime |

## Mars priority multiplier chains

PIE should initially evaluate:

1. `atmospheric CO2 -> O2`;
2. `water + CO2 -> H2/O2 -> CH4 + recycled H2O`;
3. `regolith -> metals / steel / glass / ceramics -> structures and machine components`;
4. `water + waste + atmosphere/regolith nutrients -> biological feedstocks -> food / materials / pharmaceuticals`;
5. `waste heat -> extraction / drying / preheat -> reduced power and radiator burden`.

---

# Registry-entry rules

A PIE-004/005 process record derived from this matrix should include at least:

- stable process/version ID;
- body/site/environment assumptions;
- input/output material IDs and grades;
- all matter crossing the process boundary;
- utility/power/thermal/time requirements;
- equipment and critical consumables;
- evidence source and evidence class;
- exact demonstrated/projected scope;
- numeric performance only when the source supports it;
- by-product/reject/waste disposition;
- composition/unknown-fraction data when available;
- explicit scale-up assumptions;
- downstream capabilities enabled;
- unresolved dependencies.

## Numeric-default prohibition

Do **not** set a registry-wide default throughput, yield, purity, recovery, power, or lifetime from a single demonstration. A measurement such as MOXIE's flight rate or a LuSTR ground-separation result belongs to that evidence record and exact configuration.

## Promotion path

`source index -> evidence record -> PIE-001 mass closure -> PIE-002 utility screen -> PIE-003 grade/constituent screen -> process-chain composition -> PIE-006 circularity -> PIE-007/008 dependency closure -> PIE-009 seed optimization`

Only after that should the process participate in claims about industrial closure or settlement self-expansion.
