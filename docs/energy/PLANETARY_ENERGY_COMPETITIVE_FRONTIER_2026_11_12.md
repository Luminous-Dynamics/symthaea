# Planetary Energy Competitive Frontier — 12 November 2026

**Parent:** [PLANETARY-ENERGY-SERVICE-001 #11756](https://github.com/Luminous-Dynamics/symthaea/issues/11756)  
**Related:** [PLANETARY-PV-ISRU-001 #11771](https://github.com/Luminous-Dynamics/symthaea/issues/11771)  
**Purpose:** a source-pinned competitive comparison map and benchmark translation.  
**Claim ceiling:** literature/programme-reported values and analytic targets only; no Luminous physical performance is claimed.

## 1. Main finding

Do not compete on one cell-efficiency record. The current field is already strong in several component domains:

- Space PV has mature III-V multi-junction cells with nominal 30% and reported upper offerings around 34% in NASA's 2026 small-spacecraft survey.
- Empirical spacecraft solar-array specific power is much lower than laboratory cell efficiency might suggest: NASA's flight dataset clusters around 30 W/kg and has an observed upper bound around 200 W/kg. Array mass, deployment and mission geometry matter.
- Lunar polar vertical arrays are an established NASA development path. NASA's 10-kW VSAT concept is relocatable, retractable and elevated above local terrain shadow. The second-generation concept has a 50-kW capacity and >75 W/kg goal in an FY2025 SBIR solicitation.
- NASA publicly describes Blue Alchemist as an integrated autonomous regolith-simulant process producing silicon solar cells, aluminium wire, oxygen, iron and slag. Luminous must not claim to be first to make PV from lunar regolith.
- Perovskite research is already combining ML-guided materials search with automated fabrication and closed-loop process optimization. A 2026 Nature paper reports 27.18% certified MPPT efficiency on a 0.05 cm² cell and 23.49% PCE on a 21.4 cm² mini-module, with 98.7% initial efficiency retention after 1,200 hours under the stated ISOS-L-1I protocol. A separate 2026 review identifies manufacturing cost, yield, reliability, environmental/process tolerance and recycling as major deployment constraints.

**The defensible Luminous differentiation:** join spatial/site evidence, energy/thermal co-design, storage and microgrid mission performance, materials/factory closure, lifecycle repairability and verifiable manufacturing/evidence lineage into one matched mission benchmark. This is a whole-system engineering hypothesis, not yet a demonstrated advantage.

## 2. Space PV reference — cell and array are different boundaries

NASA's 2026 State-of-the-Art: Power Systems describes current space solar-cell state of the art as 3–5 junction Group III-V devices, nominally around 30% efficiency with high values around 34%. Its manufacturer table includes these example beginning-of-life (BOL) values:

| Manufacturer / cell | NASA 2026 table BOL efficiency |
|---|---:|
| AZUR Space 3G30-Adv | 29.5% |
| AZUR Space 4G32-Adv | 31.5% |
| Rocket Lab ZTJ | 29.5% |
| Rocket Lab ZTJ Omega | 30.2% |
| Rocket Lab Z4J | 31.3% |
| Rocket Lab IMMα | 32.0% |

These are cell BOL efficiencies and do not themselves state deployed array specific power, operating life, radiation performance under a particular mission, converter efficiency or installed system mass.

The same NASA source says flown spacecraft solar-array specific power clusters around 30 W/kg, with a maximum empirical point around 200 W/kg in the dataset it reviews. This is an empirical spacecraft-array distribution, not a direct lunar ground-array target.

**PESE translation:** report separate values for cell, packaged module, blanket/wing/array, deployed array, power subsystem and complete installed/maintained plant. Include deployment hardware, support structures, hinges/motors, wiring, regulators, storage, radiators, dust management, foundations/anchoring and spares in the appropriate boundary. Never compare cell PCE with plant W/kg.

Sources:
- NASA 2026 SmallSat Institute, Power Subsystems: https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/
- NASA 2026 SOA Power Systems PDF: https://www.nasa.gov/wp-content/uploads/2026/05/3-soa-power-2026-final.pdf

## 3. Lunar vertical solar array and storage reference

NASA's current Lunar Surface Technology portfolio describes Vertical Solar Array Technology (VSAT) as an autonomously deployable/retractable and relocatable array intended to provide continuous power through lunar-day/night challenges. The current public summary describes masts up to 20 m and emphasizes uneven terrain, packing volume and mass.

NASA's earlier VSAT government reference design describes a 10-kW-class concept with array panels elevated at least 10 m above the lunar surface and planned deployment/retraction/relocation. The FY2025 NASA SBIR solicitation frames a second-generation modular 50-kW array, with a >75 W/kg goal and mechanical innovations required for packaging, transport, deployment and operation. It notes:
- near the pole, solar elevation stays very shallow (about ±1.5° at 90° S);
- high ridges/peaks may receive extended sunlight;
- some areas elsewhere can experience nights around 354 hours;
- multiple small arrays can introduce mutual shadowing, positioning, leveling, cable and deployment complexity.

These are programme design bases and solicitation goals, not completed 50-kW surface deployments. The 10-kW paper is a design reference; the 50-kW / >75 W/kg is an R&D solicitation target.

**PESE benchmark translation:**
1. Compare horizontal/tracking, vertical mast, relocatable vertical and distributed arrays on identical surveyed sites and illumination histories.
2. Measure net service after mast/structure/deployment mass, cable losses, shadow interaction, dust, stability, thermal loads, failure and relocation downtime.
3. Include storage sized to the actual darkness distribution: battery, regenerative fuel-cell/chemical storage, thermal storage, nuclear firm power and hybrid systems.
4. Report longest contiguous outage, not only average solar visibility or daily energy.
5. Compare 10-kW-class and 50-kW-class architectures at equal critical-service demand and installed boundary.

Sources:
- NASA Lunar Surface Technology: https://www.nasa.gov/lunar-surface-technology/
- NASA NTRS, Government Reference Design of the Vertical Solar Array Technology Demonstrator: https://ntrs.nasa.gov/citations/20240004914
- NASA FY2025 SBIR solicitation (50-kW, >75 W/kg goal and South Pole environment): https://www.nasa.gov/wp-content/uploads/2024/01/2025-nasa-sbir-phase-i-solicitation-final-rev.pdf

## 4. Lunar regolith-to-PV manufacturing reference

NASA's public Lunar Surface Technology page describes Blue Origin's Blue Alchemist as an integrated autonomous technology set using regolith simulants and molten regolith electrolysis plus associated processes to produce silicon solar cells, aluminium wire, oxygen, iron and slag. NASA TechPort reports that the integrated system matured to TRL 6.

NASA's 2025 ISAM State of Play says Blue Origin had produced silicon solar cells and aluminium wire using lunar regolith simulants as of 2024; its status was in development, and the first-use date was TBD. NASA's TechPort says integrated autonomous operation of eight technologies in lunar environmental conditions. These statements must remain at their source's exact maturity and demonstration boundary; they do not establish a flight/operational lunar plant or production-scale qualified arrays.

TU Berlin / JPM Silicon's SoMo project (September 2022–August 2026) also targets solar-cell manufacturing from lunar regolith with a simplified low-equipment process. A project duration and objective are not evidence of a completed, qualified industrial process unless final results are published.

**PESE / PV-ISRU benchmark translation:** compare raw feedstock, beneficiation, product purity, silicon recovery, dopant/junction quality, cell/module yield, manufacturing energy and temperature-grade heat, imported equipment/consumables, mass closure, coproduct qualification, array conversion and lifetime delivered kWh. The process must account for its own capital equipment, spares, repair and replacement capacity. A high local-mass fraction does not establish factory closure.

Sources:
- NASA Lunar Surface Technology: https://www.nasa.gov/lunar-surface-technology/
- NASA TechPort ISRUPower: https://techport.nasa.gov/projects/146991
- NASA 2025 ISAM State of Play: https://ntrs.nasa.gov/citations/20250008988
- TU Berlin / JPM Silicon SoMo: https://www.tu.berlin/en/raumfahrttechnik/research/current-projects/somo

## 5. Perovskite and autonomous manufacturing frontier

A 14 April 2026 Nature study integrates machine-learning materials discovery, automated fabrication, Bayesian optimization and symbolic regression in a closed loop. It reports:
- 27.18% certified MPPT efficiency on a 0.05 cm² cell;
- 23.49% PCE on a 21.4 cm² mini-module;
- 98.7% of initial efficiency retained after 1,200 h continuous operation under the paper's stated ISOS-L-1I protocol;
- nearly fivefold improvement in efficiency reproducibility relative to manual fabrication, in the reported experimental setup.

These are study-specific research results, not space qualification, factory throughput or lunar-regolith-derived product results.

A 19 October 2026 Nature Reviews Materials review reports perovskite PV manufacturing cost around US$0.3–0.6/W against approximately US$0.1/W for mature crystalline silicon in its analyzed context. It identifies yield, process/material reliability, material cost, fabrication simplicity and recycling as key competitiveness issues.

A 2026 Nature Reviews Clean Technology perspective argues that perovskite commercialization has moved from small-area efficiency records to large-area scale-up, operational reliability, process robustness, manufacturing yield and factory economics.

**PESE benchmark translation:** compare autonomous experiment-selection and process-control policies by qualified area/time, batch yield, variation, rejects, consumables, energy, operator intervention, reproducibility, stability and lifetime delivered service. The target is not just the highest cell PCE; it is the best manufacturable, repairable product under a declared mission.

Sources:
- Nature 2026 autonomous closed-loop perovskite paper: https://www.nature.com/articles/s41586-026-10482-y
- Nature Reviews Materials 2026 industrial manufacturing review: https://www.nature.com/articles/s41578-026-00973-9
- Nature Reviews Clean Technology 2026 product-commercialization perspective: https://www.nature.com/articles/s44359-026-00173-2

## 6. Terrestrial PV durability baseline

The NLR PV Fleet Performance Data Initiative reports a median fleet performance-loss rate of 0.75%/year across 4,915 datapoints in its current published snapshot. This is a field fleet statistic, not a pure module-degradation rate. It includes plant-level factors and its specific data/analysis population.

PESE should use it as a terrestrial system-level diagnostic baseline, not as a transferable lunar, Mars or space degradation coefficient. Use separate channels for:
- module power degradation;
- irradiance/sensor and soiling corrections;
- inverter availability and conversion loss;
- wiring/mismatch/shading;
- downtime and curtailment;
- cleaning/repair recovery;
- replacement and end-of-life effects.

Source:
- NLR PV Fleet Performance Data Initiative: https://www.nlr.gov/pv/fleet-performance-data-initiative

## 7. Competitive benchmark ladder

| Benchmark | Baseline | Target that could differentiate Luminous | Evidence required |
|---|---|---|---|
| Solar-array mass | NASA empirical spacecraft arrays: ~30 W/kg cluster, observed upper ~200 W/kg | Better delivered W/kg at complete deployed and maintained boundary on the same mission | Exact BOM, configuration and measured mass/power over mission profile |
| Space PV cell | Current III-V 3–5J nominal ~30%, high ~34% | Do not challenge at first; use as the high-performance cell reference | Independent cell data and matched spectrum/temperature/radiation conditions |
| Lunar polar generation | NASA VSAT 10-kW design / 50-kW >75 W/kg goal | Lower installed mass or higher service continuity per kg at the same site/load | Deployed/retracted prototype and shadow/thermal/dust/relocation test campaign |
| Perovskite process | Nature 2026 27.18% small cell / 23.49% mini-module | Comparable efficiency only if it also materially improves yield, throughput, cost and durability | Batch/area scaling, repeat-lot yield, long-duration and environmental tests |
| Terrestrial lifetime | NLR fleet median PLR 0.75%/yr | Improved diagnostics/recovery or net lifetime service against same fleet boundary | Field data, calibrated irradiance, exact system events and held-out sites |
| Regolith-to-PV | NASA Blue Alchemist / SoMo | More defensible process closure, less imported equipment/consumables and better end-to-end lifetime service | Source-pinned process flows, mass/energy balance, product quality and independent repeat batches |
| Thermal / storage | NASA Fission Surface Power, VSAT/RFC, lunar thermal models | Robust hybrid architecture satisfying electric + graded heat + cooling services at lower lifecycle burden | Matched service simulation then instrumented representative rig and external review |

The target column is a research programme, not present performance. Do not claim “SOTA surpassed” from a synthetic evaluator or a review document.

## 8. Highest-value paradigm shifts

1. **Terrain-aware vertical power infrastructure:** use Sol-Atlas/LOLA terrain/horizon evidence to co-design site placement, mast height, array azimuth, mutual shadowing, cables, mobility, retraction and repair. Optimize the complete site network, not each array in isolation.
2. **Joint electric–thermal–manufacturing architecture:** process heat, storage, radiation cooling, radiator sight-lines, waste heat and factory schedules share one mission envelope; preserve the different service grades.
3. **Factory-as-critical-load / factory-as-infrastructure:** include the manufacturing plant's energy and heat demand, spares, repair equipment and own replacement capacity. A factory that cannot maintain its own productive machinery is not industrial closure.
4. **Lifetime watts per imported kilogram:** optimize lifetime delivered electricity and process heat relative to imported mass and total installed mass; report multiple denominators and all replacement logistics.
5. **Autonomous process discovery with human/facility safety boundaries:** combine material/recipe discovery, high-fidelity process evidence, design-of-experiments and decision-aware selection. Proposals do not authorize hazardous equipment.
6. **Robust service before average efficiency:** include long darkness, storms, missing sources, degradation, delayed maintenance, failed deployment, storage derating and common-cause outages. Measure worst-case unserved critical energy and recovery.
7. **Repairability and reversibility:** modules, blankets, masts, radiators, power converters and process equipment must be replaceable, inspectable and recoverable with locally available tools/materials wherever demonstrated.

## 9. Next benchmark work

1. Extract source-pinned component and mission data for NASA VSAT (10-kW baseline, 50-kW target), the actual manufacturer cell products in NASA SOA, Blue Alchemist, SoMo and 2026 perovskite process papers.
2. Separate confirmed product values, programme goals, model outputs, author interpretations and unknown values.
3. Freeze three matched mission sets: terrestrial site, lunar South Pole (ridge plus shadowed location) and a 2.7-AU main-belt screening case.
4. Compare solar-only, solar + battery, solar + regenerative fuel cell, solar + thermal store, solar + nuclear and firm nuclear architectures against identical critical loads.
5. Compare PV manufacturing pathways on a matched product specification and process boundary, including a rational imported-cell baseline.
6. Build the synthetic evaluator and source register; then compare Symthaea candidate/experiment selection with random, space-filling, uncertainty-only, simple engineering and external-tool baselines.
7. Establish one low-risk terrestrial bench and one regolith-simulant process coupon test only after facility/safety and independent material/product qualification.

## 10. Machine-readable competitor ledger

The source-defined numeric values and evidence boundaries summarized above are also recorded in:

[`benchmarks/planetary-energy-service/v0.1/competitive-reference-snapshot.json`](../../benchmarks/planetary-energy-service/v0.1/competitive-reference-snapshot.json)

Each record binds a metric identifier, value, unit, exact declared system boundary, evidence class, source reference and caveats. The ledger intentionally keeps:
- cell beginning-of-life efficiency separate from module/array power-to-mass;
- flown spacecraft array empirical distribution separate from lunar surface deployment targets;
- NASA design references separate from solicitation goals;
- Blue Alchemist process maturity separate from lunar operational use;
- perovskite research-cell/mini-module measurements separate from factory yield and space qualification;
- terrestrial PV plant performance-loss data separate from module degradation in lunar/space conditions.

The ledger is a source-qualified starting register, not an automatically validated technology comparison. A future metric promotion requires the exact source revision and context, compatible system boundary, evidence rights and independently reproducible evaluation.

## 11. Claim ceiling

This document reports current source-defined SOTA, programme designs and research targets as of 12 November 2026. It does not establish that any Luminous design beats these values or is physically feasible. No simulator result, source audit, synthetic case or AI-generated architecture is physical validation or approval. Claims require exact configuration identity, comparable functional boundaries, independent measurement, held-out conditions and reproducible artifacts.
