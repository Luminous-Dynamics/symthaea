# Atelier Lightshell — robotic textile second-skin architecture v0.1

**Status:** research architecture / concept definition  
**Owner:** Symthaea ROBOT-LIGHTSHELL-001 #12688  
**Parent:** Symtropy ROBOT-PROTECTIVE-SHELL-001 #5868  
**Evidence state:** no CAD, material selection, fabricated article, physical test, or qualification claimed

## 1. Design thesis

Atelier Lightshell is a **kinematic garment for a robot**: a lightweight, removable, replaceable, textile-first outer system co-designed with the robot's geometry, joint motion, thermal system, sensing, cable routing, service access and operating domain.

It is not merely a costume or cosmetic skin. It is also not a substitute for load-bearing structure, rated guarding, or a safety-rated protective system. The design combines:
- a textile face selected for a declared environmental and wear profile;
- optional 3D spacer textile for compliant cushioning and airflow;
- local compliant polymer/lattice or rigid guards where the hazard requires them;
- articulated gaiters, gussets and service panels;
- a robot-specific attachment/interface system;
- optional sensor-ready provisions with independent safety authority.

The objective is not maximum coverage or minimum fabric cost. It is **qualified function per added mass and lifecycle burden**, with no unacceptable loss of mobility, cooling, sensing, maintainability or safety.

## 2. Why Atelier should own this

Existing Atelier work already handles apparel product requirements, fabric/yarn inverse design, pattern-to-3D interoperability, designer-in-the-loop co-creation, cloth manipulation/sewing and lifecycle evidence. Lightshell applies that engineering substrate to a different product domain: articulated robotic equipment.

Reuse:
- Symthaea #11731 — product requirements, pattern, BOM, manufacture and lifecycle thread;
- Symthaea #11746 — inverse textile design and physical experiment selection;
- Symthaea #11750 — preservation of pattern, 3D geometry and manufacturing-program semantics;
- Symthaea #11753 — human design agency and editable design alternatives;
- Symtropy #4865 — robotic cloth handling and seam execution;
- Symtropy #456 — specialized deformable/cloth model profile;
- Symthaea #5984 — textile material and product-function qualification;
- Sol-Atlas #738 — read-only spatial projection, coordinate frames and uncertainty;
- Mycelix #6092 and #5985 — textile and robotic article identity, material lots, repair and requalification.

Do not create a second CAD system, cloth solver, material registry, process scheduler, safety controller or lifecycle ledger.

## 3. Industry and research comparison

| Existing approach | What it demonstrates | Limitation / fair comparison |
|---|---|---|
| Conventional rigid robot panels | Stable geometry, hard local protection, established service interfaces | Can be heavier and less conformal; actual protection and serviceability must be measured for the chosen robot |
| Industrial robot jackets (e.g. ASP Protection) | Specialist covers for articulated/linear/Delta robots; ASP states solutions cover 1,000+ robot models and 90+ brands, with repairable, process-specific constructions | Strong benchmark for environmental protection and service; product claims still need matched tests; does not automatically provide cushioning or human-contact qualification |
| Hygienic / food-process covers | ASP describes double-coated fabric, welded seams and inflated construction intended to avoid dirt-retaining folds in high-pressure-cleaning environments | Demonstrates that hygienic cover design differs from a draped garment; requires its own cleaning/chemical/pressure test profile |
| 1X NEO soft body / washable nylon suit | Commercial humanoid precedent for a soft, textile-covered exterior and washable soft goods | Manufacturer description, not an independent head-to-head protection or lifecycle test |
| Textile robot skin research | Textile cover with proximity-sensor arrays and passive cushioning has been demonstrated as a research direction | Research prototype does not establish product-level reliability, safety integrity or generality |
| Textile + lattice cover architecture | A 2026 patent describes a textile cover adjacent to a lattice energy-attenuation assembly | Patent disclosure is not evidence of a tested, deployed or superior product |
| 3D spacer textiles | Research supports tuning cushioning, air-trapping and deformation through structure/yarn parameters | 3D spacer fabrics need structure-aware tests; flat-textile methods alone can miss shear, compression and edge effects |

Relevant sources:
- 1X NEO product description: https://www.1x.tech/neo
- ASP Protection industrial robot covers: https://www.asp-protection.eu/
- ASP food-industry covers and welded/inflatable construction: https://www.asp-protection.eu/protective-covers/protective-cover-for-the-food-industry/
- ASP spraying-robot covers: https://www.asp-protection.eu/protective-covers/spraying-robot-cover/
- Shanghai municipal report, 15 September 2026: https://english.shanghai.gov.cn/en-BusinessEnvironmentJiading/20260915/276b40bfbb32491696ba26e33ec2b166.html
- IEEE, *Prototype of an intelligent textile based robot skin to expand potential applications for cobot's* (2023): https://ieeexplore.ieee.org/document/10379882/
- Dejene & Gudayu, critical review of 3D woven/knitted spacer fabrics (2024): https://doi.org/10.1177/15280837241253614
- Schwager et al., spacer-fabric testing methods (2022): https://doi.org/10.1007/s10443-021-09959-y
- Sirris, *Guide to hygienic design of robot-based automation systems*: https://www.sirris.be/system/files/documents/2024-01/GUIDE%20TO%20HYGIENIC%20DESIGN%20OF%20ROBOT-BASED%20AUTOMATION%20SYSTEMS.pdf

**Competitive thesis:** the differentiator should be robot-specific pattern generation + zone-optimized textiles + local lattice/guards + measurable kinematic/thermal/sensor non-regression + traceable repairs. Textile robot clothing itself is not novel.

## 4. Architecture

### 4.1 Functional stack

1. **Robot interface / carrier:** approved mounting points, local frame references, keep-out zones and keyed/captive fasteners.
2. **Outer textile face:** abrasion/weather/dust surface chosen by application; replaceable by zone.
3. **Optional 3D spacer:** breathable compressible layer; candidate for low-energy cushioning only after testing.
4. **Local reinforcement:** flexible TPU/TPE or other suitable polymer patches; rigid guards where textile cannot meet the hazard.
5. **Inner liner / barrier:** smooth, inspectable interface that prevents chafe, debris ingress or unwanted contact where required.
6. **Articulated sections:** gaiters, gussets, bellows or overlapping panels shaped to the exact joint envelope.
7. **Optional sensor layer:** removable pressure/proximity/temperature/strain elements; observations never directly authorize hazardous movement.

### 4.2 Material families to compare (not an approved BOM)

- High-tenacity woven or knitted polyamide and polyester for abrasion/wear faces.
- Stretch knits for articulation zones, provided creep and fatigue are bounded.
- 3D spacer knits or weaves for compliant thickness, airflow and compression response.
- Coated or laminated textiles for a specific liquid/dust/chemical requirement; coating durability and repairability must be measured.
- TPU/TPE elastomer patches or printed lattices for local conformal reinforcement.
- Aramid-containing fabrics only where a documented heat/cut/abrasion hazard justifies their trade-offs.
- Recycled feedstock as a sourcing candidate; the exact lot and finished article still require material identity and functional testing.

Do not choose a material from a marketing label such as "heavy duty", "waterproof", "cut resistant" or "recycled". Bind fibre/yarn/construction/finish, supplier lot, specimen conditioning and method.

### 4.3 Zone-driven construction

| Robot zone | Candidate construction | Design constraints |
|---|---|---|
| Head / sensor pod | Minimal textile shroud or replaceable optical surround | No camera/FOV occlusion, microphone blockage, thermal blockage or calibration shift |
| Torso / back | Woven face + spacer/liner + access panels | Battery, compute, cooling, charging and service access |
| Shoulder / upper arm | Stretch panels + local abrasion patches | Motion envelope, cable routing, snag avoidance |
| Elbow / knee | Segmented gaiter or articulated spacer construction | No bunching, pinch, shear, migration or ROM loss |
| Forearm / shin | Replaceable textile skin + local guard | Abrasion, tool contact, impact and inspection |
| Hand / gripper | Small isolated guards only | Preserve tactile/grasp sensing, finger motion and tool interfaces |
| Hip / ankle | Pleated or gusseted panels | Repeated flex, clearance and cable/hose routing |
| Battery / power bay | Profile-specific removable cover | Heat rejection, emergency isolation, connectors and access |

The first physical design target is the forearm/elbow/joint article around Symtropy #4337 and Symthaea #4814. The exact geometry, ODD, load case, temperature map, cable routing and mass budget remain open; no dimensions or material grade are frozen.

## 5. Manufacturing hypotheses

Compare fabric-only construction with hybrid options:
- cut/sew woven panels with engineered seam topology;
- 3D knit or spacer-knit panels for articulated or compliant zones;
- thermoformed or printed TPU/TPE local reinforcements;
- bonded/laminated assemblies where the bonding process can be inspected and repaired;
- replaceable mechanically retained guard patches;
- machine knitting for repeatable shaped panels once machine capability and Knitout/backend preflight are established.

The garment should be designed around manufacturing capability and repair, not optimized only as a rendered surface. Keep source-native patterns and machine programs authoritative; use explicit sidecars for semantic links and transformation loss.

## 6. Test programme

### P0 — application and interface freeze
Record platform/article revision, task, ODD, hazard by zone, coverage, local frames, rated attachment points, ROM, keep-outs, cable bend limits, mass/inertia budget, cooling, sensor apertures, service operations, cleaning profile and inspection criteria.

### P1 — coupon characterization
For each candidate sample:
- fibre/yarn/fabric construction, finish, orientation, lot and conditioning;
- areal density, thickness and dimensional stability;
- tensile, tear, seam and attachment pull-out;
- abrasion/pilling and surface damage;
- spacer compression curve, recovery, cyclic compression set and shear/tilting;
- air permeability and water behaviour;
- thermal/moisture behaviour as relevant;
- laundering/cleaning change, shrinkage, delamination and finish loss;
- ESD, flammability or chemical exposure only when explicitly required.

Potential method references include ISO 12947-2:2016 for textile specimen breakdown by abrasion, ISO 13935-1:2014 for straight-seam tensile strength, ISO 13937-2:2000 for tear force, and ISO 6330:2021 for domestic washing/drying procedures. They characterize textile specimens or articles; they do not qualify a robot shell or certify protective performance. Spacer fabrics require structure-aware compression, shear/tilting and air-flow protocols.

### P2 — isolated module tests
Compare a bare joint/module to matched candidates:
- exposed baseline;
- existing rigid cover or application-appropriate commercial cover;
- textile-only Lightshell;
- textile + spacer;
- textile + spacer + local lattice/guard.

Measure:
- added mass, local COM/inertia, coverage;
- joint ROM, clearance, fabric migration and contact with moving components;
- torque/current/energy delta on identical trajectories;
- motor/drive temperature and cooling margin;
- cable/hose flex, abrasion and bend-radius compliance;
- seam/fastener retention, snag and entanglement;
- sensor occlusion and perception error;
- instrumented low-energy impact/compression transmission with exact fixture, velocity, impactor geometry, support, location and calibration;
- installation, removal, inspection, cleaning and repair time.

Do not report impact energy alone as protection. Report the measured force/acceleration/strain response, deformation, uncertainty and failure mode.

### P3 — integrated task and lifecycle
After module risk review and safe-motion approval, run matched robot tasks and motion cycles. Check task success, energy, temperature, sensing, safety, seam creep, wear, lattice damage, delamination, fastener retention, cleaning and repeated refit. Bind all results to the exact as-built revision and material lot.

## 7. Benchmark strategy

Freeze the task and baseline before optimization. Keep each metric separate; do not hide a thermal or safety regression inside a weighted score.

| Dimension | Metric |
|---|---|
| Lightweighting | Added mass; mass per covered area; mass as % of robot mass |
| Kinematics | Joint ROM change; minimum clearance; fabric tension/migration at extreme poses |
| Dynamics | Added reflected inertia; torque/current/energy delta on matched trajectories |
| Thermal | Temperature rise and cooling-margin change at same duty cycle |
| Mechanical | Transmitted force/acceleration/strain; residual deformation; compression recovery |
| Durability | Abrasion cycles to defined endpoint; repeated-flex failure; seam and attachment fatigue |
| Environmental | Exact-profile ingress/chemical/ESD/heat result, where required |
| Perception | Camera/FOV occlusion, optical contamination, sensor error and calibration shift |
| Serviceability | Installation, removal, inspection, cleaning and repair time; replaced area vs whole cover |
| Lifecycle | Cost per accepted service-hour; repair yield; material waste; recovery eligibility and output evidence |

Ambitious research hypotheses (not current results):
- lower added mass than a rigid cover at equal tested coverage and the same hazard profile;
- no loss of required ROM or task success;
- no thermal, sensing or safety regression;
- panel-level repairs faster than whole-cover replacement;
- better service life per unit mass under the same flex/abrasion/cleaning protocol;
- every released panel linked to exact pattern revision, material lot, manufacturing route and inspection evidence.

Only set numeric improvement percentages after a matched baseline and measurement system are frozen. That prevents an arbitrary target from becoming a false SOTA claim.

## 8. Safety, hygiene and claim boundaries

A textile cover can hide pinch/shear interfaces, alter collision geometry, add inertia, restrict cooling, snag on fixtures and change perception. Repeat the robot/system risk assessment with the cover installed. The existing safety controller and emergency stop remain authoritative.

A cover is not an ingress seal by default. Do not claim IP, IK, flame, ESD, chemical, cleanroom, medical, PPE or human-contact compliance without an exact profile and the appropriate independent test. Do not infer that a soft layer makes a robot safe to touch.

For food/pharmaceutical/cleanroom/product-contact applications, textile folds, seams and overlaps can create contamination-retaining areas. The Sirris hygienic-design guide warns that conventional flexible covers can create residue-retaining folds and reinstallation risks. Conversely, ASP describes a specialized food-industry cover using double-coated fabric, welded seams and an inflated construction intended to prevent folds where residue can settle. This is an architectural contrast, not independent proof of hygienic conformity.

Define two different product archetypes rather than one universal shell:
1. **Conformal soft garment:** washable textile, spacer and local compliant guards for domestic, service, research and selected industrial ODDs.
2. **Sealed process cover:** smooth, welded/coated, cleanable and possibly pressurized enclosure for a specific dust/liquid/chemical/food-process profile.

The same robot may need different coverings for different deployments. A washable nylon suit is not automatically a high-pressure-cleanable food cover, and a sealed industrial jacket is not automatically comfortable, lightweight or suitable for human-contact use. Consider smooth sealed surfaces or no cover in product-contact zones unless the complete installation is qualified.

## 9. Evidence and claim ladder

Keep the following states distinct:
- design hypothesis;
- CAD / pattern draft;
- synthetic or simulated result;
- material supplier declaration;
- measured coupon;
- fabricated article and as-built inspection;
- calibrated module test;
- integrated robot test;
- independent qualification for an exact scope.

Each result must bind article/pattern revision, material lot, construction, process, fixture, instrument calibration, conditioning, method, units, uncertainty, raw-data digest, acceptance rule and reviewer. A later pattern, fabric lot or process change invalidates only the dependent evidence, but never silently preserves it.

## 10. First executable work package

1. Export an exact neutral CAD representation of the #4337 joint/forearm test article, including frames, moving envelopes, approved attachment interfaces, cable routes and thermal/sensor keep-outs.
2. Create a simplified fit fixture (not a production CAD master) with named zone and motion-envelope metadata.
3. Generate three equal-coverage candidate patterns: textile-only, textile + spacer, textile + spacer + local guard.
4. Freeze a low-risk intended use case and hazard list.
5. Select a small candidate fabric matrix and obtain actual lot/specification samples.
6. Define coupon protocols and a calibrated module fixture before selecting a final material.
7. Build a removable, inspectable prototype only after fit, clearance and safety review.

No physical prototype or material has been selected or tested as part of this v0.1 document.

## 11. References

- 1X NEO: https://www.1x.tech/neo
- Shanghai municipal report, 15 September 2026: https://english.shanghai.gov.cn/en-BusinessEnvironmentJiading/20260915/276b40bfbb32491696ba26e33ec2b166.html
- IEEE textile robot skin research (2023): https://ieeexplore.ieee.org/document/10379882/
- 3D woven/knitted spacer textile critical review (2024): https://doi.org/10.1177/15280837241253614
- Spacer textile test-method research (2022): https://doi.org/10.1007/s10443-021-09959-y
- ISO 12947-2:2016: https://www.iso.org/standard/61058.html
- ISO 13935-1:2014: https://www.iso.org/standard/60678.html
- ISO 13937-2:2000: https://www.iso.org/standard/23370.html
- ISO 6330:2021: https://www.iso.org/standard/75934.html
- Sirris hygienic design of robot-based automation: https://www.sirris.be/system/files/documents/2024-01/GUIDE%20TO%20HYGIENIC%20DESIGN%20OF%20ROBOT-BASED%20AUTOMATION%20SYSTEMS.pdf


## 12. Industry refinement — two cover archetypes

The current market is more mature than a simple "fabric versus plastic" comparison suggests.

- 1X NEO publicly describes a custom 3D lattice polymer body, externally covered joints, a machine-washable nylon suit/shoes, body IP44 and hands IP68. Treat these as manufacturer-stated whole-robot architecture/product claims, not independently measured textile performance.
- ASP Protection states it has made industrial robot covers since 1988 and offers solutions for more than 1,000 robot models across more than 90 brands. Its product families include dust, spray, heat, cold and food-process covers. For food production, it describes double-coated fabric, welded seams, repairable construction and an inflated cover intended to prevent folds where residue can accumulate.
- A September 2026 Shanghai municipal report describes a garment producer shipping more than 3,000 protective-suit sets, primarily for industrial robot applications, and developing textile surfaces with temperature/pressure sensors.

These sources show three distinct markets: soft humanoid skins, industrial process jackets and robot-facing smart textiles. Lightshell should bridge capabilities without collapsing them into one product. The differentiator is to co-design the covering from robot kinematics and ODD, then bind pattern, lot, seam/process, fit, thermal/sensor impact, service and test evidence into one engineering thread.

## 13. What "better than SOTA" should mean

Do not claim superiority to 1X or ASP from product photos, a single prototype or supplier datasheets. Compare on the same robot, task, hazard envelope, coverage and test fixtures.

The first competitive experiment should include:
- bare platform;
- its existing production panel/cover;
- a commercial cover selected for the same ODD (ASP or equivalent, when platform and use case match);
- Lightshell textile-only candidate;
- Lightshell textile + spacer + local guard candidate.

The result should answer whether Lightshell improves the Pareto frontier across added mass, abrasion/impact response, mobility, thermal margin, sensing, cleanability, repair time and cost per accepted service-hour. It may win on repairability or comfort while losing on high-pressure cleaning; report that trade-off honestly. The best product family may therefore be two distinct architectures: a conformal soft garment and a sealed process cover.


## Research extensions — 17 November 2026

Two follow-on tracks now make the frontier explicit:
- [LIGHTSHELL-SENSOR-FABRIC-001 #12704](https://github.com/Luminous-Dynamics/symthaea/issues/12704) evaluates modular tactile/strain/temperature textile sensing against current fibre-based and self-powered textile-skin research. Sensor readings remain observational and cannot authorize robot motion.
- [ROBOT-GARMENT-COMPILER-001 #12710](https://github.com/Luminous-Dynamics/symthaea/issues/12710) explores generating robot-specific textile panels and seam topology from exact CAD, kinematics, swept envelopes, thermal/sensor/service keep-outs and process capabilities.

RGBench is a cloth simulation/manipulation comparator, not a protective-shell test. The January 2026 auxetic-lattice research is a candidate local reinforcement direction, not a qualified robot-cover material. These distinctions are recorded in the comparative test plan.
