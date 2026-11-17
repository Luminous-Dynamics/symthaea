# Robot Lightshell comparative test protocol v0.1

**Owner:** Symthaea ROBOT-LIGHTSHELL-001 #12688  
**Companion architecture:** `ATELIER_ROBOT_LIGHTSHELL_ARCHITECTURE_V0_1.md`  
**Status:** preregistration template / no physical test run  
**Claim ceiling:** test design only; no protection, safety, ingress, fire or qualification result

## 1. Objective

Determine whether Atelier Lightshell improves the protection/serviceability Pareto frontier of a specified robot zone without unacceptable penalty to mass, motion, energy, thermal behavior, perception, cleaning or safety.

This is not a universal textile leaderboard. A result applies only to the declared robot article, zone, material lot, construction, ODD, fixture, conditioning and method.

## 2. Test articles and matched baselines

Use the same robot geometry, covered area, attachment interfaces, task, ODD and test fixtures.

| ID | Test article | Purpose |
|---|---|---|
| B0 | Bare/uncovered robot zone | Reference for motion, energy, thermal state, sensing and exposed-surface behavior |
| B1 | Existing rigid panel / production cover | Rigid-cover baseline for mass, clearance, impact, service and maintenance |
| B2 | Commercial textile cover/jacket selected for the exact ODD | Industry cover baseline; use actual product/lot and supplier care instructions |
| L1 | Lightshell textile-only | Isolate benefits/penalties from the textile garment |
| L2 | Lightshell textile + 3D spacer | Isolate the spacer layer's incremental cushioning, airflow, mass and thermal effects |
| L3 | Lightshell textile + spacer + local lattice/guard | Evaluate the full hybrid concept |

Not every platform will have a meaningful B1 or B2. If a baseline is unavailable or incompatible, mark it **NOT_COMPARABLE**; do not substitute a different robot/task and call it matched.

## 3. Candidate construction matrix

Start with interpretable material families rather than a large black-box search.

| Candidate family | Intended role | Key uncertainty / failure modes |
|---|---|---|
| High-tenacity woven polyester or polyamide | Replaceable abrasion/environmental face | Tear, seam fatigue, snagging, moisture uptake and finish durability |
| Stretch knit | Shoulder, elbow, hip, knee and ankle articulation | Creep, bagging, local strain concentration and abrasion |
| 3D spacer knit / weave | Compliant thickness, cushioning and airflow | Compression set, shear/tilting, edge collapse and variable air paths |
| Coated/laminated woven textile | Splash, dust or specific chemical barrier | Coating cracks, delamination, stiffness, breathability and repair |
| TPU/TPE local reinforcement | Flexible guard, abrasion pad or interface patch | Flex fatigue, adhesion, temperature dependence and thickness/mass |
| Textile + 3D-printed auxetic/lattice composite | Tunable local deformation and energy management | Print-to-textile bond, anisotropy, fatigue, local pressure peaks and repeatability |
| Smooth welded/inflatable industrial cover | Sealed process-cover reference where appropriate | Cleanability, puncture/pressure failure, fit complexity and thermal behavior |

A 2025 study on 3D-printed auxetic structures combined with spacer fabric is a relevant research lead, not proof of robot-shell performance. The first campaign should vary only a few controlled factors (fabric construction, spacer thickness/structure, local guard geometry) so the effects remain interpretable.

## 4. Freeze before testing

A test is not executable until its protocol record freezes:
- article/platform ID, hardware and software revision;
- zone, coverage boundary and coordinate frames;
- ODD and hazard profile;
- attachment/retention interface and allowed structural load path;
- baseline article identity and equivalence rationale;
- candidate fabric, yarn/fibre, construction, finish/coating and supplier lot;
- specimen orientation and conditioning;
- motion trajectory, speed, load, duty cycle and repeat count;
- thermal ambient, sensor placement and component limits;
- camera/sensor configuration and calibration;
- impactor geometry, mass, velocity, support and impact location if impact tests are applicable;
- instruments, calibration validity, sampling, synchronization and uncertainty;
- pass/fail/abort criteria and responsible reviewer.

Unknown inputs remain `UNKNOWN`; they are not filled with assumed defaults.

## 5. Stage A — textile coupon characterization

Use suitable current standards where their stated scope matches the construction. Record the exact edition, method, conditioning, apparatus, deviations and uncertainty.

| Property | Candidate method/reference | Lightshell-specific addendum |
|---|---|---|
| Fabric abrasion | ISO 12947-2:2016 for applicable textile specimen breakdown; coated surfaces may need a different method | Test the actual face, seam, edge binding and local reinforcement as separate failure locations |
| Seam strength | ISO 13935-1:2014 for straight sewn seams in its scope | Add curved seam, gusset, articulated seam and attachment-point tests; ISO strip method is not a substitute for these |
| Tear | ISO 13937-2:2000 where applicable | Test warp/weft or principal directions and the finished assembly |
| Wash/cleaning | ISO 6330:2021 where domestic washing is the intended care path | Record dimensions, mass, seam, coating, colour, fit and sensor/electrical change before/after cycles |
| Spacer compression | Method selected/validated for the exact 3D structure | Record loading/unloading curve, hysteresis, recovery, compression set, rate and conditioning |
| Spacer shear/tilting | Structure-specific test rig | Record face-layer offset, tilt stability and edge collapse; do not infer from 2D textile data |
| Air permeability | Method suitable for 3D spacer geometry | Seal/guard edges consistently; side leakage can distort flat-textile measurements |
| Water/chemical | Exact liquid, exposure and cleaning profile | Check absorption, wicking, drying, delamination and trapped residue |
| ESD / electrical | Application-specific electrical method | Include dry, wet/cleaned and aged states if ESD is required |
| Heat/flame | Exact hazard-specific method | No generic flame or heat claim from fibre family or supplier label |

For 3D spacer fabrics, the literature identifies limitations in applying 2D textile tests directly to compression, air permeability, shear/tilting and abrasion. The method must be adapted and documented rather than assuming the nominal standard captures the structure's behavior.

## 6. Stage B — garment fit and kinematic screening

Use an exact CAD-derived robot zone or a clearly labeled simplified fit fixture.

Capture:
- pattern-to-surface deviation and panel edge registration;
- cloth strain/tension proxy across named joint poses;
- panel migration, folds, bunching and contact with moving components;
- minimum clearance to links, cables, hoses, connectors and environmental surfaces;
- required and achieved joint range of motion;
- attachment engagement, retention and safe removal;
- sensor/camera aperture visibility and thermal vent area.

Run static pose coverage first, then slow controlled motion. Do not proceed to higher speed or loaded motion while any garment can enter a pinch/shear zone, snag, detach, cover a sensor or restrict cooling.

A geometric/cloth simulation may prioritize patterns, but only a fitted physical article can reveal actual friction, migration, seam distortion and assembly tolerance.

## 7. Stage C — instrumented module comparison

Compare B0–B2 and L1–L3 on the same module, using the same motion and load profiles.

Measure:
1. **Mass and geometry:** added mass, mass per covered area, centre-of-mass shift, local inertia and coverage of each declared zone.
2. **Kinematics:** ROM, clearance, fabric migration, contact events and attachment movement.
3. **Actuation:** synchronized voltage/current, joint position/velocity and torque/load where instrumented; compare energy per identical trajectory.
4. **Thermal:** motor/drive/gearbox temperatures and cooling margin under matched duty cycles.
5. **Perception:** camera field-of-view, contrast/reflection, sensor occlusion, calibration shift and task-level perception error.
6. **Retention and snag:** fastener migration, pull-out, seam rupture, snag locations and failure mode.
7. **Impact/compression:** only a pre-approved low-energy fixture and profile; record impactor mass, velocity, shape, support, location, force/acceleration/strain, displacement, recovery and uncertainty.
8. **Service:** time and tools to install, remove, inspect, clean, repair and refit; panel-only repair versus whole-cover replacement.

Impact energy alone is insufficient. Report response and failure mode at the specified interface. Never call a soft layer "safe to touch" from an impact coupon or HIC-like number alone.

## 8. Stage D — flex, wear, cleaning and lifecycle

Set motion cycles from the target service duty and exact robot motion envelope; do not choose a cycle count without that use case.

After each inspection interval, record:
- seam elongation, stitch rupture, yarn breaks and edge fray;
- spacer compression set, face-layer shear and permanent deformation;
- local lattice cracks, delamination or attachment failure;
- fabric migration and any cable/hose wear;
- mass/dimension change, moisture retention and residual contamination;
- sensor continuity, calibration drift and failure detection where present;
- repair/replacement parts and post-service fit/clearance.

For a washable home/service garment, use the declared washing/drying profile. For industrial chemical/high-pressure cleaning, use the exact process chemistry, pressure, temperature, distance and exposure time; domestic wash results do not qualify industrial cleaning.

A repair is accepted only after the repaired article passes the affected inspection and regression tests. Replacing a panel creates a new as-maintained article generation and material-lot lineage.

## 9. Hard gates before optimization

The candidate is **INELIGIBLE** for Pareto ranking if any of these occur:
- garment/attachment enters the moving or pinch/shear envelope;
- required ROM or cable/hose bend constraints are violated;
- cooling, emergency isolation, charging or service access is obstructed;
- critical sensor FOV or calibration is degraded beyond its approved bound;
- detachment, snag or loose-part failure creates an uncontrolled hazard;
- temperature/current/torque or task performance exceeds approved robot limits;
- evidence is missing, stale, uncalibrated or bound to a different article/lot.

Set the numeric margins with the robot owner and safety reviewer. No universal percentage is substituted for platform limits.

## 10. Pareto analysis and SOTA comparison

Do not compress all outcomes to one scalar. First apply safety/thermal/ROM/perception hard gates, then compare the remaining candidates across:
- added mass;
- measured abrasion life;
- impact/compression transmission under the same fixture;
- mobility / actuation-energy delta;
- thermal margin;
- perception change;
- installation and repair time;
- cost per accepted service-hour;
- replacement waste and qualified recovery.

The SOTA hypothesis is a **Pareto improvement**: equal or better declared protection and service function at lower mass/lifecycle burden, without unacceptable motion, energy, thermal, perception or safety regressions.

Report all candidates, failures, inconclusive runs and confidence intervals. A candidate may win for a domestic soft-skin profile but lose for a high-pressure food-process profile. This is a valid result and supports two product families rather than a false universal winner.

## 11. Evidence package

Each coupon and module result must include:
- source design and article revisions;
- exact sample/panel/material/lot identity;
- process and assembly record;
- ODD, conditioning, method and deviations;
- instrument identity, calibration and time synchronization;
- raw data and digest;
- derived calculations and code/version;
- uncertainty and acceptance rule;
- failure mode and reviewer disposition.

Store large raw measurement data in the designated evidence store; keep a small rights-cleared manifest and digest in source control. Do not commit proprietary commercial sample designs or unlicensed standards text.

## 12. Unresolved inputs blocking a fabrication-ready test

- exact #4337 CAD and approved attachment interfaces;
- which joint/forearm geometry is the selected first article;
- robot mass, motion envelope and approved duty cycle;
- cable/hose routing and minimum bend radii;
- motor/drive thermal constraints and cooling paths;
- target ODD, hazards, cleaning and human-proximity context;
- mass allowance and acceptable torque/energy delta;
- selected candidate material lots and local prototype process;
- calibrated torque/angle/current/temperature measurement chain and lab safety review.

Until those inputs are closed, the next work is CAD/ODD capture, candidate swatches and coupon-protocol preparation—not production drawings or a fabrication order.

## 13. References

- ISO 12947-2:2016: https://www.iso.org/standard/61058.html
- ISO 13935-1:2014: https://www.iso.org/standard/60678.html
- ISO 13937-2:2000: https://www.iso.org/standard/23370.html
- ISO 6330:2021: https://www.iso.org/standard/75934.html
- Spacer-fabric testing methods (2022): https://doi.org/10.1007/s10443-021-09959-y
- 3D printed auxetic structure–spacer fabric composite (2025): https://doi.org/10.1108/IJCST-09-2024-0193
- 3D woven/knitted spacer textile critical review (2024): https://doi.org/10.1177/15280837241253614
- 1X NEO official product description: https://www.1x.tech/neo
- ASP Protection robot cover range: https://www.asp-protection.eu/


## 14. Candidate-scoped gates and evidence admission

The companion contract is implemented in `symthaea-engineering::lightshell_qualification` and documented in `ROBOT_LIGHTSHELL_QUALIFICATION_CONTRACT_V0_1.md`.

Every hard gate is now recorded per candidate (six candidates × nine gates), and each metric observation is candidate-scoped (six candidates × seven metrics in the seed fixture). One candidate's pass cannot discharge another candidate's gate. A failed or inconclusive run remains visible and blocks that candidate from ranking.

Ranking eligibility requires an exact matched B0 baseline, a common coverage boundary, all candidate-scoped gates resolved acceptably, at least one evidence-complete passed module/integrated test, no adverse run, and observed core metrics with method and uncertainty references. Core metrics must use the same units, measurement boundary and comparison direction as B0; the physical comparison run must use the same test stage, protocol revision, instrument references, calibration references and acceptance-rule revision as B0. The validator returns eligible candidate IDs only; it does not calculate Pareto dominance or claim superiority.

The seed fixture remains `inputs_pending`: all gate states are unknown, all observed/target/baseline metric values are null, and no test runs exist. This remains an explicit non-result.


## 15. Simulation, smart-textile and metamaterial frontier (17 November 2026)

### RGBench — use as a cloth-model comparator, not a protective-shell rating

The AAAI-2026 RGBench project reports a large garment-mesh corpus and its GarmentDynamics simulator, with paper-reported average simulation error reduction of about 20% and around 3× speed versus the compared simulator baselines. Its publicly released real-capture evaluation package is distinct from its full research corpus: the published GitHub release describes nine garments, 98 capture sessions and three bimanual actions (fling/fold/grasp), with a separate 2,794-row garment-asset research subset carrying a non-commercial research license.

Use existing Symtropy [RGBENCH-REPRODUCE-001 #4870](https://github.com/Luminous-Dynamics/symtropy/issues/4870) for rights-aware reproduction. Lightshell should consume an admitted cloth profile and test simulator residuals against RGBench where relevant, but must also calibrate the actual Lightshell fabric, seams, attachment tension and robot-mounted deformation. RGBench measures cloth simulation/manipulation; it does not qualify impact protection, seam durability, robot kinematics, cooling or human-contact safety.

Sources:
- RGBench project page: https://rgbench.github.io/
- Public evaluation/code release: https://github.com/hwk0809/RGBench
- RGBench evaluation dataset: https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1

### Auxetic textile/lattice hybrids

A January 2026 study optimized a 2D hybrid auxetic lattice's strut slenderness and viscoelastic properties to reduce transmitted peak force while increasing energy mitigation in its modeled/validated context. This supports a local reinforcement research lane: compare conventional elastomer/lattice patches against auxetic geometries coupled to the same textile/spacer face. It does not establish performance in a robot shell. The test must account for rate, temperature, fatigue, attachment, local pressure peaks and actual print/material process.

Source: https://doi.org/10.1016/j.ijsolstr.2025.113659

### Sensor textiles are a distinct qualification domain

The 2026 fibre-piezoresistive e-skin and self-powered minimal-channel textile papers are now tracked under [LIGHTSHELL-SENSOR-FABRIC-001 #12704](https://github.com/Luminous-Dynamics/symthaea/issues/12704). Keep sensor-free mechanical protection as the control; compare sensor architectures separately. Wash/flex/humidity/calibration/fault performance is part of sensor qualification, not inferred from base textile strength.

### Kinematic pattern generation

The next architectural step is [ROBOT-GARMENT-COMPILER-001 #12710](https://github.com/Luminous-Dynamics/symthaea/issues/12710): transform source robot CAD, joint motion, thermal/sensor/service keep-outs and protection zones into candidate panel/seam/attachment/manufacturing packages. The compiler's geometric result remains a candidate. It must not replace robot CAD, garment authoring, cloth simulation or human engineering approval.

### Evidence scope

All literature results remain external paper/project claims until independently reproduced on the declared task. No RGBench data, sensor array, auxetic coupon or Lightshell physical sample is claimed downloaded, fabricated or tested by this PR.


## 16. Smart textile architecture frontier — 17 November 2026

### SoTa: common taxel layout and cross-embodiment sensing

The October 2026 preprint **SoTa: Soft Tactile Skins for Dexterous Manipulation** reports:
- a common layout of 202 taxels across human and robot hand embodiments;
- multilayer fabric-electrode construction, with reported material cost below US$10 per skin;
- more than 97% of initial response span after 10,000 loading/unloading cycles;
- conductive traces retained through 1,280 tight-fist folding cycles;
- improved contact-rich manipulation performance when human demonstrations augmented robot demonstrations, under the paper's eight evaluation conditions.

Reference: https://arxiv.org/abs/2610.02338

This is a preprint's hand-skin result, not a Lightshell performance result or forearm acceptance threshold. The architectural lesson is to define a **shared sensor-tile coordinate and observation contract** for replaceable modules, while retaining per-panel geometry, sensor-lot and calibration metadata. A common layout may improve replaceability, cross-robot data reuse and calibration tooling, but does not make sensors physically interchangeable without calibration.

### Deployment-oriented e-skin evaluation

The 1 November 2026 review *Tactile electronic skin: From mechanical perception to intelligent interfaces* argues that peak sensitivity alone is a poor deployment metric. It emphasizes working range, durability, environmental stability, signal integrity, conformal integration and task-level reliability, and discusses hysteresis, crosstalk, wet interfaces and mechanical decoupling.

Reference: https://doi.org/10.1016/j.sna.2026.118150

This reinforces the #12704 sensor-fabric test matrix: measure calibration/drift, wet-state response, flex/wash aging, mechanical decoupling, crosstalk, fault detection and task outcomes. Keep sensor-free mechanical Lightshell as a control.

### Scalable in-situ fabrication

A 29 January 2026 *npj Flexible Electronics* paper reports a scalable, cleanroom-free, in-situ fabrication approach for multimodal electronic skin, combining capacitive tactile structures with UV-laser-patterned flexible circuitry.

Reference: https://www.nature.com/articles/s41528-026-00538-4

This is a process comparator for manufacturable sensor panels, not evidence that the same method is suitable for industrial robot garments. Evaluate process capability, textile compatibility, repairability, connector life and calibration retention before adoption.

**Sensor claim boundary:** all literature values are paper-specific. The first Lightshell pilot should use robot-only sensor calibration and physical reference instruments. It should not ingest human tactile data, assume human/robot signal equivalence, or feed textile sensor output directly into a safety or motion-authority path.


### Sensor placement: task value before taxel count

Two additional 2026 studies should inform [LIGHTSHELL-SENSOR-FABRIC-001 #12704](https://github.com/Luminous-Dynamics/symthaea/issues/12704):

- **Tactile Genesis** (21 June 2026) provides large-scale tactile simulation with configurable taxel placement and noise. Its reported robot-hand studies find sensor placement can dominate sensor type; whole-hand coverage improves on fingertip-only for its tasks, and approximately 200 taxels across the hand sufficed under those evaluation conditions. It also identifies per-taxel force/torque as useful. This is not a full-body Lightshell sizing result. https://arxiv.org/abs/2606.22332
- **Which Taxels, Not How Many** (7 November 2026) reports that contact-informed taxel placement can match a much larger uniform array under its controlled in-palm rotation and grasping tasks; rarely contacted sites add little. https://arxiv.org/abs/2611.10647

Therefore compare **uniform low-density**, **task/contact-informed**, **joint/edge/service-zone targeted**, and **sensor-free** constructions. Measure observability at the task-relevant contact zones and downstream task performance rather than using taxel count or spatial resolution as the sole objective. No sensor distribution is fixed until the robot's tasks, expected contacts, mounting, calibration and failure modes are frozen.
