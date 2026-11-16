# SENSE-TAC-001A — First Tactile Co-Design Benchmark Constitution

Status: benchmark constitution / design freeze only  
Parent: Symthaea #5821  
Related: #5820, #5823, #5826, #5827, #5831, #4918, FIELD-000 #3589, FIELD-001A #3633, ROB-INSTR #4866

## 1. Objective

Freeze a small, falsifiable first tactile-sensor campaign before selecting dimensions, material formulations, electronics, fabrication process, or a candidate winner.

Primary question:

> Can an accessible, repairable, mechanically encoded multimodal tactile patch improve contact localization and incipient-slip evidence over credible low-complexity and commercial baselines, on held-out physical conditions, without hiding regressions in calibration, saturation, hysteresis, drift, latency, power, repairability, or calibration burden?

This tranche does not select a product design or authorize a physical build. It defines the comparison and evidence needed to make those decisions.

## 2. Research position and paradigm shift

The tactile system is the complete physical information path, not the transducer alone:

    contact / load / motion
            ↓
    mechanical morphology and contact interface
            ↓
    transducer (capacitive / resistive / optical / magnetic / other)
            ↓
    excitation + analog front end + ADC/readout
            ↓
    time, calibration, environmental and fault evidence
            ↓
    raw physical observation + uncertainty
            ├──→ immutable source/evidence lineage
            ├──→ conventional signal-processing baseline
            └──→ HDC / event / temporal representation
                        ↓
                 task inference / proposal
                        ↓
              independently measured task outcome

The design hypothesis is that **mechanical morphology can be an information operator**: compliant structures, cell geometry, electrode/marker layout and mechanical decoupling can transform a contact stimulus before computation. This must be tested with fixed-electronics/fixed-algorithm ablations; it is not assumed to be true for every geometry.

A second hypothesis is that a two-path representation may be useful:

- a fast, bounded path for transient/contact/slip cues;
- a richer path for slower semantic and context interpretation.

Both are descendants of the same physical observations. They are not independent sensors, and a fast path gains no motor or reflex authority.

## 3. Frozen scope

V1 is a non-human-contact tactile coupon/patch on a bench fixture or simple robotic test surface.

Out of scope: whole-body e-skin, human-contact safety, clinical palpation, production gripper control, custom MEMS/ASIC fabrication, autonomous actuator control, and universal tactile superiority claims.

### Primary tasks

**T1 — contact localization**

Given a declared contact-indenter family and bounded fixture region, estimate contact location on held-out positions and contact geometries. Primary outcome: spatial localization error and task disposition, with the reference position produced independently of the candidate sensor.

**T2 — incipient-slip discrimination**

Under a declared normal-load and surface/friction profile, distinguish stable contact from incipient slip before gross relative motion, where the fixture can define a defensible reference event. Report event precision/recall or proper event score, detection delay from the independent reference event, missed events, and false alarms per episode.

A task score is not a force/shear calibration result.

### Secondary physical characterization

Keep each axis separate: normal response and calibration residual; shear response and cross-axis sensitivity; dynamic response/bandwidth; bias/noise; resolution; nonlinearity; hysteresis; creep; drift; temperature dependence; saturation/recovery; dropout/stuck-at behavior; remount variance; curvature transfer; and between-article variation.

### System/resource characterization

Keep separate: sensor/readout power; host and transfer energy where measurable; data rate; compute and memory traffic; end-to-end latency distribution (including tail); wiring/channel count; fabrication and repair time; calibration time; reference-equipment burden; consumables; and lifecycle cost. Unknown resource values remain unknown.

## 4. Candidate and baseline ladder

No family is pre-ranked. Select the smallest useful subset after the reference-fixture and capability audit.

| ID | Family | Purpose |
|---|---|---|
| B0 | Commodity FSR / single-point load or pressure pad | Low-complexity baseline |
| C1 | Capacitive/compliant patch with COTS readout | Accessible candidate transducer |
| C2 | C1 with changed morphology/electrode/cell geometry, same readout | Isolate mechanical/electrode encoding |
| C3 | Optical deformation sensor (DIGIT, GelSight Mini, TacTip) | Image/deformation comparator; force outputs require their own validation |
| C4 | Replaceable magnetic skin (ReSkin, AnySkin) | Repairability, cross-instance transfer and magnetic-field sensing comparator |
| C5 | Commercial distributed/multiaxis tactile sensor (XELA uSkin, Contactile PapillArray) | Direct force/displacement system comparator, subject to exact model and calibration |
| C6 | Multimodal research sensor (SuperTac) | Literature/SOTA comparator; include physically only if exact article, interfaces and task are available |

C3/C4 are not required for the first bench run. Their inclusion depends on availability, interface access, exact model/revision, and fair task applicability. A baseline must not be deliberately weakened.

### Published commercial comparison anchors (reference only)

Pin exact model, datasheet revision, SDK/firmware, units, calibration state, and capture boundary before any comparison.

- **DIGIT**: the manufacturer describes it as an open-source, compact, high-resolution vision-based tactile sensor. Its output is tactile imagery; normal-force estimates and marker-assisted shear estimates are downstream interpretations. Official product page: https://digit.ml/digit.html. Commercial assembled listing snapshot: https://www.gelsight.com/product/digit-tactile-sensor/ (USD 355 listed at the research cut; price/lead time may change).
- **GelSight Mini**: commercial optical tactile instrument with 2D/3D mapping and replaceable gel; listing snapshot USD 510: https://www.gelsight.com/product/gelsight-mini-system/. Do not treat its 3D reconstruction as automatically calibrated 3-axis force.
- **XELA uSkin uSPa44-S**: vendor datasheet describes 16 taxels, 3-axis sensing, 133 Hz sampling, and model-specific noise/range in LSB. The LSB values must not be mistaken for SI force uncertainty without exact physical calibration: https://assets-global.website-files.com/630ccaa81fd45082532631d0/65af8c18b1ee6dd10d114f9f_uSPa44_Datasheet%20-%20Jan%202024.pdf. Vendor technology page describes its 3-axis displacement approach and published 0.1 gf resolution: https://xelarobotics.com/technology/.
- **Contactile PapillArray v2.0**: vendor development-kit sheet describes 9 sensing pillars, local 3-axis displacement/force, global 3D force/torque and 1 kHz sampling. The sheet also defines a limited lab/light-manufacturing environment; preserve that limitation: https://contactile.com/wp-content/uploads/2022/01/Contactile_PTSDevKit2.0_SpecSheet_JAN22.pdf.

- **ReSkin**: Meta's open-source magnetically read tactile skin separates the deformable interface from electronics and is designed to be replaceable. Meta reports a materials cost below USD 6 per unit at 100 units, up to 400 Hz, approximately 1 mm spatial resolution and 50,000+ interactions in its published research description. These are source-reported results, not an LMF reproduction: https://ai.meta.com/blog/reskin-a-versatile-replaceable-low-cost-skin-for-ai-research-on-tactile-perception/ and https://reskin.dev/.
- **AnySkin**: an open-source, replaceable magnetic tactile skin with separated sensing electronics and replaceable interface. Its paper reports cross-instance transfer of learned manipulation policies and slip detection; the reported 92% slip accuracy is a task/model result on its own dataset and must not be compared directly with force calibration or a different slip protocol: https://arxiv.org/abs/2409.08276 and https://any-skin.github.io/.
- **TacTip**: biomimetic optoelectronic sensing through deformation of a compliant skin and internal markers. The research family includes submillimetre rolling-cylinder localization; the current project page advertises 120 Hz reporting and slip/dynamic capabilities. Treat these as task- and implementation-specific research/manufacturer claims: https://tactip.org/en/research/ and https://tactip.org/en/.
- **SuperTac**: a 2026 Nature Sensors research system combining multimodal optical/triboelectric/inertial sensing. The paper reports 0.06 N force accuracy, 0.4 mm position accuracy, temperature/proximity/vibration capabilities and greater than 94% classification accuracy on its evaluated classes. This is a high-end research comparator, not a direct baseline unless the exact hardware, measurement methods, test tasks and data are available: https://www.nature.com/articles/s44460-025-00006-y.

These are comparison anchors, not an assertion that vendor or paper claims have been independently reproduced. Commercial listing prices are time-stamped context, not a procurement recommendation. Do not mix: (a) calibrated SI-unit force/displacement results, (b) raw image resolution/frame rate, (c) task-model accuracy, and (d) durability/cost into one rank. ReSkin/AnySkin/TacTip/DIGIT/GelSight/SuperTac may produce fundamentally different native signals; a matched downstream task can compare utility, but it cannot turn their different physical observables into equivalent measurands.

## 5. Reference fixture and evidence design

### Physical phantom ownership

The physical tactile calibration phantom and its reference-instrument architecture already have a canonical owner: **Symtropy #4753 — PHRI-TAC-PHANTOM-001**. This Symthaea campaign must consume that phantom rather than define a second one.

The benchmark's first integration task is an applicability audit against the exact #4753 article and evidence package:

- phantom article/revision, surface material and geometry;
- indenter identity and measured as-built geometry;
- independently calibrated six-axis force/torque reference and calibration chain;
- surface registration/position reference;
- replaceable tile/article identity, calibration and firmware profile;
- normal/tangential load, contact migration, creep, slip and cycle conditions;
- reference uncertainty, timebase and sample lineage;
- accepted and rejected conditions / declared measurement envelope.

If any T1/T2 requirement is not supported by the exact phantom version—for example, an independently timestamped incipient-slip reference—the campaign must record "REFERENCE_CAPABILITY_NOT_ESTABLISHED" and add a scoped extension to #4753 or an owning Symtropy child. The candidate tactile sensor must never supply its own evaluator labels.

### Task-to-reference mapping

| Benchmark question | Independent reference needed | Claim ceiling |
|---|---|---|
| T1 contact localization | Measured phantom geometry + calibrated registration/position reference | Contact location under the exact surface, indenter, mount and registration profile |
| Normal-force characterization | Independently calibrated normal-force/wrench channel | Normal-force result inside its calibrated range and uncertainty |
| Shear characterization | Independent tangential wrench component under an applicable calibrated method | Shear result only where the reference and method cover it |
| T2 incipient slip | Independent displacement/motion/time evidence for the contact interface, synchronized to the force reference | Slip event timing only where the independent event definition and clock uncertainty are qualified |
| Contact area / center of pressure | Reference geometry and independent registration; any pressure comparator must itself have an independent calibration | Geometric/pressure-derived outputs only within the declared method and uncertainty |
| Drift / hysteresis / cycling | Repeated reference observations over declared load histories and cycle counts | Drift/hysteresis for the tested article and profile, not population reliability |

A commanded stage position is not measured force. A force peak alone is not necessarily a slip label. A video annotation is not automatically a calibrated displacement reference. The method and the reference channel must match the exact measurand.

### Standards and measurement uncertainty

ISO 376:2011 remains current after confirmation on 2026-08-14. It specifies calibration/classification of force-proving instruments for static verification of **uniaxial** testing machines. It is a potential reference path for an applicable axial-force proving instrument, not an automatically sufficient method for multi-axis tactile calibration, spatial pressure mapping, torque, or dynamic incipient slip. Those channels require an explicitly reviewed method and uncertainty budget.  
Standard: https://www.iso.org/standard/44661.html

The result, not merely the instrument or certificate, must have a documented unbroken traceability chain with uncertainty contributions. A calibrated reference instrument does not automatically make every subsequent measurement traceable or fit for a declared purpose.  
NIST policy: https://www.nist.gov/metrology/metrological-traceability

For uncertainty computations, bind to **Mycelix #6048** and the applicable current JCGM/BIPM guidance: GUM linearization, the 2026 amendment addressing nonlinearity in measurement models, Monte Carlo distribution propagation (JCGM 101), multiple output quantities/covariance (JCGM 102), conformity assessment/guard bands (JCGM 106), and measurement-model development (GUM-6). Choose the propagation method from the measurement model and input distributions; do not assume unrecorded covariance is zero.  
Official catalogue: https://www.bipm.org/en/committees/jc/jcgm/publications/

### Cross-project owner integration (no duplicate ontologies)

- **Symtropy PR #4743 — PHRI-CONTINUOUS-COUPLING-001:** provides the existing standalone flat 32×32 calibrated normal-pressure field reducer (integrated normal force, thresholded area and centre of pressure). It rejects incomplete/saturated fields and is not a live sensor, shear/slip estimator, curved-surface or multi-tile implementation. Reuse it as the existing normal-pressure reference primitive; do not fork or duplicate it.
- **Symtropy #4753 — PHRI-TAC-PHANTOM-001:** owns the non-human tactile calibration phantom, physical fixture/reference geometry, reference instrument setup and phantom measurement campaign.
- **Symtropy #4776 — PHRI-TAC-TRANSFER-001:** owns cross-unit/cross-family/cross-revision/aging calibration-transfer regimes and sensor/lot/contact-sequence split protocol. Symthaea consumes transfer results for representation/task evaluation; it must not create a second calibration-transfer evidence service.
- **Symtropy #4754 — PHRI-QUAL-BENCH-001:** owns the matched physical-interaction/tactile-coupling comparative benchmark and its scorecards. This Symthaea campaign contributes sensor co-design arms and must not duplicate pHRI controller/passivity or human-agency benchmark semantics.
- **Symtropy #4790 — PHRI-CONTACT-EXPOSURE-001:** owns phantom material-stack/contact-exposure characterization. Its non-human pressure/deformation results do not establish human injury, comfort, consent or safety thresholds.
- **Symtropy #4531 / #5335:** own the unoccupied room-scale service test cell and WaterLoop physical demonstrator. A water-quality/flow instrument is a later, separate sensor application.
- **Mycelix #6034 — MYC-BUILD-METROLOGY-002:** owns instrument/channel capability, calibration, installation and sample-level evidence records. Use its canonical records rather than creating another instrument passport.
- **Mycelix #6047:** owns digital calibration-certificate interoperability/verification. Keep source bytes, signature result, signer trust, accreditation scope and calibration applicability distinct.
- **Mycelix #6048 — MYC-BUILD-UQ-001:** owns measurement-equation, covariance/common-cause and nonlinear/Monte-Carlo uncertainty computation. Bind its exact method/receipt rather than implementing an uncertainty engine here.
- **Mycelix #6054 / #6065:** own stream identity/timebase/replay and measurement-condition applicability. Exact sample epoch, sequence, condition values and condition-profile digest must survive every adapter.
- **Symthaea FIELD/SE-OBS/SE-SEM and #4918:** remain the semantic and evidence-dependence authorities. Do not add local quantity, unit, observation, calibration or common-mode ontologies.
- **Sol-Atlas #633 / #700:** consume owner-native spatial and service-outcome projections. They do not recalculate sensor calibration or reinterpret source sample time.

If an owner contract is not implemented or is absent from the exact consuming branch, preserve the dependency as unresolved. Do not imply cross-repository integration is executable merely because the contract is documented.

## 6. Comparison and causal attribution

### External benchmark alignment — reuse, do not create another leaderboard

The first tactile campaign should have two distinct tracks:

1. **Physical sensor/metrology track:** Symtropy #4753 phantom, independently qualified reference channels, Mycelix instrument/calibration/uncertainty/timebase/condition records, and the P1 physical characterization defined here.
2. **Representation/manipulation track:** adapt to public benchmark protocols and datasets below, preserving their task definitions, splits, licenses and model baselines. These datasets cannot establish SI-unit calibration or physical traceability.

| External benchmark / dataset | Published scope | What Luminous should test | Critical limitation |
|---|---|---|---|
| **TacO (2026)** | Four sensor modalities (visual, acoustic, magnetic, resistive) and three contact-rich manipulation tasks: unknown-mass pick-and-place, object reorientation and plug insertion | Sensor modality × material friction × task interaction; compare sensor utility under matched manipulation policies | Task-policy performance is not physical calibration; no universal best modality is implied |
| **TacMind (30 Oct 2026)** | 128 simulation tasks, Precise / Recognition / Dynamic regimes, static and dynamic variants, 12 perturbation settings; action-conditioned tactile-history world model | Temporal prediction, contact-state discrimination, policy robustness and whether tactile history improves decision quality | Simulation benchmark only; physical transfer still required |
| **RCT (2026 preprint)** | 29,279 tactile frames, per-press contact sequences, force signals, indentation depth, materials, categories, sensors and contact positions; held-out presses/materials/sensors/positions | Contact-sequence generalization and sensor/material transfer using its official sequence-level splits | Frame-random splitting leaks near-duplicate frames from the same press; preserve press/article lineage |
| **GenForce (Nature Communications, 28 Jan 2026)** | Cross-sensor spatiotemporal force prediction across GelSight, uSkin and TacTip using marker-space translation | Mandatory learned-transfer comparator for force estimation; reproduce on exact hardware/data and evaluate target-unit calibration, uncertainty and held-out sensor articles separately | Published force MAE and transfer results are paper-specific; successful learned transfer is not itself target-device metrological traceability |
| **TacVerse (2026)** | 106,800 images across 7 vision-based tactile platforms; within-sensor, zero-shot cross-sensor and few-shot adaptation protocols | Cross-sensor representation transfer, calibration/adaptation sample efficiency and degradation under sensor shift | Vision-based tactile family only; it does not compare all physical modalities |
| **HT-Bench (2026)** | 10M RGB frames, 7.8M tactile frames, 226 tasks; retrieval, masked tactile inpainting, vision-to-tactile synthesis and multimodal prediction | Full-hand tactile representation, temporal/multimodal alignment and transfer to unseen tasks | Representation-task metrics are not force/pressure metrology |
| **Sparsh / TacBench (Meta, 2024)** | Self-supervised tactile representation pretraining on 460k+ tactile images and six downstream tasks; reports strong gains over task-specific training | Mandatory learned-representation baseline for vision-based tactile inputs, including DINO/I-JEPA-style methods | The reported average relative gain is specific to the paper's tasks and baselines; do not compare it as a universal sensor accuracy |
| **TactiDex (2026)** | Real-world tactile-guided dexterous manipulation, with real and simulated task demonstrations | Real-world manipulation confirmation after synthetic and dataset qualification | Task/embodiment scope must be matched before comparative claims |

Source registry:
- TacO paper: https://arxiv.org/abs/2605.21976 and project: https://tacobench.github.io/
- TacMind: https://arxiv.org/abs/2611.00455 and https://tacmind-bench.github.io/
- RCT: https://arxiv.org/abs/2606.31694 and https://faerber-lab.github.io/RCT/
- TacVerse: https://lannwei.github.io/Tactile_Database/
- GenForce: https://www.nature.com/articles/s41467-026-68753-1
- HT-Bench: https://arxiv.org/abs/2606.19161
- Sparsh / TacBench: https://ai.meta.com/research/publications/sparsh-self-supervised-touch-representations-for-vision-based-tactile-sensing/
- TactiDex: https://tactidex.github.io/ and https://arxiv.org/abs/2607.09190

No third-party benchmark result is a Symthaea result. Use the original benchmark license and official split/capture protocols. Do not silently redistribute external data or claim reproduction from a paper abstract alone.

### Transfer novelty boundary

Cross-sensor tactile force prediction is already an active published result, not a Luminous novelty claim. GenForce is a required comparison baseline. Luminous's testable differentiation is **metrology-preserving transfer**: lower target-unit labelled calibration burden while retaining target-unit calibration identity, uncertainty coverage, guard-band qualification, held-out sequence/unit/lot splits, aging/revision applicability and correct recalibration/abstention. Symtropy #4776 owns this transfer protocol; the Symthaea task/representation campaign consumes its qualified outputs. A strong predictor remains a model estimate unless the target measurement chain independently qualifies the claimed measurand.

### Luminous competitive hypothesis

The strongest defensible research target is not “best tactile classifier”. It is:

> A repairable sensor system that preserves calibrated physical observations and uncertainty, diagnoses its own evidence failures, transfers representations across articles/sensors/materials, and improves a real manipulation task under matched power, latency, calibration and lifecycle constraints.

To test this without double-counting evidence, report five distinct result families:

- **P1 metrology:** force/displacement calibration, uncertainty coverage, hysteresis, drift, crosstalk, saturation and physical transfer.
- **P2a representation:** sequence prediction, contact geometry, retrieval, sensor/material generalization, calibration sample efficiency and OOD detection.
- **P2b task:** localization, slip discrimination, unknown-mass pick, reorientation, insertion and other exact manipulation outcomes.
- **P3 resource/lifecycle:** complete-system energy, bandwidth, compute/memory, latency tails, calibration time, repair, replacement and useful measurement availability.
- **P4 evidence integrity:** held-out leakage, common-mode detection, source/clock identity, calibration currentness, raw-to-derived lineage, fault diagnosis and correct abstention.

A result can improve P2b while P1 remains unresolved. It is a valid task result, not a metrology upgrade.

### Representation baselines and ablations

For each public tactile representation task, compare at least:

1. raw sensor input + the strongest simple task-specific baseline;
2. sensor-native engineered features / conventional temporal filtering;
3. published learned tactile encoder applicable to that sensor family (e.g. Sparsh, HandTouch where supported);
4. Symthaea HDC encoding alone;
5. Symthaea HDC + temporal/CfC context;
6. a matched learned temporal baseline;
7. fast HDC/event path plus slower contextual path.

Use the same training/evaluation examples, task labels, adaptation budget and measured system boundary. Report both sample efficiency and total training/inference cost. A compressed representation must not silently erase amplitude, timing, uncertainty, saturation or channel-fault information required by the downstream claim.

### Leakage-resistant evaluation

- Use RCT's press/contact-sequence grouping rather than random frame splits.
- Hold out physical sensor article, sensor model where possible, material, contact object, contact position, fixture/remount session and environment separately.
- Keep calibration/development, design search and confirmatory partitions distinct.
- Use TacVerse's zero-shot and few-shot cross-sensor protocols as separate strata; adaptation examples must be counted in the resource budget.
- Do not train on hidden evaluator truth, held-out task labels, post-hoc calibration residuals or future observations.
- Preserve every failed and excluded episode with its reason. Repeated frames from one physical press are not independent replications.

### Proposed competitive gates (targets, not achieved results)

- 100% deterministic agreement on frozen synthetic sensor-fault and representation-loss known answers.
- Zero false-valid measurement claims for seeded saturation, missingness, stale/article-mismatched calibration, unit mismatch, common-reference faults or invalid timebase.
- Held-out P1 uncertainty coverage and calibration residuals meet preregistered, measurand-specific criteria; no universal numerical force threshold is selected before the exact range and reference chain exist.
- On each declared P2 task, beat or match the strongest applicable published/open baseline under matched data, sensor, compute and adaptation budgets; report paired uncertainty and per-stratum results.
- On cross-sensor/material transfer, publish the full zero-shot-to-few-shot adaptation curve, not only the best post-adaptation score.
- HDC/CfC earns deployment consideration only if it preserves task-relevant signal and physical-evidence lineage while improving a declared latency, energy, memory, sample-efficiency or robustness axis.
- No task improvement may compensate for a protected P1, evidence-integrity or operational regression.

This is how we can improve on SOTA responsibly: retain the strongest public baselines, find a measurable gap (especially metrology-aware transfer, fault diagnosis, or complete-system resource cost), and demonstrate a Pareto improvement on held-out evidence—not by selecting an easier task or collapsing incompatible metrics.


Minimum arm set where physically feasible:

| Arm | Physical system | Processing | What it can isolate |
|---|---|---|---|
| A0 | B0 simple pad | simple threshold/filter | incumbent low-complexity baseline |
| A1 | C1 candidate patch | same simple processing | total hardware-system difference |
| A2 | C1 candidate patch | stronger task model | algorithm contribution on candidate hardware |
| A3 | B0 where signal interface permits | same stronger task model | algorithm benefit independent of new hardware |
| A4 | C2 morphology variant | same readout and algorithm as C1 | morphology/electrode contribution |

If exact parity is impossible, state the confound. Use factorial/ablation logic where it can distinguish morphology, transducer, front-end, calibration, representation and task model contributions.

### Industry comparison rule

The 2026 literature emphasizes a persistent transition from device-level sensitivity demonstrations to application-level systems balancing dynamic range, hysteresis, crosstalk, environmental stability, fatigue, integration, energy and task reliability. The comparison programme must therefore include:

- a low-cost baseline (B0);
- one open, replaceable research sensor (ReSkin or AnySkin);
- one optical/deformation baseline (DIGIT, GelSight Mini or TacTip);
- one direct multiaxis commercial sensor (XELA or Contactile) if the measurand and interface match;
- the best task-specific published research method (SuperTac or a more relevant method selected at protocol freeze).

The review literature identifies crosstalk, environmental stability, signal decoupling and scalable integration as persistent challenges, rather than one universally dominant transduction method: https://doi.org/10.1016/j.snr.2026.100459 and https://doi.org/10.1016/j.sna.2026.118150.

For each comparator, freeze exact sensor model/revision, firmware/SDK, mounting, native data representation, filtering, calibration status, usable range, acquisition boundary, power boundary, and test conditions. Report native-signal and common-task comparisons separately. A 1 kHz sample rate, a 90 fps camera, 400 Hz magnetic readout, 92% task accuracy and a force resolution in LSB are not interchangeable measures.

The comparison must report three non-substitutable planes:

- **P1 physical:** calibration error/residual, hysteresis, drift, cross-axis, range, bandwidth, etc.
- **P2 task:** localization, slip detection, timing and task utility.
- **P3 system:** energy, latency, data, size, maintenance and calibration burden.

Use profile-relative dispositions: ParetoImprovedUnderProfile, TradeoffUnderProfile, EquivalentWithinDeclaredTolerance, RegressionOnProtectedAxis, EvidenceInsufficient, ComparisonInvalid, OutOfProfile, or Dominated. Never reduce the result to one universal sensor-quality score.

## 7. Development, confirmation and replication

Separate calibration/development, design search, and held-out confirmation. Freeze exact physical article generation, mount/remount session, contact episode, surface/indenter, fixture session, calibration root, and held-out condition identity.

Do not split adjacent samples from one physical episode into training and test and call them independent trials. A second independently realized patch is needed before claiming article-level reproducibility. Multi-lab reproducibility and production-lot variation are separate later evidence strata.

After held-out reveal, a material change to geometry, material, front end, firmware, calibration model, threshold, representation or task model creates a new comparison generation.

## 8. Adversarial controls

At minimum include:

1. high sensitivity but early saturation;
2. low normal error but high shear cross-talk;
3. strong task score with invalid physical calibration;
4. better physical calibration with no task improvement;
5. morphology variant tested with changed algorithm (causal attribution unresolved);
6. held-out surface/position leaking into tuning;
7. candidate-defined force/slip labels (circular reference);
8. flat calibration transferred to curved mounting without evidence;
9. remount shift not reflected in calibration generation;
10. high sample rate with worse host buffering/tail latency;
11. high-resolution array with no decision benefit but higher data/power;
12. channels sharing a failing ADC/reference rail;
13. algorithm hides a physical bias without repairing metrology;
14. one physical patch producing many samples presented as multi-article replication;
15. drift after cycling while the sensor remains electrically responsive;
16. self-test passes while scale/calibration drift remains;
17. missing energy/reference/timing data silently treated as zero;
18. human-contact or actuator authority inferred from the non-human bench.

## 9. Benchmark reporting

Publish, per profile and stratum:

- calibration residual and independent reference uncertainty;
- bias, repeatability, hysteresis, linearity/nonlinearity, drift and cross-axis response;
- force/displacement range, saturation and recovery;
- bandwidth and sensor-to-host/task latency p50/p95/p99 with clock uncertainty;
- T1 spatial error distribution, worst tested strata and out-of-profile cases;
- T2 event precision/recall, false alarms per episode and delay from independent slip reference;
- valid sample coverage, dropout and fault-detection results;
- per-run energy, total data rate, compute and operator/calibration time;
- repair/remount and recalibration burden;
- article/session/condition variance and missingness;
- uncertainty budget and unexplained residual.

### Proposed benchmark gates (targets, not achieved results)

- **Synthetic correctness:** 100% frozen known-answer/adversarial disposition agreement; 0 false-valid results from seeded saturation, missingness, stale calibration, channel swap, shared-reference fault or held-out leakage.
- **Physical metrology:** every claimed P1 axis has an independent reference, a documented uncertainty budget, and a frozen acceptance rule before data collection. Numerical thresholds are selected only after the specific range, method and reference capability are chosen.
- **Task utility:** compare to strongest feasible incumbent on identical held-out episodes; report effect sizes and uncertainty, not only accuracy. No predeclared universal superiority percentage.
- **Timing:** include the whole sensor/readout/transport/representation/task path and the actual deadline. Do not compare source sample-rate to end-to-end response.
- **Reproducibility:** report within-article repeatability, between-session/remount, between-article, between-lab, and field transfer separately.
- **Resources:** complete-system energy and calibration/maintenance labour included; no core-only power claim.
- **Robustness:** no protected-axis regression may be hidden by task gains.

## 10. Stage gates and claim ceiling

1. **Design freeze:** this constitution, method audit and reference-fixture uncertainty plan.
2. **Synthetic fault model:** #5831 known-answer corpus; no physical claims.
3. **Design subject:** bounded C1/C2 parametric candidates; no fabrication claim.
4. **M0 engineering review:** fixture design, hazards, reference chain, safe load boundaries.
5. **M1 bench characterization:** only after reviewed fixture and admitted channels/reference instruments.
6. **Held-out confirmation:** locked design and evaluation protocol.
7. **Second-article/lab transfer:** before broader reproduction claims.
8. **Task/control integration:** proposal-only until the relevant robotics authority/safety owner qualifies any actuation route.

This document authorizes no purchase, fabrication, energization, human contact, physical actuation, certification, or deployment. A synthetic pass is not a physical sensor pass. A physical patch pass is not human-contact safety, production reliability, or universal sensor superiority.

## 11. Research / incumbent references

- Xue & Du (June 2026), review of multimodal tactile-sensor mechanisms and signal-decoupling strategies: https://doi.org/10.1016/j.snr.2026.100459.
- 2026 review of tactile electronic skin and intelligent interfaces, including hysteresis, crosstalk, scalability, mechanical decoupling and application trade-offs: https://www.sciencedirect.com/science/article/pii/S0924424726007016.
- 2026 review, *Tactile robotics: Past and future*: https://doi.org/10.1177/02783649261421615.
- ISO 376:2011 force-proving instrument calibration/classification scope (current as reviewed in 2026): https://www.iso.org/standard/44661.html.
- NIST metrological traceability: https://www.nist.gov/calibrations/traceability.

Research literature informs hypotheses and method selection; commercial specifications are vendor claims; neither is Symthaea physical performance evidence.
