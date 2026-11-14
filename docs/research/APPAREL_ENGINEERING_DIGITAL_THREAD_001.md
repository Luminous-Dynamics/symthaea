# APPAREL-ENG-TWIN-001 — Apparel Engineering Digital Thread v0.1

**Status:** research architecture and benchmark contract. The first Luminous textile process software slice exists as draft work in Symtropy; the cross-system apparel thread specified here is not yet implemented or physically qualified.

**Parent:** [Symthaea #11731 — APPAREL-ENG-TWIN-001](https://github.com/Luminous-Dynamics/symthaea/issues/11731)

**Companion work:** [Symtropy #4865 — textile robotics](https://github.com/Luminous-Dynamics/symtropy/issues/4865) · [Sol-Atlas #738 — apparel spatial geometry](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/738) · [Mycelix #6092 — textile DPP projection](https://github.com/Luminous-Dynamics/mycelix/issues/6092)

---

## 1. Executive thesis

Build an evidence-native engineering thread that can carry one textile product from intent to design, geometry, physical material behavior, construction, manufacturing, inspection, use, repair and recovery.

The differentiator is not a text-to-fashion image generator. It is the ability to preserve the meaning and evidence of a physical product while it moves through different specialist tools, models, machines, suppliers and lifecycle events.

```text
Intent
  -> requirements and constraints
  -> material / fabric alternatives
  -> pattern and garment geometry
  -> physical model and manufacturability analysis
  -> process plan and qualified machine capabilities
  -> execution and physical observation
  -> independent inspection / test
  -> product + batch lineage
  -> repair / reuse / recovery
  -> next design iteration
```

The platform must distinguish:
- proposed design from released design;
- simulated behavior from measured behavior;
- supplier declaration from independently measured composition;
- plan from actual execution;
- one accepted sample from a qualified production process;
- reported recovery from verified secondary-material output;
- internal engineering record from regulatory product-passport projection.

## 2. The actual state of the public repositories

This programme composes existing owners and must not duplicate them.

| System / existing work | Existing responsibility | Role in this thread |
|---|---|---|
| Symthaea [ENG-DESIGN-001 #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005) | Evidence-bound physical engineering design, requirements, assumptions, V&V, configuration change | Requirements, design reasoning, trade studies, test selection, engineering decision records |
| Symthaea [TEXTILE-HYG-MFG-000 #5984](https://github.com/Luminous-Dynamics/symthaea/issues/5984) | Textile/clothing product-function qualification | Material/product evidence, fit/use boundaries, mechanical, absorbent, barrier and lifecycle qualification domains |
| Symthaea [Industrial Alternatives Fabric #4224 in Mycelix](https://github.com/Luminous-Dynamics/mycelix/issues/4224) and implementation anchor [Symthaea PR #6856](https://github.com/Luminous-Dynamics/symthaea/pull/6856) | Evidence-qualified functional substitution and multi-axis alternatives | Material/process alternatives without a scalar "green score" |
| Symtropy [textile PR #1679](https://github.com/Luminous-Dynamics/symtropy/pull/1679) | Garment profiles, process objectives, feasibility gates and Pareto comparison | Candidate process evaluation |
| Symtropy [textile PR #1699](https://github.com/Luminous-Dynamics/symtropy/pull/1699) | Bounded single-stage scheduler, disruption stress and demand/surplus evaluation | Scheduling and production policy comparison |
| Symtropy [textile PR #1716](https://github.com/Luminous-Dynamics/symtropy/pull/1716) and [#1724](https://github.com/Luminous-Dynamics/symtropy/pull/1724) | Empirical evaluation protocol, paired trial records and fail-closed typed validation | Evidence-qualified field evaluation |
| Symtropy [PHYS-SOFT-05 #456](https://github.com/Luminous-Dynamics/symtropy/issues/456) and [PHYS-ARCH-00 #457](https://github.com/Luminous-Dynamics/symtropy/issues/457) | Specialized deformable/cloth profiles outside the rigid-body core | Cloth simulation and model coupling only through qualified profiles |
| Mycelix [MFG-EVID #3084](https://github.com/Luminous-Dynamics/mycelix/issues/3084), [MFG-PROC #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085), [ERG-CORE #3097](https://github.com/Luminous-Dynamics/mycelix/issues/3097), [CIRC-MFG #3095](https://github.com/Luminous-Dynamics/mycelix/issues/3095) | Product identity, process lineage, material/economic event graph, lifecycle composition | Product/article/lot identity, custody, manufacturing and lifecycle events |
| Mycelix [textile recovery PR #5057](https://github.com/Luminous-Dynamics/mycelix/pull/5057) | Textile recovery eligibility and physical output lineage | Recovery evidence; it is draft and not a qualified recycling result |
| Sol-Atlas [body/device geometry #708](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/708) and [ITIP-FED #359](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/359) | Spatial frames, uncertainty, privacy and inter-twin geometry semantics | Read-only pattern/garment/fit spatial projection |

The Symtropy textile PRs are a draft stack, not a deployed textile factory simulator. The original #1679 exact-head run failed at rustfmt before its contract tests, Clippy and build steps; formatter-generated changes have been applied to the parent branch, but no current-head CI/test pass is asserted here. Mycelix #5057 is also draft; its observed D6S workflow failed before the independent verifier because the requested `d6s_canon_1_golden_vectors` test target was missing, while ordinary Mycelix CI was skipped. Neither status is a textile-performance result.

## 3. Paradigm shifts to pursue

| Conventional approach | Luminous research direction |
|---|---|
| Design around whatever fabric is in the catalogue | Inverse textile engineering: start with required behavior, search material/fiber/fabric/process combinations |
| Pattern is a drawing or proprietary file | Pattern is versioned manufacturable geometry with piece identity, grain, notches, seam allowances, topology and tolerances |
| 3D garment is a visual prototype | Digital engineering twin with calibrated material inputs, declared model fidelity and a physical discrepancy record |
| One size chart represents fit | Fit is a task-, pose-, layer-, garment- and user-contextual geometry question with uncertainty and privacy boundaries |
| Fabric robot grasps a visually convenient point | Robot selects actions from measured fabric state, task, material properties, tension and expected deformation |
| Factory optimizes individual machines | Factory policy optimizes qualified functional service across yield, quality, service, time, energy, water, material and waste constraints |
| Sustainability is a single opaque score | Non-compensatory gates plus explicit Pareto alternatives, functional units and burden-transfer checks |
| Passport is a QR code or supplier claim | Passport is a controlled projection over a source-preserving evidence/lifecycle record |
| Recycling begins after disposal | Product construction, component joining, repair and recovery routes are engineered at design time |

A key principle: no objective may compensate for violating a hard safety, quality, legal, environmental or evidence gate. Missing evidence is not zero, and unknown is not pass.

## 4. First demonstrator: non-PPE woven overshirt

Choose a deliberately bounded, low-risk sewn product. The first article is a research demonstrator, not PPE, medical apparel or a protective product.

Freeze the following before measurement:
- intended use, user task and product non-goals;
- garment profile and declared functional unit;
- size/fit vocabulary and fit-manikin revision;
- pattern pieces, 2D geometry, seam topology and exact pattern revision;
- grain directions, notches, seam allowances, fasteners and trims;
- fabric construction, composition evidence and material-lot identity;
- BOM and assembly/process sequence;
- machine/tool capability assumptions and operation constraints;
- dimensional and visual quality criteria;
- inspection/test plan and acceptance limits;
- repair, replacement, reuse and recovery pathways.

Start with a synthetic garment configuration and calibrated reference manikin. Make one physical sample after the digital thread contract is stable. Expand to a small size range only after the first product identity, revision and measurement lineage are coherent.

## 5. Cross-system contracts

### 5.1 Symthaea — requirements and design reasoning

Consumes the intended use, performance constraints, material evidence, pattern/geometry references, production constraints and prior test observations. It may propose alternatives, estimate trade-offs, detect contradictions and recommend the next informative test.

It must not:
- claim measured fabric properties from an unvalidated language-model estimate;
- publish a pattern as physically manufacturable without the relevant checks;
- authorize machine operation;
- turn supplier claims into independent physical test results.

### 5.2 Sol-Atlas — spatial/geometry projection

Consumes owner-native pattern, garment mesh, fit-manikin/body-frame, pose, material profile reference and revision-bound measurements. It projects:
- 2D pattern-piece identity and 3D garment correspondence;
- seam path and join topology;
- grain/notch/seam-allowance geometry;
- fit/ease/clearance envelopes under named poses;
- collision/interference and motion envelopes;
- spatial uncertainty and source lineage.

It is read-only with respect to CAD and physical state. Raw identifiable body scans remain local by default; export only reviewed, purpose-limited derived geometry. Geometric correspondence is not comfort or pressure safety.

### 5.3 Symtropy — physical/process simulation and robotics benchmark

Consumes versioned material parameters and product geometry under declared solver profiles. Distinguish:
- cloth state and mechanical model;
- process plan and machine constraints;
- simulated operation from observed physical execution;
- single-stage scheduler from a complete multi-stage factory model;
- cloth manipulation benchmark from autonomous sewing claims.

The existing XPBD seed in #456 must not be represented as a production textile simulator. Rigid-core physics remains lean; cloth/deformables use qualified specialized profiles and explicit coupling. Calibration requires measured material coupons and held-out validation specimens.

### 5.4 Mycelix — identity, lifecycle and evidence projection

Owns canonical identity/lineage and admitted manufacturing/lifecycle events. Bind:
- product design/configuration revision;
- manufactured article and batch/lot identity;
- input material and supplier claims;
- operation/process events;
- actual test/inspection records;
- repair/replacement and custody;
- verified reuse, recovery and secondary-material output.

A hash or signature can support integrity and attribution; it does not prove the truth of a supplier assertion or the physical existence/grade of recovered material.

## 6. External datasets and baseline evidence

Use datasets as separately versioned benchmark inputs. Review terms, licensing, attribution, privacy and redistribution conditions before download or reuse. Do not commit bulky raw datasets or make CI depend on a multi-gigabyte download.

| Source | What it contributes | Limitation / use |
|---|---|---|
| [ICRA 2024 Cloth Competition paper](https://doi.org/10.1177/02783649251414885) and [dataset DOI](https://doi.org/10.5281/zenodo.14372573) | 679 real-world unfolding demonstrations across 34 garments; 11 participating teams | Strong starting point for grasp selection/unfolding; not sewing or general garment assembly |
| [TexSortML dataset](https://doi.org/10.5281/zenodo.22143896) | 751 post-consumer garments, 11 categories and three handling-difficulty stages | Sorting/perception robustness; one 2026 model comparison reports smaller performance degradation for a ViT, but model sizes differ substantially, so parameter/compute-matched reproduction is required |
| [NIST NIR-SORT 2.0](https://www.nist.gov/news-events/news/2026/03/nist-releases-nir-sort-20-enhancing-model-validation-textile-feedstock-0) | NIR spectra for textile feedstock/fiber identification; released 19 March 2026 | Material identification only; check exact data package, metadata, usage terms and lab-to-field transfer |
| [2026 cloth manipulation review](https://doi.org/10.3389/frobt.2026.1752914) | Identifies irregular garment states, data collection, simulator realism, Sim2Real and metrics as open problems | Review evidence, not a benchmark result |
| [2026 robotic sewing study](https://doi.org/10.1007/s41315-026-00590-3) | Vision, learned grasping and force control applied to robotic sewing | Experimental comparator under its own setup; not proof of general-purpose production autonomy |

A 2026 TexSortML paper reports a 9.3 percentage-point degradation for its ViT between clean and difficult garment presentations, versus 16.6–16.9 points for its CNN comparators. However, the reported ViT has approximately 305.6M parameters, versus 23.5M for ResNet-50 and 11.2M for ResNet-18. Treat this as a robustness hypothesis to reproduce with parameter/compute-matched baselines, not as a fair architecture victory.

### Dataset split rules

- Split by garment identity, source/collection batch and physical specimen—not random video frames alone.
- Keep calibration/training specimens separate from held-out evaluation specimens.
- Preserve source revision, file checksums, license, attribution and transformation code.
- Do not tune a physical model on the validation trial whose error is reported as held out.
- Report per-class, per-material and lower-tail results; no pooled mean may hide a fabric type that fails systematically.

## 7. Physical standards and measurement anchors

Use the exact current edition and scope as recorded on the test record; do not reproduce copyrighted standard text in the repository.

- [ISO 11092:2026](https://www.iso.org/standard/85998.html): steady-state thermal resistance and water-vapour resistance for fabrics and multilayer assemblies. The method does not itself define wearer comfort.
- [ISO 6330:2021](https://www.iso.org/standard/75934.html): domestic washing and drying procedures for textile testing. It defines procedures, not universal durability or care acceptance.
- [ISO 12947-2:2016](https://www.iso.org/standard/61058.html): Martindale abrasion specimen breakdown.
- [ASTM D737-18(2023)](https://store.astm.org/standards/d737): air permeability of textile fabrics.

Depending on the declared article, later test lanes may include tensile/tear, seam strength, dimensional change, pilling, colorfastness, hydrostatic/chemical barrier, thermal response, laundering and repair durability. Select methods with competent domain owners; do not presume all are required for every garment.

Every physical test record should bind:
- specimen and batch identity;
- exact design/material revision;
- method and edition;
- apparatus and calibration;
- conditioning/environment;
- sample preparation and orientation;
- measured values, units and uncertainty;
- operator/lab and time;
- deviations, exclusions and disposition.

## 8. Benchmark programme

### B0 — identity and semantic interchange

Test round-trips across pattern, geometry, BOM, process and lifecycle representations:
- all IDs/revisions survive;
- units, coordinate frames, handedness and grain direction survive;
- seam allowance, notches, labels, topology and piece relationships survive;
- every unsupported/lossy field is reported, never silently discarded;
- stale evidence is invalidated only for its actual dependency scope.

### B1 — garment geometry and physical fidelity

Measure:
- pattern-to-3D reconstruction error;
- local ease/clearance deviation against known reference fixtures;
- seam-path correspondence and topology;
- dimensional deviation between digital model and manufactured article;
- drape prediction against measured reference poses;
- uncertainty calibration and coverage;
- sensitivity to fabric properties, conditioning, size and pose.

Do not reduce the result to global mesh Chamfer distance alone; critical seams, notches, fit constraints and local clearances need task-specific measures.

### B2 — process optimization

Reuse the Symtropy textile process benchmark contracts. Report:
- feasibility and hard-gate disposition;
- Pareto frontier and exact small-instance optimality gap;
- makespan, due-date service and changeover;
- accepted/rejected units, rework and stockouts;
- fabric utilization, offcut/scrap and mass closure;
- measured energy/water per declared functional unit;
- demand/surplus outcomes by garment profile.

Do not compare a one-off couture or ceremonial product to a high-frequency basic as if "wears per year" were a universal value metric.

### B3 — cloth robotics and seam execution

Use separate suites for unfold, spread, align, feed and sew. Report:
- task completion, coverage and grasp success;
- edge/corner localization;
- regrasp and recovery;
- time/action count;
- tension, slip and damage;
- seam deviation and post-seam quality;
- Sim2Real degradation;
- abstention and safe-stop behavior.

The ICRA dataset is the initial unfolding baseline; it is not used to claim sewing ability.

### B4 — material identification and sorting

Compare NIR/SWIR/visual/metadata inputs only where the actual sensor/data supports them. Report:
- per-class precision/recall and confusion;
- unknown/open-set rejection;
- blended-fiber and contamination cases;
- calibration and domain shift by presentation condition;
- parameter/compute budget and throughput;
- downstream purity and recovery yield once physically sorted.

### B5 — circularity and product evidence

Report:
- composition/lot provenance completeness;
- physical custody and quantity/mass closure;
- sort/disassembly success;
- secondary output grade and physical functional acceptance;
- repair/reuse/transfer receipts;
- product-passport projection accuracy and disclosure;
- avoided-virgin-material or lifecycle claims only with a separate qualified counterfactual/LCA method.

## 9. Industry comparison and how to beat the frontier

Compete with specialist tools on matched tasks, not vague product-category claims.

| Benchmark family | Industry comparators / references | Luminous differentiator to test |
|---|---|---|
| Pattern CAD and production preparation | Lectra/Modaris, Gerber AccuMark or equivalent | Cross-tool semantic preservation; bind pattern, material, process and test revision |
| 3D garment development | CLO and Browzwear | Physical calibration, uncertainty, as-built discrepancy and independently measured garment fit/drape |
| Robotic unfolding | ICRA Cloth Competition participating baselines | Held-out garment/material generalization and consistent failure reporting |
| Robotic sewing | Published 2026 vision/grasp/force-control system | Robustness across fabric classes, seam quality and recoverable failure handling |
| Fiber identification/sorting | NIST NIR-SORT 2.0 and TexSortML | Open-set/unknown handling plus downstream physical sort purity and recovery qualification |
| Product passports | Manual and template-based records; future textile DPP mapping | Every emitted claim traces to owner-native source, applicability, evidence class and disclosure authority |

A valid comparison freezes the same product, requirements, materials, specimen, machine, operator training, sensors, functional unit and measurement procedure. Report vendor/tool version, parameter budget, human intervention and license constraints. Reproduce the selected reference baseline before claiming an improvement.

### Proposed ambitious targets (not achieved)

Freeze only after baseline repeatability is established:
- 100% of required product claims bind to exact product/configuration revision and evidence scope.
- 100% preservation of supported pattern semantics in the declared interchange profile; zero silent unit/frame/grain/notch/seam-allowance mutation.
- zero model/supplier/synthetic values promoted to independent physical measurement.
- zero stale, revoked or restricted data emitted as current/public without an explicit policy decision.
- ≥30% reduction in engineering iteration time versus a matched baseline, with equal or better verification coverage.
- ≥20% reduction in avoidable material waste on the matched cutting/nesting task, with no loss of accepted output or product function.
- ≥25% fewer physical sample iterations to reach the same independently accepted dimensional/quality result.
- for a narrow, frozen robot task set, ≥95% task completion on held-out garment identities before expansion to broader fabric classes; damage and false-safe handling remain hard gates.
- reduce Sim2Real performance degradation by ≥50% relative to the frozen initial model/policy on the same held-out tasks.
- no single aggregate score may offset a failed material/quality/rights/safety gate.

These are challenge targets, not predictions or results. If a baseline, test method or functional unit makes a target invalid, replace it before preregistration and preserve the reason.

## 10. Digital Product Passport boundary

The European Commission lists Q4 2027 as the *planned* adoption of the textile ESPR delegated act and notes that timelines may evolve: [official textile DPP page](https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en).

Until the actual textile-specific act and technical specifications are published:
- do not hardcode a guessed mandatory data-point list;
- keep legal applicability, evidence state and disclosure audience separate;
- build a generic evidence projection over the actual product/lifecycle record;
- do not claim legal compliance or registry acceptance;
- keep publication/registration authority with the responsible economic operator or explicitly delegated party.

## 11. Execution sequence

1. **Digital thread and synthetic corpus:** freeze object IDs, revisions, typed references, semantic-preservation rules and adversarial fixtures.
2. **Benchmark data intake:** approve source rights, create manifests and small parser fixtures; no bulk downloads in CI.
3. **Spatial projection:** connect 2D/3D pattern/garment geometry, frames and uncertainty.
4. **Physical fabric coupons:** measure the minimum material properties needed to calibrate one declared textile profile; reserve held-out specimens.
5. **First physical overshirt:** manufacture one revision, inspect, test and bind all observations to the exact article.
6. **Cloth manipulation:** reproduce the public unfolding baseline; then add aligned feeding and seam tasks as separate physical lanes.
7. **Recovery and passport projection:** connect Mycelix textile-recovery evidence and test a versioned, disclosure-aware projection.
8. **Matched industrial study:** only after measurement, sample size, baseline and analysis plan are frozen.

## 12. Global invariants and claim ceilings

```text
design revision != manufactured article
simulation PASS != physical test PASS
supplier declaration != independently measured composition
hash present != source bytes verified
planned operation != executed operation
one accepted sample != qualified production process
recycling claim != physically verified secondary output
passport projection != legal compliance or registry publication
geometry match != comfort, safety or clinical fit
```

Synthetic fixtures qualify only deterministic schema/logic behavior. Simulation qualifies only its declared model and applicability domain. Physical testing qualifies only the tested samples, methods, configurations and conditions. Human-use, protective, medical, environmental and regulatory claims require separate owners and evidence.

**The success criterion is a reproducible chain from a defined garment requirement to an inspected physical article and traceable lifecycle record—not a visually impressive render or an unqualified AI benchmark score.**


## 13. Machine-readable seed fixture

The first synthetic cross-system record is now committed alongside this specification:

- `docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.schema.json`
- `docs/research/fixtures/apparel_engineering_thread_v0_1/apparel_thread.example.json`

Fixture ID: `APPAREL-TWIN-SYNTH-0001`. It binds a synthetic overshirt product, design revision, pattern pieces, fabric/thread lots, BOM, planned cut/sew/inspect operations, geometry projections, unperformed physical tests and four claims. Its expected result is **structurally complete but physically unqualified**. The fabric's key mechanical properties remain unknown/not measured, the physical article has not been manufactured, and an incomplete modelled drape projection cannot support a physical drape claim.

This is deliberately a negative epistemic example: a structurally connected digital thread must not be mistaken for a physically qualified garment. The schema and example were locally JSON-parsed after authoring. An independent JSON Schema Draft 2020-12 validator and cross-field semantic validator have **not** yet been run; no validation result beyond JSON parsing is claimed.


## 14. Competitive benchmark refresh — 12 November 2026

The first draft benchmark list has been sharpened with the latest public research. A machine-readable source register is committed at:

`docs/research/fixtures/apparel_engineering_thread_v0_1/benchmark_sources_2026-11-12.json`

It records dataset/research identity, public scope, license state, limitations, split policy and claim ceilings. It does **not** assert that the data have been downloaded, licensed for redistribution or reproduced.

### RGBench — real garment physics, not just simple cloth

The AAAI-26 RGBench paper describes a 6,000+ garment-mesh corpus, a measured material-property pipeline and the GarmentDynamics FEM-based cloth simulator. It evaluates dual-arm grasp, fling and fold and reports bidirectional Chamfer/Hausdorff measures. The paper reports approximately 20% lower simulation error and approximately 3× faster simulation than its evaluated baselines. These are author-reported results, not a Luminous result.

There are now **two distinct public RGBench datasets**, and their scope and rights must not be conflated:

1. **Real-capture evaluation package:** [RGBench-Cloth-Sim2Real-v1](https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1) contains nine captured garment identities, three bimanual actions and 98 capture sessions (the card describes approximately 100 evaluation samples). It contains camera calibration/initial pose, robot joint/end-effector CSVs, segmented cloth point clouds and matching OBJ/USD meshes. The dataset card lists **CC-BY-4.0**. The two non-manifold garments `grey_sunwear` and `khaki_blazer` are excluded from the paper's published baseline results.

2. **Garment asset research subset:** [RGBench/garment_assets](https://huggingface.co/datasets/RGBench/garment_assets) is a separate release with 2,794 rows, 504 classes and approximately 25.5 GB of OBJ and Style3D assets. Style3D files carry measured stretch stiffness, bending stiffness, area density and thickness. Its license is a **project-specific non-commercial research license**, not CC-BY-4.0. The dataset page describes it as a research subset of the industrial asset library; the full 4,000+ industrial models remain contact/licensing-only. Parsing and simulator post-processing scripts are described as forthcoming. Do not infer that 2,794 rows equal 2,794 unique garment designs.

The paper's 6,000+ mesh count also draws on other sources, including ClothesNet; those source datasets must be obtained and used under their own original terms. The **GarmentDynamics simulator remains forthcoming** as a separate open release; the RGBench GitHub repository already supplies an evaluation harness and PyBullet/Isaac Sim wrappers.

**Our immediate goal:** reproduce the real-capture evaluation package and exact metrics first, then inspect the asset research subset under its restrictive non-commercial license to calibrate material/mesh intake. Keep these two lanes separate. No external RGBench data has been downloaded or ingested by Luminous.

The next qualification layer must extend beyond global point-cloud distance:
- local geometry around seams, hems, cuffs, collars, closures and fold lines;
- self-intersection, penetration and invalid-topology counts;
- time-series deformation, phase lag, settling and contact/fold event timing;
- physical parameter calibration for thickness, areal mass, warp/weft stretch, bending and surface friction;
- uncertainty and held-out specimen validation;
- downstream manipulation success, damage and recovery;
- initialization/step time, memory, numerical failures and accuracy/runtime Pareto fronts.

The published bidirectional CD/HD metrics remain necessary for comparability, but no global geometric average may compensate for a catastrophic local seam/topology or dynamics failure.

### ClothDojo — learned trajectory generation, not a substitute for physical truth

The 31 October 2026 [ClothDojo preprint](https://arxiv.org/abs/2611.01190) reports nearly 8,000 successful simulated trajectories for bimanual garment flattening/folding over hundreds of garments. It uses canonical mesh-coordinate cues, a learned data-generation policy and appearance-randomized RGB observations for downstream policy training. It reports improvements over direct human-demonstration training in its evaluated splits, while explicitly leaving unseen-garment generalization as an open problem.

The Luminous research question is whether generated trajectories improve **held-out physical garment performance**, not simply the number of successful simulated rollouts. Appearance randomization must not mutate material mechanics; rollouts must bind exact garment/material/solver/action/seed identity; and simulation success must remain distinct from physical trial success.

### Robotic sewing — a real seam-error comparator

The 21 September 2026 [robotic sewing study](https://doi.org/10.1007/s41315-026-00590-3) reports a single-arm SCARA platform with an unmodified industrial sewing machine, vision-based contour/trajectory generation, demonstration-trained grasp prediction, vision seam tracking and force-regulated tension. It reports seam deviation within ±2.5 mm on its experimental tasks, while noting that limited user intervention remains and that generalization to large/highly elastic fabrics needs more work.

Use ±2.5 mm as a published reference only when the same task geometry, seam type, metrology definition, material and control envelope are matched. It is not a universal seam-quality tolerance or proof of autonomous production.

### Visual garment sorting and fibre-composition sensing are separate tasks

The 30 September 2026 [TexSortML paper](https://doi.org/10.3390/s26196236) contains 751 post-consumer garments in 11 visual categories and three presentation-difficulty stages, with 2,253 annotated frames. Its garment-level five-fold split keeps each specimen's three presentation images together. The directly fine-tuned ViT reports 95.23%, 91.90% and 85.94% accuracy across the three stages, a 9.3 percentage-point drop. ResNet-18 and ResNet-50 report drops of 16.9 and 16.6 points. The authors note the sample is small and class-imbalanced and came from collection containers that had already been pre-sorted at one facility.

Crucially, TexSortML labels **visual garment category and presentation state**, not fibre composition. Do not turn “jacket” or “shirt” into a material assertion.

[NIST NIR-SORT](https://catalog.data.gov/dataset/near-infrared-spectra-of-origin-defined-and-real-world-textiles-nir-sort-a-spectroscopic-a) contributes a separate material-sensing lane: 113 specimens (70 known-provenance known-fibre, 20 known-provenance undyed blends, 11 post-consumer, 12 pre-consumer), nominally seven NIR replicates, benchtop 833–2500 nm and handheld 1550–1950 nm ranges, plus microscopy data. The catalog lists roughly 193 MB spectroscopy and 10.6 GB microscopy.

**Paradigm:** a sorting system should produce separate, calibrated evidence for visual garment family, spectral composition, contamination/material state, and route eligibility. Sensor disagreement or out-of-domain samples should lead to a second measurement or an explicit unknown—not a forced class. The actual sort stream must be evaluated by purity, mass balance and qualified recovery output, not classification accuracy alone.

The associated work is [Symthaea #11743 — TEXTILE-SORT-FUSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/11743).

### Research execution order

1. Reproduce RGBench on the currently public nine-garment evaluation set; preserve its published split and exact two-direction metric.
2. Independently calibrate Symtropy's named cloth profile on physical coupons; use held-out specimens and report uncertainty.
3. Compare generated-data policies (including ClothDojo-style methods where reproducible) on unseen garment and material identities.
4. Reproduce TexSortML's garment-level split and NIR-SORT specimen-level grouping as separate lanes.
5. Add multimodal fusion with open-set rejection, then measure physical sort purity and qualified recovery outputs.
6. Compare robotic seam tracking with the published ±2.5 mm study only on matched tasks and measurements.

No dataset has been declared ingested or reproduced by this specification. Rights/terms, attribution and exact file checksums must be reviewed and recorded before any external data is committed or redistributed.


## 15. Semantic adversarial corpus

A synthetic semantic-control corpus now accompanies the structural schema and example:

`docs/research/fixtures/apparel_engineering_thread_v0_1/semantic_cases.example.json`

It contains 16 negative controls and two positive controls. The negative cases cover pattern-revision drift, unresolved BOM material references, duplicate pattern-piece IDs, measured tests without result evidence, model-to-physical evidence promotion, prototype-to-production scope inflation, material-lot substitutions, unknown composition, visual-category/fibre conflation, recovery without physical output-lot proof, expired NIR calibration, lot over-allocation, unit conversion without a basis, synthetic-to-physical evidence promotion, disclosure violations and repair without dependent requalification.

The positive controls make two things explicit:
- one article can be supported within its own tested scope without claiming batch-wide production capability;
- a sensor can correctly abstain and request a remeasurement rather than force an unsupported material class.

These cases are expected outcomes for a future semantic validator; they are not yet executable tests. The current structural schema alone cannot enforce the cross-field causal/applicability rules. A dedicated semantic validator remains a P0 implementation task.


## 16. Inverse textile design: beyond generative pattern aesthetics

A separate research lane is now tracked in [TEXTILE-INVERSE-DESIGN-001 #11746](https://github.com/Luminous-Dynamics/symthaea/issues/11746).

The field already has real inverse-design work; we must reproduce it rather than imply the area is empty:
- [Data-driven prediction and inverse design of fabric tensile properties via machine learning](https://doi.org/10.1016/j.mtcomm.2025.114433), January 2026, combines finite-element-generated training examples and DNN surrogate models for woven tensile-property prediction/inverse design.
- [Inverse design of personalized textiles with architected curved yarns](https://doi.org/10.1016/j.matdes.2026.116330), July 2026, uses a streamline/spacing-controlled curved-yarn construction, ANN design-space compression and surrogate-driven gradient optimization, with computational and experimental validation in three case studies.
- [Machine Learning-Based Inverse Design for Functional Materials](https://doi.org/10.1002/adfm.75070), 9 April 2026, surveys direct inverse mappings, topology optimization, surrogate and hybrid pipelines, and challenges involving data scarcity and coupled physical constraints.
- The [2026 fabric-design review](https://doi.org/10.1057/s41599-026-08095-x) covers 65 studies through 2024. It reports promising aesthetic-generation and material-property results, but also notes heterogeneity, dataset quality, interpretability and workflow-integration limitations. Its cross-study numerical summaries are not a matched benchmark leaderboard.

The Luminous opportunity is to connect inverse design to calibrated physical models, manufacturing constraints and active experiment design:
1. propose several non-dominated fibre/yarn/fabric architectures for a declared performance envelope;
2. estimate uncertainty and detect out-of-domain candidates;
3. reject candidates that cannot be manufactured on the declared machine/process route;
4. select the next physical coupon test by expected information gain/value-of-information;
5. test candidates and update only from admitted physical evidence;
6. preserve held-out specimens for final assessment.

A surrogate prediction is not a physical material property. The benchmark must report prediction error, uncertainty calibration, physical target hit-rate among top-k candidates, hard-constraint violations, number/cost of experiments and Pareto trade-offs. Compare FEM-only, simple regression, random forest/boosting, Gaussian-process or equivalent uncertainty-aware surrogate, FEM/ML hybrid, direct inverse mapping and expert/space-filling experiment selection under equal data and candidate budgets.

## 17. Design-for-sorting and design-for-disassembly

[DESIGN-FOR-SORTING-001 #11747](https://github.com/Luminous-Dynamics/symthaea/issues/11747) adds an upstream circularity lane.

A November 2026 study, [Design–sorting mismatch limits circularity in fast-fashion garments](https://doi.org/10.1016/j.spc.2026.09.008), analysed 47,522 colour-specific H&M/Uniqlo product variants using a rule-based sorting-compatibility framework. It reports that 50.3% violated at least one study-specific baseline sorting rule (sensitivity range 45.0–64.0%). Main reported barriers included recognition-scope mismatch, multi-fibre blends, low-share components, optical-risk proxies and surface/hidden-material mismatch.

This result is a product-data/rule analysis, not a measured sorting-line yield and not a universal estimate for all garments.

The paradigm is to design around the actual target sensor and recovery plant:
- make fibre composition detectable, including hidden and low-share components;
- disclose and physically verify linings, membranes, coatings, trims and attachments;
- prefer joining methods that can be separated by the declared repair/disassembly route;
- model the damage and labour/tooling required to remove components;
- verify the output feedstock actually meets the downstream plant's acceptance envelope.

Evaluate false-safe design decisions, material identification coverage, disassembly success, labour/tool time, damage, stream purity, mass closure, secondary grade and functional acceptance. A product is not "recyclable" simply because a rule checker says its design is compatible; physical recovery and output-grade evidence remain necessary.

## 18. Current machine-readable research corpus

The benchmark source manifest is now at revision `2026-11-12-r2` with 12 distinct sources. The current source set spans:
- real garment dynamics and cloth manipulation;
- visual garment-category sorting;
- NIR specimen-level and hyperspectral textile composition;
- inverse textile/material design;
- robotic sewing;
- design-to-sorting compatibility.

Each source record distinguishes paper-reported results, publicly downloadable scope, licensing/redistribution status, domain limitations and intended use. The manifest does not claim any dataset has been downloaded or reproduced.

The source split rules are strict: repeated pixels/spectra/frames from the same physical specimen remain in one split; garment identities and material families are held out where possible; simulation outputs never become physical labels; and source-domain transfer is evaluated explicitly.


## 19. Multi-stage factory scheduling and source-load coordination

Symtropy's existing textile scheduler is intentionally a one-stage parallel-machine benchmark. Keep that baseline frozen and extend it through a separate workstream: [Symtropy #4875 — TEXTILE-MULTISTAGE-FACTORY-001](https://github.com/Luminous-Dynamics/symtropy/issues/4875).

The research landscape includes concrete comparators:
- [Dual-resource flexible job-shop scheduling in garment manufacturing (2026)](https://doi.org/10.1080/00207543.2026.2711058): machine/operator assignment, sequence-dependent setup and skills; the reported LCS heuristic is compared with MIP/CP and an analytical upper bound. The reported >200-operation / <23%-upper-bound-gap figures are specific to that study's industrial instances and must be reproduced.
- [Steam-aware multi-objective textile scheduling (August 2026)](https://doi.org/10.1016/j.swevo.2026.102498): Q-learning-driven multi-objective evolutionary search for flexible job-shop scheduling with steam use.
- [Graph RL source-load scheduling for textile production (2025)](https://www.sciencedirect.com/science/article/pii/S2950550X25000275): the authors report 10.1% lower carbon emissions, 2.9% lower energy cost and 1.1% higher makespan in a Shanghai enterprise case; data are reported available on request.
- [Multi-line garment scheduling (2026)](https://doi.org/10.1016/j.procs.2026.02.164): flow-shop, lot-streaming and heterogeneous-workforce assignment, with MILP and a genetic algorithm.
- [Apparel scheduling systematic review (2026)](https://doi.org/10.1108/RJTA-09-2025-0213): 82 studies across 2005–2025.
- [Human-centered garment scheduling (2025)](https://doi.org/10.1007/s11740-025-01349-8): integrates skills and ergonomic considerations into flexible garment manufacturing scheduling.

The next benchmark progression should be:
1. Preserve the existing single-stage exact/dispatch-rule comparison as M0.
2. Add multi-stage route and precedence constraints (cut, sew/assembly, finish, inspect).
3. Add dual-resource constraints (machines plus operator skills/availability), sequence-dependent setup and workload/ergonomic bounds.
4. Couple time-dependent electricity and steam/thermal capacity, freshwater and wastewater constraints to the production plan.
5. Add dynamic order, material, worker, machine and utility disruptions with replayable robust rescheduling.
6. Connect a read-only observed-state interface before considering any operator-approved actuation or plant connection.

Compare exact MILP/constraint programming where tractable, EDD/setup-aware heuristics, the published dual-resource heuristic, genetic and multi-objective evolutionary algorithms, and RL/graph-RL only after the deterministic baselines are frozen. Measure optimality gaps, Pareto hypervolume, service and quality, worker workload/fairness, energy/steam/water peaks, WIP, plan churn, recovery latency and compute budget.

A lower carbon or energy objective cannot compensate for unsafe workload, unqualified process windows, a missed hard delivery commitment, off-spec output or missing environmental evidence. Published case-study savings must not be stitched across different papers into a single SOTA number.

The machine-readable benchmark register has been extended to 18 sources at revision `2026-11-12-r3`; it records scheduling literature separately from cloth physics, material sensing, inverse design and robotic sewing.


## 20. Executable apparel fixture qualification

A standalone Rust harness has now been added:

- `tools/apparel-benchmark-harness/Cargo.toml`
- `tools/apparel-benchmark-harness/src/main.rs`
- `.github/workflows/apparel-engineering-fixtures.yml`

It uses a focused workspace so it does not depend on the broader Symthaea workspace manifest glob. The `jsonschema` crate is pinned to `0.59.0` with default features disabled. The auditor:
1. requires the Draft 2020-12 dialect;
2. rejects all `$ref`, `$dynamicRef` and `$recursiveRef` to keep validation self-contained and prevent external reference reads;
3. validates the schema document against the Draft 2020-12 meta-schema;
4. validates the synthetic apparel example using the Draft 2020-12 validator;
5. checks cross-field invariants such as unique object IDs, BOM-to-material references, measured-test evidence, design-revision freshness, claim/evidence binding and the synthetic-not-physically-qualified boundary;
6. checks the 30-case semantic-intent corpus and 18-source benchmark register for internal consistency;
7. emits an exact-head, content-hashed JSON receipt with an explicit claim ceiling.

The focused test suite contains structural negative controls (unknown property and missing pattern-piece ID) and semantic negative controls (unknown BOM material, modelled drape promoted to physical claim, measured test without an evidence artifact, stale geometry projection and synthetic fixture promoted to physical qualification).

**Important execution boundary:** the 30 semantic cases remain expected outcomes and descriptions, not 30 executable mutation tests. The harness executes selected adversarial mutations directly in unit tests and checks the remaining case corpus structurally. The receipt records `intent-corpus-validated-not-executed` for that distinction. A follow-on should convert the highest-risk remaining cases into typed mutation operations and executable golden fixtures.

The GitHub workflow is triggered by relevant research/harness paths, checks the exact commit, uses pinned Rust 1.96.0, generates a separate lockfile for the isolated harness, runs tests, Clippy and formatting, then writes/uploads a content-addressed receipt. No hosted-runner result is claimed until the workflow has executed on the exact PR head.

Even a passing receipt validates the declared schema and synthetic contract only. It does not validate dataset rights, scientific predictions, physical textile properties, fit, quality, manufacturing yield, recycling yield, safety or regulatory compliance.


## 21. Pattern-to-machine-program interoperability

A newly identified gap is the lack of a complete semantic bridge from a sewn 2D pattern or knitted garment design through 3D product geometry to a machine-specific executable program. Two important open/standardized primitives already exist, but they solve different layers:

- **ASTM D6673-10 / DXF** provides a 2D sewn-pattern exchange convention and associated grade-rule exchange. Its own scope excludes pattern-piece relationships and 2D-to-3D sewn-pattern geometry correspondence, and does not carry numerical cutter/plotter instructions or full marker/spreading/product-specification data. It is therefore a useful CAD interoperability baseline, not a full garment engineering twin. See [ASTM's standard listing](https://store.astm.org/d6673-10.html).
- **Knitout v0.6** is a public, machine-independent, UTF-8 line-oriented representation of low-level knitting instructions. It is an intermediate instruction format for design tools, simulators and machine backends, not a high-level garment specification or guarantee that every machine supports every operation. See the [Knitout v0.6 specification](https://textiles-lab.github.io/knitout/knitout.html) and [format repository](https://github.com/textiles-lab/knitout).
- **Textiles Lab's parametric sweater** is a concrete public reference that maps body measurements/ease into a parametric complete-garment pattern and Knitout. Its README says it was developed for and tested on a Shima Seiki machine; it references Kniterate conversion and says STOLL backend support was under development. This is a comparator with a declared machine boundary, not universal knitting interoperability. See [parametric-sweater](https://github.com/textiles-lab/parametric-sweater).

The architecture and two new work items are:
- [Symthaea #11750 — APPAREL-INTEROP-001](https://github.com/Luminous-Dynamics/symthaea/issues/11750): semantic preservation across 2D pattern, 3D garment and machine-program revisions.
- [Symtropy #4878 — TEXTILE-KNIT-PREFLIGHT-001](https://github.com/Luminous-Dynamics/symtropy/issues/4878): deterministic Knitout parsing and preflight against versioned machine capabilities.

The new source register revision is `2026-11-12-r4` with 21 sources. It explicitly marks ASTM text as copyrighted and requires checking the parametric-sweater repository/code/asset license before reuse.

This interoperability work should preserve source-native formats and add typed, versioned sidecars for seam topology, piece-to-piece relationships, pattern-to-3D correspondence, machine profile, backend/compiler, program identity and actual execution evidence. Unsupported semantics must be reported as unsupported/lossy, never silently dropped.

The key benchmark is not merely whether a file opens. It is whether the supported product semantics survive a deterministic round trip and whether the named machine profile accepts the resulting program without unsupported operations or hidden conversion.

## 22. Human–AI fashion co-design and designer agency

A dedicated workstream is now open: [FASHION-CO-DESIGN-001 #11753](https://github.com/Luminous-Dynamics/symthaea/issues/11753). This expands the programme from product engineering into the creative process itself, while preserving the distinction between aesthetic ideation, editable design geometry, manufacturing feasibility and physical product validation.

Relevant 2026 comparators include:
- [Human–AI co-design for sustainable fashion](https://doi.org/10.1177/24723444261488123), 23 September 2026: the study reports 4,200 design records, 350 expert-rated designs, 864 consumers and 426 production patterns. Its evaluation spans novelty, colour harmony, production complexity, waste/circularity potential and brand fit, and studies consumer response/willingness to pay. Reproduce its exact sampling, tasks, rating rubric and splits before comparing; production pattern records are not proof of physical manufacture.
- [Sketch-to-parametric garment generation with dynamic fit evaluation](https://doi.org/10.1177/15589250261441600), 28 April 2026: reports 0.87 mm Chamfer distance, 91% detail matching, 45 ms for a 100k-vertex mesh and a designer task-time reduction from 390 to 55 minutes under its own evaluation. These are study-specific results. Local seam, notch, grain, pattern and assembly errors need their own metrics; mesh similarity alone is insufficient.
- [EASE spatially varying garment fit and 3D-to-2D pattern extraction](https://doi.org/10.1016/j.cag.2026.104644): a comparator for local ease fields, anisotropic ease and local-global fitting/pattern extraction. Computational ease is not wearer comfort.
- [Controllable fashion diffusion](https://doi.org/10.1002/cav.70155): pose/edge conditioning, designer-refined masks and controllable concept/prototype generation. It is an image-ideation comparator, not a pattern/production comparator.
- [Multimodal sketch/text-to-pattern generation](https://doi.org/10.3389/frai.2026.1828627): landmark-guided pattern generation and sewing-detail representation with CLO3D digital validation. Digital validation does not establish physical fit or manufacturing acceptance.

The benchmark should compare five workflows on identical design briefs, reference rights, time budgets and participant conditions:
1. Expert designer without AI.
2. Designer using conventional CAD/3D tools.
3. Designer using image-only generative AI.
4. Designer using parametric pattern/geometry generation.
5. Designer using the full Luminous thread: concept reasoning, geometry, material/process feasibility and evidence/provenance.

Score distinct outcome families rather than a single composite:
- aesthetic/creative quality and brief adherence: blinded expert ratings, novelty, silhouette, colour/material harmony, brand/style fit and inter-rater reliability;
- designer agency: editability, reversibility, ability to override, suggestion acceptance/rejection, perceived authorship, control over constraints and diversity of explored options;
- engineering: valid patterns, grain/notch/seam semantics, local ease/fit, assembly feasibility, machine/material compatibility and manufacturability;
- physical outcomes: pattern-to-article dimensional deviation, drape/fit under named conditions, seam quality and physical test evidence;
- sustainability/circularity: measured material use, waste, route-specific sortability and repair/recovery feasibility;
- workflow efficiency: time to viable concept, time to accepted production package, edit/rework count and physical sample iterations;
- rights/provenance: source asset permissions, source inspiration records, AI assistance disclosure, similarity/near-duplicate review and full revision lineage.

FID, CLIP similarity or a consumer preference score must not be used as a substitute for editable pattern validity or engineering qualification. Cultural and brand preferences are task-specific and should not be flattened into a universal aesthetic score.

The new source register is revision `2026-11-12-r5` with 26 distinct sources. Its rights fields distinguish article access from permission to reuse datasets, generated assets, code or training material; no dataset is claimed ingested or licensed for redistribution.
## 23. Technical textiles for digital construction

The textile platform also opens a construction-engineering vertical. A new owner is open in [Symtropy #4880 — TECH-TEXTILE-CONSTRUCTION-001](https://github.com/Luminous-Dynamics/symtropy/issues/4880). This is a distinct material/structural qualification domain: an apparel textile property profile cannot be reused as a TRC/CRC, geotextile or architectural-membrane design allowables set.

Recent construction research gives concrete baselines:
- [Textile- and carbon-reinforced concrete in digital fabrication](https://doi.org/10.1617/s11527-026-03285-0), published 25 September 2026, compares extrusion, non-planar deposition, robotic spraying, digital casting, flexible/printed formwork, particle-bed and injection routes with flexible textiles, stabilized grids, mineral-impregnated rovings and spatial preforms. Its synthesis stresses matrix penetration, bond, cover/position, access, inspectability, automation and structural evidence; process-parallel integration can improve performance while also creating voids and inspection barriers.
- [Concrete printing with textile reinforcement: approaches, classification, and testing](https://doi.org/10.1016/j.cemconres.2026.108340), October 2026, classifies reinforcement as single-step during shaping versus two-step before/after shaping and connects placement sequence to bond, anisotropy and mechanical test methods.
- [Automated hybrid textile + SHCC reinforcement](https://doi.org/10.1016/j.cemconcomp.2025.106361), January 2026, demonstrates a three-channel nozzle for mortar and carbon-textile/SHCC bonding-layer placement and reports flexural/ductility/crack-resistance improvements under its specific specimen and load configuration.
- [Robotic rebar insertion and grouting](https://doi.org/10.1016/j.addma.2026.105078), 5 February 2026, combines surface detection and force control with a robot arm for vertical reinforcement insertion/grouting and characterizes pull-out/splitting/rebar-fracture modes. It is an adjacent robotic comparator, not textile placement.
- [In-process mesh reinforcement for printed concrete](https://doi.org/10.1016/j.autcon.2026.106958), July 2026, reports a dedicated printhead that inserts mesh into concrete layers, with improved green strength and beam deformability in its configuration.
- [Coordinated robotic printing and spatial steel reinforcement/welding](https://doi.org/10.1016/j.autcon.2026.107113), October 2026, presents a two-robot mock-up integrating concrete printing, spatial steel reinforcement and arc-stud welding. It demonstrates coordinated fabrication steps, not universal structural qualification.
- [Textile waste reinforced cementitious composites review](https://pmc.ncbi.nlm.nih.gov/articles/PMC13476324/), 2026, synthesizes 23 studies through February 2026. It reports promising flexural/toughness/post-cracking/thermal behavior in some material systems, but compressive strength can fall due to porosity and weak fibre–matrix interfaces. Long-term durability, full-scale evaluation and comprehensive LCA remain gaps.

The design paradigm is reinforcement architecture as a **co-designed, robotically placeable, inspectable and qualified structural subsystem**. Co-optimize textile/yarn/grid architecture, matrix rheology, aggregate size, placement sequence, tension, curvature, laps/anchors, local cover, bond and inspection access. A cloth/XPBD solver is not a substitute for fibre–matrix interface, concrete rheology or structural mechanics.

The proposed benchmark sequence starts with low-risk coupons and small non-load-bearing panels: placement-accuracy coupons, matrix penetration/bond coupons, and a small curved panel. Report target-to-as-built placement and orientation error, cover/spacing, waviness/sag, damage, matrix penetration and void fraction, bond–slip/pull-out, crack width/spacing, post-crack response, durability, NDT detectability and mass/energy per declared functional unit. Compare conventional pre-installed reinforcement/casting, in-layer textile placement, post-installed reinforcement, robotic spraying/winding and hybrid textile+SHCC only under matched geometry, material, cover, matrix, loading and conditioning.

The source register is now revision `2026-11-12-r6` with 33 sources. Structural use requires independent material and component testing, durability evidence, code review and authorized professional sign-off; this research issue does not qualify a load-bearing building product.

## 21. Sorting must be measured as a physical process, not a classifier score

[TEXTILE-SORT-TO-STREAM-001 #11767](https://github.com/Luminous-Dynamics/symthaea/issues/11767) now separates recognition from physical routing and downstream material acceptance.

The 2026 systematic review [Intelligent and Automated Technologies for Textile Recycling Pre-Processing](https://www.mdpi.com/2227-7080/14/4/200) included 21 primary studies from 2020–2025; it reports 19 studies on identification, only four on explicit physical sorting and three on contaminant detection/removal. These are review-sample counts, not a census of industrial deployments. The research gap is nevertheless important: better classification accuracy does not prove that a sorting cell generates cleaner, accepted feedstock.

Additional 2026 comparators:
- [Effects of surface contamination on automated textile sorting using NIR-spectroscopy](https://doi.org/10.1016/j.wasman.2025.115227) reports increased misclassification risk from moisture/heavy contamination and improved accuracy when classifiers are tailored to textile fractions.
- [Challenges and spectra interpretability in textile sorting: NIR hyperspectral images and chemometrics](https://doi.org/10.1016/j.saa.2025.126665) studies material discrimination and minor elastane detection.
- [NIST NIR-SORT 2.0](https://www.nist.gov/news-events/news/2026/03/nist-releases-nir-sort-20-enhancing-model-validation-textile-feedstock-0) expands blend, dye and instrument diversity.
- [Mechanical recycling and industrial reintegration of post-industrial stretch textile waste](https://doi.org/10.1016/j.jclepro.2026.149662), 22 October 2026, examines a specific 68 wt% polyester / 29 wt% viscose / 3 wt% elastane stream. It is a specific post-industrial route study, not a universal post-consumer benchmark.
- [Untangling textile waste flows](https://doi.org/10.1016/j.resconrec.2026.109030), 30 August 2026, reports a Dutch clothing-waste material-flow scenario in which reuse and recycling could reach 40% under specified assumptions (22±9% reuse, 5±2% closed-loop and 13±5% open-loop recycling). This is a scenario estimate, not an observed achieved national rate.

Benchmark the full process separately:
1. identification (composition, colour, hidden components, contamination);
2. route selection with open-set uncertainty and abstention;
3. physical separation and verified collection;
4. output-stream purity, yield, damage and mass closure;
5. downstream recycler/process acceptance and secondary functional grade.

Report false-safe routing, over-reject, abstention, measured purity, accepted output mass per input tonne, downstream rejection and cost/resource intensity. A syntactically valid route decision or actuator command is not evidence of a successful physical output.

## 22. Format interoperability: a semantic IR, not a legacy file as the canonical model

[TEXTILE-INTEROP-001 #11768](https://github.com/Luminous-Dynamics/symthaea/issues/11768) addresses file and semantic interchange across pattern CAD, 3D geometry, simulation and manufacturing.

Important standards correction:
- ASTM D6673-10 was withdrawn in 2019 according to ASTM's current textile standards index. It is a legacy 2D pattern/grade-rule exchange comparator, not a current active standard. Its own scope excludes 2D/3D geometry correspondence and several manufacturing/product-data areas.
- NIST's 1996 [STEP/EXPRESS apparel pattern information-model research](https://www.nist.gov/publications/data-sharing-implementation-based-information-model-apparel-pattern-making) is a useful neutral-information-model precedent, not proof of a current deployed industry-wide apparel exchange standard.
- [Knitout v0.6](https://textiles-lab.github.io/knitout/knitout.html) is a machine-independent low-level knitting instruction format. It still requires a supported backend and target-machine capability checks; successful parsing does not authorize or prove physical execution.

The architectural goal is a Luminous semantic garment IR that preserves product revisions, 2D pattern pieces, notches, grain, seams, allowances, grade rules, 2D/3D correspondence, units/frames, BOM/material identities, process profiles and evidence references. Format adapters may import/export legacy or vendor-specific representations. Unsupported critical semantics must be preserved as unresolved or cause refusal; no silent field loss is permitted.

The benchmark should test round-trip semantic retention, ID/revision/frame/unit integrity, geometric deviation, grade-rule preservation, machine/backend compatibility and deterministic exact-head file receipts. File-open success is not semantic interoperability.

## 23. Current expanded benchmark register and semantic coverage

The source manifest is now revision `2026-11-12-r8` with 41 unique research/reference entries across:
- cloth simulation, real garment manipulation and sewing;
- visual and NIR/Hyperspectral fibre/material identification;
- sorting, contamination, minor-component detection and recovery routes;
- inverse textile design and fabric engineering;
- apparel scheduling and human-centered production;
- technical textiles and construction;
- pattern interchange and machine-instruction formats.

The semantic corpus contains 30 expected cases (26 negative and four positive). The harness now records explicit case coverage: eight cases are mapped to direct executable negative-control tests; 22 remain intent-only/not yet executed. Additional cross-field Rust unit tests cover operation-sequence collisions, measured-material-property evidence, measured-composition evidence, bound-lot quantities, claim evidence resolution and a positive physical-test binding control. Do not report the whole 30-case corpus as fully executed.
