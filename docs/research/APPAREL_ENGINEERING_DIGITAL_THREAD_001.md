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
