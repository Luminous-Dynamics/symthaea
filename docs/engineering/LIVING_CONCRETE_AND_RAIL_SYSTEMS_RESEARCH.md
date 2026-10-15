# Living Concrete and Better Rail Systems: Research Program

**Status:** Research plan and hypothesis registry. This document does not establish a concrete formulation, biological safety, rail-system fitness, code qualification, or authority to construct, inspect, maintain, or operate infrastructure.

**Date:** 2026-10-15

## Executive direction

Yes: Symthaea should help search for *living or bioactive concrete*, but it should not begin by optimizing bacterial growth or inventing a biological recipe. It should compare damage-prevention and self-healing concepts against matched conventional controls, then select candidates only after independent tests show improved performance under the intended exposure conditions.

In parallel, the highest-value near-term railway work is likely **condition-based maintenance and whole-network reliability**, not a speculative new rail alloy or a general-purpose AI dispatch controller. Federal Railroad Administration (FRA) reports RR 26-01 and RR 26-02 describe development and implementation of track-geometry forecasts, including statistical point/segment approaches and the value of encoding maintenance events as explanatory variables. These methods provide a strong baseline Symthaea should attempt to match before claiming additional predictive value.

## Program ownership

- **Symthaea:** research question generation, candidate representation, calibrated property-specific prediction, candidate ranking, uncertainty, experiment manifests and claim boundaries.
- **Symtropy:** deterministic counterfactual scenarios where its documented physical capabilities are sufficient. Its existing rigid-body engine is not presumed to be a validated cementitious fracture, geotechnical, finite-element, wheel–rail contact, or rail-track degradation solver.
- **Sol-Atlas:** read-only geospatial context: climate, terrain, catchment, flood exposure, materials sources, communities, corridor dependencies and observed asset-condition projections with source/time/resolution metadata.
- **Mycelix:** canonical provenance/source rights, review lineage, materials/maintenance history, procurement and commons governance. A vote is not an engineering test.
- **Laboratories, operators, licensed engineers and authorities:** choose and interpret methods, qualify materials, approve designs, and authorize field trials or operational changes.

Compose existing work rather than creating duplicate roots:
- Symthaea [TRANSPORT-ENG-001 #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244), [owner coverage PR #6246](https://github.com/Luminous-Dynamics/symthaea/pull/6246), and [TRANSPORT-RAIL-001 #6248](https://github.com/Luminous-Dynamics/symthaea/issues/6248) for mode-specific rail dynamics/qualification. The proposed track-geometry forecasting benchmark is a read-only condition-maintenance input and must not duplicate that vehicle/guideway owner.
- Mycelix [Living-City concept contracts PR #5018](https://github.com/Luminous-Dynamics/mycelix/pull/5018) and [source/evidence contracts PR #5029](https://github.com/Luminous-Dynamics/mycelix/pull/5029).
- Sol-Atlas [evidence-bounded transport projection PR #111](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/111).
- Symtropy [engineering digital-twin boundary #1467](https://github.com/Luminous-Dynamics/symtropy/issues/1467).

## Canonical field-ownership map

This table is an adapter/research checklist, not a proposal to instantiate parallel Rust types. Verify each interface against the current qualified successor before coding; some historical MAT/split implementations are explicitly not current qualified dependencies.

| Fact needed | Existing owner to compose | Contribution of this research | Must not duplicate |
|---|---|---|---|
| Material/formulation identity | MAT-007 (#4336) and its repaired/current successor | State which candidate/material the experiment concerns | Material identity type |
| Conditioned property, unit, method and uncertainty | MAT-008 (#4339), SE-SEM-001 (#4861), FIELD/SE-OBS | Name application-specific outcome planes and measurements | Quantity/unit/measurement system |
| Campaign/objective and trade space | MAT-009 (#4340), qualified MAT-OPPORTUNITY/MAT-PORTFOLIO successors | State baselines, objectives and Pareto trade-offs | Campaign or generic optimizer |
| Null/failure/OOD results | MAT-010 (#4341) | Keep unsuccessful, contradictory and out-of-domain evidence visible | Search-memory store |
| Physical specimen/process lineage | MAT-013 (#4344) | Require sample, batch, precursor, protocol and scale identity for physical work | Physical lineage from literature rows |
| Human-authorized experiment proposals | MAT-014 (#4345) | Restrict active-learning output to reviewable proposals | Physical authorization from model confidence |
| BIO-CEM observations, biological state, output planes, controls, extraction and comparability | BIO-CEM #5751, corpus #5753, contract/fixture owner #5767 | Define the experiments and adversarial cases that matter | Second living-concrete candidate ontology |
| Materials mechanism hypotheses | MAT-HYPOTHESIS #5254 | Define falsifiable candidates and competing explanations | Common-mechanism claims from similarity |
| Evidence derivation and replication | ENG-EVID-INDEP #4918, ENG-EVID-CLAIM #5771, BIO-CEM profile #5757 | Preserve controls, lineage, competing explanations and claim ceilings | Local evidence graph or scalar “healing score” |
| Study/dependency split groups | ENG-EVID-INDEP #4918 and a current qualified split-contract successor consumed through #5756 | Require whole-study/lineage holdouts and disclose common modes | Local split engine; distinct IDs do not prove independence |
| Actual living-material lot/custody/lifecycle | Mycelix BIO-MFG #3113 plus manufacturing/circularity owners when available | Identify future operational provenance needs | Mycelix batch/lifecycle records |
| Road/rail field inspection and calibration | FIELD / FIELD-001A / SE-OBS (#3589/#3633/#3695) | Treat track geometry as a distinct physical observation plane | Generic infrastructure-observation ontology |
| Units, time and spatial frames | SE-SEM-001 #4861 and relevant geospatial/frame owners | Declare units, temporal coverage and coordinate semantics | Ad hoc unit/time/frame system |
| Rail dynamics, braking and qualification | TRANSPORT-RAIL-001 #6248 composed under #6244/#6246 | Keep track-condition forecasting as read-only maintenance input | Rail dynamics solver, signaling system or approval authority |
| Maintenance/work-order history | Mycelix transport operations #3395 and canonical maintenance owners | Use verified maintenance events as forecast covariates | Operational event ledger |
| Spatial projection | Sol-Atlas transport projection PR #111 / SOL-ATLAS-009 #106 | Show condition, source freshness, uncertainty and dependencies | Competing logistics database |
| Counterfactual simulation | Symtropy ENG-SYNTH-TWIN #1467 and its credibility gates | Compare bounded hypothetical cases after model-suitability review | Synthetic trajectory as field observation or qualification |

The ownership gate #5768 requires a demonstrated semantic gap before adding any new common production type. This research PR is documentation-only and deliberately follows that rule.

## A. Living concrete / bioactive cementitious materials

### A1. Why it is plausible

Cementitious materials can undergo limited autogenous healing: unhydrated cement and carbonate precipitation may close some small cracks when water and suitable conditions are present. Additional concepts attempt to improve this behavior using fibers that limit crack opening, crystalline or mineral admixtures, healing-agent capsules, and microbial biomineralization.

Microbial approaches often aim to precipitate calcium carbonate at a crack. Dormant spores or other protected biological systems may be carried in a porous aggregate, capsule, hydrogel or other encapsulant and reactivate when the crack environment provides suitable moisture and substrates. The goal is to seal a transport path and improve durability; it is not safe to assume that visible crack closure equals restored load capacity.

Reviews published in 2024–2025 identify promising laboratory results but also describe major comparability and scale-up gaps: strain survival, alkaline/desiccating conditions, nutrient access, crack width and depth, encapsulation damage during mixing, cost, environmental conditions, inconsistent measurements, and limited long-term field evidence. Different studies often test different formulations and endpoints, so reported percentages must not be combined into a universal performance claim.

### A2. Candidate tracks: establish comparators before optimizing microbes

**Track 0 — Conventional baseline and autogenous-healing control**
- A well-characterized concrete/mortar reference using the project's currently accepted cement, aggregates, reinforcement/fibers and curing regime.
- A matched control to quantify ordinary autogenous crack closure.
- Internal curing as a damage-prevention comparator. FHWA reports that internal curing can reduce early-age shrinkage cracking and may extend bridge-deck service life; its effects and any life-extension claim remain application/context specific.
- A fiber-reinforced, controlled-crack-width comparator when appropriate.

**Track 1 — Non-living autonomous-healing control**
- Crystalline/mineral admixture or encapsulated non-biological healing agent.
- Measure whether the healing system changes workability, setting, strength, shrinkage, durability and repeatability before crediting any crack-sealing benefit.
- This is an essential comparator because it may deliver some intended repair benefits without viable microbes or environmental persistence.

**Track 2 — Encapsulated microbial mineralization**
- Evaluate published, appropriately characterized research systems using non-pathogenic, non-engineered strains selected by qualified microbiology partners and documented risk assessment.
- Focus on compatibility with cementitious alkalinity, moisture limitation, temperature cycling and the actual crack microenvironment.
- Treat carrier/encapsulant and nutrient/precursor choices as separate factors. Encapsulation can improve survival but can also introduce voids, alter strength or fail during mixing.
- Evaluate the biomineralization pathway and its by-products. Ureolytic pathways, for example, can create ammonia-related environmental concerns; a high carbonate yield alone is not an adequate environmental score.
- Do not proceed to deliberate environmental release or engineered-strain deployment on the basis of a software ranking.

**Track 3 — Biological/fiber/mineral hybrid**
- Investigate whether crack-width control, biological or mineral healing, and reduced early-age cracking are complementary.
- Use a predeclared factorial or other justified design to test interactions instead of multiplying the best result from separate studies.
- Treat fungal systems and novel engineered living materials as research hypotheses until material-level and biosafety evidence justify further investment.

### Evidence maturity and cross-study comparability (2025–2026 update)

Recent reviews sharpen the roadmap but do not remove the need for application-specific evidence. A 2026 review classifies autogenous healing as the most established approach, capsule/vascular systems as more advanced experimental or pilot-stage approaches, and many microbial, nanomaterial and shape-memory approaches as developmental because of long-term durability, healing-agent survival, cost and scale-up gaps. A 2025 review reports promising laboratory and selected field applications while still identifying gaps in standardized testing, long-term stability, compatibility and cost. Treat maturity labels as literature-triage signals, not qualification or proof that a specific formulation is fit for a bridge deck, road slab, sleeper or other asset.

A separate inter-laboratory study of capsule-based healing found that specimen pre-cracking procedures and waterproofing around the test zone affect measured outcomes; in that study, sealing-efficiency comparisons were more consistent across laboratories than raw reference-specimen flow rates. The result is method-specific, but it demonstrates why test protocol identity and boundary preparation belong in the evidence, not in footnotes.

**Keep evidence maturity separate from technology maturity.** Grade each *claim × property × exposure regime* independently; never award one composite maturity label to a formulation when its endpoints have different support:

- **E0 — Reported lead:** literature claim or secondary-review summary; protocol, raw observations or comparator may not yet be independently reconstructed. Useful for hypothesis generation only.
- **E1 — Controlled single-laboratory evidence:** identifiable specimens, formulation/batch, baseline, damage protocol, conditioning, outcome method, units, replicates and failures are available; results remain limited to that study and exposure.
- **E2 — Independently replicated evidence:** preregistered outcome and analysis, a frozen protocol, appropriate blinded assessment where feasible, and replication by an independent team/laboratory. Method inter-comparability is reported rather than assumed.
- **E3 — Owner-authorized, application-relevant pilot:** a non-critical or explicitly approved asset/use case, pre-existing baseline, exposure-relevant monitoring, qualified inspection, and a documented stop/withdrawal plan. This remains a bounded pilot, not general service-life validation.
- **E4 — Sustained field evidence:** independent review of maintenance, durability and adverse outcomes across a declared duration and exposure envelope. This still does not replace the responsible engineer's asset-specific design, standards and authorization requirements.

The grade is attached to the supported endpoint, not the paper or candidate as a whole. For example, crack closure may have E2 support while permeability recovery is E1, mechanical retention is E0 and biosafety is unresolved. A missing endpoint stays missing; it is not imputed from an adjacent endpoint.

Before comparing rows from different studies, normalize or explicitly stratify by initial crack width and its distribution, specimen geometry and scale, age at pre-cracking, crack induction/loading method, curing and healing environment, wet/dry or thermal cycle, healing duration, transport-test geometry/boundary sealing, mechanical test and loading direction, comparator composition, replicate count, uncertainty and missingness. If two protocols cannot be compared credibly, retain separate strata rather than manufacturing a pooled ranking. Preserve reported outcomes as reported; derived unit conversions and computed ratios must retain method/version and calculation lineage.

This supports a practical decision sequence: **established low-risk comparator first; non-living self-healing comparator second; microbial or more complex hybrid candidates only when they add measured value without unacceptable fresh-state, intact-strength, lifecycle or biosafety penalties.** Model-generated rankings remain proposals for qualified review, not material recipes or deployment authorization.

### A3. Candidate data contract (reference composition, not a duplicate ontology)

This is a requirements checklist for composing existing owners, not permission to add a parallel Rust candidate/material/evidence ontology. Before implementation, complete the field-by-field zero-duplication audit required by [BIO-CEM-001A1 #5767](https://github.com/Luminous-Dynamics/symthaea/issues/5767) and [MFG-FOUNDATION zero-new-type gate #5768](https://github.com/Luminous-Dynamics/symthaea/issues/5768). Reuse MAT-007/008/009/010/013/014, MAT-HYPOTHESIS #5254, FIELD/SE-OBS, ENG-EVID-INDEP #4918, ENG-EVID-CLAIM #5771, and Mycelix BIO-MFG/lifecycle references where they own the corresponding facts. If a canonical owner represents a field, reference it instead of copying the value into a new type. The existing BIO-CEM campaign #5751 and literature corpus #5753 are the canonical research workstreams.

Each candidate/evidence projection should resolve, where applicable, to:

- Stable candidate/formulation subject and version through the canonical material subject owner.
- Material scope: mortar, cast concrete, precast, repair mortar, road slab, bridge deck, sleeper, culvert or another explicitly defined use.
- Full composition/source/batch provenance and component reuse rights.
- Healing class: autogenous, internal curing, non-living encapsulated, microbial, fiber/mineral hybrid, or other named mechanism.
- Whether a viable organism is present; taxonomic/strain identity and qualified biosafety assessment references where relevant; whether it is engineered; contained-use/release status.
- Encapsulant/carrier, nutrient/precursor class, expected activation conditions and declared by-product risks.
- Fresh-state constraints: workability, setting time, air content/voids, mixing/survival damage and construction compatibility.
- Target response with unit, method, conditioning, specimen and replicate IDs.
- Model version, measured-versus-predicted status, calibration domain and uncertainty.
- Standards edition/project acceptance criteria supplied by the responsible engineer; no unversioned thresholds.

Do not store a generic scalar called “healing efficiency” as the only endpoint. Distinguish crack-width closure, reduction in water permeability, reduced ingress of relevant agents, retained/recovered mechanical response, durability through cycles, and the ability or inability to heal a reopened crack.

### A4. Experimental design

Use a staged, preregistered comparison:

1. **Question:** define whether the target is crack sealing, durability preservation, corrosion-risk reduction, or another measurable endpoint.
2. **Matched baseline:** use the same matrix, specimen geometry, curing history, loading and environmental cycle across controls and candidates wherever feasible.
3. **Positive and negative controls:** ordinary autogenous healing; uncracked reference; deliberately damaged but unhealed control; a candidate with the healing mechanism disabled or absent where feasible; image/measurement negative controls.
4. **Replicates and blind analysis:** plan replicate counts with a statistician/materials engineer; blind specimen labels during crack-image and outcome assessment where practical.
5. **Environment:** test the intended service envelope for wet/dry cycles, relevant temperatures, load cycling, water chemistry, exposure chemistry and crack widths. Don't choose conditions after seeing which candidate wins.
6. **Measure more than appearance:** crack width and closure distribution, water permeability/penetration, healing depth, mechanical response and stiffness recovery, cyclic/reopening behavior, durability, reinforcement corrosion indicators where relevant, leachate/by-products, cost and life-cycle impacts.
7. **Microbiology:** where viable organisms are involved, have the qualified lab assess viability and activity under the cementitious conditions, identity/purity, persistence, and relevant worker/environmental exposure. Microbial presence or a positive staining image does not prove structural benefit.
8. **Replication and uncertainty:** report per-specimen results, variability, failures, outliers, missing observations and uncertainty intervals; retain unsuccessful formulations.
9. **Independent verification:** an independent checker validates input manifests, units, calculations and held-out assignments, not merely rerunning the same candidate generator.
10. **Promotion:** lab promising → independently replicated lab result → application-specific engineering review → bounded, owner-authorized monitored pilot → extended field follow-up. No direct jump from model to infrastructure deployment.

### A5. Predeclared performance gates

Every candidate is subject first to mandatory engineering and environmental requirements; a Pareto ranking must not trade away a safety constraint.

Relevant measurements, selected for the application, can include:
- compressive/flexural/tensile response as appropriate, stiffness, shrinkage and creep;
- crack-closure rate/distribution and permeability reduction under declared exposure;
- retained response after repeated wet/dry, thermal, load or chemical cycles;
- durability, bond/compatibility with substrate and reinforcement corrosion indicators;
- fresh-state performance and placement/curing robustness;
- biological viability/containment, leaching/by-products, worker exposure and end-of-life behavior;
- material and maintenance cost, greenhouse-gas/lifecycle effects, water/material demand and repairability.

Use minimum practically important improvement margins defined before running the blinded evaluation. Require confidence intervals and predeclared non-inferiority bounds for essential baseline properties, along with a cap on tolerable failure or variance. Do not specify numerical thresholds until the target application, test method, sample design and acceptance authority are known.

### A6. Environmental and worker safety

Potentially living construction materials span formulation, mixing, curing, service life, demolition and end-of-life. Before any open field deployment, consider strain identity and provenance, unintended persistence, genetic stability where relevant, horizontal gene transfer where relevant, interaction with soil/water microbiomes, runoff/leachate, by-products, demolition dust, aerosols, worker protection and waste treatment. Apply qualified institutional and jurisdictional biosafety review. Favor contained laboratory evaluation and non-engineered, appropriately screened systems for an initial research path.

This is preventive risk assessment, not a claim that all proposed strains are harmful. Conversely, ordinary use of a familiar organism is not evidence that a viable strain in a long-lived construction product is automatically safe.

### A7. Evidence model and ML policy

Symthaea should begin with a transparent tabular baseline (for example a simple regularized regression or a tree-based comparator appropriate to sample size) before testing HDC/CfC-based representations. HDC similarity is useful for candidate retrieval, not as a substitute for physical property prediction.

- Split data by study/site/formulation/batch, not random rows alone, to reduce leakage.
- Publish performance by study and exposure regime, not only aggregate error.
- Distinguish interpolation from transfer to a new cement, aggregate source, climate or crack regime.
- Calibrate prediction intervals and measure performance on an untouched holdout.
- If evidence coverage, calibration or domain checks fail, return **insufficient_evidence** and preserve the reason.
- Use active learning to propose the next most informative *approved, bounded* experiment; never let the model autonomously order cultures, release organisms or change a field material specification.

### A8. First dataset candidate: krkCMd crack-width measurement benchmark (admission gate pending)

A concrete first benchmark candidate is Jakubowski & Tomczak's *Dataset for developing deep learning models to assess crack width and self-healing progress in concrete* (Scientific Data, 2025): https://doi.org/10.1038/s41597-025-04485-z. The associated current Zenodo record is https://zenodo.org/records/14568863 (DOI: https://doi.org/10.5281/zenodo.14568863). Its description lists **19,098 brightness-profile records**, manual reference crack-width measurements, and comparison outputs from a deep-learning metasensor and an analytic edge detector. It concerns high-strength concrete undergoing **autogenous** self-healing, with image series across stages. The listed CSV is approximately 56.3 MB; the image archive is approximately 38.6 GB and the scripts archive approximately 273 kB.

The current Zenodo landing page (record version 1.1; opened 15 October 2026) labels the record “Dataset Open” and lists downloadable files, but the rendered **Rights → License** field is blank. The 2025 Scientific Data article itself is CC BY 4.0; that article-level license must not automatically be imputed to the separately deposited dataset and every bundled third-party asset. Thus file access appears open, but reusable-data/model-training/redistribution permission is **not yet established from the record metadata**. Seek an explicit dataset license or written rights clarification from the depositors before extraction or model use. Until resolved, `data_access_unresolved` is the correct status; this does not mean the currently listed files are access-restricted.

The published table schema names 510 variables: case number (`No`), profile key (`Profile`), test-series label (`Series`), image number within series (`Image`), grid-line label (`Grid`), stage number (`Stage`), 501 interpolated brightness-profile samples (`x1…x501`), manual reference width (`MANwidth`), and existing model outputs (`DLMwidth`, `AEDwidth`). The article says `Profile` comprises series/image/grid/stage labels, and `Series` combines concrete-mix label and self-healing duration. The table description does **not** list dedicated columns for scanner position, moisture condition, or operator identity, although the paper investigates those factors in technical validation. Do not report those subgroups from the CSV unless a documented external key maps rows or source images to those factors.

The paper describes two high-strength mix designs (water/binder ratios 0.28 and 0.23), material ages of 2 and 20 months at crack induction, cracks induced across a stated 0–800 µm range, repeated scanning stages, and experiments on scanner position, surface moisture, and operator effects. These make within-study measurement robustness testable but limit claims about new mixes, ordinary concrete, other scanners/cameras, environments, or field structures. The authors' `MANwidth` is an operator-derived **reference**, not error-free physical ground truth. The first benchmark should report agreement with that reference and its method limits—not “absolute accuracy” or healing efficacy.

This is enough to motivate a bounded *measurement-method* benchmark, not to claim a broad living-concrete dataset. The current Zenodo record's rights/terms must be inspected and captured at admission: Zenodo is a hosting service and does not grant blanket reuse rights. A previously indexed deposit of the same title (record 11408399) explicitly marked its files restricted, so don't assume the same file access or license conditions across record versions. If current file access, license, derivative-work permission or citation terms cannot be resolved, record `data_access_unresolved` and stop any data extraction, redistribution or model evaluation requiring those permissions.

**Benchmark A — crack-width measurement agreement (first, low-cost lane):**
1. Capture the canonical paper and dataset DOIs, record URL/version, exact license/rights text, allowed use and redistribution, file names/sizes/checksums, retrieval timestamp, scripts and tool versions in a read-only source manifest. Keep raw files outside Git; do not commit the 56 MB table or 38.6 GB images.
2. Start with metadata and the table only after the rights gate. Independently inspect units, column definitions, record keys, test-series/specimen identity, grid-line/location keys, stage/time labels, reference-measurement provenance and benchmark-output provenance. Reconcile row counts and key uniqueness against the data descriptor; do not infer grouping keys from names alone.
3. Freeze a group-aware holdout before fitting. Partition by the strongest available independent acquisition unit (at minimum test series/specimen; do not split repeated locations, image stacks, or stages from one physical crack across train and test). Report the actual number and distribution of independent groups, not the 19,098 profile-row count as if it were 19,098 specimens.
4. Compare the published analytic edge detector and metasensor outputs with a transparent simple baseline and `MANwidth`, if permitted and exactly reproducible. Report MAE/RMSE, signed bias, calibration/error distribution in micrometres and error by crack-width range and stage where derivable from documented keys. Only report moisture, scanner-position, or operator subgroup metrics if separate labels are explicitly documented and joined by a validated key. Treat `MANwidth` as a fallible operator reference and explain the paper's measurement-uncertainty results; don't call agreement with it absolute accuracy. Compute uncertainty at the image/specimen or other justified independent-group level, never as if every profile row were independent.
5. Add negative controls for shuffled reference widths, duplicate/near-duplicate grid lines across folds, hidden sample/test-series identity leakage, wrong unit conversion, repeated stages randomly spread across train/test, and reference labels used as input features. Fail closed if independence cannot be reconstructed.
6. Freeze predictions, labels, group assignment and report manifest; independently recompute row/group counts, split integrity, metrics and uncertainty. Record script/environment hashes and every excluded row with a reason.

**Claim ceiling:** successful reproduction would validate only the specified crack-width metrology task on this dataset's acquisition distribution. The source study is about autogenous healing in high-strength concrete; crack-width change is not itself proof of crack-depth closure, permeability/ingress recovery, mechanical strength recovery, durability, biological viability or suitability for rail/road infrastructure. This image-derived dataset cannot qualify microbial concrete or replace laboratory tests. A later visual model should not begin until this table-only audit establishes access rights and group/label structure.

**Benchmark B — healing-progression analysis (later, gated):** only after Benchmark A passes and stage/specimen lineage is verified, evaluate temporal crack-width progression with whole-specimen/test-series holdouts. Distinguish observed narrowing from causal attribution to healing: moisture state, scanner/operator effects, elastic regain after cracking, image registration and repeated-location dependence must be considered. No causal healing claim follows from temporal image width alone.

The initial implementation should use a small, rights-cleared local working copy of the metadata/table and the existing BIO-CEM corpus/observation contracts. Use standard tabular baselines before HDC/CfC; keep source data, labels, derived variables, model predictions and synthetic augmentation explicitly separated. Synthetic rows can test pipeline behavior but must never enter the measured-data evaluation denominator.

Execution is tracked in [BIO-CEM-001A2 #7462](https://github.com/Luminous-Dynamics/symthaea/issues/7462), including the dataset-rights gate, grouped split requirements and explicit claim ceiling.

### A9. Rights-cleared multi-endpoint starter dataset: University of Bath BATH-01087

The University of Bath Research Data Archive provides a more direct, explicitly licensed alternative for a first *material-outcome* audit: Justo-Reinoso et al. (2022), [BATH-01087](https://researchdata.bath.ac.uk/1087/), associated with [the original paper](https://doi.org/10.1016/j.conbuildmat.2022.126877). The archive page explicitly marks each of its three Excel files **Creative Commons: Attribution 4.0 (CC BY 4.0)**. Record the file-level license text and attribution, cite the dataset DOI [10.15125/BATH-01087](https://doi.org/10.15125/BATH-01087), and retain a version/URL/checksum manifest. This removes the explicit license-metadata uncertainty encountered for the current krkCMd Zenodo record, although normal source/version and any embedded third-party notices still need checking.

The archive describes three files:
- **12 MB workbook** (`Data Set AEA … Feb2022 IJR.xlsx`): reported results for several figures, including tests associated with air-entraining admixtures used to encapsulate bacterial spores.
- **112 MB workbook**: image-binarization data for healing-ratio measurement after 56 days for specimens cracked at 28 days.
- **117 MB workbook**: the corresponding 56-day assessment for specimens cracked at 9 months.

The repository description identifies potential outcome families: air content, flowability, compressive and flexural strength, image-based whole-crack healing ratio, water-penetration resistance, Raman spectroscopy and micro-bubble area/size. Confirm each variable's specimen-level availability and method against the associated publication; do not assume every endpoint is measured for every experimental unit. The 12 MB workbook is the preferred schema-audit entry point; retrieve larger files only if necessary.

**Benchmark A — controlled material outcome audit:**
1. Build a source manifest and workbook inventory (sheet names, dimensions, headers, units, formulas, controls/treatments, test method, age at cracking, healing duration, specimen/mix identity, replicates, exclusions and missingness). Reconcile worksheets to the published figures/tables and distinguish raw numeric observations from aggregated/figure-derived outputs.
2. Establish the *true experimental unit* for every endpoint before estimating effects. If data contain specimen summaries, count specimens—not image pixels, grid lines, technical repeats or spreadsheet rows—as independent units.
3. Predeclare separate estimands for (a) crack-area/width healing ratio, (b) compressive strength, (c) flexural strength, (d) water-penetration resistance and (e) fresh-state properties. Raman/mineralization and bubble metrics remain distinct where the actual data support valid interpretation.
4. Compare treatment against its experiment-defined control, stratified by the experiment's actual cracking age and mixture/treatment groups. Do not combine the early- and later-age tests into one effect without a justified model. Do not pool this single research program with krkCMd: that separate dataset focuses on image-profile metrology for autogenous healing, uses different outcomes and remains under its own unresolved reuse-rights gate [#7462](https://github.com/Luminous-Dynamics/symthaea/issues/7462).
5. Report absolute differences and uncertainty in native units as well as any relative/ratio measures; keep image closure separate from mechanical and transport outcomes. An increase in healing ratio cannot stand in for strength recovery or permeability recovery.
6. Add hostile controls for shuffled treatment labels, treatment/control identity lost in a join, mixed units, duplicate specimen rows, missing repeats coerced to zero, image pixels counted as replicates, and a healing-ratio outcome accidentally used as a strength/permeability label.
7. Only after the audit, compare simple and well-justified statistical baselines on documented group structure. This single-study archive cannot on its own establish cross-study generalization; HDC/CfC results remain exploratory until they beat transparent baselines on frozen, group-held-out evaluation with calibrated uncertainty and an independent recomputation.

#### A9.1 Published-claim prior registry (prior hypotheses, not workbook reanalysis)

The University of Bath research portal's indexed abstract for the associated 2022 paper reports the following as **author-reported results after 56 days**, to be treated as replication targets rather than accepted data-analysis outputs:
- Of three evaluated air-entraining admixtures (AEAs), only one improved healing performance relative to the comparator using bacterial spores protected in aerated concrete granules (ACGs).
- For that AEA, reported healing ratios were 59.6% higher than the ACG comparator for cracks formed at 28 days, and 46.2% higher for cracks formed at 9 months.
- The abstract reports water-penetration resistance as increased by 18.1% or remaining similar at approximately 84%, for early- or later-formed cracks respectively, after 56 days.
- A historical study cost analysis estimated the AEA route at about 13 times lower cost than the ACG route.
- The article's Results section also reports approximately fourfold air content and approximately 56% lower compressive strength and 48% lower flexural strength for AEA mixes relative to the reference. These figures are literature-reported and still need workbook-level reconstruction; they must be considered alongside healing and water-flow results, not collapsed into a single score.

Source: Justo-Reinoso et al. (2022), [article DOI](https://doi.org/10.1016/j.conbuildmat.2022.126877), [University of Bath publication record](https://researchportal.bath.ac.uk/en/publications/air-entraining-admixtures-as-a-protection-method-for-bacterial-sp/), [data record BATH-01087](https://doi.org/10.15125/BATH-01087).

These numbers are not findings of the current Symthaea audit. Their denominators, uncertainty, exact calculation, specimen/replicate hierarchy, significance tests, and compatibility between endpoints must be recovered from the paper and workbooks before reproduction. In particular, “59.6% higher” and “46.2% higher” are relative comparisons, not absolute percentage-point gains unless the source explicitly defines them that way. The historic “13 times lower cost” is not a current cost estimate and should not be ported into another geography or construction supply chain.

Use the abstract only to preregister a narrow replication question: **Can the workbook reproduce the authors' endpoint-specific contrasts for the selected AEA versus ACG at each crack-induction age, with the original unit of analysis and uncertainty preserved?** Do not select the apparently best AEA or tune models against these expected summaries; freeze the extraction and evaluation before comparing results. If the accessible tables cannot identify independent specimens/replicates or reconstruct the reported summaries, report that exact limitation rather than reverse-engineering unsupported sample counts.

#### A9.2 Experimental units, outcome equations, and mechanism-attribution safeguards

The full-text article is available in the authors' [ResearchGate full-text record](https://www.researchgate.net/publication/358862745_Air-entraining_admixtures_as_a_protection_method_for_bacterial_spores_in_self-healing_cementitious_composites_Healing_evaluation_of_early_and_later-age_cracks), with the publisher's [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S0950061822005633). It adds critical details for the workbook audit:

- The programme has three experimental groups. Group I tests fresh/hardened physical properties using 40 × 40 × 160 mm prisms; Groups II and III assess cracks in 40 × 40 × 65 mm prisms after either 28 days or 9 months of curing.
- The article says mixes were made in triplicate. Group I flexural tests use three prisms per mix; the six resulting prism halves then supply compressive tests. Those six compressive measurements are not six independent casting replicates: they are nested within three parent prisms and likely share mix/batch effects. The reported air content is the mean of two individual readings per mix. Preserve those different replication levels, and distinguish physical specimens from the number of measurements in a workbook. For Groups II/III, the paper reports three specimens per mix and 18 location readings per mix per group (three specimens × six readings), giving 288 marked-location readings across eight mixes × two groups. The same locations were revisited at post-crack, 7, 28 and 56 days; reconcile the workbook's per-stage record count against the paper rather than assuming the 288 count covers every repeated stage. This is still only three physical specimens per mix/group; repeated locations and times do not multiply the independent sample size.
- Crack area was measured from images taken immediately after cracking and after 28 and 56 days of healing, with a black pixel threshold of 100. The area-based healing ratio for specimen i is `100 * (A_initial_i - A_final_i) / A_initial_i`; the paper then reports the mean of the per-specimen ratios for the mix. Do not recalculate it as `100 * (1 - sum(A_final) / sum(A_initial))` unless the estimand is explicitly changed and reported as a separate metric.
- Crack width was measured at two permanently marked positions, with three width readings at each position (six readings per specimen) at post-crack, 7, 28 and 56 days. These are repeated observations within a specimen and location, not 6 × 4 independent samples. Retain specimen and location pairing and use a repeated-measures or cluster-level summary appropriate to the declared target.
- Water-flow measurements were taken immediately after cracking and at 28 and 56 days. The article's recovery calculation is `100 * (k_0 - k_t) / k_0`, where `k_0` is the post-cracking flow coefficient and `k_t` is the coefficient at healing time `t`. This is recovery relative to the same specimen's post-cracking reference; values depend on the source method and are not directly interchangeable with absolute strength or permeability.
- SEM-EDX/Raman characterization was reported for representative specimens after 56 days; only selected mixes are described in the methods. Treat those images/spectra as selected mechanistic evidence, not a full-random sample-level outcome for every treatment group. A qualitative mineral signature does not by itself quantify healing or prove causal biological activity.

**Confounding and attribution caveat:** the discussion notes that tap water and non-sterile curing/healing conditions were used, and proposes environmental bacteria as one possible contribution alongside yeast extract, calcium hydroxide and carbonation. Thus the treatment-level result may be useful even if the precise mechanism is not fully identified. Do not attribute the observed healing entirely to the deliberately added strain unless the experiment has controls or independent evidence that distinguish intended-strain activity from environmental microbes and abiotic/autogenous healing.

**Cross-endpoint trade-off that must not get averaged away:** the same paper's Results section reports the AEA mixes' air content as roughly four times the reference mix, alongside approximately 56% lower compressive strength and 48% lower flexural strength relative to the reference. These are author-reported figures and have not been recomputed from the workbook. They make this a useful cautionary case: an improvement in a crack-area endpoint against one comparator can coexist with a major baseline-strength penalty. A Pareto chart may visualize that trade-off, but essential intact-strength and constructability requirements are hard feasibility constraints, not objectives that can be compensated by a high healing score. Do not call the 2022 AEA candidate structurally superior or infrastructure-ready from the healing ratios alone.

**Workbook audit order:** (1) map every sheet to Group I / II / III and the paper's figure/table; (2) recover the hierarchy mix/batch → parent specimen/prism → halves or image/location/time repeats; (3) distinguish raw readings, per-specimen summaries and plotted/aggregated values; (4) recompute published formulas on per-specimen values; (5) only then compare treatment effects and uncertainty. If identifiers are absent from a figure-derived workbook, report that the hierarchy cannot be recovered instead of inferring independence.

A small auxiliary source is Paine et al. (2017), [University of Bath BATH-00448](https://researchdata.bath.ac.uk/448/), which explicitly licenses its 17 kB workbook and documentation CC BY 4.0. It contains perlite particle-size data, bacterial growth/calcite productivity in three media, mortar mix designs, and initial surface absorption. Treat it as a distinct experimental programme that can inform hypothesis/context comparison, not as rows interchangeable with BATH-01087 or as direct validation of the same treatment effect.

**Claim ceiling:** these openly licensed lab datasets permit a reproducible evidence audit, but not a claim of field readiness, long-term durability, ecological safety, suitability for road/rail assets or a universal microbial-concrete recipe. No workbook has been downloaded or analysed as part of this metadata review. Execution is tracked in [BIO-CEM-001A3 #7471](https://github.com/Luminous-Dynamics/symthaea/issues/7471); this section creates no new production schema.

### A10. Non-biological comparator: self-healing via large clinker particles

A 2025–2026 sequence of open-access papers by Wang and colleagues proposes a compatible, non-living autogenous-healing route by partially replacing fine cement with larger clinker particles. This is an important comparator for the microbial programme because it may avoid carrier survival, viability, inoculum and environmental-persistence questions. The two papers form a linked research programme, **not independent replication by unrelated laboratories**. Neither is an openly downloadable raw-data benchmark: both article records state that data will be made available on request. They are currently literature priors until the underlying data are obtained and rights/use terms are confirmed. The experimental system is cement paste/specimens, not a full aggregate concrete element or a field-qualified bridge, pavement, sleeper or rail asset. It is **not automatically lower-carbon or better overall**.

Source: Wang et al. (2026), *A mechanistic study of a novel self-healing concept for cement enabled by large clinker particles: Recovery of transport properties and microstructure characteristics*, *Construction and Building Materials* 516, 145707, DOI https://doi.org/10.1016/j.conbuildmat.2026.145707. The publication page labels it open access under a Creative Commons licence.

**What the two papers report (literature priors, not Symthaea reproduction):**

- The 2025 preliminary paper reports that 40–60 μm clinker particle variants achieved 28-day tensile and compressive strength recovery ratios of 1.12 and 0.91, respectively, and crack-sealing widths of 300–400 μm with area healing of 83.4–94.4% after 56 days. Treat these as source-defined ratios and outcomes; do not assume the 1.12 ratio implies 112% recovery from an arbitrary baseline without checking the original equation and denominator.
- The 2026 follow-up examines transport-property recovery and microstructure, described below. Its reported transport outcomes must remain distinct from the earlier study's strength and crack-area results; these are not automatically one common cohort or pooled effect.
- Particle-size families included 40–60 μm, 60–90 μm and 0.5–1.0 mm; the study varied partial-replacement contents from 20% to 40% by mass.
- For the 40–90 μm variants, reported recovery rates were approximately 67.0–67.6% for initial water absorption, 83.2–83.4% for secondary water absorption, and 78.6–82.6% for gas permeability, compared with 46.2%, 49.5% and 64.1% for the control, respectively.
- The study reports surface cracks starting near 200 μm narrowing to roughly 20–60 μm, and internal crack widths of 57–111 μm by XCT for the highlighted sizes.
- XCT and microstructural analyses support the study's proposed mechanism of continued hydration from residual unhydrated clinker. These are study-specific findings, not a general guarantee of crack-depth sealing or mechanical recovery in a different mix.

The 2026 study's methods report cylindrical cement specimens (diameters 50 mm or 100 mm, height 30 mm), initially water-cured at 75 °C for 14 days; pre-cracks were induced by heating at 150 °C for two hours. The reported crack widths in this experiment were mostly controlled in the 80–200 μm range. This is a specialized accelerated laboratory protocol, not a direct proxy for ordinary ambient-cured concrete cracks, load-induced cracks or rail/pavement service environments.

Both the 2025 and 2026 article records state **“Data will be made available on request.”** No underlying raw-data package has been identified in the sources inspected. To reproduce those published summaries, the work item must first request the source tables, experimental-unit hierarchy, protocols, and permission/terms for analysis and derived outputs. Until then, only article-level literature extraction is possible.

The authors also explicitly note a **carbon trade-off**: replacing cement with clinker may increase the clinker factor, particularly in cements with high supplementary-cementitious-material content. They argue that possible service-life extension could offset that impact, but this is conditional on field durability, actual mix chemistry, repair frequency, manufacturing/transport burdens, and the functional unit. Do not assert a climate benefit without a transparent whole-life comparison and sensitivity analysis for assumed service-life extension.

**Common comparator requirements:** compare this family against the conventional/autogenous baseline, internal curing or crack-width control, non-living repair systems and microbial candidates only after harmonizing initial crack width, curing and healing conditions, conditioning and test methods. Keep initial/secondary absorption and gas permeability as different outcomes. Also measure fresh-state properties, intact compressive/flexural strength, post-healing mechanical properties where appropriate, durability through cycling, material quantities and emissions. The paper's reported outcome percentages cannot be directly pooled with the BATH-01087 dataset's different protocol or treated as additional independent observations.

Use mandatory feasibility gates for the intended application. No amount of sealing performance compensates for a violation of required strength, workability, durability, or owner/standard criteria. Only candidates that pass those hard gates may enter multiobjective comparison for lifecycle cost, emissions, repairability and maintenance burden.

Cross-mechanism comparator design and lifecycle assumptions are tracked in [BIO-CEM-001A4 #7488](https://github.com/Luminous-Dynamics/symthaea/issues/7488). This literature addition creates no material ontology, carbon-accounting schema or production type; reuse canonical MAT/FIELD/SE-OBS/SE-SEM/ENG-EVID owners and record source-level claims separately from measured data.

### A11. Cross-study comparability matrix: what can and cannot be compared

This matrix is a research triage aid, not a data schema or a claim that the studies share an experimental population. A cell marked **not reported / not verified** means the current source-level audit did not establish it; do not silently impute it.

| Source / system | Crack generation and specimen context | Crack/image endpoint | Transport endpoint | Mechanical / fresh-state endpoints | Data & independent replication | Permitted comparison now |
|---|---|---|---|---|---|---|
| Bath BATH-01087 (2022), AEA / bacteria-spore encapsulation | Mortar; cracks targeted around 0.50 mm; induction at 28 days or 9 months; follow-up 56 days | Whole-crack image-binarization healing ratio; crack-healing vs initial width; separate replicate/location structure must be recovered from workbook | Water penetration resistance using RILEM 11.4 | Air content, flowability, compression/flexure; Raman and microbubble characterization for documented subsets | Three workbook files listed CC BY 4.0. Single study; actual workbook not yet inspected | Suitable first licensed within-study reproduction. Do not pool raw effect sizes with other rows until methods and experimental units are reconciled |
| krkCMd (2025), autogenous high-strength concrete | High-strength concrete; image stacks and later healing stages; full physical-specimen mapping requires documented key audit | 19,098 brightness profiles; manual `MANwidth` reference; DLM/AED outputs | None established in the described profile table | Not in the core 510-column profile table | Current Zenodo record files listed, but rendered license field blank; prior record 11408399 explicitly restricted | Crack-width metrology benchmark only, after explicit dataset-use rights and group keys; not a mechanical-healing benchmark |
| Wang & Wu (2025), large clinker | Cement paste/specimens; source reports controlled cracking; detailed raw series not provided in the article record | Source reports crack-sealing width and area closure | Not an outcome in the preliminary paper's abstract | Source reports 28-day tensile/compressive strength recovery ratios; definitions must be recovered before interpreting | Data available on request; linked group programme, not independent replication | Literature prior only; request raw tables/methods; do not pool with 2026 paper or Bath |
| Wang et al. (2026), large clinker | Cement specimens Ø50×30 and Ø100×30 mm; water cure at 75 °C for 14 days; artificial through-going thermal cracks at 150 °C for 2 h | OM surface and XCT internal geometry; distinction between initial crack width, residual width, and the article's “crack-healing width” needs source-defined metric confirmation | Initial/secondary capillary water absorption and gas permeability; separate methods | Microstructural/chemical characterization; a matched mechanical recovery endpoint not established by the abstract alone | Data available on request; same research group as 2025 work | Mechanism/endpoint prior only; not an open raw-data benchmark or independent replication |
| Van Mullem et al. (2020), polyurethane glass macrocapsules | Mortar and reinforced concrete; differing crack-control approaches; six laboratories | Crack-width preconditioning/measurement plus healing-agent spread | Mortar water permeability; concrete capillary absorption with alternate waterproofing methods | Bending/cracking setup differed between specimen types | Inter-laboratory round robin across six labs | Use to specify protocol sensitivity and metrology quality gates; capsule-specific performance values are not a universal comparator |
| SARCOS mineral-agent inter-lab study (2021), MgO expansive agents | Concrete prisms/discs; multi-lab cracking and healing over six months | Crack-width evolution; opening variability affected cross-lab comparisons | Sorptivity and water-flow/permeability; application errors mattered for absorption | Compressive strength / hardened density reported by batch; other scope is method-dependent | Multi-lab round robin; exact common dataset availability must be checked separately | Use as method-reproducibility evidence; don't treat reported aggregate outcomes as row-compatible with Bath or clinker data |

**The current compatible overlap is much smaller than “all studies measure self-healing”:**
- Crack width / closure appears across several sources, but initial crack geometry, imaging method, width statistic, time points, and specimen/independent unit vary. Report per-study, use a shared harmonized endpoint only after protocol-level review.
- Transport recovery exists in Bath, Wang 2026 and inter-lab studies, but water penetration, initial/secondary sorptivity, water flow/permeability and gas permeability are distinct test methods. Do not collapse them into a generic “impermeability” value.
- Mechanical evidence is explicit in Bath and Wang 2025's abstract, but the comparator, test timing, denominator and whether it is intact strength or recovery of strength after cracking must match before cross-study use.
- Fresh-state / air and workability measurements are available in Bath; no common measure is established for the other rows above.
- Biosafety and lifecycle carbon are not jointly established by any of these datasets. These remain separate qualification/LCA workstreams, not missing values to be filled by a model.

**Metric-definition guard:** the Wang 2026 abstract uses the phrase “crack-healing width” with values 203–225 μm, while its body also discusses initial ~200 μm cracks and residual widths around 20–60 μm at 28 days (depending on mix and whether OM or XCT is used). These are not necessarily contradictory measurements, but the phrase is not self-defining. Do not map 203–225 μm to “remaining open crack width”; recover the figure caption/equation and endpoint meaning before using it in the common matrix.

**Stop rule:** if a study-specific unit, denominator, control, specimen hierarchy, measurement method, or reuse right is unresolved, keep the claim as source-reported literature evidence and mark that cell not comparable. Do not fit a cross-study pooled model merely because rows can be converted to the same numeric unit.

### A12. Dataset admission queue and next-experiment order (2026-10-15)

This is a source-admission triage table, not a scientific result. **Downloadable/open access is not by itself a reuse license**, and a paper's open-access licence does not automatically license unpublished source data. For records whose rendered license field is blank, use `license_scope_unresolved`: that is a metadata/rights clarification gate, **not** a conclusion that all reuse is forbidden and not a presumption of permission. [Zenodo's current help](https://help.zenodo.org/docs/deposit/describe-records/licenses/) says public records require a license and describes CC BY 4.0 as the default, so a blank rendered field is anomalous and should be checked against the specific record/version and depositor. Until clarified, don't treat the source as cleared for training, redistribution or derived publication.

| Source | Endpoint relevance | Current access/rights finding | Next allowed action |
|---|---|---|---|
| University of Bath BATH-01087 (2022; linked in #7471) | Strongest licensed multi-endpoint lead in this registry: crack-area/width, transport, strength and fresh-state outcomes, subject to actual workbook contents | Archive page explicitly marks all three workbooks CC BY 4.0. Payload retrieval has not succeeded in this work session; no sheet, row, formula or checksum audit has been performed | Keep as first target for within-study outcome reproduction once the original bytes can be retrieved; compute and record local SHA-256, map sheets to paper figures, and recover the specimen hierarchy before effects |
| University of Bath BATH-00448 (2017; https://doi.org/10.15125/BATH-00448) | Mix composition, perlite grading, bacterial growth optical density, carbonate productivity and mean initial surface absorption at 10–120 minutes | The 17 kB workbook and 382 kB documentation PDF are explicitly CC BY 4.0 at the archive. Documentation was readable, but workbook bytes could not be retrieved here; **no workbook results were analysed** | Use only as an auxiliary data-ingestion / formulation-provenance pilot, not a crack-healing or structural-recovery benchmark. Resolve the documentation's apparent units/volume inconsistency before calculations (it prints loose bulk densities as 1660/665/625 g/cm³ and gives 0.813 cm³ for 1350 g divided by 1660 g/cm³). Treat these as source-document defects requiring article/author reconciliation—not as permission to silently change units |
| krkCMd (2025; https://doi.org/10.5281/zenodo.14568863) | Crack-width metrology, not strength or transport recovery | Current record is labelled Dataset Open with files listed, but the rendered License field is blank; an earlier version was explicitly Dataset Restricted. Reuse terms remain unresolved | Do not use for model training or redistribute until record owner clarifies the dataset licence and intended scope |
| SARCOS RRT2 / MgO expansive-agent inter-lab data (2021; https://doi.org/10.5281/zenodo.4559868) | Cross-lab method repeatability and transport/measurement variability | Record describes nine participating labs and downloadable raw/supplementary files, but its Rights → License field is blank | Rights query first. If licensed, use for a method/repeatability study with lab and specimen clustering—not as rows exchangeable with the Bath or clinker studies |
| SARCOS RRT5 / polymeric macrocapsules (2020; https://doi.org/10.5281/zenodo.3865635) | Six-lab study of controlled cracks, capillary absorption and mortar water-flow methods | 219.3 MB file is listed; record's Rights → License field is blank | Rights query first. If licensed, use to study measurement protocol and inter-lab variability, keeping waterproofing methods and specimen types separate |
| Sulfate-attack supplementary results (2026; https://doi.org/10.5281/zenodo.18277063) | Expansion, mass change, microscopy, and mercury-intrusion-porosimetry-related results under sulfate-rich exposure | Five downloadable files are listed (including four XLSX/DOCX files totalling 5.6 MB in the record view), but the record's Rights → License field is blank | Treat as access-visible but reuse-unresolved; do not infer permission from the paper's open-access status. If permission is secured, map exposure, mix, specimen and timepoint before comparing durability |
| Tran-SET bacterial-spore concrete (2019; https://doi.org/10.5281/zenodo.3471961) | Crack-sealing and stiffness recovery under wet/dry cycles in a hot subtropical-climate study | Dataset Open and a 16.5 MB archive is listed, but the rendered Rights → License field is blank | Rights query first. Preserve a crucial source-reported confound: the record says alginate-bead groups, including beads without bacteria, improved outcomes versus controls without beads. Do not attribute the full effect to bacterial activity absent discriminating controls |
| OnboardEU railway positioning sample (2025; https://doi.org/10.5281/zenodo.16368240) | GNSS/IMU track-selective georeferencing and positioning output, not rail-condition or defect labels | Record-level License field is blank. A bundled file named `licencse.txt` contains CC BY 4.0 text, but README says the licence is in `license.txt`, which does not match that filename; named copyright holders include DLR, AIT and i4M | Record as `license_scope_unresolved` pending confirmation of the filename/scope mismatch. If cleared, test positioning/georeferencing separately; do not treat the processed output as independent truth or geometry-defect labels |
| Train-track environmental sensing (2025; https://doi.org/10.5281/zenodo.17607068) | 40 Hz on-board inertial/environmental time series and GPS-tagged points; potential signal-quality exercise only | Record-level License field is blank | Same gate: explicit reuse permission before extraction/training/release; it is not track-geometry ground truth |

#### Execution order and stopping conditions

1. **Metadata-only source manifest first.** For each record, store citation/DOI, owner, exact record/version URL, observed access state, licence evidence URL/text, allowed purpose, retrieval attempt/status, record-supplied hashes, and unresolved questions. Do not mark a checksum verified unless it was recomputed from retrieved bytes; do not check in raw workbooks or large source archives.
2. **BATH-01087 is the first target for healing-outcome reproduction** because it directly spans healing, strength, transport and fresh-state endpoints and has clear CC BY 4.0 archive terms. Its current blocker is payload retrieval and schema/experimental-unit verification, not licence discovery.
3. **BATH-00448 is an auxiliary ingestion/provenance probe only.** It has useful ingredients/growth/carbonate/absorption context but no established post-crack healing series in the described workbook. Its documentation contains apparent physical-units inconsistency; resolve it from the source paper or data steward before any derived quantity is calculated.
4. **SARCOS RRT2/RRT5 are the highest-value candidates for protocol repeatability once rights are cleared.** Their multi-lab structure may inform measurement variance and test-method selection, but lab/specimen nesting must be recovered and protocol differences must remain explicit.
5. **Do not let a downloadable, apparently well-documented source jump the queue when the licence scope is unclear.** Request written clarification from the data steward/record owner and preserve the request/response in the evidence lineage. For OnboardEU, the embedded CC BY 4.0 text and README filename mismatch require specific confirmation rather than either silently ignoring the file or claiming the scope is settled.
6. Model only after provenance, units, specimen grouping, endpoints and negative controls are independently checked. Missing inputs remain missing; source summaries are literature priors, not results reproduced by Symthaea.

No source file was retrieved and analysed for this admission update. This section changes source prioritization and claim boundaries only; it adds no production types, data schema, benchmark score or material recommendation.

### A13. Download-failure ledger and dimensional-consistency gate (2026-10-15)

This section separates an access-tool failure from source unavailability or rights restriction. It records what was attempted; it does not claim that the data owners removed access.

#### Bath spreadsheet retrieval attempts

The official [BATH-01087 archive page](https://researchdata.bath.ac.uk/1087/) exposes the 12 MB workbook at the link target `https://researchdata.bath.ac.uk/1087/4/Data%20Set%20AEA%20Manuscript%20Feb2022%20IJR.xlsx` and explicitly labels it CC BY 4.0. Following the archive's file link in the browser tool returned **HTTP/tool error 400: unsupported XLSX content type**. A separate direct-download attempt from the execution container failed before HTTP transfer with a DNS resolution error. These failures establish only that these retrieval paths/tools did not deliver workbook bytes in this session; they do not show that the archive file is missing, access-restricted, corrupt, or unlicensed. No workbook checksum was computed and no sheet/cell/result was inspected.

The [BATH-00448 archive](https://researchdata.bath.ac.uk/448/) similarly exposes `https://researchdata.bath.ac.uk/448/1/Data.xlsx`, licensed CC BY 4.0, but the browser tool returned the same unsupported-XLSX-content-type error. Its [official 2-page data documentation](https://researchdata.bath.ac.uk/448/3/Data_documentation.pdf) was retrievable. Next attempt should use a normal file-capable HTTP client or browser download path, save the original bytes outside Git, and calculate SHA-256 locally. If that environment remains unavailable, ask Bath Research Data Service for an alternate delivery URL or repository export rather than treating a search result or derived conversion as the source file.

#### Worked dimensional check: BATH-00448 documentation

The archive's documentation, at lines 16–20 of the PDF, states that 1350 g of sand divided by a loose bulk density of 1660 g/cm³ gives a sand volume of 0.813 cm³, and lists the loose bulk densities of coated perlite with nutrients (CPN) and with spores (CPS) as 665 g/cm³ and 625 g/cm³. Taken literally, those density magnitudes are physically implausible for loose mineral aggregates. The documentation is therefore not safe to use as a numerically trusted unit source without reconciliation.

A **hypothesis for author/steward verification only**, not an accepted correction: if the intended density units were kg/m³, then (1350\,\mathrm{g}=1.350\,\mathrm{kg}) and
`1.350 kg / (1660 kg·m⁻³) = 0.00081325 m³ ≈ 813 cm³ = 0.813 L`.
That differs by a factor of 1000 from the printed volume of 0.813 cm³. The CPN/CPS numerical densities would likewise have plausible-looking scale as kg/m³ rather than g/cm³, but plausibility alone cannot prove the intended units. Do **not** silently patch a workbook, use these reconstructed units in model input, or publish derived mix quantities before the paper and/or data steward confirms the density units, target volume unit and exact source workbook version. Record the answer with the source/version and any correction history.

#### Admission states for repeatable work

Use a small, reviewable status vocabulary in the existing source manifest (no new production schema implied):
- `metadata_seen`: an official landing page or citation was reviewed;
- `license_scope_unresolved`: the relevant file/reuse terms are not unambiguously established for the proposed purpose;
- `download_attempt_failed_tooling`: the request failed at the client/tool layer;
- `bytes_retrieved_unverified`: a file exists locally but expected digest, format and source identity are not verified;
- `bytes_verified`: exact source identity and recorded checksum were independently verified from local bytes;
- `schema_audited`: sheets/columns, units, identifiers, missingness and formula provenance have been inventoried;
- `analysis_reproduced`: a frozen, cited calculation reproduces a named source result within a predeclared tolerance and with experimental units intact.

These are sequential audit states, not a new evidence type. Later states do not erase unresolved rights, and a checksum only validates byte identity—not correctness of the original measurement. Do not claim a state from a citation or file listing alone.

### A14. Paradigm shifts: from “self-healing material” to verified adaptive infrastructure (2026-10-15)

The strategic opportunity is not to claim a universally self-healing concrete or a universally superior railway predictor. It is to build a research-to-operations loop that **prevents avoidable damage, senses what cannot be prevented, selects a bounded intervention, and independently verifies whether that intervention improved the asset over its service life**. The following are hypotheses and programme design principles, not demonstrated Symthaea capabilities.

This reframing responds to known limitations in the literature. A six-laboratory study of glass-macrocapsule systems found that crack-preconditioning and waterproofing choices materially influenced measurements; reference flow rates were comparable in only four of six labs, while sealing-efficiency results were more comparable across all six. A separate MgO interlaboratory study reported high variability in crack creation, which complicates comparison across labs. These findings make metrology and protocol reproducibility part of the product, not administrative overhead ([Van Mullem et al., 2020](https://doi.org/10.1080/14686996.2020.1814117); [SARCOS MgO interlaboratory study, 2021](https://doi.org/10.3390/ma14082024)). Research on active-learning materials laboratories similarly emphasizes managing epistemic and stochastic error, reproducibility, reconfigurability and interoperability ([Ren et al., 2023](https://www.nature.com/articles/s41578-023-00588-4)). A 2026 review of rail geometry forecasting identifies local-exceedance under-detection from label scarcity and smoothing, and proposes separating exceedance events from anomalies rather than relying on one aggregated condition score ([Railway track geometry degradation: a review of prediction methods, 2026](https://doi.org/10.1016/j.rineng.2026.109583)).

#### A14.1. Eight design shifts

| Paradigm shift | Old question | Better question | Evidence gate / failure condition |
|---|---|---|---|
| **1. Material-first → service-first** | Which ingredient or recipe maximizes crack closure? | Which intervention portfolio delivers required asset function at acceptable safety, repair burden, cost and environmental impact over a declared service period? | Use a declared functional unit, baseline design, exposure/loading, maintenance policy and end-of-life. No universal “best material” claim from a coupon result. |
| **2. Static leaderboard → causal experiment** | Which formulation has the highest reported healing percentage? | Which component of the intervention caused the outcome relative to matched controls, and under which boundary conditions? | Pre-register control groups, batch/specimen hierarchy and estimand. Where safe and appropriate, distinguish carrier effects, added-agent effects, ordinary autogenous healing and environmental contributions. If the design cannot separate explanations, report association rather than mechanism. |
| **3. Scalar healing score → property-specific envelope** | How much did the crack “heal”? | For what crack width/depth, moisture, temperature, exposure chemistry, age and cycling regime does each endpoint remain within requirements? | Keep appearance/closure, transport/ingress, mechanical capacity, fatigue, durability, workability, biological viability and biosafety separate. Unknown or out-of-domain conditions return insufficient evidence; no compensatory weighted score may override a hard requirement. |
| **4. One-lab result → reproducible measurement system** | Can one lab produce an attractive result? | Can independent labs reproduce both the physical treatment and the measurement under a shared protocol? | Freeze crack induction and width-control methods, conditioning, sealing, instrumentation, calibration, endpoint definitions, raw-data shape and analysis. Include reference specimens and report between-lab as well as within-lab variation before claiming transferable performance. |
| **5. Maximize predicted improvement → maximize useful information** | Which next candidate has the best model-predicted outcome? | Which experiment most reduces uncertainty that could change a real decision, per unit cost/time/risk? | Compare expected information gain or decision value against simple baselines; include feasibility, experiment cost, expected duration, risk and OOD likelihood. The value is in changing a decision, not merely reducing model error. Preserve negative, null and contradictory results. |
| **6. Embodied-carbon snapshot → whole-life functional comparison** | Is the new mix lower-carbon per cubic metre? | What are emissions, cost and service risks for equivalent accepted function over the same horizon, including construction, repairs, closures, replacements, transport and end-of-life? | State system boundary, functional unit, inventories and service-life assumptions; vary uncertain repair frequency and life extension in sensitivity analysis. A recent SHC LCA reports 30–50% higher embodied carbon for studied concepts while modelling potential maintenance/design benefits—the trade-off is formulation- and assumption-specific, not a general result ([Rengaraju et al., 2025](https://doi.org/10.1016/j.jobe.2025.113105)). |
| **7. Autonomous repair → repairable, inspectable material** | Can the material repair itself without a human? | Can damage be detected early, remain inspectable, and be repaired or safely taken out of service if healing fails? | Preserve crack/degradation visibility, fallback repairs, inspection intervals, repairability and failure modes. “Autonomous” must never mean “self-certifying” or “inspection not required.” |
| **8. Model-as-oracle → evidence-bounded digital thread** | Can the model output a confident answer? | Can a reviewer trace every decision to source rights, physical specimens/assets, units, assumptions, calibration domain and independent checks? | Compose existing MAT/FIELD/SE-OBS/SE-SEM/ENG-EVID owners and Mycelix provenance controls. Distinguish literature, lab observation, simulation, field evidence and independent review. A hash proves byte integrity/lineage, not physical truth; governance approval is not engineering validation. |

#### A14.2. The experiment-selection engine should optimize information, not excitement

Once lawful, versioned source bytes can be retrieved, rank candidate experiments by **expected decision value** rather than only expected healing performance. For each proposed experiment record: (i) decision it could change; (ii) competing hypotheses it distinguishes; (iii) outcomes and controls required; (iv) expected uncertainty reduction; (v) equipment, time and financial cost; (vi) safety/biosafety and environmental review; (vii) whether a null result remains interpretable; and (viii) the result that would stop the line of research.

The first use should be human-reviewed batch selection from a pre-registered queue, not an autonomous wet lab. Start with source-data and analysis experiments (file identity, formulas, replicate hierarchy, published-result reproduction), then matched materials experiments, then inter-laboratory replication, and only then a low-consequence pilot. An experiment with modest expected performance but high ability to discriminate between rival explanations may be more valuable than another high-scoring formulation.

Use simple, transparent baselines for acquisition policies and compare decision-relevant outcomes, including experiments needed to reach a reliable decision, calibration, failure-to-reject poor candidates, and replication success. Any active-learning gain should be measured on a frozen candidate pool or prospective protocol; retrospective ranking of known winners is not evidence that the method improves discovery.

#### A14.3. Concrete-specific paradigm: define a “healing envelope” and require witness evidence

Do not define an ingredient as “self-healing” in the abstract. Define a bounded envelope of conditions and endpoints in which a specific material/process has passed independent, repeated tests. Each proposed envelope should name:

- initial crack geometry (width distribution, depth and location), crack-generation method, and load history;
- mix/batch and specimen lineage, exposure, moisture regime, temperature, relevant chemistry and healing duration;
- separate endpoints for visual closure, water/ion/other relevant transport, post-healing mechanical behaviour, repeated-load/cycle durability, and fresh-state/constructability;
- the uncertainty interval, failure fraction, missingness, measurement sensitivity and out-of-domain boundary;
- controls that distinguish the candidate agent/carrier from crack-width control, substrate effects, autogenous healing, carbonation, environmental organisms or other plausible contributors;
- the independent replication condition that would falsify transfer to a new mix, lab, exposure or scale.

Use a small, predeclared reference specimen set and calibrated imaging/transport apparatus as **witnesses** for each experimental campaign. A witness confirms measurement setup and process stability; it does not replace structural tests. For any biological candidate, biosafety, environmental persistence, worker exposure and by-product assessment remain independent hard gates controlled by qualified specialists. No deliberate environmental release or engineered-organism field deployment follows from a model ranking.

The most defensible early result may be a **negative design rule**—for example, documenting a condition where crack closure coexists with an unacceptable mechanical, transport or constructability penalty. A reproducible exclusion rule can save more money and risk than a marginal gain in a healing percentage.

#### A14.4. Rail-specific shift: prediction → event detection → intervention-effect evaluation

A more accurate forecast is only useful if it improves inspection and maintenance decisions without hiding localized defects. Keep three tasks separate:

1. **Detection/segmentation:** identify localized exceedance events and anomalies in measured geometry, with location and instrument uncertainty; do not let a segment average smooth away a short, severe defect.
2. **Forecasting:** estimate future condition using only information available at the declared issue time, with chronological and segment/route holdouts. Future realised maintenance must not leak into an operational forecast.
3. **Intervention-effect evaluation:** estimate whether a specific intervention (e.g. tamping or grinding) changes subsequent deterioration relative to a defensible comparison. Because interventions are targeted at poor condition, naive before/after comparison is confounded by indication, regression to the mean, weather, traffic and inspection timing. Pre-register suitable matched or quasi-experimental analyses only when the data support their assumptions; disclose residual confounding rather than calling association a causal effect.

Use two distinct reporting layers: (a) local event-level recall and false-negative analysis for safety-relevant exceedances, and (b) segment-level trend/forecast metrics for planning. Report site/route variability, lead time, false-alert burden, data staleness, uncertainty and operational decision consequences. The owner/operator still defines authoritative thresholds and dispatch/inspection decisions.

#### A14.5. Execution sequence and go/no-go gates

1. **Reproduce before predicting.** Retrieve BATH-01087 bytes from the licensed archive; preserve the original, calculate SHA-256, inventory sheets/formulas/units/IDs, reconcile each endpoint to the source paper, and independently recompute a small set of pre-specified published results. If file retrieval remains blocked, request alternate official delivery; do not substitute a screenshot or hand-transcribed chart for specimen data without labeling it as such.
2. **Pre-register one discriminating test.** Select an outcome where sources overlap meaningfully and write the estimand, independent experimental unit, controls, endpoint, tolerance and analysis before checking confirmatory results. Do not start with a pooled model across incomparable studies.
3. **Establish metrology repeatability.** Use known/reference specimens, controlled crack widths and repeat measurements; separate instrument, operator, batch, specimen, lab and method variance. If the method cannot distinguish an effect from its measurement variability, improve the measurement protocol before optimizing the mix.
4. **Replicate across a real domain boundary.** Change one declared factor (such as lab, mix batch, exposure or scale) at a time where feasible. A second paper from the same programme is not automatically independent replication.
5. **Evaluate decision-level value.** For only candidates passing hard constraints, compare lifecycle cost, carbon, repair/inspection burden and robustness under sensitivity cases. Reject a claimed benefit that depends on an unsupported service-life extension.
6. **Pilot only with reversible safeguards.** Choose a non-critical, inspectable, owner-approved asset with a conventional fallback, pre-registered monitoring, intervention/withdrawal triggers and a professional sign-off. Keep the model read-only until evidence supports a stronger operational role.

**The first measurable success criterion is not “Symthaea discovered a superior concrete.”** It is a third-party reviewer independently reproducing a published result from the original data, verifying the experimental-unit hierarchy and negative controls, and then seeing a prospective experiment whose decision value is demonstrated against a transparent baseline.

This is a programme hypothesis and test plan only. It does not establish a new concrete recipe, model advantage, railway safety capability, a causal maintenance effect or an approved deployment. It creates no new production type or parallel evidence ontology.

**Additional research anchors for this paradigm section**
- Ren et al. (2023), [Autonomous experiments using active learning and AI](https://www.nature.com/articles/s41578-023-00588-4), *Nature Reviews Materials*.
- Wang et al. (2022), NIST, [Benchmarking Active Learning Strategies for Materials Optimization and Discovery](https://www.nist.gov/publications/benchmarking-active-learning-strategies-materials-optimization-and-discovery).
- Galal, Mourad & Rana (2026), [Challenges and advancements in self-healing concrete: mechanisms, applications, and scalability](https://doi.org/10.1016/j.jobe.2025.115097).
- Rengaraju et al. (2025), [Environmental life cycle assessment of self-healing concrete with microcapsules and impregnated lightweight aggregates](https://doi.org/10.1016/j.jobe.2025.113105).
- [Railway track geometry degradation: a review of prediction methods](https://doi.org/10.1016/j.rineng.2026.109583) (2026).

## B. Better railway systems

### B1. Highest-value first move: predict deterioration, not just geometry

FRA research reports offer immediate baselines:
- [RR 26-01 — Track Geometry Predictions Using Point and Segment Processing](https://railroads.fra.dot.gov/elibrary/track-geometry-predictions-using-point-and-segment-processing). The agency reports that point-based SARIMAX and segment-based Prophet approaches forecast track-geometry behavior, and that adding unexpected maintenance events as explanatory data materially improved predictions. The report described case studies as future work at publication.
- [RR 26-02 — Implementation of Advanced Track Geometry Forecasting Models](https://railroads.fra.dot.gov/elibrary/implementation-advanced-track-geometry-forecasting-models), last updated by FRA on July 30, 2026. It describes operational implementation work improving track-geometry filtering and forecasting methods.
- [RR 21-18 — Feasibility of Applying Model-Assisted Probability of Detection to Track Inspection](https://railroads.fra.dot.gov/elibrary/feasibility-applying-model-assisted-probability-detection-track-inspection). This is a useful reference for evaluating inspection effectiveness, especially ultrasonic rail-flaw detection.

These are baselines to reproduce and learn from, not evidence that Symthaea already outperforms them. Public reports do not guarantee unrestricted access to the underlying raw, geolocated measurements. Every data source still needs access/licensing and field-of-use review.

### B2. Railway workstreams

**Railway track condition and preventive maintenance**
- Forecast geometry by location and time; retain spatially correlated defects rather than treating every reading as independent.
- Incorporate verified maintenance history, tamping/grinding/renewal events, weather, load/axle passages and inspection frequency.
- Compare forecasts with the owner's current model and simple persistence/threshold baselines.
- Optimize inspection and maintenance windows under access/possession constraints, using verified work duration and resource availability.
- Measure calibration, missed deterioration, false alarms, lead time and avoided service disruption; don't optimize only forecast RMSE.

**Inspection quality and detection confidence**
- Preserve sensor/instrument identity, calibration, data-quality status, coverage gaps, inspection modality and operator/method version.
- Study probability of detection, false-negative risk and uncertainty by defect type, track/speed/condition and inspection setup.
- Corroborate sensors using independent physical inspections and appropriate non-destructive testing.
- Missing or low-quality inspection evidence must stay unresolved, not become a “no defect” observation.

**Ballast, subgrade, drainage and transition zones**
- Correlate recurring track geometry deterioration with drainage, ballast condition/fouling, settlement, earthworks, bridge approaches and culvert/flood exposure.
- Combine terrain/catchment/climate context in Sol-Atlas with track-condition and maintenance evidence.
- Prefer root-cause and whole-life comparisons over repeatedly treating the same symptom with tamping alone.
- Keep hydrogeological/geotechnical claims bounded until site-specific observations and a suitable validated model exist.

**Rail/wheel interface and high-wear assets**
- Track rail-head wear, rolling-contact fatigue indications, wheel condition, rail grinding history, curves, loads and temperature.
- Compare maintenance timing and inspection strategies against measured defect recurrence, wear and service impacts.
- Prioritize turnouts, crossings, joints/welds and other complex assets as distinct asset classes with their own inspection/maintenance profiles. A single generic “rail health” model is inadequate.

**Timetable, capacity and passenger/freight reliability**
- Only after route/capacity data are trustworthy, model dwell-time variability, conflicts, junction and terminal bottlenecks, missed handoffs, temporary speed restrictions and maintenance possessions.
- Improve schedule robustness and freight transfer reliability rather than maximizing nominal trains/hour at the cost of recovery margin.
- Separate published/planned capacity, observed use, temporary restrictions and hypothetical capacity improvements.
- For energy, evaluate traction efficiency, regenerative braking where the rolling stock and power network support it, coasting/speed profiles, terminal dwell, and electrification only against real service, power-system and lifecycle constraints.

**Resilience and maintainability**
- Assess heat, flood, wildfire/vegetation, embankment movement, power outage, equipment failure, supplier delays and corridor disruption with explicit scenarios.
- Keep alternative routes, repair stock, crew availability, inspection coverage and recovery time visible.
- Measure service continuity, delay minutes, cancellations, time-to-detect, time-to-repair, repeat defect rate, lifecycle cost and access/equity impacts.
- Apply human operational authority and fail-safe procedures; no simulation is dispatch authority.

### B3. A concrete first pilot

**Pilot:** a read-only track-geometry deterioration benchmark with maintenance-history covariates. Execution is tracked separately in [TRANSPORT-RAIL-001A #7452](https://github.com/Luminous-Dynamics/symthaea/issues/7452) under the canonical rail owner #6248; this research note is not a duplicate implementation spec or split engine.

Inputs:
- exact segment/chainage or stable pseudonymous location ID;
- timestamps and survey method, instrument/source identity, units, quality flags and coverage;
- measured geometry values and owner-authoritative limits only if a current source references them;
- traffic/load/axle exposure if available and rights-cleared;
- maintenance events (date, location, intervention class, recorded magnitude and uncertainty);
- local weather/environmental covariates where source and resolution permit;
- split/manifest and source rights.

Models:
1. simple persistence and standard alert-threshold baseline;
2. implementation-faithful SARIMAX/segment Prophet baseline where assumptions fit;
3. Symthaea models and HDC/CfC feature experiments;
4. a held-out, independent evaluator that consumes frozen predictions and labels.

Evaluate by *future* time periods and held-out segments/routes; report site-level variation and maintenance-induced resets. Prevent near-duplicate segment leakage. Track MAE/RMSE where useful, but prioritize threshold-crossing calibration, under-prediction of deteriorating segments, lead time, false-alert burden and decision usefulness. Preserve manual review before maintenance scheduling.

The first prototype may use openly permitted synthetic data to validate the pipeline, but it must label those results synthetic and must not claim real-rail predictive performance from them.

**Supplemental sensor-data lead (separate scope; reuse rights unresolved):** the 2025 [Train track environmental data for predictive maintenance dataset](https://zenodo.org/records/17607068) (DOI https://doi.org/10.5281/zenodo.17607068) lists a 65.6 MB CSV of 40 Hz on-board inertial and environmental measurements plus GPS-tagged points from a Madrid–Barcelona run. This can be considered for a distinct schema/signal-quality pipeline exercise—not as track-geometry labels or proof of detecting physical defects. The current Zenodo record says “Dataset Open” and lists a download, but its rendered Rights → License field is blank; use for extraction, model training or redistribution remains `data_access_unresolved` until explicit rights terms are confirmed. The listed MD5 is `cc24e0419d1bb820abb4bce388350518`. Do not publish GPS-linked route traces or derivative location outputs without a rights and release review. See [TRANSPORT-RAIL-001A #7452](https://github.com/Luminous-Dynamics/symthaea/issues/7452).

#### Preregistered benchmark execution contract

**Data-admission gate (run before modeling):**
1. Find an actual dataset with documented permission for the proposed use, or record **data access unresolved**. An official research report is evidence about a method, not proof that the underlying telemetry can be downloaded or redistributed.
2. Freeze a source manifest and extraction/normalization code identity. Record licence/terms, unit definitions, coordinate/chainage semantics, segment identity, measurement channel, direction, instrument/method, calibration, missingness, quality flags, resampling, time coverage, and maintenance-history availability.
3. If chainage/segment alignment or the meaning of a field cannot be resolved, quarantine it or mark it unresolved. Do not silently merge tracks, interpolate missing inspections into facts, or coerce unknown units to a default.
4. Record whether maintenance events are actual retrospective events, interventions known at forecast time, or planned interventions. These are different information sets.

**Two separate forecast estimands (never combine):**
- **Operational forecast:** use only data that would have been available at the declared forecast issue time. Past maintenance can be used. Planned future work can be used only if it was recorded as known at issue time. Unexpected maintenance that occurs later must not be leaked into the predictor.
- **Post-intervention/conditional forecast:** after an intervention actually occurs, update the state using the recorded intervention and subsequent measurement. Test whether the model correctly represents the new condition trajectory/reset. Do not describe this as forecasting the unknown intervention in advance.

This distinction matters because FRA RR 26-01 says maintenance-event covariates improved forecasts but also describes the work as an initial investigation with case studies as a next research step. Reproducing that finding requires clarifying the event timing/information available in the data; it is not by itself field validation of a general predictor.

**Frozen comparison plan:**
1. Use rolling-origin evaluation in chronological order. Select training/validation/test cutoffs before model fitting; no random row split as the primary evaluation.
2. Add whole-segment/route holdouts where source identity and population size permit; use the current qualified split/independence owners rather than a local split engine. Declare whether the claim is future-window, segment, route, region or operator transfer.
3. Compare against persistence/last value and a simple trend baseline, then a reproducible SARIMAX and/or Prophet-style model if data shape and assumptions fit, the current owner/operator model when supplied, and only then Symthaea/HDC/CfC experiments.
4. Define target geometry channels and forecast horizons before looking at test results. If the operationally meaningful horizon is unknown, run a labelled exploratory horizon analysis and preregister the confirmatory horizon later.
5. Primary evaluation should include predeclared threshold-crossing detection/underprediction and warning lead time at owner-supplied, versioned limits; secondary results can include MAE/RMSE, calibration, false-alert burden and subgroup/site variation. Thresholds come from the applicable current owner/operator rules, not an AI-chosen value.
6. Keep results separated by route/segment, asset class, inspection spacing, instrument/source, maintenance state and operating/environment regime where samples allow. Show the population and uncertainty for each slice; don't infer absent subgroup performance.
7. Freeze predictions and target labels before the independent evaluator runs. The evaluator recomputes data checks, split membership, metrics and summaries from the frozen artifacts. It must include null/missing cases and inspect data leakage.

**Mandatory hostile/negative controls:**
- shuffled target and a source/study/segment ID-only model;
- future actual maintenance event accidentally exposed to an operational-forecast feature;
- duplicated/adjacent segment windows split across train and test;
- reversed chainage/direction or mismatched units;
- a segment whose inspection becomes stale or missing;
- one route/instrument dominates the aggregate score while another deteriorates;
- post-maintenance reset wrongly counted as model success or failure without recording intervention timing;
- a model that predicts average error well but systematically under-predicts critical deterioration;
- synthetic rows mixed into the real evaluation denominator.

**Stop / advance rules:**
- Stop data claims if access rights, source lineage, target definition, unit mapping or intervention timing remains unresolved.
- Stop any accuracy claim if baseline replication cannot be reconciled to the described input population or if leakage controls fail.
- Advance to a reviewable research result only if the declared held-out evaluation, uncertainty report, adverse-slice report and independent metric receipt are complete.
- A model result may inform an inspection-priority proposal, but cannot alter allowable operating limits, suppress an inspection, authorize a speed restriction/removal, or dispatch a train. Those remain operator/authority decisions.

**Additional railway-positioning lead (rights scope unresolved):** the 2025 [OnboardEU track-selective georeferencing sample](https://zenodo.org/records/16368240) exposes a railway-network GeoPackage, GNSS/IMU metadata, `onboardeu_001.h5`, and a georeferenced output file. Its README says the licence is in `license.txt`, but the record lists a file named `licencse.txt` (misspelled) containing CC BY 4.0 text, while the record-level License field is blank and DLR/AIT/i4M are named copyright holders. Treat this as `license_scope_unresolved` until the record owner confirms the intended scope/filename. If resolved, its appropriate first use is a separate positioning/georeferencing replay and uncertainty audit—not a track-geometry deterioration benchmark. The reported algorithm output is not independent ground truth. See [the source README](https://zenodo.org/records/16368240/files/README.md) and [the bundled licence text](https://zenodo.org/records/16368240/files/licencse.txt?download=1).

### B4. Cross-project contracts

- **Symthaea:** canonical asset/observation identity; separate as-designed/as-built/as-operated/observed/inferred/synthetic states; predictor identity/calibration/uncertainty; condition trajectories and decisions.
- **Symtropy:** exact read-only projections, scenario/configuration identity, simulation capability declaration, numerical integrity, replay and counterfactual outputs. It must not write simulator output back as an observed track state.
- **Sol-Atlas:** separate physical rail/road graph, communication graph, evidence/reconciliation graph and infrastructure-dependency graph. Every layer carries source, timestamp, resolution/CRS and confidence; unresolved contradictions remain visible.
- **Mycelix:** source rights, evidence lineage, operators/reviewer identities, maintenance/procurement/work orders and community/service outcomes.
- **Independent evaluator:** independently recompute data integrity, splits, metrics and report totals; code path or organization independence must be stated accurately.
- **Operator and authority:** own allowable limits, inspection policy, maintenance actions, route safety and operational authorization.

### B5. South African data and standards path

For a South African deployment, the first useful conversation is likely with the asset owner/operator and the Railway Safety Regulator (RSR), not an attempt to scrape live operational data.

- The RSR's [NIIMS NextGen update (22 June 2026)](https://www.rsr.org.za/niims-nextgen-strengthening-railway-safety-through-digital-oversight/) describes an Operator Asset Management Module for uploading asset registers and maintenance plans, alongside occurrence and safety oversight functions. This is a **potential authorized interoperability/source-discovery lead**, not evidence that the data are public or available to Symthaea.
- That same RSR update says Phase 2 was planned for **1 November 2026** for Safety Permit, Audit and Inspection Management. As of 15 October 2026, this date is still in the future; do not assume those modules are live. Until an official go-live notice confirms it, operators must continue using the then-current approved processes.
- The RSR's [railway safety standards index](https://www.rsr.org.za/categories_rt/railway-regulatory-tools/railway-safety-standards/) lists, among others, SANS 3000-2-2 for track and associated civil infrastructure and SANS 3000-2-2-1 for level crossings. Its [2026 standards-development information](https://www.rsr.org.za/wp-content/uploads/2026/03/Standards-Development.pdf) describes a modernization effort covering track infrastructure, rolling stock, signalling/telecommunications, operational principles and overhead traction equipment. Treat standards as versioned, authority-owned references; do not freeze a threshold from an old index page.
- South African evidence should be segregated from FRA-derived methods and US data. A forecast trained on US instrumenting, track practices, maintenance records and geography is an external prior, not a locally validated model.
- Mycelix may preserve source links, approved access terms, lineage and maintenance-event references, but must not create a competing regulator/operator asset register. Sol-Atlas may display an authorized read-only projection. Access control, purpose limitation and operator permission govern any real records.

## C. Joint opportunity: living materials for railway infrastructure

Rail offers an interesting application for self-healing concrete, but the specific asset matters:

1. **Concrete sleepers / precast elements:** potentially compare bioactive/self-healing repair formulations for small cracks, but include fatigue, fastening/interface behavior, moisture ingress and dimensional stability. Qualification requirements for rail components are not interchangeable with ordinary concrete cubes.
2. **Slab-track and track-support concrete:** evaluate crack-width control, internal curing and non-living or microbial sealing systems against stiffness, fatigue, settlement compatibility, drainage and repairability.
3. **Culverts, drainage channels, retaining structures and bridge elements:** these can be more appropriate early monitored cases for crack-sealing studies than primary load-critical track elements, subject to the asset owner and engineer's approval.
4. **Trackside sensor mounting and enclosures:** durability, ingress and maintainability may be a useful non-structural pilot with a lower consequence of failure, if the owner agrees.

Do not start a biological concrete pilot in a critical sleeper, turnout, bridge or tunnel merely because it is an exciting application. Select the first pilot by consequence of failure, available baseline evidence, inspectability, reversibility and owner permission.

## D. Prioritization

| Priority | Work package | Why first | Gate |
|---|---|---|---|
| P0 | Upgrade Symthaea material and infrastructure evidence contracts; distinguish measured, lab, literature and simulated claims | Prevents fabricated confidence from entering later models | Focused Rust tests and independent malformed-input controls on exact source head |
| P1 | Track-geometry benchmark and conventional forecast baseline | Existing external methods give falsifiable comparisons and a direct maintenance use case | Rights-cleared dataset, frozen manifest, future/route-held-out evaluation |
| P1 | Concrete healing literature/evidence registry and conventional control matrix | Avoids optimizing biology without physical baseline | All candidate claims linked to methods/source; comparability and risk fields complete |
| P2 | Calibrated property-specific predictor | Enables honest ranking and transfer checks | Predeclared holdout/calibration criteria and independent evaluator |
| P2 | Laboratory self-healing comparisons | Tests whether crack sealing improves material behavior rather than image appearance alone | Qualified lab, matched controls, replicates, biosafety review where relevant |
| P3 | Whole-life track maintenance optimizer and materials-aware digital twin | Uses qualified predictors and constrained model outputs | Site-specific verification/validation, owner review and safe read-only operation |
| P4 | Monitored pilot / real infrastructure recommendation | Highest consequence; must follow prior gates | Qualified professional approval, owner/regulator permission, field QA and withdrawal/rollback plan |

## Research reference set

### Living/self-healing concrete
1. Ranjith & Das (2025), “Self-healing bacterial concrete: A bibliometric overview and state-of-the-art review on fundamentals, techniques and future perspectives,” *Journal of Building Engineering*, 104, 112324. https://doi.org/10.1016/j.jobe.2025.112324
2. Wong et al. (2024), “Advances in microbial self-healing concrete: A critical review of mechanisms, developments, and future directions,” *Science of the Total Environment*, 947, 174553. https://doi.org/10.1016/j.scitotenv.2024.174553
3. “State-of-the-art review on self-healing in mortar, concrete, and composites” (2024). https://doi.org/10.1016/j.cscm.2024.e03298
4. FHWA, “Internally Curing Concrete Produces EPIC² Results” (2023). https://www.fhwa.dot.gov/innovation/innovator/issue98/page_01.html
5. FHWA, “Enhance Concrete Performance with Internal Curing” (2025). https://www.fhwa.dot.gov/innovation/innovator/issue103/page_06.html
6. “Effect of bacteria on uncracked concrete mechanical properties correlated with damage self-healing efficiency — A critical review” (2024), *Developments in the Built Environment*, 17, 100301. https://doi.org/10.1016/j.dibe.2023.100301
   This review reinforces that intact-material mechanical performance must be evaluated alongside post-crack healing; a candidate that seals cracks but degrades the uncracked matrix is not an acceptable improvement.
7. Tumwiine et al. (2025), “On the use of self-healing materials in concrete technology: A comprehensive review of materials, mechanisms, and field applications,” *Materials Today Chemistry*, 48, 103001. https://doi.org/10.1016/j.mtchem.2025.103001
8. “A Review of Self-healing Concrete: Mechanism, Performance, Techniques and Machine Learning Approaches” (2026), *International Journal of Concrete Structures and Materials*. https://doi.org/10.1186/s40069-026-00932-2
9. “Addressing the need for standardization of test methods for self-healing concrete: an inter-laboratory study on concrete with macrocapsules” (2020). https://doi.org/10.1080/14686996.2020.1814117
10. Jakubowski & Tomczak (2025), “Dataset for developing deep learning models to assess crack width and self-healing progress in concrete,” *Scientific Data*, 12, 165. Data descriptor: https://doi.org/10.1038/s41597-025-04485-z. Candidate Zenodo deposit: https://doi.org/10.5281/zenodo.14568863; exact file rights/license still require verification before use.

### Biosafety and environmental persistence

11. “Molecular and Biosafety Perspectives of Bacterial Self-Healing Concrete: From Sporulation and Biomineralization to Public Health Implications” (2026 review). https://pmc.ncbi.nlm.nih.gov/articles/PMC13566579/

This recent review highlights strain persistence, genetic stability, horizontal gene transfer, environmental microbiome interaction and occupational/environmental exposure as matters requiring assessment. It is a narrative review and does not itself establish field-scale exposure risk for any specific concrete formulation.

### South African rail governance and data
12. Railway Safety Regulator, [NIIMS NextGen | Strengthening Railway Safety Through Digital Oversight](https://www.rsr.org.za/niims-nextgen-strengthening-railway-safety-through-digital-oversight/) (22 June 2026).
13. Railway Safety Regulator, [Railway Safety Standards index](https://www.rsr.org.za/categories_rt/railway-regulatory-tools/railway-safety-standards/).
14. Railway Safety Regulator, [Standards Development overview](https://www.rsr.org.za/wp-content/uploads/2026/03/Standards-Development.pdf) (March 2026).

### Rail condition, detection and maintenance
15. FRA, RR 26-01, “Track Geometry Predictions Using Point and Segment Processing.” https://railroads.fra.dot.gov/elibrary/track-geometry-predictions-using-point-and-segment-processing
16. FRA, RR 26-02, “Implementation of Advanced Track Geometry Forecasting Models” (page updated 30 July 2026). https://railroads.fra.dot.gov/elibrary/implementation-advanced-track-geometry-forecasting-models
17. FRA, RR 21-18, “Feasibility of Applying Model-Assisted Probability of Detection to Track Inspection.” https://railroads.fra.dot.gov/elibrary/feasibility-applying-model-assisted-probability-detection-track-inspection
18. Network Rail, CP7 Route Services Delivery Plan Year 2 update (2025), describing predictive asset-maintenance and earthworks-insight programmes. https://www.networkrail.co.uk/wp-content/uploads/2025/07/Route-Services-CP7-Delivery-Plan-Year-2-update.pdf

### South African context
19. South African Department of Transport, COTO Standard Specifications for Road and Bridge Works portal. Its page lists October 2020 chapter versions and states the earlier August versions were superseded. Always verify the current applicable edition and project-specific contractual requirements before using numerical limits. https://www.transport.gov.za/?page_id=1155

### Explicitly licensed self-healing datasets for empirical follow-up

20. Justo-Reinoso et al. (2022), University of Bath Research Data Archive, “Air-entraining admixtures as a protection method for bacterial spores in self-healing cementitious composites: Healing evaluation of early and later-age cracks,” BATH-01087; data record lists CC BY 4.0 for all three Excel files. https://doi.org/10.15125/BATH-01087. Related publication: https://doi.org/10.1016/j.conbuildmat.2022.126877
21. Paine et al. (2017), University of Bath Research Data Archive, “Dataset for application of expanded perlite encapsulated bacteria and growth media for self-healing concrete,” BATH-00448; workbook and data documentation list CC BY 4.0. https://doi.org/10.15125/BATH-00448
22. Wang et al. (2026), “A mechanistic study of a novel self-healing concept for cement enabled by large clinker particles: Recovery of transport properties and microstructure characteristics,” *Construction and Building Materials*, 516, 145707. https://doi.org/10.1016/j.conbuildmat.2026.145707
23. Wang & Wu (2025), “A preliminary study on an effective and simplistic self-healing concept for cement using coarse clinker particles as the healing agent,” *Cement and Concrete Research*, 193, 107859. https://doi.org/10.1016/j.cemconres.2025.107859. The article record states that data will be made available on request.

### Supplemental rail sensing source (rights pending)

24. Bustamante & Trincado Castán (2025), “Train track environmental data for predictive maintenance,” Zenodo dataset, 65.6 MB 40 Hz onboard inertial/environmental data with GPS-tagged samples; DOI https://doi.org/10.5281/zenodo.17607068. The current record's License field is blank and does not establish reuse rights; source use is gated pending clarification.
25. Van Mullem et al. (2020), “Addressing the need for standardization of test methods for self-healing concrete: an inter-laboratory study on concrete with macrocapsules,” *Science and Technology of Advanced Materials*, DOI https://doi.org/10.1080/14686996.2020.1814117.
26. “Evaluation of Methodologies for Assessing Self-Healing Performance of Concrete with Mineral Expansive Agents: An Interlaboratory Study” (2021), *Materials*, DOI https://doi.org/10.3390/ma14082024.

## Claim ceiling

This plan establishes a research direction, a candidate/measurement vocabulary and falsifiable next steps only. It does not establish that any proposed bioactive concrete is safe, structurally superior, economical or field-ready; that Symthaea outperforms track-geometry forecasting baselines; or that Symtropy's existing physics engine can simulate fracture or railway dynamics adequately. Those claims require measured evidence and an independently reviewed, application-specific validation argument.
