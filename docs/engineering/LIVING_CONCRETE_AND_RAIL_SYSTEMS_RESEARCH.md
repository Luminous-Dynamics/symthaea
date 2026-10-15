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

## Claim ceiling

This plan establishes a research direction, a candidate/measurement vocabulary and falsifiable next steps only. It does not establish that any proposed bioactive concrete is safe, structurally superior, economical or field-ready; that Symthaea outperforms track-geometry forecasting baselines; or that Symtropy's existing physics engine can simulate fracture or railway dynamics adequately. Those claims require measured evidence and an independently reviewed, application-specific validation argument.
