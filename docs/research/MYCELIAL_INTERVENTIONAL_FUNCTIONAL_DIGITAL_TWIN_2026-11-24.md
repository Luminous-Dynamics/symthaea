# Mycelial Interventional Functional Digital Twin
## Research protocol and benchmark specification v0.1 — 24 November 2026

**Status:** research proposal and software-integration protocol.  
**Implementation status:** this document does not claim that raw datasets have been ingested, biological experiments performed, functional predictions validated, or benchmark targets achieved.  
**Scope:** connect the existing reconstruction, acquisition, morphogenesis, transport, model-adequacy and value-of-information work into a falsifiable experimental programme.

## 1. The research question

Can we predict, before observing the outcome, how a living mycelial network will remodel and redistribute resources when substrate availability, sink demand, hydration or network connectivity changes?

The objective is not simply to reconstruct a visually plausible graph. It is to build an **evidence-qualified, forward-predictive and intervention-testable digital twin** whose claims remain traceable to microscopy, calibration, independent functional assays and biological replicates.

The central cycle is:

1. Observe the specimen and attest what was actually acquired.
2. Reconstruct a distribution of feasible structural hypotheses, including unresolved and unobserved regions.
3. Predict future morphology and separately typed transport outcomes.
4. Identify which additional measurement would most reduce decision error at a declared cost.
5. Predict the consequences of a specified intervention before its outcome is revealed.
6. Compare predictions with independently measured, blinded outcomes.
7. Update the model only through a versioned, replayable evidence record.

A graph edge is not a measured flow path. A shortest path is not a resource flux. A cord is not a hydraulic conductance measurement. A posterior normalized over a candidate inventory is not proof that the inventory contains the biological truth.

## 2. What the current evidence supports

### 2.1 Dynamic fungal networks and cytoplasmic flow

The 2025 study *A travelling-wave strategy for plant–fungal trade* reports large-scale tracking of more than 500,000 fungal nodes and approximately 100,000 cytoplasmic-flow trajectories. It provides a reference for time-dependent network growth and intrahyphal transport, while its plate-based network geometry should not be mistaken for a complete model of three-dimensional soil networks.

Reference: https://doi.org/10.1038/s41586-025-08614-x

### 2.2 Demand-responsive cordogenesis

The 2026 *Agaricus bisporus* cordogenesis study describes changes in cord formation and widening during new-substrate colonization and fruiting development. It also reports substantial cultivation-cycle variation and no significant general correlation between cord abundance and total mushroom yield.

Reference: https://doi.org/10.1016/j.funbio.2026.101799  
Dataset record: https://figshare.com/s/b241753378d5252ff808

This rules out a simple “more cords means more production” objective. Cord geometry should instead be tested as a conditional predictor of resource delivery under specified distances, local depletion, hydration, developmental demand and network state.

### 2.3 Isotope-based source-region attribution

The 2026 stable-isotope study of *A. bisporus* reports source-region-labelled nitrogen incorporation into fruit bodies, with effects related to colonization time, cord abundance, local depletion and hydraulic resistance. It also reports substantial heterogeneity between individual mushrooms. Its experimental unit must be respected: mushrooms sampled from the same cultivation box are subsamples, not independent biological replicates.

Reference: https://doi.org/10.1111/1462-2920.70345  
Raw-data DOI reported by the article: https://doi.org/10.6084/m9.figshare.30818828

**Important assay interpretation:** isotope enrichment in a fruit body is evidence of incorporation of label originating from the labelled substrate region. It is not automatically proof of direct uptake by the fungus: microbial assimilation and transfer within compost may contribute. The benchmark must preserve the assay's exact interpretation, source region, recovery and uncertainty. Do not label this outcome “direct fungal uptake” unless an assay-specific control supports that stronger attribution.

The cordogenesis and isotope studies are distinct cohorts. Do not join their records, replicate IDs or outcomes unless an independently verifiable sample-level key proves that they refer to the same biological units.

### 2.4 Microbial dispersal is a separate functional layer

The 2026 Hyphal Highways Device study examines bacterial dispersal along *Pythium ultimum* hyphae under unsaturated conditions. *P. ultimum* is an oomycete, not a true fungus. The experiment is a valuable reference for extracellular liquid constraints and microbial movement, not a surrogate measurement for fungal intrahyphal cytoplasmic flow or isotope transfer.

Reference: https://doi.org/10.1093/femsml/uqag035

A related benchmark issue has cited the suffix uqag017; the DOI above is the identifier to use for this Hyphal Highways Device study. Verify citations at ingestion time.

### 2.5 Smart microscopy and interoperability

The 2026 review *Smart microscopy: adaptive microscope control to improve the way we see life* defines smart microscopy as real-time analysis, feedback control and automated actuation. It distinguishes quality-, event-, target-, information- and outcome-driven acquisition, and emphasizes trade-offs among resolution, speed, signal-to-noise, field of view, photon budget and specimen viability: https://doi.org/10.1038/s44303-026-00145-y.

A separate 2025/2026 *Nature Communications* study demonstrates outcome-driven microscopy by combining image feedback and optogenetic actuation to control cell migration and nucleocytoplasmic transport: https://doi.org/10.1038/s41467-025-67848-5. This is an important field direction, but it does not mean Mycelial should manipulate fungal biology with light. For Mycelial, “outcome-driven” initially means selecting observations that resolve a prespecified biological decision; sample interventions remain separate, preregistered experiments.

Symthaea should not recreate mature microscope drivers. It should interoperate with hardware-neutral experiment descriptions and established image formats, beginning with offline and recorded-session adapters.

References:
- Adaptive smart-microscopy review: https://doi.org/10.1038/s44303-026-00145-y
- Smart Microscopy Working Group interoperability roadmap: https://doi.org/10.1515/mim-2025-0029
- useq-schema: https://github.com/als-computing/useq-schema
- OME-NGFF: https://ngff.openmicroscopy.org/
- Micro-Manager: https://micro-manager.org/

## 3. Paradigm shift: one specimen, several non-interchangeable processes

Represent the living specimen as a shared, time-indexed spatial substrate with distinct biological state and observation layers.

| Layer | State or observations | What it does not prove |
|---|---|---|
| Structural network | Hyphal segments, junctions, candidate fusions, length, radius, branching, extension, retraction and uncertainty | Viability or transport |
| Intrahyphal cytoplasmic transport | Separate forward and reverse particle-stream trajectories, speed distributions, direction switches and observation windows | Net nutrient delivery or extracellular liquid flow |
| Extracellular/apoplastic hydraulics | Liquid connectivity, saturation, water potential, pressure gradients and assay-specific hydraulic measurements | Intracellular streaming |
| Resource source–sink transfer | Source-region nutrient availability, depletion, sink development, isotope ratios, recovery and arrival/censoring | A unique path through a reconstructed graph unless independently established |
| Microbial dispersal | Organism identity, inoculation point, liquid conditions, cell counts and time-to-arrival | Fungal nutrient transfer or cytoplasmic flow |
| Morphogenesis | Colonization proxies, cord area, width distributions, growth-front displacement and primordium/fruiting-body area | Biomass, conductance or sink strength without independent calibration |

Each layer shares immutable specimen, coordinate-frame and timebase identities, but retains its own assay protocol, calibration, evidence digest, uncertainty and missingness state.

Do not collapse forward and reverse cytoplasmic streams into one signed net-flow value: simultaneous antiparallel streams can coexist. Do not use bacterial arrival as a nutrient-transfer label. Do not use optical intensity as biomass without independent calibration.

## 4. Model ladder: earn each additional mechanism

Every model is evaluated on the same frozen split, forecast horizon, source evidence and functional endpoint. More complicated models do not receive credit merely for fitting the training data better.

| Model | Inputs | Question |
|---|---|---|
| M0 — persistence | Last observed value / event state | Is there useful signal beyond doing nothing? |
| M1 — trend and covariates | Time trend, substrate, hydration, treatment and developmental covariates | Do simple trends explain the outcome? |
| M2 — static morphology and geometry | Frozen image features, measured lengths, widths and coordinates | Does the current measured structure add value? |
| M3 — dynamic structure | Time-indexed topology, growth, branching, cordogenesis and retraction | Does network remodeling improve forward prediction? |
| M4 — source–sink state | M3 plus independently observed source availability, depletion and sink-development indicators | Does resource demand explain structural adaptation and transfer? |
| M5 — hydraulic/resource coupling | M4 plus measured liquid state, water-potential or pressure evidence and independently measured tracer outcomes | Does the coupled physical model improve functional predictions? |
| M6 — multiplex dynamics | M5 plus independently observed intrahyphal forward/reverse streams and, in a separate task, microbial dispersal | Do distinct transport processes add predictive value when modeled separately? |

Use a physical resistance approximation only where the relevant geometry, boundary conditions, fluid properties and compartment assumptions are justified. A Hagen–Poiseuille-style relation may be a useful sensitivity model for a declared tube-flow regime; it is not a universal law for every hyphal or extracellular transport pathway. Record the assumptions and test sensitivity to their violation.

Symthaea-native HDC and CfC/LTC remain optional components:
- HDC may retrieve local structural motifs and analogous episodes, but similarity is not a biological posterior or evidence of physical contact.
- CfC/LTC may predict irregularly sampled dynamics, but the output is a model prediction, not a measurement.
- A hybrid is accepted only if it improves held-out proper scores or decision value over M0–M5 at equal measurement cost and compute budget.

## 5. The decisive experiments: predict interventions, not just correlations

The digital twin should eventually answer questions such as:

- If a source region becomes depleted, where does resource attribution shift?
- If a new substrate patch is encountered, where and when should directional extension or cordogenesis occur?
- If a sink develops or expands, does the network remodel in a way that improves measured transfer?
- If hydration changes, which predictions change and which remain stable?
- If a branch is interrupted, what is the predicted effect on a measured downstream outcome, and how does the network recover?

Potential collaborator-designed, controlled interventions:

| Intervention | Prediction to freeze before outcome reveal | Essential controls |
|---|---|---|
| Local source depletion or replenishment | Source-region attribution and time course of tracer incorporation | Matched sham, nutrient and isotope controls, manipulation-success verification |
| Hydration / saturation change | Morphology, liquid state and functional-transfer changes | Matched moisture control, temperature and off-target monitoring |
| New substrate patch | Direction and timing of extension, cordogenesis and colonization | Matched substrate patch, sham placement, baseline imaging |
| Sink appearance / developmental demand | Structural remodeling and source-to-sink redistribution | Developmental-stage matching, independent sink measurements |
| Selective branch interruption | Change in independently measured transfer and rerouting | Sham manipulation, unaffected-region control, verified interruption and regrowth observation |

For every intervention, preregister:
- the biological question, primary endpoint and smallest effect of interest;
- independent experimental unit, block, randomization and blinded outcome procedure;
- baseline evidence and graph-hypothesis inventory digests;
- intervention target, time, dose/intensity and spatial footprint;
- manipulation-success and off-target observations;
- post-intervention imaging and independent functional observations;
- missingness, censoring, exclusions and analysis plan.

Experiments require qualified biological collaborators, approved protocols and suitable laboratory equipment. This software protocol neither authorizes physical interventions nor creates microscope actuation.

### 5.1 Active acquisition under hard sample-health constraints

The active-sensing policy should not optimize only information value or collapse all costs into one scalar. The policy first filters actions against specimen and instrument constraints, then ranks only admissible actions by the existing decision-value measure.

\[
a^* = \arg\max_{a \in \mathcal{A}_{\mathrm{admissible}}} \operatorname{NetVOI}(a)
\]

subject to:
- cumulative upper-bound photon dose per specimen × spatial cell × illumination channel remains within its calibrated assay-specific limit;
- session duration and action count remain within the approved acquisition budget;
- ROI area and conservative upper-bound local temperature rise remain within policy;
- every required illumination channel has a calibrated wavelength and dose estimate;
- coordinate frame, spatial footprint, acquisition model and calibration identities match the source evidence.

A hard limit is not a cost to be traded against expected information gain. An ineligible action stays ineligible even if it would resolve every remaining graph hypothesis. Overlapping ROIs accumulate exposure on shared spatial cells; they must not reset dose simply because the ROI name changes.

The proposed Rust implementation is split into two layers:
- PR #13379 computes expected reduction in Bayes decision risk minus declared acquisition cost.
- Issue #13486 / draft PR #13487 adds specimen-health admissibility using cumulative dose, session time, ROI, temperature and channel-calibration evidence.
- Issue #13488 separates fungal photobiological signalling from phototoxicity and defines a species/strain/stage-specific exposure/response model.
- The sample-health gate accepts only a replay-qualified VOI evidence bundle: the upstream report is independently recomputed against the exact source skeleton, candidate inventory, posterior, decision table and acquisition-response models.
- The final sample-health receipt binds both the upstream report and qualified source-evidence digest. Its own `validate_against` operation replays the hard-constraint filtering and final recommendation against the exact policy, cumulative exposure ledger and acquisition proposals; a changed recommendation or receipt is rejected.

Hard constraints require exact calibration binding:
- the health-model digest in every acquisition proposal must match the versioned sample-health policy;
- the exact compiled acquisition-plan digest and device-capability-manifest digest are bound to the active recorded session;
- the spatial-footprint digest commits to coordinate frame, compiled plan, device capability and canonical source-region cell IDs;
- each illuminated channel has a policy-bound wavelength band and calibration digest;
- each channel has a specimen-plane maximum irradiance and maximum uninterrupted illumination duration;
- each proposed optical action is scheduled against a monotonic session timebase and must satisfy the minimum recovery interval since the last specimen illumination;
- each optical action supplies a conservative dose and irradiance upper bound, plus actual continuous illumination duration, for every required illumination channel;
- each historical exposure records its source receipt and calibration digest; an unqualified or incompatible historical exposure blocks further optical acquisition until reconciled;
- incomplete exposure history blocks optical acquisitions, but does not block non-optical human review;
- overlapping spatial footprints are accumulated by canonical source-region cell identity, not by ROI label.

The policy is advisory-only. It does not infer biological viability from photon counts, guarantee absence of phototoxicity, issue microscope commands or override a laboratory protocol. Dose, irradiance, continuous-dwell, recovery and temperature bounds must be established for the specific specimen, fluorophore, modality, instrument and assay by qualified personnel.

### 5.2 Photobiological signalling is not phototoxicity

A second fungal-specific issue prevents treating all illumination effects as damage. Fungi sense light as an environmental/developmental signal. The 2025 review *How fungi see the world* describes UV, blue, green, red and far-red photoreception, with blue light acting as a major regulatory cue and responses depending on fungal photoreceptors and signalling pathways: https://doi.org/10.1128/mmbr.00149-22.

Recent experiments illustrate why responses cannot be pooled into one generic curve:
- *Gibberella zeae* work reports wavelength- and irradiance-dependent effects of 385/405-nm illumination on mycelial growth and deoxynivalenol production, with ROS observations at high-irradiance 405-nm exposure: https://doi.org/10.1016/j.funbio.2025.101690.
- A *Fusarium solani* experiment reported shorter hyphal growth under 454-nm illumination while conidial quality/viability increased under its tested exposure protocol: https://pubmed.ncbi.nlm.nih.gov/32112251/.
- A 2025 *Hericium erinaceus* study reported different mycelial-growth responses under different LED spectra: https://doi.org/10.3389/ffunb.2025.1684852.

These results motivate a strict distinction:
1. **Phototoxicity:** stress, damage, bleaching, reduced viability and recovery.
2. **Photobiological response:** regulated changes in extension, branching, cordogenesis, sporulation, fruiting, pigments or metabolites.

An exposure can preserve viability while changing developmental behaviour; opposite endpoints can move in different directions. An imaging condition must therefore be treated as a possible biological treatment until its effect is shown negligible for the endpoint being predicted.

Issue #13488 specifies a separate, source-bound photobiological exposure ledger and response model. Draft PR #13492 implements the first typed Rust record: species, strain, stage, cultivation unit, substrate/hydration, spectrum, specimen-plane irradiance, fluence, photon dose, duty cycle, longest continuous dwell, ambient photoperiod phase, spatial footprint, acquisition plan and device identities. It stores the exact per-channel on/off interval inventory and a canonical schedule digest; validation rejects interval overlap, undeclared channels, and disagreement between the interval inventory and duty cycle, maximum continuous dwell or minimum fluence implied by the irradiance upper bound. Equal-fluence continuous and fractionated schedules therefore remain distinct and independently replayable. Its downstream benchmark is the observer-effect protocol: estimate endpoint-specific imaging-induced changes under randomized dark/ambient, handling-sham, low-exposure and matched-fluence schedule conditions. It adds separate response-observation types for extension, branching, cord area, conidiation, viability, ROS, analyte-specific metabolites and fruiting latency, with exact exposure binding and pre/during/post/recovery observation windows. This is an evidence schema, not a fitted light-response model. Unknown species/strain/stage or out-of-domain exposure schedules must produce an explicit unqualified response state in the future response-model layer; no wavelength-only correction factor is permitted.

The first controlled fungal study should include dark/ambient and low-exposure imaging controls, species-appropriate spectral bands, calibrated irradiance levels, continuous versus fractionated schedules at equal integrated fluence, temperature/handling shams, randomized independent cultivation units and post-exposure recovery observations. Such work requires qualified mycologists and approved protocols. PhotoFiTT and the cited fungal studies provide research design precedents, not universal Mycelial safe-dose thresholds.

#### Observer-effect benchmark

The microscope itself must be included in the causal graph. The acquisition policy is a treatment assignment, not just a logging covariate.

Design a randomized, cultivation-unit-level comparison with:
- dark/ambient reference;
- handling/device sham with no added optical exposure;
- the lowest experimentally validated imaging exposure;
- one or more higher, still laboratory-approved exposure patterns;
- continuous and fractionated schedules matched on integrated fluence;
- species-appropriate spectral bands and photoperiod phase.

Keep all fields, frames and follow-up observations from one cultivation unit in the same partition. Do not repeatedly expose the same specimen to sequential conditions and then treat the observations as independent; cumulative carry-over requires a separately randomized sequence/crossover design with a justified washout period.

Predeclare a smallest biologically relevant change separately for hyphal extension, branching, cordogenesis, conidiation, viability, ROS and each metabolite. Use blinded/manual or orthogonal assay measurements where possible, and record when the endpoint assay itself requires illumination. A dark reference must not be scored using an unrecorded optical procedure that silently reintroduces the treatment.

Proposed metrics and gates:
- **Observer-effect detection:** detect a predeclared exposure-induced endpoint shift with sensitivity ≥95% at ≤5% false alerts on a synthetic known-effect suite; these are software test targets, not biological performance.
- **False neutrality:** ≤5% of held-out treatment contrasts may be labelled “biologically negligible” when the independently measured effect exceeds the preregistered equivalence margin.
- **Equivalence, not non-significance:** an imaging protocol may be called endpoint-neutral only when a prespecified equivalence test places its confidence interval wholly inside the biologically justified margin. A non-significant difference alone is insufficient.
- **Matched-fluence discrimination:** predict endpoint differences between continuous and fractionated schedules on held-out cultivation units, even when integrated fluence is equal.
- **Dose-response calibration:** report endpoint-specific predictive intervals and calibration by species, strain, developmental stage, hydration and photoperiod phase.
- **Acquisition interference regret:** compare downstream morphogenesis/transport prediction error when illumination history is omitted versus included, using the same frozen holdout units.
- **Decision policy:** at equal specimen-health and acquisition budgets, minimize both biological observer-effect and downstream decision regret; report both rather than hiding one in a composite score.

The target is not “zero light effect” by assumption. It is to know when the microscope has changed the biology, which endpoints moved, and whether the resulting observations remain suitable for the scientific question.

PhotoFiTT (Del Rosario et al., *Nature Communications*, 13 December 2025; https://doi.org/10.1038/s41467-025-66209-6) provides an important methodological reason not to reduce specimen illumination to one integrated dose number. In mammalian-cell confocal experiments, longer low-intensity scans produced greater mitotic delay than shorter high-intensity scans at the same total light dose. This supports treating irradiance and continuous illumination duration as separate dimensions, and recovery as a variable to test. It does **not** provide fungal phototoxicity thresholds; none of its numerical values should be copied into Mycelial policy. Fungal-specific dose-response and recovery experiments are required before any instrument actuation could be considered.

Reproducible fluorescence microscopy guidance also emphasizes irradiance at the sample plane and illumination pattern, rather than nominal lamp power alone: https://doi.org/10.1038/s41592-021-01156-w.

Evaluation must compare fixed schedules, random sampling, confidence-only, entropy-only, VOI-only and VOI-plus-health-constraints at equal measurement budgets. Report both:
1. downstream decision regret at equal photon/time budget; and
2. photon/time budget required to reach equal decision quality.

Plot a Pareto frontier of decision regret against cumulative photon dose, acquisition time and independently measured specimen-health outcomes. The admissible-policy hard-constraint violation count must remain zero in synthetic and recorded-session replay. Do not infer specimen viability from dose alone: health outcomes need independent assay measurements and matched sham controls.

Also report dose-bound violations, sample-health outcome drift, missed rare events, calibration, abstention and human overrides separately. A single composite score must not conceal a policy that improves inference by damaging the specimen.

## 6. Three-tier evaluation programme

### Tier 0 — synthetic known-truth oracle

Build bounded synthetic networks in which the true graph and process state are known. Include:
- missing candidate edges, unobserved branches and false junctions;
- crossings, occlusion, z-order ambiguity and coordinate-transform errors;
- common-cause focal-plane, illumination and segmentation artifacts;
- dynamic branching, fusion, retraction and time-varying edge geometry;
- simultaneous antiparallel streams;
- source depletion, multiple competing sources, sinks and incomplete tracer recovery;
- hydration-dependent extracellular connectivity;
- bacterial dispersal with explicit liquid-film constraints;
- sham interventions, off-target damage and unsuccessful interventions.

Synthetic fixtures qualify inference, conservation bookkeeping and failure detection. They do not establish biological performance.

### Tier 1 — retrospective reproduction

Create separate, rights-qualified lanes:
1. Sten 2D fungal images and crossing/overlap data.
2. Sten 3D z-stack reconstruction and printed-prototype geometry, scored separately from biological images.
3. Fungal Network Analysis (FNA) images and reference processing.
4. SkelPy's published *Botrytis cinerea* 2D time series, with separate manual-trace, workflow-derived node, spatial-cell and trait-agreement tasks; the full raw series must be sourced separately, and its published preprocessing parameters must not leak from the 180-image tuning cohort into a test set.
5. 2025 AM-fungal travelling-wave growth and flow trajectories.
6. 2026 *Agaricus* cordogenesis/fructification.
7. 2026 *Agaricus* stable-isotope source-region attribution.
8. Hyphal Highways Device microbial-dispersal measurements.

Never merge tasks just because all involve fungal or fungal-like networks. In particular, HHD is a microbial-dispersal lane; isotope incorporation is a source-region transfer lane; cytoplasmic trajectories are an intrahyphal-flow lane.

### Tier 2 — prospective paired imaging and function

The highest-value study is a paired dataset in which the same independent cultivation unit has:
- calibrated time-resolved microscopy and an acquisition-domain manifest;
- source and sink identities with a measured nutrient/depletion profile;
- an independently measured hydration/liquid state;
- a prespecified tracer or functional-transfer assay;
- biological replicate and cultivation-unit identity;
- synchronized observation times and all assay-specific uncertainty.

This is the missing bridge between the current separate morphology and isotope cohorts. Design the pilot to estimate between-box and within-box variability, missingness and assay noise; choose sample size using a preregistered power or precision analysis rather than an arbitrary number of images. Individual mushrooms, fields of view, crops and time frames must not be treated as independent replicates when nested in one box or cultivation cycle.

### Tier 3 — blinded intervention validation

Only after the paired observational protocol is stable, test preregistered source depletion, hydration, new-substrate or branch-interruption interventions. Freeze model outputs and uncertainty before unblinding. Report both successful and failed manipulations.

## 7. Benchmark suite and proposed gates

All values below are **research targets, not achieved results**. Report denominators, independent-unit counts, effect sizes, clustered uncertainty intervals, missingness and all exclusions.

| Lane | Baseline / comparison | Primary metrics | Proposed gate |
|---|---|---|---|
| 2D filament and crossing reconstruction | Sten published protocol; FNA/reference annotations | Filament F1, crossing F1, precision/recall, length error | First reproduce the published task and splits. Then target >=0.94 filament F1 and >=0.92 crossing F1 on a larger replicate-disjoint set, without increasing false fusion |
| 3D geometry and topology | Sten 3D, printed known-geometry controls, adjudicated real z-stacks | Centerline/branch recall, crossing-versus-fusion error, geometric RMSE, graph-edit/topological error | Reproduce the narrow published 0.91–0.92 overlap result on its exact six-image scope before expanding; do not call it universal 3D SOTA |
| Skeletonization / temporal phenotyping | SkelPy, FNA and independent manual traces on matched images | Centerline distance, independently adjudicated node precision/recall, spatial-cell presence/absence F1, trait agreement, component fragmentation, tip/branch counts, width error, runtime and peak memory | Reproduce the 15-image manual comparison, 30-image workflow-derived node check, 30-cell single-user presence/absence check, trait correlations and separate throughput runs; freeze parameters on train/dev only and obtain an untouched replicate-disjoint test set |
| False fusion | Adjudicated crossing-negative and z-order controls | False-fusion rate, precision by contact type | <=0.5% on a sufficiently large, independently adjudicated difficult-negative set |
| Graph adequacy | Exact bounded enumerator; deliberately incomplete candidate families | Exact agreement, severe-miss detection, false adequacy, false abstention | 100% exact-oracle agreement on bounded cases; >=95% detection of predefined severe misspecification with <=5% false adequacy; <=5% false abstention on matched adequate controls |
| Morphogenesis forecasting | Persistence, linear trend, covariate-only, morphology-only and dynamic-structure baselines | MAE/RMSE, CRPS, event precision/recall, interval coverage | Beat persistence and linear trend on frozen forward-time, independent-unit holdouts; target >=10% relative error reduction over best matched morphology-only baseline |
| Cytoplasmic flow | Last-value, direction persistence, geometry and time-series baselines | Direction accuracy, forward/reverse speed distribution score, switch timing, calibration | Demonstrate added value over simple baselines on held-out biological networks; preserve simultaneous forward/reverse streams |
| Source–sink transfer | Source availability/distance, static topology, measured geometry, depletion-only, hydraulic model | Source-region attribution score, tracer-incorporation prediction error, arrival/censoring score | Target >=10% relative error reduction over strongest matched topology-only baseline on a preregistered independent cohort |
| Microbial dispersal | Distance-only, liquid-only, hypha-only and combined predictors | Arrival probability, time-to-arrival score, censored survival score | Test the incremental contribution of hyphae and liquid state on held-out device/biological replicates; do not substitute this for nutrient-transfer validation |
| Interventional prediction | Sham, covariate-only and observational model | Effect-sign accuracy, effect-size error, interval coverage | Predict held-out intervention-effect direction and magnitude better than matched non-interventional baselines |
| Active acquisition | Random, fixed schedule, confidence, entropy, topology, expert heuristic and existing decision-value evaluator | Decision regret, error at equal budget, acquisition time/photons/dose | Target >=10% lower decision loss at equal acquisition budget, or >=10% lower acquisition cost at equal decision quality |
| Calibration | Uncalibrated and matched probabilistic baselines | Proper log score, Brier score, CRPS, interval coverage/width, calibration by subgroup | Nominal 90% intervals assessed at biological-unit level, with coverage and width reported together |
| Evidence and leakage | Source inventory and split-manifest adversarial fixtures | Inventory coverage, evidence-chain validity, unit overlap, time violations | 100% accounting against declared acquisition scope; zero independent-unit overlap; zero future-to-context leakage; unsupported rights block processing |

The Sten published 2D results (filament detection F1 0.88–0.94 and crossing F1 0.89 on 135 images) and the narrow Sten 3D overlap result (F1 0.91–0.92 on six manually annotated real images) are comparison anchors, not universal state-of-the-art ceilings. The new benchmark must reproduce comparable tasks and data splits before claiming improvement.

For the three-cycle longitudinal *Agaricus* series, leave-one-cycle-out evaluation is necessary but statistically weak: three held-out folds do not support a definitive generalization claim. Report every fold and its uncertainty. A larger prospective set of independent cultivation cycles is needed before asserting robust superiority.

## 8. Industry position: integrate where others are mature, compete where evidence is missing

| System | Established strength | Symthaea strategy |
|---|---|---|
| ZEISS arivis | Multidimensional image analysis, 3D/4D workflows and AI-assisted segmentation | Import exports and preserve source/calibration evidence; add hypothesis adequacy and downstream functional qualification |
| Imaris | Mature 3D/4D visualization, segmentation and quantitative microscopy workflows | Interoperate with its outputs; do not rebuild its visualization stack |
| Fiji SNT | Open, scriptable multidimensional tracing and time-lapse analysis | Use as a reproducible tracing baseline and preserve tracing provenance |
| Fungal Network Analysis | Fungal-specific image processing, skeletonization, width estimation and weighted graph construction | Reproduce its pipeline; compare connectivity-preserving reconstruction and functional prediction |
| SkelPy | GUI-based fungal skeletonization, customizable preprocessing, manual trace comparison and high-throughput image processing | Reproduce its skeleton-distance and runtime/memory measures; add topology, uncertainty and independent functional endpoints rather than comparing unlike metrics |
| Sten methods | Explicit filament-overlap and 3D z-stack reconstruction reference methods | Reproduce published narrow evaluations before extending the benchmark |
| Smart-microscopy ecosystems | Hardware interfaces, acquisition orchestration and interoperable formats | Reuse useq-schema, OME-TIFF/OME-Zarr and recorded-session adapters; Symthaea owns evidence qualification and advisory decisions |

### 8.1 Capability snapshot at 24 November 2026

The comparison is intentionally a division of labor, not a claim that open software automatically outperforms commercial platforms.

- ZEISS arivis Pro 4.6 (released 2 October 2026) advertises annotation-to-cloud training workflows, use of externally trained 3D ONNX semantic models, and analysis tools in its 4D viewer. This is a meaningful segmentation and workflow capability; Mycelial should accept its outputs and independently qualify their provenance and downstream biological consequences. Release notes: https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-6
- Imaris 11.1 (released 8 September 2026) adds a Training Hub and native CZI reader integration; its release history also lists machine-learning filters for spots and surfaces in 11.0. It is a mature visualization and analysis environment, not a weak straw-man baseline. Release notes: https://imaris.oxinst.com/support/imaris-release-notes/
- Fiji SNT provides automated and semi-automated tracing, multidimensional data support, 3D visualization, time-lapse workflows, scripting and a big-data mode described as experimental for TB-scale datasets. It is a useful open tracing and curation reference, although its primary design domain is neuronal morphology. Documentation: https://imagej.net/plugins/snt/
- Fungal Network Analysis (FNA) is a specialist reference for fungal image processing: noise/background correction, network enhancement, segmentation, skeletonization, width estimation and graph representation. Reproduce its standard pipeline and preserve parameters rather than treating graph metrics as a substitute for function. Record: https://zenodo.org/records/14931863
- SkelPy (2026) is a current GUI-based fungal skeletonization and phenotyping comparator. The study used three *Botrytis cinerea* isolates grown in a 96-well plate in triplicate, imaged every four hours over 72 hours, yielding 180 2D bottom-plane images. Default preprocessing parameters were empirically optimized on all 180 images, so that cohort is not an untouched test set. The paper reports three distinct validation tasks: (1) a manual skeleton-overlay comparison on 15 images from isolate 1.04.25 (mean distance 0.0046 image-width units, SD 0.0009; maximum nearest-point distance 0.0195); (2) node precision of 80.8% on 30 images against cleaned networks generated by the SkelPy preprocessing workflow, not independent ground truth; and (3) presence/absence F1 of 0.905 on 30 sampled spatial cells, scored by one user. Trait checks report LinesInImage versus manual hyphal-length estimate r=0.99 and fractal dimension versus gridline colonization r=0.81, but these are convergent morphology measures, not independent functional validation. The 180-image run took 102 minutes and 234 MB on an Apple M4 MacBook Pro; the separate 3,842-image throughput run took 12.5 hours. Images are 2D projections at the same focal plane. The data statement releases code and example images, not the full raw 180-image or 3,842-image series. Paper: https://doi.org/10.1002/aps3.70049

The field-level opportunity is the gap between image-analysis outputs and scientific claims. A commercial or open segmentation package may provide a very good graph; Mycelial should test whether that graph is adequately supported by the acquired source, whether alternative structures remain plausible, and whether the graph predicts independent functional outcomes.

The defensible competitive position is not “better at every image-processing operation.” It is:

**A scientific evidence and decision layer that can explain what was acquired, what remains unresolved, which biological hypotheses fit, what each hypothesis predicts, which measurement is worth acquiring next, and whether an independent intervention supports the prediction.**

A mature commercial segmentation tool can be an upstream provider and a benchmark competitor at the same time.

## 9. Integration with the existing public work

This protocol connects rather than duplicates the current portfolio:

- PR #13449 — acquisition-domain manifest and extraction coverage.
- PR #13459 — source-bound typed morphogenesis observations.
- PR #13464 / issue #13460 — replicate-disjoint forward-time split contract.
- PR #13369 — feasible graph-hypothesis posterior.
- PR #13379 — decision-value active acquisition.
- Issue #13486 / draft PR #13487 — hard cumulative specimen-health admissibility after VOI ranking.
- Issue #13488 / draft PR #13492 — photobiological exposure and response observations, distinct from phototoxicity.
- Issue #13419 — posterior adequacy, candidate-generation coverage and common-cause residuals.
- Issue #13352 — independent functional observations and transport benchmarks.
- Issue #13423 — multiplex transport and causal perturbation programme.
- Issue #13435 — HDC/CfC/LTC evaluation.
- Issue #13450 — demand-responsive morphogenesis.
- Issue #13451 — hardware-neutral microscopy interoperability.
- Issue #13431 — evidence-gated acquisition, initially in offline/shadow mode.

### Execution sequence independent of CI availability

1. Repair the actual rustfmt drift on PR #13464 and rerun its exact-head renderer contract; do not infer test status from a formatting failure.
2. Keep the acquisition and observation contracts as separately qualified foundations; do not mark skipped general CI as passing.
3. Qualify registry records at file level: source identity, specimen mapping, independent unit, exact version, hash, rights, annotation provenance and permitted use.
4. Build the synthetic multiplex transport oracle and independent exact enumerator; keep them separate from production inference code.
5. Implement functional observation and split adapters only after the measurement semantics and source identity are fixed.
6. Reproduce Sten/FNA baselines on rights-cleared data, then run the morphogenesis forecast baselines with the correct independent-unit split.
7. Design the prospective paired imaging–tracer–hydration pilot with a mycology collaborator and preregister it before collecting outcomes.
8. Add intervention prediction only when manipulation success and sham/off-target controls can be independently measured.

## 10. Claim discipline

Every public result should identify which of these levels it has reached:

1. **Specified:** a research question, protocol or schema exists.
2. **Implemented:** source code and tests exist at a named commit.
3. **Structurally validated:** a named contract and its exact-head checks pass.
4. **Data qualified:** rights, source hashes, annotations and independent-unit metadata are audited.
5. **Reproduced:** a published result is independently reproduced on its declared task and data.
6. **Predictively validated:** a frozen model predicts held-out independent biological outcomes.
7. **Intervention validated:** a blinded, controlled intervention supports the prediction.
8. **Externally replicated:** an independent laboratory reproduces the result.

Never use “validated” without naming the level. CI success is software evidence, not biological validation. Simulation is not a biological experiment. Prediction is not causation. A statistically non-significant result is not evidence of equivalence.

## 11. The milestone that matters

The next decisive milestone is not a larger graph or a higher segmentation score. It is a reproducible, forward-in-time prediction on an unseen biological unit, followed by a blinded functional measurement that can falsify it.

The first compelling demonstration would show that a model using source/sink state, hydration and measured network dynamics predicts source-region tracer incorporation better than persistence, distance-only and topology-only baselines, with calibrated uncertainty and no leakage. A later intervention should test whether the model predicts how that relationship changes when depletion or hydration is deliberately altered.

That would move Mycelial from network visualization toward a genuinely testable model of living infrastructure.

---

## References and implementation resources

- Travelling-wave plant–fungal trade (2025): https://doi.org/10.1038/s41586-025-08614-x
- *Agaricus* cordogenesis and fruiting (2026): https://doi.org/10.1016/j.funbio.2026.101799
- *Agaricus* stable-isotope resource transfer (2026): https://doi.org/10.1111/1462-2920.70345
- Hyphal Highways Device (2026): https://doi.org/10.1093/femsml/uqag035
- Smart microscopy interoperability roadmap (2026): https://doi.org/10.1515/mim-2025-0029
- Adaptive smart-microscopy review (2026): https://doi.org/10.1038/s44303-026-00145-y
- PhotoFiTT illumination-pattern / phototoxicity framework (2025): https://doi.org/10.1038/s41467-025-66209-6
- Fungal photoreceptor / photobiology review (2025): https://doi.org/10.1128/mmbr.00149-22
- *Gibberella zeae* wavelength/irradiance response (2025): https://doi.org/10.1016/j.funbio.2025.101690
- *Hericium erinaceus* LED-spectrum / mycelial growth response (2025): https://doi.org/10.3389/ffunb.2025.1684852
- *Fusarium solani* 454-nm blue-light growth and conidial viability study: https://pubmed.ncbi.nlm.nih.gov/32112251/
- Reproducible fluorescence microscopy recommendations (2021): https://doi.org/10.1038/s41592-021-01156-w
- Outcome-driven optogenetic microscopy (2025/2026): https://doi.org/10.1038/s41467-025-67848-5
- Sten 2D mycelium analysis (2024): https://doi.org/10.1016/j.ecoinf.2024.102670
- Sten 3D z-stack reconstruction (2025): https://doi.org/10.1016/j.ecoinf.2025.103317
- Fungal Network Analysis (Fricker): https://zenodo.org/records/14931863
- FNA multi-taxa image and parameter archive: https://doi.org/10.5281/zenodo.5725751
- FNA four-species grazer time-series archive: https://doi.org/10.5281/zenodo.14931953
- SkelPy paper: https://doi.org/10.1002/aps3.70049
- SkelPy source: https://github.com/AaronMoseley/SkelPy
- SkelPy downstream R analyses: https://github.com/melaniemadrigal07/Post.ImageProcessing
- Sten mycelium-network-analysis-tool (2D filament / crossing reference): https://gitlab.iit.it/oscar.sten/mycelium-network-analysis-tool
- Fiji SNT: https://imagej.net/plugins/snt/
- OME-NGFF: https://ngff.openmicroscopy.org/
- useq-schema: https://github.com/als-computing/useq-schema
- Micro-Manager: https://micro-manager.org/
