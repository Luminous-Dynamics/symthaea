# Mycelial Posterior Model Adequacy and Open-World Residuals
**Date:** 2026-11-23  
**Status:** Research protocol / synthetic-first contract; no real-data adequacy score or model-performance claim

## 1. Objective

The feasible graph posterior v1 enumerates candidate subsets exactly under a conditional-independent Bernoulli prior conditioned on explicit pairwise incompatibilities. That is a useful bounded inference contract, but exact normalization does not establish that the candidate inventory contains the true graph or that the supplied model family adequately explains the observations.

The question for this protocol is:

> Can the system detect when its own candidate set or generative assumptions are insufficient to explain the observed mycelial evidence?

This is distinct from asking whether the posterior is concentrated. A narrow, normalized posterior can be confidently wrong under candidate-generation failure or shared acquisition artifacts.

Related implementation and benchmark tracks:
- #13369 — feasible graph hypothesis posterior
- #13379 — decision-value active sensing
- #13367/#13368 — functional observation and biological-replicate split contracts
- #13352 — measured transport and tracer validation
- #13409 — source–sink flow under depletion and hydraulic resistance
- #13419 — model adequacy and open-world residuals (this protocol)

## 2. Claim boundaries

Keep these quantities separate:

1. **Candidate probability:** classifier or annotator-derived support for a specific candidate under a declared calibration model.
2. **Graph posterior:** probability over feasible combinations within the declared candidate inventory and prior.
3. **Candidate coverage:** evidence that the generator considered the relevant observed contact/branch domain.
4. **Model adequacy:** whether the model family predicts independent evidence with acceptable residuals.
5. **Biological truth:** an independently adjudicated or experimentally supported connection.

None implies the next automatically.

A residual is not an edge. A highly probable candidate is not biological acceptance. A graph score is not a flow measurement. A connectivity claim is not transport or viability.

## 3. Open-world observation coverage ledger

Create a source-bound coverage ledger for every observation region eligible for reconstruction.

Each observation/contact record must be assigned exactly one state:
- examined and candidate generated;
- examined and candidate rejected;
- examined and candidate unresolved;
- examined, no candidate generated;
- unavailable or unmeasurable;
- outside the declared acquisition scope.

The ledger binds:
- raw image/volume content digest and source identity;
- sample, biological replicate, acquisition session and timepoint;
- objective, pixel/voxel scale, z-step and imaging modality where known;
- coordinate-frame and spatial-transform digests;
- calibration and annotation protocol digests;
- candidate-generator identity, version, configuration and checkpoint digest;
- explicit observation IDs and any generated candidate IDs.

Unmatched observations must retain their original identity and geometry. Do not silently discard them or promote them to synthetic graph edges.

### Residual classes

Represent at least:
- apparent crossing with unresolved depth ordering;
- possible contact absent from the candidate inventory;
- branch/junction residual;
- segmentation or centerline discontinuity;
- occluded/out-of-focus region;
- field-of-view boundary;
- calibration/registration conflict;
- temporal mismatch;
- unknown/unclassified residual.

Each class must carry a bounded explanation and source evidence. The system may conclude that evidence is insufficient; it must not invent missing physical geometry.

## 4. Model adequacy and predictive checks

Define observation signatures predicted by a graph hypothesis and acquisition model, then compare them against evidence not used to select that hypothesis.

Possible signature families:
- focal-plane-specific crossing and z-order observations;
- local filament continuity, tangent and diameter profiles, only where measured;
- contact-group counts and conflict patterns;
- temporal branch/fusion persistence;
- edge-localized cytoplasmic-flow direction and speed;
- tracer arrival, recovery and source-to-sink outcomes;
- microbial dispersal under measured saturation/liquid movement.

Use independent biological-replicate holdouts. Do not split frames, crops, trajectories, sessions or fruiting bodies from the same cultivation box/replicate across train/test.

Use proper predictive scores appropriate to the observation type:
- held-out log predictive score / expected log predictive density;
- Brier score for declared binary events;
- calibration intercept, slope and reliability curves;
- CRPS or another declared proper score for continuous outcomes;
- predictive-interval coverage and width;
- residual checks stratified by organism, device, modality, focal depth, treatment and acquisition group.

Posterior predictive checks are diagnostics, not proof of model truth. Prefer a held-out predictive check when the same data would otherwise be used both to fit and to assess the model.

Reference: Moran, Blei and Ranganath, “Holdout predictive checks for Bayesian model criticism,” Journal of the Royal Statistical Society: Series B (2024), https://doi.org/10.1093/jrsssb/qkad105. Their holdout predictive check compares posterior-predictive draws with an independent held-out population sample, addressing the data-reuse problem of ordinary posterior predictive checks.

## 5. Common-cause and dependent evidence

Candidate decisions are often not independent. Several contacts can share one acquisition artifact or one learned failure mode.

Represent explicit evidence/acquisition groups for:
- z-registration and focal-plane calibration;
- illumination, contrast, stain and sample preparation;
- objective, detector and acquisition device;
- segmentation/checkpoint/model version;
- local occlusion, saturation or projection artifacts;
- annotator, annotation protocol and review session.

Compare, using adjudicated multi-contact groups:
1. v1 conditional-independent Bernoulli posterior;
2. grouped calibration / joint-label baseline;
3. constrained pairwise or log-linear factor model;
4. explicit latent common-cause model, only where the latent cause is identifiable from independent evidence.

Do not double-count evidence. A calibrated classifier posterior is not automatically a likelihood that can be multiplied by the same evidence-derived prior. Every probability input must declare whether it is a prior, likelihood, posterior score or calibrated marginal and which evidence it conditions on.

Exact inference on bounded groups should remain the oracle for validating approximate inference. Any new approximate posterior must report approximation error, effective sample size, convergence diagnostics, and failure/abstention conditions.

## 6. Calibrated abstention

Model adequacy output must distinguish:
- adequate under the tested evidence scope;
- unresolved / insufficient evidence;
- candidate-coverage failure;
- predictive/model-family inadequacy;
- unsupported coordinate or calibration state;
- evidence dependency not modelled;
- numerical/resource-bound refusal.

“Not detected” is not “absent”. “Not measured” is not zero. “No tracer arrival” must be distinguishable from right-censored, below-detection, failed-assay and missing observations.

Any acceptance threshold must be preregistered against the cost of false adequacy and false abstention. Confidence or posterior concentration alone must never choose the adequacy status.

## 7. Synthetic misspecification benchmark

Use frozen seeds, generator version, source digests and exact ground truth for controlled cases:

1. true fusion omitted from the candidate inventory;
2. branch or junction omitted by segmentation;
3. two candidate contacts share a focal-plane registration error;
4. a common illumination/contrast artifact biases several candidates;
5. projected crossing is mistaken for contact because depth is unavailable;
6. several candidates share a segmentation checkpoint failure;
7. contradictory annotations / unresolved depth;
8. observation outside the training organism, stain, modality or device;
9. censoring and unavailable regions that must not be treated as negative labels;
10. adequate-model controls with independent acquisition groups and known feasible graph truth.

A separate independent reference implementation should generate truth and verify exact small cases. Do not reuse the production posterior routine as its own oracle.

## 8. Functional scientific benchmark

Functional validation is downstream and independent of graph construction.

### Evidence families

- **Cytoplasmic flow:** edge-localized direction, speed, trajectory count, time and measurement method.
- **Stable-isotope tracer:** source stratum, sink/fruiting-body identity, tracer and isotope calibration, injection/recovery quantities, colonization time, flush, treatment and censoring.
- **Hyphal highway dispersal:** organism, hypha/bacteria identity, source/destination regions, cell counts, arrival time, liquid saturation and device geometry.

### Baselines

At equal data access and the same biological-replicate splits compare:
- image-only model without reconstructed graph;
- Euclidean distance-only model;
- unweighted topology/reachability;
- shortest path / weighted geodesic;
- MST and Delaunay structural controls where meaningful;
- degree-preserving rewired graphs;
- geometric resistance model only with measured radii/geometry and declared physical assumptions;
- local source–sink demand/depletion model;
- measured-flow/tracer-constrained model.

Do not infer conductance from segment length, treat graph cycles as flow capacity, or turn a shortest path into a biological transport label.

### Outcomes

- flow-direction accuracy and speed MAE/RMSE;
- calibrated edge-flow likelihood;
- tracer arrival-time and source-attribution error;
- recovered quantity prediction and interval coverage;
- per-sink source contributions;
- bacterial dispersal/arrival probability and censored time-to-arrival;
- held-out intervention/perturbation prediction;
- false-delivery, missed-delivery and decision-abstention rates.

Use biological replicate or cultivation box as the split and resampling unit. Keep samples, fruiting bodies, frames, crops, trajectories and repeated assays from a shared biological unit together.

## 8A. Matched benchmark tracks against open-source tools

Run these as separate task families; the reported scores below are not directly comparable across task types.

| Comparator | Matching task | Required controls and measures |
|---|---|---|
| Sten 2024 | 2D centerline and crossing detection | Reproduce its manual adjudication policy; report both automated-only and adjudicated scores |
| Sten 2025 | Z-stack reconstruction and overlap | Separate six-image real overlap test from printed-prototype geometry RMSE |
| Fungal Network Analysis | Enhancement, segmentation, skeletonization, width and graph | Same raw images, calibration, masks and parameter-tuning budget; report connectivity preservation, false merges, width/length error and graph-edit topology |
| SkelPy | Time-series skeletonization and structural traits | Same *B. cinerea* images where obtainable; report centerline distance, node precision/recall, false junctions, topology over time, physical calibration and processing time |
| Fiji SNT | Tracing and time/channel-preserving import/export | Same traces/volumes; check ID/provenance preservation, temporal metadata and explicit loss on SWC/other projections |
| arivis / Imaris | Volumetric segmentation and object handling | Pinned trial/product version, same volume/model weights and operator time; report overlap, crossing/contact errors and export metadata |

**SkelPy score interpretation:** its paper reports a mean nearest-skeleton distance of 0.0046 normalized image-width units (SD 0.0009; maximum reported nearest-point distance 1.95% of image width). A separate validation on 30 images reported 80.8% of predicted nodes as true biological nodes. A different 30-cell sample from 4×4 spatial grids reported F1 0.905, sensitivity 90.5% and specificity 75.0%. These are different evaluation units/protocols; do not collapse them into one headline accuracy or compare the grid-cell F1 to Sten's crossing F1. The 180-image time series is three isolates and 20 timepoints, but only code/example images are stated publicly available. The complete raw series therefore requires author access before a matched re-benchmark.

**FNA score interpretation:** it already takes explicit steps to preserve network connectivity by over-segmenting before graph construction and edge pruning. A meaningful challenge is not whether our graph has more edges, but whether it preserves true biological junctions while reducing false fusion, and whether width/length/topology outputs remain calibrated on held-out organisms and devices.

The intended result is a **Pareto comparison** across topology fidelity, geometry fidelity, uncertainty calibration, runtime/throughput, data requirements and information preservation—not a single synthetic leaderboard number.

## 9. Initial research targets

These are hypotheses for preregistered gates, not achieved results or claims of superiority:

- 100% agreement with an independent exact enumerator on every bounded feasible-posterior fixture.
- Zero posterior mass on explicitly infeasible graph assignments.
- At least 95% detection of the predefined severe synthetic misspecification cases at no more than 5% false adequacy, with uncertainty intervals and prevalence reported.
- No more than 5% false abstention on matched adequate-model synthetic controls.
- Nominal 90% held-out graph-event predictive coverage within a preregistered tolerance at biological-replicate level.
- On dependent-evidence cohorts, a common-cause model improves held-out proper score over the v1 independence baseline without degrading calibration on independent-evidence controls.
- A functional model must beat the strongest matched non-graph baseline on a held-out biological replicate cohort before a graph-specific predictive benefit is claimed.
- A transport improvement target of at least 10% relative error reduction versus the strongest topology-only baseline is an aspirational gate from #13352, not a present result.

Report denominators, missingness, assay failure, confidence intervals, effect sizes and subgroup performance. Do not report only aggregate F1 or a single scalar score.

## 10. External comparison and interoperability position

### Reconstruction and microscopy benchmarks

- Sten et al. (2024) evaluated 135 real mycelium images across varied datasets/acquisition conditions. They reported filament-detection F1 0.88–0.94 and crossing F1 0.89: https://doi.org/10.1016/j.ecoinf.2024.102670. Their released data/source provide a reproducible starting point.
- Sten et al. (2025) introduced z-stack-based 3D filament reconstruction and reported overlap-detection F1 0.91–0.92 on six manually annotated real fungal images: https://doi.org/10.1016/j.ecoinf.2025.103317. The small annotated real-image subset makes this a narrow baseline, not a universal 3D graph-reconstruction SOTA.
- Do not compare our proposed F1 ≥0.94 target to either result without reproducing their exact task, annotation rules, denominators, image conditions and split protocol. Then expand to replicate-held-out 3D geometry, topology and calibrated false-fusion outcomes.

### Industry and adjacent open-source platforms

- **ZEISS arivis Pro 4.6** (released 2 October 2026) supports externally trained ONNX models for volumetric semantic segmentation, plus a guided training workflow and segment-contact handling (merge, overlap, touch). Official release notes: https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-6.
- **Imaris 11.1** (released 8 September 2026) adds the Training Hub, extended file-type recognition and native CZI reader integration. Official release notes: https://imaris.oxinst.com/support/imaris-release-notes/.
- **Fiji SNT** already provides multidimensional tracing, semi-automated/automated workflows, batch processing and analysis of non-neuronal filamentous structures. Its documentation recommends TRACES over SWC for some 4D/5D workflows because TRACES retains channel/time-frame context: https://imagej.net/plugins/snt/ and https://imagej.net/plugins/snt/faq.
- **Fungal Network Analysis (FNA)** is an important direct baseline, not merely a visualization tool. Its documented pipeline includes noise reduction, background correction, curvilinear enhancement, segmentation, skeletonization, width estimation and graph construction. It intentionally over-segments to preserve connectivity, then prunes candidate edges, and emits weighted graph edges with width, length and orientation attributes. The public software record provides a Windows standalone build, MATLAB app and manual: https://doi.org/10.5281/zenodo.14931863. The associated 3.2 GB multi-taxa image/parameter dataset is at https://doi.org/10.5281/zenodo.5725751; a distinct 10.2 GB four-species soil-grazer time-series and processed-data collection is at https://doi.org/10.5281/zenodo.14931953. These are substantial comparators for segmentation, width/length extraction, topology and ecological-trait analysis. The software record describes a specific packaged version; pin and report the exact archive/build used. Zenodo's visible license field is blank for the data records, so verify per-artifact rights before reuse.
- **SkelPy** (Madrigal et al., 24 March 2026) is a newer, open-access GUI for skeletonization and time-resolved fungal phenotyping. It reports 180 images from three *Botrytis cinerea* isolates across 20 timepoints over 72 hours, plus a larger 3,842-image throughput run. Its workflow uses configurable radial thresholding, noise cleanup, Gaussian smoothing, Canny edge detection and Zhang–Suen skeletonization; it supports user-modified pipelines, normalized/physical calibration, structural traits and export to CSV/JSON with downstream R cleaning. Paper: https://doi.org/10.1002/aps3.70049; code and example images: https://github.com/AaronMoseley/SkelPy; downstream R analysis: https://github.com/melaniemadrigal07/Post.ImageProcessing. The complete 180-image raw time series is not stated to be publicly available; the paper states that code and example images are available. Its imaging captures the same bottom focal plane and represents planar 2D projections, not complete 3D networks. The article is CC BY-NC 4.0; do not assume the code or sample images have identical permissions.
- These systems are strong in microscopy workflows, tracing, segmentation, visualization, learning, manual parameterization and graph/trait extraction. Their public feature pages do not establish whether every evidence-assurance capability proposed here is absent from their products. A fair comparison requires pinned versions, matched data, matching annotation protocols and recorded workflows.

**Competitive recommendation:** do not rebuild the viewers or claim superior segmentation. Integrate with arivis, Imaris, SNT, FNA and SkelPy, preserve foreign object IDs and provenance, and compete on the evidence layer: alternative graph hypotheses, source-bound uncertainty/calibration, open-world model adequacy/abstention, explicit information loss, decision-value acquisition, counterfactual resilience and independent functional/interventional qualification.

### Independent biological evidence and accessible benchmark artifacts

The data-access and adapter work is tracked separately in [#13425 — rights-aware benchmark registry](https://github.com/Luminous-Dynamics/symthaea/issues/13425). The following dataset families are not interchangeable.

1. **2D reconstruction — Sten et al. 2024.** Paper: https://doi.org/10.1016/j.ecoinf.2024.102670. Public code/data repository: https://gitlab.iit.it/oscar.sten/mycelium-network-analysis-tool. The paper reports 135 real images, filament detection F1 0.88–0.94 and crossing F1 0.89. The repository states that selected branch/crossing cases require manual visual adjudication/correction. Reproduce that protocol explicitly; do not compare an automated-only score to their corrected score without reporting the distinction. The repository's permissive software license must not be assumed to license all third-party source images.

2. **3D z-stack reconstruction and synthetic geometry — Sten et al. 2025.** Paper: https://doi.org/10.1016/j.ecoinf.2025.103317. Code/data package: https://doi.org/10.5281/zenodo.15382388. The record describes *Rhizophagus irregularis* z-stacks, overlap annotations, time-lapse images and 3D-printed filament prototypes. The paper's 0.91–0.92 overlap F1 is based on six manually annotated real fungal images; physical 3D-printed prototypes are a separate known-geometry task. File-level image rights require auditing: the record identifies some subcollections as copyrighted by Cardini et al. (2020) or Oscar Sten. The software package license is not a blanket license for every embedded image.

3. **Dynamic topology and bidirectional cytoplasmic flow — 2025 Nature plant–fungal trade.** Paper: https://doi.org/10.1038/s41586-025-08614-x. Source data for the main analysis and figures: https://doi.org/10.6084/M9.FIGSHARE.27889143. Analysis code is available under MIT at https://github.com/Cocopyth/AMFTravellingWave. The study links ~100,000 flow trajectories to >500,000 tracked nodes across three AM fungal strains. It reports bidirectional/antiparallel streams. The plate-based growth geometry is 2D, so this is a high-value dynamic/functional benchmark, not full 3D soil-network ground truth. Audit per-artifact data terms before redistribution.

4. **Host-independent AM-fungal flow — 2026 *Fungal Biology*.** Paper: https://doi.org/10.1016/j.funbio.2026.101775. It reports 5,000 videos and ~25,000 trajectories across two *Rhizophagus irregularis* strains. The article's data statement is “all data available upon request”; this is a collaboration/request-only benchmark, not an assumed open download.

5. **Isotope source-to-sink transfer — 2026 *Agaricus bisporus*.** Paper: https://doi.org/10.1111/1462-2920.70345. Raw data: https://doi.org/10.6084/m9.figshare.30818828. The article treats a cultivation box, not each mushroom, as the experimental unit in treatment analyses. Mushrooms within the same box/flush can have strongly heterogeneous tracer uptake, so all mushrooms from a box must remain in the same split. The raw-data artifact's own license/redistribution terms must be recorded separately from the article's CC BY license.

6. **Bacterial dispersal along hyphae — 2026 Hyphal Highways Device.** Paper: https://doi.org/10.1093/femsml/uqag035. Data: https://doi.org/10.5281/zenodo.20071448; semiautomated analysis macros are included in the supplement. The article reports bacterial outlet CFUs in only 4 of 8 co-inoculated devices. Preserve non-arrival and assay outcomes instead of conditioning the benchmark on success. This is bacterial dispersal along *Pythium ultimum* hyphae under unsaturated conditions, not intracellular fungal flow.

7. **FNA multi-taxa images — Fricker et al. dataset.** The public 3.2 GB Zenodo record contains mycelial images from basidiomycetes, ascomycetes and zygomycetes plus species-key and processing-parameter files: https://doi.org/10.5281/zenodo.5725751. It directly supports matched comparisons of network enhancement, segmentation, skeletonization, width/length extraction and graph construction. Its Zenodo License field is blank; OpenAIRE indexes CC BY, so resolve file-level terms before reuse rather than treating the apparent license as independently verified.

8. **FNA soil-grazer time series — 2025 Zenodo release.** The 10.2 GB archive contains time-series images for four fungal species growing from wood-block resources over compressed soil, with and without invertebrate grazers; each ZIP includes original images, FNA processing parameters and processed outputs: https://doi.org/10.5281/zenodo.14931953. The repository identifies FNA v1.00.24 for the processed files. This is useful for multi-species longitudinal topology and treatment-response baselines. Keep all frames from a wood-block/soil-box replicate together. The Zenodo record's visible license field is blank and the imagery derives from an earlier study, so verify original-source rights.

9. **SkelPy — 2026 code and example images.** Paper: https://doi.org/10.1002/aps3.70049; GUI source and example images: https://github.com/AaronMoseley/SkelPy; downstream R analysis: https://github.com/melaniemadrigal07/Post.ImageProcessing. The paper describes 180 images from three *Botrytis cinerea* isolates at 20 timepoints and a separate 3,842-image throughput run. The raw 180-image time series is not stated to be publicly downloadable; only code and example images are. Its article reports a mean nearest-skeleton distance of 0.0046 normalized image-width units, node precision of 80.8% on a 30-image subset, and a separate spatial-grid F1 of 0.905 from 30 sampled cells. These are distinct protocols, not one accuracy score. The image data are planar bottom-focal-plane projections, not full 3D reconstructions. Article license is CC BY-NC 4.0; inspect code/example-image rights independently.

**Benchmark consequence:** use separate task cards and metrics:
- reconstruction: filament/crossing detection, false-fusion rate, graph-event calibration and geometric error;
- dynamic flow: forward and reverse speed distributions, event timing and temporal graph prediction;
- isotope: per-sink tracer uptake, source attribution, depletion and arrival/recovery;
- hyphal highways: bacterial arrival probability/time, along-channel signal and CFU recovery with censoring.

No random frame/crop/trajectory/mushroom split may cross its independent biological unit (replicate, cultivation box or device/culture run). Keep source artifacts immutable; only build adapters after file-level rights and provenance are checked. This distinction is the point of [#13425](https://github.com/Luminous-Dynamics/symthaea/issues/13425).

### Interchange and OME-NGFF status

Maintain the canonical mycelial graph sidecar independently from SWC. SWC is useful for tree-like projections but cannot preserve every cyclic graph or assay-specific property. SNT's TRACES format can retain channel/time-frame context for 4D/5D tracings.

Bind OME-NGFF coordinate systems and spatial transforms by digest. As of the current NGFF RFC index:
- RFC-5, Coordinate systems and transformations: S3 (Implementation), associated with OME-Zarr 0.6;
- RFC-8, Collections: R1 (Under review);
- RFC-7, Channel provenance: D1 (Initial idea).

The current status list is authoritative here: https://ngff.openmicroscopy.org/rfc/. Do not describe RFC-8 as D1; that status belongs to RFC-7. Do not claim vendor round-trip validation until a real import/export has been executed against pinned versions, source files and metadata.

## 11. Implementation sequence

1. Build a source-bound observation-coverage and residual contract.
2. Add deterministic synthetic fixtures for open-world omissions and common-cause artifacts.
3. Add a separate model-adequacy report; do not alter v1 posterior semantics.
4. Add held-out predictive signatures and calibration metrics.
5. Implement a joint/common-cause baseline only after evidence grouping and double-counting semantics are specified.
6. Connect model adequacy outcomes to the active-sensing evaluator as a reason to acquire or review additional evidence.
7. Ingest licensed real data only after file-level rights, provenance and replicate identity audits.
8. Validate functional prediction using independent measured flow, tracer or dispersal outcomes.

## 12. Acceptance conditions

- Versioned Rust contracts, bounded resources and deterministic canonical digests.
- Original source, acquisition model, calibration, replicate and residual identities are committed into reports.
- Exact small-case oracle is independent from production inference.
- Adversarial tests cover missing candidate, shared acquisition artifact, evidence double-counting, censored observations, tampered digests and unavailable geometry.
- Model adequacy is reported separately from posterior concentration.
- No microscope actuation, automatic biological acceptance, physical transport claim or SOTA claim is inferred from this protocol alone.
