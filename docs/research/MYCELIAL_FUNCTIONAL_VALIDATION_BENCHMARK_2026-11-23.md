# Mycelial Functional Validation Benchmark v0.1
**Date:** 2026-11-23
**Status:** Research protocol / proposed targets, not benchmark results

## 1. Purpose

A structurally plausible mycelial graph is not yet a functional model. Functional validation must ask whether a graph and its associated measurements predict independently observed biological outcomes.

This benchmark separates three task families that must not be collapsed into a generic "transport" score:

1. **Intrahyphal cytoplasmic flow** — direction, velocity and temporal/spatial variability within observed hyphae.
2. **Nutrient / isotope transfer** — uptake, recovery, transfer timing and source-to-sink outcomes.
3. **Hyphal-highway microbial dispersal** — bacterial movement through liquid films associated with fungal or fungal-like hyphae.

These processes may interact, but one is not a proxy for another. In particular, graph shortest paths are not measured flow paths, a connected component is not a viable transport corridor, and observed cytoplasmic motion does not by itself quantify nutrient delivery.

## 2. 2026 evidence and potential benchmark datasets

### 2.1. Hyphal-highway single-cell dispersal

Moratto et al., *A microfluidic platform reveals the role of liquid movement on the dispersal of Pseudomonas putida via Pythium ultimum hyphal highways*, microLife 7 (2026), uqag035, published 21 September 2026:
- Article: https://doi.org/10.1093/femsml/uqag035
- Dataset: https://doi.org/10.5281/zenodo.20071448
- The paper reports a microfluidic Hyphal Highways Device for long-term single-cell quantification.
- Its results connect bacterial dispersal with hyphal presence, biomass and liquid movement; the paper reports bacteria moving between liquid pools along hyphae in 20 of 26 observations.
- It supplies semiautomated image-analysis macros in supplementary material.

Use this as a **microbial dispersal / liquid connectivity** benchmark. Do not relabel bacterial displacement as fungal cytoplasmic flow. Audit the Zenodo record's dataset-level license and individual file metadata before ingestion; article rights do not automatically establish rights for every deposited file.

### 2.2. Isotope-labelled nutrient uptake

Van Iersel et al., *Feeding of Agaricus bisporus mushrooms is impacted by mycelium colonization time and network architecture*, Environmental Microbiology 28(6), e70345 (2026):
- Article: https://doi.org/10.1111/1462-2920.70345
- Raw data: https://doi.org/10.6084/m9.figshare.30818828
- The paper uses stable-isotope labelling to quantify uptake from the lower compost stratum and discusses hydraulic resistance, cord abundance, colonization time and local substrate depletion.
- The Figshare record identifies CC BY 4.0.

Use this as a **tracer uptake / source-to-sink resource-transfer** benchmark. The endpoint is measured tracer recovery/uptake, not an inferred graph path length. Preserve cultivation box, flush, compost stratum, colonization time, treatment and individual fruiting-body identity.

### 2.3. Network reorganization and cord morphology

Van Iersel et al., *The mycelial network organization adapts to facilitate new substrate colonization and fruiting in Agaricus bisporus*, Fungal Biology 130(5), 101799 (2026):
- Article: https://doi.org/10.1016/j.funbio.2026.101799
- Raw data: https://figshare.com/s/b241753378d5252ff808
- The study uses longitudinal scan imaging and quantifies optical colonization density and cord abundance. It describes cord formation at new substrate fronts and increased cord abundance / widening during fruiting-body development.
- The article and published version are listed as CC BY on the Utrecht University research portal.

Use this as a **longitudinal architecture / demand-associated remodeling** benchmark, not as direct flow ground truth. Its surface scans and cord masks must retain the study's imaging limitations; surface-visible density is not a complete 3D hyphal graph.

### 2.4. Cytoplasmic flow trajectories

Klein et al., *Cytoplasmic flow dynamics in arbuscular mycorrhizal fungi are intrinsic and independent of plant hosts*, Fungal Biology 130(4), 101775 (June 2026):
- Article: https://doi.org/10.1016/j.funbio.2026.101775
- The study reports 5,000 cytoplasmic-flow videos and approximately 25,000 trajectories from two Rhizophagus irregularis strains, using kymograph analysis. It reports strong temporal and spatial variation.
- Its data availability statement says data are available upon request.

This is a high-value **flow dynamics** benchmark candidate, but it is not presently an open raw-data ingestion target. Seek author permission / data access and agree reuse conditions before attempting a reproduction. Do not infer a public license from the article's open-access status.

## 3. Evidence contract and graph binding

The companion Rust module `mycelial_functional_observations.rs` defines a first intake contract. Each observation set binds:
- biological replicate, organism, treatment, session and assay-protocol identity;
- graph digest, coordinate-frame identity and calibration digest;
- a distinct functional-measurement digest;
- monotonic timebase and acquisition window;
- typed measurement records.

The three measurement families remain distinct:
- `CytoplasmicFlow` binds to a graph edge and records direction relative to the edge's annotated orientation, speed in µm/s and trajectory count;
- `TracerTransfer` binds source and target graph nodes, tracer identity, injection/arrival times and unit-labelled quantities;
- `MicrobialDispersal` retains assay-local source/target region IDs, cell counts, arrival time and optional measured liquid-saturation fraction. These assay-local IDs are not silently converted into graph node identities.

The current schema is an intake foundation, not a complete scientific ontology. Before real-world adoption, add marker identity (e.g. lipid tracer vs bulk cytoplasm proxy), per-trajectory records or distribution summaries, assay-specific missingness/censoring, and explicit uncertainty for each measurement. Version those additions rather than silently changing v1 semantics.

## 4. Evaluation protocol

### 4.1. Reconstruction and graph baselines

Evaluate the same observation cohort against:
1. independently annotated graph (where available);
2. current reconstructed graph;
3. graph with uncertain contacts rejected;
4. graph with a candidate contact accepted for analysis only;
5. Euclidean MST and Delaunay reference graphs;
6. degree-preserving edge-rewired null graphs;
7. image-only models that do not receive reconstructed graph features.

Topology-derived and physically weighted models must be reported separately. A resistance/conductance model is admissible only when required measurements and boundary conditions (e.g. lumen radius, viscosity, pressure/potential gradient) are actually available. Never impute conductance from edge existence alone.

### 4.2. Cytoplasmic flow metrics

Report per-strain and per-replicate:
- edge-level flow-direction accuracy / macro-F1, with unresolved and bidirectional labels retained;
- speed MAE, median absolute error and error by speed quantile;
- predictive log score or CRPS for velocity distributions where trajectory-level observations are available;
- 50%, 80% and 90% interval coverage and interval width;
- calibration by time, location, edge class, strain and assay batch.

Do not average away rare high-speed bursts. Report tail prediction separately from mean speed. Distinguish the speed of a fluorescently tagged transported marker from bulk cytoplasmic motion when the assay does so.

### 4.3. Tracer transfer metrics

Report:
- tracer recovery / uptake error in the original quantity unit;
- source-to-sink event precision, recall and calibration;
- time-to-arrival error for observed arrivals;
- censored time-to-arrival metrics for non-arrivals or arrivals beyond the observation window (e.g. concordance and time-dependent Brier score);
- predictive interval coverage for uptake and arrival times;
- treatment-, flush-, stratum- and replicate-specific performance.

A missing arrival is not a zero quantity unless the assay protocol establishes that interpretation. Treat right-censored observations explicitly.

### 4.4. Microbial dispersal metrics

Report:
- probability of successful dispersal between assay regions;
- cell-count error and count-distribution score;
- first-arrival / transit-time error and censoring;
- performance stratified by liquid saturation, liquid-patch connectivity, hyphal biomass and device geometry;
- false-connected and missed-connected event rates.

The 2026 Hyphal Highways Device work emphasizes the role of liquid films and liquid connectivity. A fungal graph alone cannot account for all such conditions; retain device and liquid-state covariates.

### 4.5. Architecture / remodeling metrics

For longitudinal cord and hyphal architecture:
- cord-abundance and cord-width measurement error;
- temporal event precision/recall for cord formation, branching and remodeling;
- growth-front displacement error in calibrated physical units;
- correlation / predictive score for independently measured sink-demand or resource-transfer changes;
- stratification by substrate, depth, cultivation stage and imaging surface.

Do not treat optical density as a complete 3D reconstruction. Preserve the imaging modality and observed coverage.

### 4.6. Local source–sink flow under depletion and hydraulic resistance

The 2026 *Agaricus bisporus* isotope-uptake study changes the functional benchmark from a generic "transport prediction" task to a **local source–sink allocation** task. Individual mushrooms within the same flush can draw different resource proportions from substrate strata. Cord abundance and wider cord pathways can reduce resistance for longer-distance transfer, but this becomes more important when nearby substrate is depleted; extra cords do not necessarily change the source distribution while local resources remain available.

This creates a critical negative control for our graph work: two sinks can be equally connected and have comparable shortest paths, yet receive different measured resource flows because local resource availability, hydraulic resistance and sink demand differ.

Required model comparison:
1. distance-only graph model;
2. unweighted topology / reachability model;
3. measured-geometry resistance model;
4. local source availability + sink demand + colonization / flush model;
5. joint model with measured flow or tracer constraints;
6. image-only / substrate-covariate baseline without graph topology.

Only use a Poiseuille-style radius prior where its assumptions are defensible and the relevant lumen radius, viscosity and boundary conditions are measured. Edge existence and segment length alone do not specify conductance. Do not infer physical hydraulic resistance from graph weights unless the measurement contract records what those weights mean.

Per-sink metrics:
- individual fruiting-body tracer recovery / uptake error and distributional CRPS;
- source-stratum attribution Brier score and calibrated probabilities;
- time-to-arrival and censored-arrival scores;
- predictive interval coverage for individual sink uptake;
- calibration and error stratified by flush, substrate depth, colonization time, treatment and cultivation box;
- counterfactual prediction under held-out local resource-depletion regimes.

The cultivation box is the grouping unit for splits and bootstrap intervals. Keep each fruiting body, layer, flush and timepoint nested under its originating biological box. Do not randomly split mushrooms from the same box into train and test.

Related implementation programme: [#13409 — local source–sink flow under depletion and hydraulic resistance](https://github.com/Luminous-Dynamics/symthaea/issues/13409).

## 5. Split design and leakage controls

Primary split unit is the **biological replicate**, not a random frame, crop, edge or trajectory.
- Hold out complete biological replicates / cultivation boxes.
- Where sample size supports it, add leave-one-strain/species-out, leave-one-device-out and leave-one-lab-out evaluations.
- Keep every timepoint and derivative crop of one biological replicate in the same split.
- Keep calibration and preprocessing fits within training folds only.
- Freeze annotations and candidate contact adjudication before functional outcomes are unblinded.
- Report cluster-bootstrap confidence intervals over biological replicates, not frame-level pseudo-replication.
- Report sample counts, number of independent replicates, number of trajectories, and missing/censored counts per metric.

For small published cohorts, label estimates exploratory and do not claim cross-species generalization from pooled frames.

## 6. Nulls, negative controls and counterfactual tests

Required controls:
- degree-preserving rewiring;
- edge direction permutation;
- time-shifted / replicate-shuffled outcome labels;
- geometry-only and graph-only feature ablations;
- image-only baseline;
- no-hypha / no-liquid / disconnected-patch controls when present in the assay;
- candidate fusion rejected vs accepted-for-analysis evaluated against held-out functional outcomes.

The candidate must not be called beneficial because its own graph-derived score improves. The acceptance hypothesis is supported only when it improves prediction of an independent held-out measurement or a registered intervention outcome.

## 7. Proposed benchmark targets (not achieved)

Targets should be frozen after a baseline reproduction and power analysis. Initial hypotheses:
- Flow: direction macro-F1 ≥ 0.85 on resolvable direction labels; calibrated 90% intervals cover 85–95% at replicate level; reduce velocity-distribution CRPS by ≥10% relative to the best topology-only baseline.
- Tracer: source-to-sink discrimination AUROC ≥ 0.80 and C-index ≥ 0.75 for time-to-arrival, with calibrated uncertainty and censoring handled; reduce uptake error ≥10% relative to the strongest non-graph baseline on held-out biological replicates.
- Microbial dispersal: dispersal-event AUPRC ≥ 0.80 on a predeclared class-balanced / natural-prevalence reporting pair; improve cell-count distribution score ≥10% over the best non-graph baseline after liquid-state covariates are included.
- Remodeling: cord-abundance/width error lower than an image-only baseline by ≥10% on held-out cultivation boxes, without sacrificing replicate-level interval coverage.
- Cross-domain: no single pooled score may hide a regression in one assay family. Publish each task's denominator, confidence interval and missingness.

These are proposed targets, not measured performance or claims of superiority to SOTA.

## 8. Data rights, governance and provenance

- Maintain a per-file rights manifest: repository record, file path, license, citation, access date, permitted use, derivatives and any attribution/share-alike obligations.
- The article license is not sufficient evidence that every raw image or video file is licensed for redistribution or commercial use.
- Flow trajectories from Klein et al. are "available upon request"; obtain permission before ingestion.
- Preserve original records and hashes; derived annotations must retain parent file hashes and transformation/version receipts.
- Do not upload unpublished biological images, human-linked or institution-restricted files without explicit permission.
- Keep public reproduction data separate from any restricted research collaboration data.

## 9. Execution sequence

1. Audit the Figshare raw uptake dataset and its associated published imaging/morphology data; record each file-level license and hash before ingestion.
2. Inspect the Hyphal Highways Zenodo record and supplementary macros; produce a reproducible intake manifest before converting anything.
3. Request access to the AM fungal flow dataset, including the raw kymographs / per-trajectory data and reuse terms.
4. Implement assay-specific benchmark adapters and grouped split manifests.
5. Publish baseline performance and uncertainty before tuning graph-aware models.
6. Couple functional prediction scores to active-sensing value-of-information and fusion counterfactuals only after independent outcome bindings exist.
7. For isotope-tracer studies, fit source–sink allocation by individual fruiting body and stratum; test the depletion-conditioned resistance hypothesis separately from topology-only predictions.

## 10. Claim boundary

Until the corresponding independent measurements are actually ingested and evaluated, report only that the schema and protocol exist. Do not claim a validated transport simulator, biological viability inference, functional network superiority, or benchmark score.
