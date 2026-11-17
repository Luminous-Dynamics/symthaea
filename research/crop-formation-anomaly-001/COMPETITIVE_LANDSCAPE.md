# CROP-INTEL-001 — Competitive landscape and benchmark positioning

**Audit date:** 2026-11-18  
**Status:** desk research; not an exhaustive literature review  
**Important:** the figures below describe each source's own declared catalogue or benchmark scope. They are not interchangeable measurements of unique, independently verified physical formations.

## Executive assessment

The field is not empty. Several archives have substantial historical collections; a newer open-data effort publishes parametric geometry; and an open-source atlas has implemented a sophisticated provenance-aware catalogue and exact-field geolocation workflow.

The strongest opportunity for Symthaea is therefore **not** to win on catalogue size, duplicate another geolocator, or claim that general-purpose computer vision is inadequate. It is to establish a reproducible research benchmark spanning:

1. measurement-grade geometry reconstruction;
2. preregistered mathematical-structure and information tests;
3. multiple-comparison-controlled decoder evaluation;
4. open-world causal hypothesis comparison;
5. active experiment selection;
6. independent, rights-aware scientific reproduction.

No recognized community-wide leaderboard covering these tasks was identified in this desk audit.

## 1. Existing archives and datasets

| Project | Publicly stated scope | What it is strong at | What it does not establish |
|---|---|---|---|
| Crop Circle Research Archive (CCRA) | Nearly 17,000 cross-referenced information items; over 2,850 formations; nearly 8,000 photographs; exact locations for more than 1,400 formations | Historical reports, photographs, correspondence, surveys, source research and UK distribution | Complete global coverage, uniform geolocation accuracy, consistent image rights, or a labelled causal ground-truth set |
| Crop Decoder | 2,982 formation records across 35 countries, with source/archive links | Cross-archive discovery, recent chronology and attribution links | That every record is a distinct physical event or independently validated |
| FEEDIT / Field Edit | 11 parametric formations and 1,195 geometric elements; records in metres; dataset declared CC BY-SA 4.0 | Accurate, reproducible vector descriptions of a small set of selected formations | Representative sampling or ground truth for the whole phenomenon; aerial-photo rights are separate |
| Crop Circle Archive (CCA) | Explicit observation-first record methodology: accession ID, place/date, geometry and cited sources; unknowns stay unknown | An emerging repeatable record standard | A validated causal or information-decoding benchmark |
| MYTbrain/crop-circle-atlas | 8,391 source assertions reconciled into 7,745 conservative catalogue entities; 419 field candidates/sites (including four corroborated fields and six registered sites); distinct locality references and unresolved entities retained | Provenance-first reconciliation, geolocation review, rights handling and a fail-closed exact-field research worker | 7,745 proven distinct physical formations or real-world geolocator effectiveness; its own benchmark explicitly says real cases have not yet run end-to-end |

Always retain source counts, source assertions, catalogue entities, geolocated records, reviewed fields and independently verified events as distinct denominators. A larger number is not automatically a better catalogue.

## 2. Strongest engineering comparator: MYTbrain crop-circle-atlas

This project is a serious baseline and should be treated respectfully. Its published architecture includes:
- bounded search polygons and normalized historical imagery metadata;
- physical, metre-based imagery tiles and content-addressed caching;
- deterministic CPU retrieval;
- SIFT feature matching and RANSAC registration proposals;
- spatial-distribution and homography diagnostics;
- control points kept separate from held-out checkpoints;
- checkpoint error measured in metres;
- a human review and state-transition gate;
- rights-gated output and a rule that machine matches never mutate canonical atlas coordinates.

Its published geolocator benchmark is particularly instructive:

- Nine real-world cases are preregistered.
- Eight are metadata-only because image rights are not cleared.
- The one pixel-permitted candidate lacks the required historical reference raster.
- The project reports **0/9 real cases run end-to-end** in that benchmark.
- One synthetic positive case has a correct tile rank of 1 and sub-metre synthetic recovery metrics.
- False-acceptance is explicitly not measured because there are no negative synthetic cases.

This is good experimental transparency. It also shows why a synthetic positive alone cannot support operational effectiveness.

### Symthaea's relationship to this work

Do not clone the catalogue or geolocator. Prefer a compatible research interface and a comparative evaluation:
- use its documented source assertions and field/site distinctions as an external catalogue comparator;
- compare candidate georeferencing methods only on permissioned, independently surveyed assets;
- retain its separation of search controls from evaluation checkpoints;
- do not copy its private imagery, caches, credentials, or rights-restricted assets;
- ask for collaboration or a rights-cleared benchmark exchange before reusing any data not explicitly licensed.

Symthaea should add a higher-level layer that can ingest *qualified* geometry and provenance, then test mathematical structure, candidate encodings, mechanism predictions and experiment value. It should never silently promote another project's candidate coordinate or catalogue assertion into ground truth.

## 3. Adjacent industry and SOTA baselines

### Optical remote-sensing computer vision

A 2026 systematic review of 189 optical remote-sensing object-detection studies reports broad method-dependent performance ranges. On DOTA-style tasks, two-stage detectors are reported around 70–76% mAP with slower inference, one-stage approaches around 60–74% mAP, and deep-learning approaches broadly around 65–80% mAP. The review also reports material cross-region/domain degradation for some methods.

These values **must not** be represented as a crop-circle baseline: DOTA and related datasets contain different object classes, image conditions, annotation protocols and task definitions. They are useful evidence that rotated-object detection, domain adaptation, annotation quality, uncertainty and throughput matter in remote sensing.

For CROP-INTEL, compare classical contours/Hough/ellipse fitting, feature registration, manual annotators, and suitable trained segmentation/detection models on exactly the same crop-formation task. Report precision/recall, boundary error, registration error, calibration, latency and abstention—not borrowed mAP alone.

### Pattern mining and information theory

Minimum Description Length (MDL) is a mature model-selection principle with established research on pattern mining, including itemsets, sequences and graph patterns. It is not a novel Symthaea invention.

Use MDL as a baseline for comparing the compactness of a geometry-generating model against matched nulls. Compare it with simpler measures (entropy, recurrence, symmetry residuals) and with model families that account for geometric complexity. Penalize search over many encodings and parameterizations. A shorter description is evidence of regularity relative to a declared model family—not evidence of intention or authorship.

### Scientific inference

Use accepted statistical controls as baselines, not proprietary AI novelty:
- preregistration and blinding;
- hierarchical models for repeated plants nested within locations and events;
- family-wise error control for a decoder/transform tournament;
- proper predictive scoring and uncertainty calibration;
- held-out event-level evaluation;
- negative controls and independent replication;
- explicit out-of-domain abstention.

## 4. What "better than SOTA" should mean here

There is no single SOTA number. The goal is **Pareto superiority on matched tasks**, not a composite score that hides weaknesses.

| Track | Strongest baseline to beat | CROP-INTEL differentiation | Evidence required before claiming superiority |
|---|---|---|---|
| Historical catalogue | CCRA, Crop Decoder, MYTbrain reconciliation | Assertion-level provenance, conflict retention, rights and event/entity uncertainty | Audited sampled reconciliation with false-merge and false-split rates |
| Exact-field geolocation | MYTbrain SIFT/RANSAC and human review; GIS/photogrammetry workflows | Independent benchmark participation, explicit uncertainty and cost/latency | Permissioned real imagery, independent checkpoints, hard-negative fields and held-out sites |
| Geometry reconstruction | Manual vector tracing, contour/Hough, feature-based and learned vision methods; FEEDIT parametric plates | Primitive-level uncertainty, dual reconstruction and descriptors linked to source transforms | Blind held-out surveyed field geometry |
| Pattern/information analysis | Symmetry, signal processing, graph mining, MDL and conventional encoding pipelines | Pre-registered decoder tournament with complete search log and max-statistic null calibration | Positive, negative and adversarial controls; independent reproduction |
| Causal explanation | Explicit physical models, crop mechanics, weather/terrain analysis and controlled experiments | Multiple competing predictions plus active experiment selection | Held-out spatial predictions and physical/biological replication |
| Scientific assistant | Search/LLM tools and conventional Bayesian/statistical workflows | Evidence graph, model uncertainty, falsification, cost-aware information gain and abstention | Matched prospective benchmark with blinded hypotheses and independent graders |

## 5. Proposed performance goals

These are internal research targets, not published SOTA results.

1. **Catalogue:** zero silent merges; every event has source assertions, rights state and uncertainty. Report entity precision/recall against an adjudicated sample rather than maximize record count.
2. **Geometry:** F1 ≥ 0.95 on synthetic surveyed controls and median boundary deviation ≤ 1% of formation diameter on held-out surveyed examples. Publish real and synthetic scores separately.
3. **Registration:** keep optimizer control residual separate from held-out checkpoint error. Use median, RMSE, p95 and maximum error in metres. Include hard-negative fields and false-acceptance rate.
4. **Information:** family-wise α ≤ 0.01 for a preregistered S2 code-like claim; exact held-out payload recovery and independent blind reproduction for S3.
5. **Calibration:** publish Brier/log score, calibration curves, selective risk/coverage and false-confident anomaly rate.
6. **Physical models:** compare predictive log score, interval coverage, physical budget closure and identifiability on withheld observations.
7. **Efficiency:** report end-to-end cost, memory, human review time and latency; do not let cheap incorrect matches count as wins.
8. **Reproducibility:** every headline number tied to a frozen dataset version, source hashes, software commit, configuration, seed, exclusions and independent review.

## 6. Strategic advantage

Do not compete on the whole pipeline with every archive and geolocator. Form an interoperability strategy:

- import *references and assertions*, not other people's copyrighted images;
- use rights-cleared parametric geometry as the first scientific data;
- publish our own synthetic controls, geometry test suite and decoder benchmark;
- build adapters for independent catalogue/geolocator outputs while preserving source identity and confidence;
- invite independent researchers to reproduce the results;
- keep all origin hypotheses open but require increasingly strong evidence for increasingly strong claims.

The defensible asset would be a transparent, independently reusable scientific benchmark and evidence system—not the assertion that Symthaea has solved crop circles.
