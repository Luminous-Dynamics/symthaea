# CROP-INTEL-001 — Crop-formation structure, information, and physical anomaly research

**Status:** proposed research programme; initial protocol only  
**Branch:** research/crop-formation-anomaly-001  
**Claim ceiling:** no formation has been adjudicated by this work; no non-human origin, unknown physical mechanism, or encoded semantic message is established.

## Mission

Build a reproducible, open-world investigation workflow for reported crop formations. Treat each formation as a set of independently testable questions:

1. What geometry is actually present in the field?
2. What mathematical and statistical structure does that geometry contain?
3. Does it support a stable, pre-specified information encoding?
4. What physical process could produce the crop deformation and any independently measured plant effects?
5. Which observations discriminate between competing explanations?
6. What remains unknown because the evidence is insufficient?

The programme must be useful even if every case turns out to be human-made, naturally lodged, or misreported. Better cataloguing, geometry reconstruction, crop-mechanics studies, and calibrated anomaly detection are valuable outcomes in their own right.

## The central distinction: structure is not automatically a message

All patterns contain information in an information-theoretic sense. That does **not** mean that every pattern is an intentional message.

Use four separate claim levels:

| Level | Claim | Evidence needed |
|---|---|---|
| S0 — observed structure | The image/field contains measurable shapes, regions, boundaries, repetitions or spatial relationships | Source image, calibration, extraction method, uncertainty |
| S1 — mathematical regularity | The structure has symmetry, recurrence, fractal-like scaling, ratios, graph motifs or measurable compressibility | Declared descriptors and comparison against suitable null models |
| S2 — code-like structure | A consistent mapping from geometric features to symbols/bits is supported | Predeclared decoder family, controls, multiple-testing correction, robustness |
| S3 — semantic payload | The decoding yields a stable, independently interpretable message or data payload | Exact held-out recovery, independent decoding, predictive or checksum-like validation |

A golden-ratio fit, a binary-looking grid, a recognizable pictogram, or a clever textual interpretation is not by itself evidence of S2 or S3. The decoder, orientation, symbol mapping, segmentation, and allowed transformations must be specified before inspecting the proposed payload. Every attempted decoder must be logged, including failures.

## Keep the causal questions orthogonal

Do not collapse the following into a single "genuine / hoax" label:

- **Authorship:** documented human authorship, probable human authorship, unknown, or conflicting evidence.
- **Formation mechanism:** mechanical flattening, weather/lodging, other supported mechanism, unresolved, or mixed.
- **Plant response:** no measured difference, difference detected, inconclusive, or not sampled.
- **Information content:** S0, S1, S2, S3, or not assessable.
- **Evidence quality:** source provenance, time certainty, geolocation certainty, measurement quality, custody, and independent review.

"Unexplained" means only that the current evidence and model set do not resolve the case. It is not a synonym for paranormal, extraterrestrial, or non-human.

## Research questions

### RQ1 — What structures are present?

Reconstruct formations as calibrated geometry, not merely as attractive aerial photographs. Extract circles, arcs, straight segments, paths, filled regions, intersections, repeated components, crop-flow boundaries, and standing/flattened regions. Preserve the image-to-field transformation and uncertainty.

Measure:
- rotational and reflection symmetry, including symmetry-breaking residuals;
- radii, angular spacing, chord lengths, intersections and ratios;
- adjacency and topology graphs;
- radial, angular and spatial frequency spectra;
- repeated motifs and hierarchical/self-similar structure;
- scale-dependent fractal descriptors, with finite-resolution limits stated;
- minimum-description-length (MDL) / compression scores;
- deviations from matched human-art, algorithmic-art and random/geometric null sets.

Do not report a fractal dimension, exact ratio, or mathematical constant without scale, uncertainty, fitting range, and alternative-model comparison.

### RQ2 — Is there information encoded in the geometry?

Establish a decoder tournament with bounded, pre-registered families:
- binary occupancy of standing versus flattened regions;
- polar/radial and concentric-ring readings;
- rectangular grids and both axis orientations;
- graph-based symbol and path encodings;
- geometric alphabets based on arcs, intersections, counts and orientation;
- known public encodings used as positive controls (for example, generated Arecibo-style matrices);
- deliberately human-authored messages hidden in geometry.

The primary challenge cases should include the reported 2001 Chilbolton/Arecibo-like formation and 2002 Crabwood disc, but mark these as **claimed or disputed decodings**, not known-positive encoded messages. Preserve the original imagery, all segmentation alternatives, and every decoder attempt. Do not choose the decoding scheme because it gives an appealing answer.

A meaningful S2/S3 result must survive:
1. independent geometric reconstruction from source imagery;
2. pre-registered orientation and segmentation alternatives;
3. a family-wise null test across the entire decoder search;
4. image perturbations and independent reconstructions;
5. negative controls with similar complexity and visual density;
6. a held-out prediction or exact recovery test;
7. independent replication by analysts who were not given the proposed message.

### RQ3 — What physical processes can produce the formation?

Model at least:
- human mechanical construction and crop-lodging methods;
- wind, gusts, rain, terrain and ordinary lodging;
- crop species, growth stage, moisture and stem mechanics;
- machinery/foot traffic and access paths;
- other physically specified hypotheses only where they generate discriminating predictions.

A mechanism is not supported because a simulation can draw a similar shape. It must predict independent observables: deformation direction, boundary sharpness, plant break/bend profiles, spatial gradients, energy/momentum requirements, and effects under controlled conditions.

### RQ4 — Are there reproducible biological effects?

Historical literature includes Levengood's 1994 report of plant anatomical and developmental differences, later debate about sampling and statistical interpretation, and a 2014 study that compared a reported formation with an experimentally made control circle. This record is disputed and does not establish a unique energy source or non-human mechanism. The 2014 authors reported that their man-made control did not reproduce the original case's reported node pattern; that is a reason to design a better replication, not a general proof.

A new field study must be:
- permitted by the landowner and non-destructive outside agreed sampling;
- prospectively registered;
- randomized and blinded at sample labeling and laboratory analysis;
- matched by cultivar, field, crop stage, soil zone, moisture and weather;
- spatially resolved inside, at boundaries, and across multiple outside distances;
- controlled against hand/board/roller/foot flattening and naturally lodged crop;
- powered from a declared smallest effect of interest and variance model;
- independently replicated.

Measure plant geometry, node and internode dimensions, bend angle, breakage, tissue histology where feasible, seed viability/germination, and relevant soil/environmental covariates. Do not infer radiation, plasma, or any special energy source from a biological difference alone.

## Data sources and current research landscape

These are discovery sources, not ground truth:

- [Crop Circle Research Archive database](https://circleresearcharchive.com/mast/the-database/) describes nearly 17,000 cross-referenced information items, over 2,850 formations, and nearly 8,000 photographs, mostly spanning 1927 to the early 2000s.
- [Crop Decoder](https://cropdecoder.com/formations/) currently lists 2,982 formations from 35 countries and links records to source archives. Treat it as a cross-archive discovery index; entity deduplication and source completeness still need auditing.
- [FEEDIT field geometry archive](https://www.feedit.info/) reports 11 parametric formations, 1,195 geometric elements and machine-readable geometry in metres under CC BY-SA 4.0. It is a valuable small geometry seed corpus, not a representative sample of all formations. Referenced aerial photographs have separate rights.
- [MYTbrain/crop-circle-atlas](https://github.com/MYTbrain/crop-circle-atlas) is an important external engineering baseline: provenance-first catalogue reconciliation, geolocation review, bounded image registration, SIFT/RANSAC proposals, metric checkpoints, and rights-gated map outputs. Its README explicitly limits its reported benchmark to synthetic mathematics and internal pipeline QA; do not promote those numbers into real-world crop-circle geolocation performance. This programme does not claim affiliation with that project.

The public sources located so far do not establish a recognized, community-wide crop-circle benchmark or leaderboard. Therefore, do not claim "state of the art" by comparing one private score with an unrelated paper. First publish a transparent benchmark with frozen inputs and baselines.

## Paradigm shifts

1. **From visual interpretation to measurement-first geometry.** Every derived shape retains source pixels, calibration, coordinate transforms and uncertainty.
2. **From post-hoc decoding to preregistered decoder tournaments.** Count every tested decoder and correct for the entire search, not just the final attractive result.
3. **From binary genuine/hoax labels to orthogonal causal dimensions.** Authorship, physical process, plant response, information content and evidence quality are separate.
4. **From anomaly scores to experiment selection.** Symthaea should choose the next observation that best distinguishes live hypotheses, subject to power, cost, permission and safety constraints.
5. **From an image archive to a spatiotemporal evidence graph.** Connect each formation, source report, photograph, geolocation, weather record, crop condition, geometry extraction, physical model and sample without merging conflicting assertions.
6. **From AI interpretation to independent replication.** A result only graduates when a second pipeline or analyst reproduces it from frozen inputs.

## Architecture fit

- **Symthaea:** open-world hypothesis generation, evidence graph reasoning, uncertainty/calibration, counterfactual prediction, decoder tournament orchestration, and abstention.
- **Symtropy:** physical simulation and sensitivity analysis for wind/crop mechanics, only after boundary conditions and constitutive assumptions are explicit.
- **Mycelix/evidence layer:** source assertions, immutable evidence IDs, custody/transfer events, signatures and reproducible receipts. Provenance does not establish truth.
- **Sol-Atlas / geospatial tools:** map alignment, field context, remote-sensing provenance and rights-aware imagery handling where appropriate.

Reuse the existing Symthaea falsification/power reference and open-world work rather than implementing a duplicate statistical framework. The branch now includes a small dependency-free Rust reference crate for calibrated point-set geometry, explicit binary-grid transforms, raw-bit decoder attempts, and simple statistical helpers. It is not a production image-analysis pipeline and has not yet been compiled or executed.

## Work packages

| WP | Deliverable | Exit condition |
|---|---|---|
| WP0 | Source, rights and entity audit | Every imported assertion has a source, rights state and duplicate/uncertainty disposition |
| WP1 | Geometry extraction + parametric representation | Independent reconstructions agree within declared uncertainty on surveyed controls |
| WP2 | Information/decoder benchmark | Decoder tournament passes negative controls and held-out payload tests |
| WP3 | Physical mechanism benchmark | Competing mechanisms produce discriminating, falsifiable predictions |
| WP4 | Plant study preregistration | Blinded controls, power analysis, sample custody and independent lab plan accepted |
| WP5 | Integrated evidence qualification | Complete reproducible receipt; independent review; claim level stated explicitly |

## Initial execution sequence (no CI dependency)

1. Freeze this research contract and its claim ceilings.
2. Create a source-assertion inventory from CCRA, Crop Decoder, FEEDIT and the external atlas; do not copy source photographs.
3. Start with FEEDIT's 11 measured formations and known synthetic encodings to validate the geometric pipeline.
4. Add synthetic positive and negative controls before analysing claimed message cases.
5. Run the decoder tournament blind, with all attempts and multiplicity corrections retained.
6. Select one information-rich case only after the benchmark protocol is frozen.
7. Design field measurements only when landowner permission, sample custody, laboratory capacity and prospective registration are real.

## Non-goals and safeguards

- No trespass, unauthorized drone operation, crop damage, or collection without permission.
- No scraping or redistribution of copyrighted aerial photographs; retain source links, credits and rights states.
- No disclosure of private landowner details or sensitive site information.
- No extraordinary-origin inference from an image, ratio, biological difference, witness report or anomaly score alone.
- No runner/CI status implied by a document commit. Tests and physical experiments remain unrun until separately evidenced.

## References

### Primary / disputed plant research
- Levengood (1994), [Anatomical anomalies in crop formation plants](https://doi.org/10.1111/j.1399-3054.1994.tb05348.x), *Physiologia Plantarum* 92, 356–363.
- Levengood & Talbott (1999), [Dispersion of energies in worldwide crop formations](https://doi.org/10.1034/j.1399-3054.1999.105404.x), *Physiologia Plantarum* 105, 615–624.
- Grassi, Cocheo & Russo (2005), [Balls of Light: The Questionable Science of Crop Circles](https://www.cicap.org/crops/en/jse_19_2_159-170_2005.pdf), *Journal of Scientific Exploration* 19(2), 159–170 — methodological and physical-model critique.
- Haselhoff, Boerman & Bobbink (2014), [An Experimental Study for Reproduction of Biological Anomalies Reported in the Hoeven 1999 Crop Circle](https://www.robertboerman.nl/pdf/experimental-study.pdf), *Journal of Scientific Exploration* 28(1), 17–33 — a single-case study with a man-made control, not independent multi-site replication.

### Claimed information-bearing cases
- Paul Vigay, [The Chilbolton 'Arecibo message' Formation](https://www.vigay.com/cropcircles/articles/arecibo.html) — contemporaneous claimed decoding.
- The Skeptic (2023), [The Arecibo Reply: How We Know Aliens Aren't Calling Us Through Crop Circles](https://www.skeptic.org.uk/2023/11/the-arecibo-reply-how-we-know-aliens-arent-calling-us-through-crop-circles/) — skeptical analysis and alternative interpretation.
- [Crabwood formation archive / claimed binary-ASCII reading](https://www.cropcirclearchives.co.uk/archives/2002/Crabwood/crabwood2002a.html) — a claimed decoding to be treated as contested input, not confirmed payload.
- Hafdi (uploaded March 2026), [The Mathematics of Crop Circles and Their Applications to Complex Problems](https://www.researchgate.net/publication/403159866_The_Mathematics_of_Crop_Circles_and_Their_Applications_to_Complex_Problems_Geometry_Number_Theory_Encoding_and_Engineering_Implications_of_Agrarian_Formations) — exploratory manuscript; claims and decodings require independent verification.

---
**Current result:** research programme specification committed. No cases classified; no data ingestion, image processing, code execution, field sampling, or scientific replication claimed.


## Research extensions and implementation

- [Competitive landscape and SOTA positioning](COMPETITIVE_LANDSCAPE.md) — archives, exact-field geolocation, remote-sensing baselines and a matched comparison strategy.
- [Active experiment selection](EXPERIMENT_SELECTION.md) — expected information gain, statistical power, cost, feasibility and model-misspecification safeguards.
- [Standalone Rust reference lab](lab/README.md) — dependency-free geometry and crop-flow descriptors, explicit binary-grid D4 × traversal × bit-order decoder family, 7-bit decoding and complete attempt logging, row/column-margin-preserving grid null generation, maximum-statistic calibration helper, entropy and Wilson intervals.

The current reference crate includes crop-flow circular statistics as well as geometry and grid decoding. It preserves per-sample direction and does not force a single mean direction when the resultant is near zero.

The Rust lab is an initial reference implementation, not a production image pipeline. It currently accepts calibrated 2D point sets, preconstructed binary grids, and bit sequences; it does not read aerial images, infer crop masks from photographs, classify formation origin, or infer semantic meaning. Its unit tests have been authored but **not executed** in this environment because no Rust toolchain is installed. Do not treat the crate as compiled or qualified until `cargo fmt`, `cargo test` and `cargo clippy` run on the exact head.
