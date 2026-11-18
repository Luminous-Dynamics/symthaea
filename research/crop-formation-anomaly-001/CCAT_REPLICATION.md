# CROP-INTEL-001 — CCAT Zeals Knoll geometry replay and cross-case benchmark

**Version:** 0.1  
**Status:** source dataset imported with attribution; exploratory independent calculations performed; Rust known-answer tests authored but not run.

## 1. Why this fixture matters

CCAT v1.0.0 provides a strong, directly relevant external baseline for geometric construction-grammar research. The project applies its toolkit to the Zeals Knoll 2026 formation and publishes a derived 60-circle geometry CSV under CC BY 4.0.

CROP-INTEL imports only the geometry CSV. It does not import the photograph, source imagery, image tiles, or any other rights-restricted media.

## 2. Dataset provenance

- Author: Salah-Eddin Gherbi.
- Software/research release: CCAT v1.0.0, released 2026-07-24.
- Citation: [CCAT v1.0.0, Zenodo DOI 10.5281/zenodo.21527026](https://doi.org/10.5281/zenodo.21527026).
- Concept DOI: [10.5281/zenodo.21527025](https://doi.org/10.5281/zenodo.21527025).
- Original repository: https://github.com/salahealer9/ccat-crop-circle-analysis-toolkit
- Original data file: `data/circles_drag_60.csv`.
- Source Git blob SHA: `97e95ce39b8e5b0e1f663d3ab80c635add24222a`.
- Data licence declared by source: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Source software licence: MIT; this is separate from the data licence.
- CROP-INTEL copy: `datasets/ccat-zeals-knoll-2026/circles_drag_60.csv`.
- CROP-INTEL manifest: `datasets/ccat-zeals-knoll-2026/manifest.json`.

The imported CSV was fetched from the declared source and compared line-by-line with the source content after import; it is identical. Row order, identifiers, precision and values are preserved. Attribution and restrictions are recorded in the dataset directory.

## 3. Source claim boundary

CCAT's own claim-boundary document supports a *plausible geometric construction grammar* for this one digitized formation: 60 circles, five radius classes, a near-regular 17-circle outer structure, radial organization, A–E adjacency enrichment and repeated A/B/E motifs.

The source explicitly does **not** claim symbolic meaning, authorship, physical construction method, intentional message, astronomical decoding, or proof that the formation was built according to the recovered grammar. CROP-INTEL adopts those limits.

The source's predictive validation uses 80 stratified masks of the same 60-circle event, with 15 class labels hidden per split. Its reported radial-prior accuracy is 0.515 and macro-F1 0.289; the pair/motif grammar has accuracy 0.493 and macro-F1 0.419. CCAT itself concludes that radial position gives the best raw accuracy, while the local grammar helps recover minority classes, especially E. This is a useful **within-event, transductive masked-label result**, not cross-event generalization.

## 4. Independent descriptive replay

A separate calculation was performed from the imported 60-row CSV, using the source's declared general method:
- arithmetic mean of circle centres as the radial reference centre;
- one-dimensional quantile-initialized k-means with five radius classes and five radial bands;
- pairwise tangent gap (g_{ij}=d(c_i,c_j)-(r_i+r_j));
- an edge when (|g_{ij}| le 45) input coordinate units;
- centred motifs counted from unordered pairs of neighbours in that fixed contact graph.

The exploratory replay produced:

| Quantity | Recomputed value |
|---|---:|
| Circle records | 60 |
| Five radius-class centres | 25.6264, 34.5994, 44.9150, 52.2998, 76.8023 px |
| Contact edges at (|g|le45) px | 66 |
| A–E contact edges | 12 |
| A–E–B centred motifs | 10 |
| B–A–E centred motifs | 10 |
| A–E–A centred motifs | 8 |

The radius-class centres and the 12 / 10 / 10 / 8 contact/motif counts reproduce the source-reported values visible in its analysis outputs. The total of 66 proximity edges is an additional descriptive output from this replay.

**Interpretation:** this is a computational cross-check of the published digitized geometry and method family. It is not an independent measurement of the field, not independent validation of the circle digitization, and not evidence of authorship, intent, message, or unusual physical origin.

## 4A. Independent threshold and measurement-sensitivity audit

The CCAT source already performs an explicit contact-tolerance sweep: 30, 35, 40, 45, 50, 55 and 60 px. Its `radius_contact_tolerance_sweep.py` defaults to 5,000 null trials per tolerance and null model, evaluates five contact metrics, uses global and within-ring label shuffles, and applies Benjamini–Hochberg correction across the 35 metric-by-tolerance tests separately for each null model. This is a strength of the source methodology; CROP-INTEL does not claim CCAT ignored tolerance sensitivity.

Our independent descriptive replay makes the graph-density consequences visible:

| Tolerance (px) | Contact edges | A–E edges | A–E–B | B–A–E | A–E–A |
|---:|---:|---:|---:|---:|---:|
| 30 | 6 | 0 | 0 | 0 | 0 |
| 35 | 24 | 3 | 1 | 0 | 0 |
| 40 | 42 | 5 | 4 | 3 | 1 |
| 45 | 66 | 12 | 10 | 10 | 8 |
| 50 | 84 | 14 | 16 | 10 | 11 |
| 55 | 93 | 18 | 24 | 14 | 17 |
| 60 | 103 | 18 | 25 | 16 | 17 |

These are descriptive counts on the fixed imported geometry and five-class assignment, not p-values. The source's null sweep is the appropriate source for its reported inferential comparisons.

### Radius-class-count sensitivity

With the contact graph fixed at 45 px, changing the requested number of quantile-initialized radius classes changes the class vocabulary and the motif labels:

| Classes (k) | Class populations, smallest to largest radius | Same-class contact edges | Most frequent centred motif |
|---:|---|---:|---|
| 3 | 25 / 28 / 7 | 20 | 1–3–1 (19) |
| 4 | 15 / 13 / 25 / 7 | 11 | 1–4–2 (11) |
| 5 | 15 / 10 / 8 / 20 / 7 | 7 | 1–5–2 (10) |
| 6 | 15 / 9 / 4 / 7 / 18 / 7 | 8 | 1–6–2 (10) |
| 7 | 9 / 10 / 6 / 6 / 11 / 11 / 7 | 4 | 1–7–2 (9) |

Class labels are ordinal bins generated by a chosen algorithm and class count, not intrinsic symbols. The A–E motif vocabulary is therefore conditional on selecting five classes. This sensitivity test does not invalidate the five-class model; it shows why the class count and its selection rationale belong in the preregistered model family.

### Coordinate/radius perturbation stress test

Because the source CSV has no measured per-circle uncertainty, we ran an explicitly hypothetical stress test. For each replicate, each circle's x, y and radius were independently perturbed by a uniform offset in ([-\delta,+\delta]) pixels. We then recomputed the arithmetic centre, five radius classes, 45 px contact graph and centred motifs. Each bound used 300 replicates with seed 20261118.

| Assumed bound (px) | Mean contact edges ± SD | Mean A–E edges ± SD | Mean A–E–B motifs ± SD | Radius-class label agreement with unperturbed case |
|---:|---:|---:|---:|---:|
| ±0.5 | 65.43 ± 1.05 | 11.90 ± 0.30 | 10.17 ± 0.56 | 99.27% |
| ±1 | 64.38 ± 1.56 | 11.47 ± 0.80 | 10.30 ± 1.00 | 98.29% |
| ±2 | 63.38 ± 2.51 | 10.18 ± 2.04 | 9.81 ± 2.07 | 94.41% |
| ±5 | 61.43 ± 4.52 | 8.36 ± 2.47 | 9.03 ± 3.29 | 82.04% |

Exact simultaneous recovery of all five baseline counts (66 edges, 12 A–E edges, 10 A–E–B, 10 B–A–E and 8 A–E–A) occurred in 28%, 8%, 0.33% and 0% of replicates for ±0.5, ±1, ±2 and ±5 px respectively.

**Interpretation ceiling:** these are chosen perturbation envelopes, not estimates of the real digitizer's standard error. They cannot tell us whether CCAT's actual digitization is accurate to 0.5, 1, 2 or 5 pixels. They show that the exact contact/motif counts can change under small hypothetical coordinate/radius changes while the broad radius-class assignments remain comparatively stable.

The next empirical step is to obtain rights-cleared source imagery and independent digitizations, then estimate centre/radius uncertainty from annotator disagreement and repeat measurements rather than assigning a convenient error bound.

## 5. Rust replay implementation

The standalone dependency-free crate adds `circle_layout.rs` with:
- strict four-column `id,x,y,radius` CSV parsing;
- duplicate-ID, non-finite-coordinate and non-positive-radius rejection;
- arithmetic-centre and radial-distance calculation;
- pairwise tangent-gap graph construction at a caller-declared tolerance;
- quantile-initialized 1D k-means for radius and radial-band descriptors;
- centred neighbour-pair motif counting.

The Rust lab also exposes `audit_circle_measurement_sensitivity`: callers provide explicit centre/radius pixel perturbation bounds, class count, contact tolerance, replicate count and seed. The routine reruns centre estimation, clustering, contact graph and motif counts for every perturbed dataset, then reports sample spreads, radius-label agreement and exact baseline-signature fraction. It does not estimate measurement uncertainty; the bounds remain declared stress-test assumptions.

The Rust test suite contains known-answer checks against the imported data for:
- 60 records;
- 66 contact edges at a 45-pixel threshold;
- 12 A–E contacts;
- A–E–B = 10, B–A–E = 10, A–E–A = 8.

These tests have been authored but **not compiled or executed** in this environment. The numerical descriptive and NumPy perturbation replays above were performed separately in Python; the new Rust sensitivity routine is a deterministic project-native implementation but has not yet been executed, so its numerical output is not claimed to match the exploratory NumPy stream. No CI pass or Rust execution is claimed.

## 6. What the current comparison still cannot establish

1. **Digitization error:** the CSV has no per-circle covariance, annotator disagreement, independent re-digitization or source-image calibration record.
2. **Coordinate meaning:** the values are image-space pixels, not field-local metres or georeferenced positions.
3. **Selection effects:** the formation, 60 circles, five classes, five radial bands, 17-circle boundary subset, 45-pixel contact threshold and motif families were part of a research process. Testing only the final fixed configuration does not correct for the entire sequence of analyst choices.
4. **Conditional nulls:** shuffling classes over fixed centres/graphs can test conditional associations, but it does not test whether the complete image-to-circles-to-classes-to-rings-to-contacts pipeline would find similar structure in unrelated formations.
5. **Generalization:** one event cannot establish a universal crop-formation grammar. Masking points within that same graph is not a substitute for leaving entire formations out.
6. **Mechanism:** a compact grammar that describes a geometry is not a physical construction model and does not identify who or what made it.
7. **Semantics:** mathematical regularity, motif recurrence and compressibility do not establish a symbolic message.

## 7. Next benchmark: from one formation to a family of formations

### Track A — exact replay

- Freeze the imported CSV and source Git blob identity.
- Reproduce source-defined preprocessing and outputs.
- Compare source and independent outputs field-by-field.
- Preserve deviations and all failed assertions.
- Report runtime, method parameters and numerical tolerances.

### Track B — full-pipeline model-selection nulls

The null replicate must rerun the same entire analysis family, not merely shuffle labels after the final geometry has already been selected. Predeclare bounded search ranges for:
- centre estimator;
- radius-class count;
- radial-band count;
- boundary subset/outer-count rule;
- contact threshold, both absolute and radius-normalized;
- motif length and motif families;
- orientation/reflection and permitted reconstruction variants.

For each null replicate, retain the maximum selected statistic over the complete declared family. Use family-wise calibration or a qualified sequential method. Include nulls that preserve different nuisance structures: global class counts, radial counts, geometry/point density, contact degree distribution, perimeter/connected-component structure, and human/algorithmic geometric designs. No single null is sufficient.

### Track C — uncertainty and digitization sensitivity

Because the source CSV supplies no measurement covariance, do not invent one. Run explicit *stress-test envelopes* (for example, declared pixel perturbation magnitudes) and label them as assumptions, not empirical standard errors. Once rights-cleared source imagery and independent annotations are available, estimate measurement uncertainty from inter-annotator and repeat-digitization differences.

### Track D — event-held-out generalization

The primary split unit must be the formation event, not the circle or photograph. Multiple images of the same event must remain in one split. Compare:
- global radius-count baseline;
- radial-only model;
- contact/pair grammar;
- pair-and-motif grammar;
- simpler geometric/MDL baselines;
- an unconstrained or flexible baseline with matched training data and complexity accounting.

Report event-held-out accuracy, macro-F1, per-class recall, proper scores where probabilistic predictions exist, calibration, and risk/coverage. Include negative formations/designs and "no stable grammar" as valid outcomes.

### Track E — independent geometry and causal field science

For selected cases, obtain permissioned source imagery and independent digitizations. Require image calibration, per-circle uncertainty, and inter-annotator agreement. Keep this separate from the crop-flow/plant-response field protocol. A geometric grammar can be evaluated without claiming a physical mechanism.

## 8. Acceptance gates

- Exact source dataset bytes/rows are preserved and attributed.
- Independent replay reproduces the declared descriptive statistics on the frozen fixture.
- Rust tests execute on the exact candidate commit before any test-pass claim.
- Full-pipeline nulls include all predeclared preprocessing and model-selection choices.
- Event-held-out performance is reported separately from within-event transductive masking.
- No general grammar claim until multiple formations and suitable controls exist.
- No semantic, authorship or physical-origin claim from geometry-only evidence.

**Current status:** CCAT geometry fixture imported with attribution; exploratory descriptive replay performed; independent Rust replay code and known-answer tests committed; Rust tests pending execution; no original image re-digitization or field validation performed.
