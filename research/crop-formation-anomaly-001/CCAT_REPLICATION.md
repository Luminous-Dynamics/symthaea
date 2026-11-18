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

## 5. Rust replay implementation

The standalone dependency-free crate adds `circle_layout.rs` with:
- strict four-column `id,x,y,radius` CSV parsing;
- duplicate-ID, non-finite-coordinate and non-positive-radius rejection;
- arithmetic-centre and radial-distance calculation;
- pairwise tangent-gap graph construction at a caller-declared tolerance;
- quantile-initialized 1D k-means for radius and radial-band descriptors;
- centred neighbour-pair motif counting.

The Rust test suite contains known-answer checks against the imported data for:
- 60 records;
- 66 contact edges at a 45-pixel threshold;
- 12 A–E contacts;
- A–E–B = 10, B–A–E = 10, A–E–A = 8.

These tests have been authored but **not compiled or executed** in this environment. The numerical replay above was performed separately in Python. No CI pass or Rust execution is claimed.

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
