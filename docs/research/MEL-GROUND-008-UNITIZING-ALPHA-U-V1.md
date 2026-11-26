# MEL-GROUND-008 — Corrected multi-rater unitizing alpha-u for musical event intervals

**Status:** synthetic-first Rust implementation; awaiting exact-head qualification  
**Date:** 26 November 2026  
**Implementation:** `tools/mel-temporal-grounding-qualifier/src/temporal_unitizing.rs`

## 1. Why this is a separate measurement

The existing annotation programme already reports:
- nominal Krippendorff alpha for categorical dispositions (Present / Absent / Ambiguous / NotEstimable);
- descriptive pairwise event F1, exact occurrence-count agreement and matched IoU;
- per-event onset/offset boundary distributions.

None of those is a chance-corrected coefficient for independently unitizing stretches of a continuum. Framewise kappa would make the result depend on arbitrary time-bin width. Pairwise interval F1 uses the production matcher and an IoU threshold, but it is not chance-corrected and does not model the full marked and unmarked continuum.

This module adds the **Krippendorff 2019 fourth-edition §12.6 alpha-u family**, based on the corrected multi-rater developments in Krippendorff, Mathet, Bouvry & Widlöcher (2016). It does not implement the withdrawn simplified first-printing c_u-alpha equation from the earlier version.

## 2. Selected coefficient family

For each query-item, eligible raters partition the same finite continuum `[0, L)` into labeled event intervals and explicit gap intervals, where `L` is the exact source audio frame count. Within a query, all Present event intervals share the category `event`. An Absent annotation has no event intervals and therefore contributes a full-continuum gap partition.

Let:
- (m) be the number of eligible unitizers;
- (g,h) be partition segments, each with category (c) or (k);
- (|g \cap h|) be the length of their intersection in integer frames;
- (phi) be the distinguished gap category.

The observed ordered coincidence matrix is

[
\ell_{ck}
=
\frac{1}{m-1}
\sum_i\sum_{j\ne i}
\sum_{g\in P_i(c)}\sum_{h\in P_j(k)}
|g\cap h|.
]

Thus every annotator is compared with every other annotator, and each ordered rater pair is normalized by (m-1). Its total mass is (N=\sum_{ck}\ell_{ck}=mL).

For each category, the self-pairing correction is

[
C_c =
\begin{cases}
\sum_i\sum_{g\in P_i(\phi)} |g|, & c=\phi,\\
\sum_i\sum_{g\in P_i(c)} |g|^2, & c\ne\phi.
\end{cases}
]

The expected coincidence matrix is

[
E_{ck} =
\frac{\ell_{c\cdot}\ell_{\cdot k}
-\mathbf 1[c=k]C_c}
{N-\frac{\sum_c C_c}{N}}.
]

For either the observed or expected matrix, disagreement is total off-diagonal mass:

[
D_o=N-\sum_c\ell_{cc},
\qquad
D_e=N-\sum_c E_{cc}.
]

The overall nominal coefficient is

[
\alpha_{u,nominal}=1-\frac{D_o}{D_e}.
]

The profile keeps four distinct outputs:

1. **Overall nominal u-alpha** — agreement on unit boundaries and category labels, with gaps included.
2. **Binary u-alpha** — collapse all non-gap event categories to one `!gap` value, then apply the same coefficient. It asks whether the same regions were marked at all, independently of event category.
3. **Coding-conditional c-u-alpha nominal** — restrict coincidences to non-gap/non-gap overlap and apply the corrected pairable-length self-correction. It measures category coding conditional on pairable marked regions.
4. **Per-value u-alpha** — category-specific reliability and overlap coverage. Values with little or no pairable overlap must remain visibly sparse; they must not be silently dropped.

When all events in a query use one category, overall nominal and binary u-alpha are expected to coincide. The coding-conditional coefficient answers a narrower question and should be read with its pairable-coverage field.

Degenerate expected disagreement is explicit: when both observed and expected disagreement are zero, alpha is 1; when expected disagreement is zero but observed disagreement is positive, alpha is undefined (`null` in JSON), never NaN or infinity.

## 3. Audio-specific input policy

- The unitizing continuum is the complete source recording, not a cropped excerpt with an implicit local origin.
- Coordinates are zero-based, half-open sample-frame intervals: `[start_frame, end_frame)`.
- All raters on one item must share the exact query ID, audio digest, sample rate, channel count, total frame count and annotation policy (inherited from the corpus item).
- One item represents one query/event category. Different instruments, event categories or questions are separate items; do not merge overlapping polyphonic classes into one single-label unitizing continuum.
- Within one rater, intervals must be valid, sorted and non-overlapping. The implementation rejects overlap; it never unions or repairs a rater's labels.
- Gaps are the exact complement of that rater's marked intervals over `[0,L)`. Gaps are not pairwise evidence of agreement by themselves; the coefficient's coincidence equations account for gap mass.
- Present with no interval is invalid. Absent with a non-empty interval list is invalid.
- Ambiguous and NotEstimable annotations are retained with their original digests and explicit reasons, but excluded from the unitizing panel. They are not recoded as Absent or as a gap.
- The minimum eligible rater count is declared before calculation. The current implementation supports 2 or more eligible raters per item.

Coordinates and overlap lengths remain integer frames throughout segmentation and intersection. Only the final coincidence/disagreement arithmetic uses floating point. The maximum continuum length is bounded to the exact-integer range of the JSON/Rust f64 implementation.

## 4. Golden reference and synthetic controls

The unit test transcribes the published Krippendorff 2019 fourth-edition Figure 12.11 into zero-based half-open frame coordinates. It uses three raters over a continuum of length 60.

Expected reference values:
- overall nominal u-alpha: approximately **0.510**;
- binary u-alpha: approximately **0.413**;
- coding-conditional c-u-alpha: approximately **0.72835**;
- per-value alpha anchors: labels 1, 5 and 6 are 1.0; labels 2, 3 and 4 are 0.0;
- label-5 overlap coverage: approximately (8/18);
- label-7 overlap coverage: 0.

Other Rust tests cover identical two-rater unitizations, a one-category audio query, unanimous absence, disjoint unitizations, a one-rater invalid panel and overlapping/unsorted intervals.

The CLI fixture includes a three-rater Present panel with disagreement, two preserved excluded records, a unanimous Absent panel, and an item with insufficient eligible raters. The fixture tests report structure and exclusion accounting; it is not musician data.

## 5. Aggregation and statistical limits

This v1 reports item-level coefficients and descriptive means/medians/min/max across scored items. It also groups summaries by source-work lineage.

It intentionally does **not**:
- pool frames, coincidence matrices or continuum lengths across unrelated recordings into one “corpus alpha”;
- compute population confidence intervals or bootstrap intervals;
- resample frames, intervals or excerpts as if they were independent works;
- create consensus labels or adjudicate who is right;
- score model predictions;
- authenticate annotation receipts or prove rater identity/consent.

Source-work summaries prevent a long recording split into many excerpts from silently dominating the display. Inferential source-work bootstrap, if later added, must be preregistered and treat source work as the sampling unit. Whether a coefficient-specific CI procedure supports the intended population estimand must be established before adding it.

## 6. Relation to the other MEL measures

Keep these columns separate:

| Measurement | Question |
|---|---|
| Nominal disposition alpha | Do raters agree on Present / Absent / Ambiguous / NotEstimable across query-items? |
| Overall nominal u-alpha | Do raters independently mark the same stretches with the same label, including gaps? |
| Binary u-alpha | Do raters mark the same stretches, ignoring event identity/category? |
| Coding-conditional c-u-alpha | Where marked regions are pairable, do raters assign the same category? |
| Per-value u-alpha / coverage | How reliable and cross-rater-covered is each category? |
| Pairwise event F1 / IoU | How well do two specific event interval sets overlap under the frozen matcher? |
| Boundary MAE / P95 | How far apart are their onset/offset frames? |

No single coefficient is “the music annotation truth.” These are different reliability views with different chance models and denominators.

## 7. References

- Krippendorff, K. (2019). *Content Analysis: An Introduction to Its Methodology*, 4th ed., §12.6, Sage. ISBN/DOI: 10.4135/9781071878781.
- Krippendorff, K., Mathet, Y., Bouvry, S. & Widlöcher, A. (2016). “On the reliability of unitizing textual continua: Further developments.” *Quality & Quantity* 50, 2347–2364. DOI: [10.1007/s11135-015-0266-1](https://doi.org/10.1007/s11135-015-0266-1).
- Quallmer 0.6.0 (CRAN, 21 Oct 2026), independent native implementation and worked-example tests for the 2019 §12.6 family: [CRAN package](https://cran.r-project.org/package=quallmer). Its R code is GPL-3; this repository contains an independent Rust transcription of the published equations, not copied source.
- Krippendorff, K. (1995). “On the Reliability of Unitizing Continuous Data.” *Sociological Methodology* 25, 47–76. DOI: [10.2307/271061](https://doi.org/10.2307/271061). This is historical context; the present implementation targets the later corrected formulation.

## 8. Claim ceiling

This implementation estimates annotation agreement under a specified one-category unitizing protocol. It does not establish that annotators are correct, culturally representative, equally expert or free of shared bias. It does not measure model accuracy, musical understanding, or musician agency. No SOTA or human-performance claim is made by passing synthetic equations and CLI fixtures.


The per-item report preserves the original query text, audio digest/sample rate/channels, annotation policy, source-work lineage, every annotator digest/state, the original interval arrays and each status reason. Excluded Ambiguous and NotEstimable records remain recoverable with explicit exclusion reasons; they are not turned into gaps or negative labels.