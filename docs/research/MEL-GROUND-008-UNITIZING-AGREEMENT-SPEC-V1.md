# MEL-GROUND-008 — Unitizing agreement for musical event intervals (specification v1)

**Status:** specification and reference-input fixture only. No alpha-U coefficient is implemented by this increment.

**Owner relationship:** extends #13676 / PR #13742 (corpus-level annotator reliability) and #13560 / PR #13563 (per-query annotations). The implementation must reuse their annotation roster, missingness and source-work lineage conventions, but must not change their nominal disposition alpha, pairwise event-F1, IoU or boundary-error outputs.

## 1. Why this is a separate statistic

A temporal annotation is a *unitization*: an annotator chooses both the locations and durations of units on a shared continuum. This is not the same as coding a fixed list of pre-existing events. A framewise kappa/alpha after discretizing audio into arbitrary windows creates dependence on the chosen bin width and is not a valid substitute for a unitizing reliability coefficient.

Keep the measurement levels separate:

| Question | Metric family | What is not implied |
|---|---|---|
| Did annotators agree that an item is Present / Absent / Ambiguous / NotEstimable? | Nominal Krippendorff alpha in #13742 | Does not measure where event intervals start/end |
| Did annotators mark matching event sets? | Symmetric production-scorer F1, count agreement, IoU thresholds in #13742 | Not a chance-corrected unitizing coefficient |
| How far do onset/offset boundaries differ? | Boundary MAE, median, P95 and support distributions | Not a chance-corrected unitizing coefficient |
| Did annotators independently select the same stretches of the continuum, accounting for gaps and chance placement? | A specified unitizing-alpha family | Does not adjudicate which event is musically correct |

## 2. Reference-method selection

The initial acceptance target is the corrected **u-Alpha software family v1.1** documented by Yann Mathet in consultation with Klaus Krippendorff, based primarily on Krippendorff, Mathet, Bouvry & Widlöcher, *On the Reliability of Unitizing a Continuum: Further Developments* (Quality & Quantity 50, 2347–2364, 2016; DOI 10.1007/s11135-015-0266-1).

The paper's abstract states that the earlier formulation behaved unexpectedly for more than two annotators and that the revised method provides a family of four coefficients with confidence intervals and probability of failing to reach a target reliability. The v1.1 manual presents five report views (including category-specific values) for segmentation/unitizing/coding. The implementation must reproduce the *named measures and their documented scope*, not make up a single value and label it alpha-U.

At minimum the report adapter should reserve distinct fields for:
- All identified units;
- All segments;
- Irrelevant segments relative to units;
- Intersections of valued units (cu-alpha);
- Individual category values in unit intersections.

Exact family/version semantics and how they map to the 2016 paper's four coefficient family must be confirmed against the full reference specification and the uAlpha v1.1 implementation. No coefficient code should be merged until that mapping is frozen.

References:
- 2016 paper: https://doi.org/10.1007/s11135-015-0266-1
- uAlpha v1.1 software manual: https://mathet.users.greyc.fr/agreement/uAlphaSoftware.pdf
- uAlpha software distribution: https://mathet.users.greyc.fr/agreement/
- qdaR unitizing guide (the simpler Krippendorff 1995/2004 formulation, not a substitute for the corrected multi-rater reference family): https://qdar.zotqda.org/latest/en/guide/unitizing.html
- qdaR unitizing source: https://rdrr.io/cran/qdaR/src/R/unitizing.R
- qdaR testing/reference-vector approach: https://qdar.zotqda.org/latest/en/dev/testing.html

## 3. Strict V1 input contract

A unitizing case is for one query/category (or an explicitly defined nominal value set) on one shared finite continuum.

The JSON Schema is:
`tools/mel-temporal-grounding-qualifier/schemas/mel-unitizing-continuum-v1.schema.json`

The source-format reference case is:
`tools/mel-temporal-grounding-qualifier/fixtures/unitizing-reference-alex-paul-suzan.synthetic.json`

It transcribes the three-annotator Alex/Paul/Suzan example in the uAlpha v1.1 manual section 8.1, with anonymized coder identifiers and a synthetic artifact clock. It is a mathematical/software reference fixture, not an audio recording or a Melothaea result.

The schema binds:
- exact artifact digest and audio clock;
- finite half-open continuum `[continuum_start_frame, continuum_end_frame)`;
- query and annotation-policy identity;
- nominal/interval/ordinal/ratio unit value metric;
- independent opaque annotator and annotation digests;
- each rater's independently chosen half-open unit intervals and values;
- source-method provenance and reference-output status.

Additional runtime validation (JSON Schema cannot express all of these invariants):
1. `continuum_start_frame < continuum_end_frame <= artifact.total_frames`.
2. Every unit is nonempty and contained in the continuum.
3. Units within a single annotator's unitization must not overlap. Reject overlap; never union, split, or normalize it silently.
4. Adjacent units remain separate if the annotator encoded two units, even when values match.
5. Gaps are preserved, including a trailing gap up to the declared continuum end. Do not infer continuum length from the latest unit end.
6. Annotator IDs and annotation digests are unique within a continuum.
7. Missing annotation, declared nonresponse, Ambiguous and NotEstimable are distinct states; they are never coerced to an empty unitization or Absent without the coefficient-specific rule.
8. All computations use integer source-frame coordinates. Convert to seconds/milliseconds only for display.
9. A category's unitizing score must not mix different queries, instruments, or event definitions simply because their intervals overlap.

## 4. Reference fixture status

The Alex/Paul/Suzan reference input contains 16 units, three coders and a continuum of length 60. The exact input is stored in the fixture.

**Numeric coefficient outputs are intentionally not present yet.** The v1.1 manual documents the example, but this commit does not transcribe or infer the coefficient values from the screenshots. The reference manifest explicitly says `not-yet-transcribed`.

Before implementation, obtain the uAlpha v1.1 software distribution and run the fixture in its unitizing format. Capture:
- application/version and exact settings (metric, confidence level, target alpha);
- all five output rows, coverage percentages, confidence intervals and target-failure probabilities;
- observed/expected coincidence data where exposed;
- checksums of source CSV and captured output;
- at least one independent implementation replay (e.g. a second language/reference implementation) with agreed tolerances.

Do not treat visual OCR of a screenshot as the primary numeric oracle. Keep the input file and exact output values machine-readable, with manual transcription reviewed independently.

## 5. Implementation sequence

1. **Specification freeze:** map the five uAlpha v1.1 report views to the 2016 paper's coefficient definitions and missing-data semantics.
2. **Reference adapter:** document and replay the published Alex/Paul/Suzan vector, without production alpha calculations.
3. **Rust input validator:** enforce the frame/continuum/overlap rules and emit structured refusal reasons.
4. **Pure Rust coefficient port:** one function per equation/component, preserve intermediate observed/expected values, and compare against frozen reference outputs.
5. **Statistical uncertainty:** port the reference bootstrap only after its resampling unit and target-probability semantics are confirmed. For music-corpus studies, add source-work cluster sampling only if mathematically compatible with the coefficient; fixed annotator panel vs sampled annotator population must be reported separately.
6. **Corpus integration:** join source-work lineage, roster and nonresponse data from #13742; keep unitizing reliability alongside nominal alpha and pairwise temporal metrics, never as a composite.

## 6. Required controls

- Reproduce the Alex/Paul/Suzan published input exactly.
- Two-rater and three-rater panels.
- Identical unitizations; disjoint unitizations; partial overlap; boundary splits/merges; a missing coder on an item.
- One coder with overlapping intervals (must be rejected).
- Adjacent same-valued units (must stay distinct).
- Empty unitization and trailing gap with explicit continuum length.
- Category-only vs identification-only metric selection.
- Degenerate expected-disagreement denominator (typed undefined result; no NaN/Inf in serialized JSON).
- Different continuum extents with identical observed units (must not silently compare as if the extents were equal).
- Exact parity of every output measure, observed/expected data and confidence result to the frozen reference.

## 7. Reporting and claim ceiling

Report the coefficient name/version, metric, continuum length, coders/ratings, missingness, unit coverage, observed disorder, expected disorder, point estimate, confidence interval and probability of failing a preregistered target where applicable.

A unitizing coefficient measures agreement under a specified coding protocol. It does not prove that annotations are correct, that annotators are expert/representative, or that a model is accurate. Small synthetic examples are reference tests, not a reliability study. No model-confidence or causal-musician-benefit claim follows from alpha-U.

## 8. Implementation status

- [x] Issue opened: #13806.
- [x] Strict V1 input schema authored.
- [x] Published uAlpha v1.1 Alex/Paul/Suzan input transcribed as a source fixture.
- [ ] Coefficient family mapping independently checked against the 2016 paper/manual.
- [ ] Numeric reference output transcribed and independently reviewed.
- [ ] Rust implementation and frozen reference tests.
- [ ] Source-work/bootstrap design validated for this statistic.
