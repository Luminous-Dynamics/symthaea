# Food quantity conversion qualification benchmark v1

**Status:** synthetic contract implementation and proposed qualification protocol. No empirical yield, density, edible-fraction, or local-cuisine performance is claimed. Rust tests are authored but must be run in a Rust-enabled environment.

## Paradigm shift

Food matching says which identity a record refers to. Quantity conversion says whether two values are comparable and, if so, how one is transformed into another. Keep the operations separate.

Every conversion must make its transformation chain explicit: input quantity and denominator, raw/processed state, factor uncertainty, output quantity, source locator, source version, and the time the evidence was known. Arithmetic validity is not empirical validity. A successful validator result establishes internal consistency, not source authenticity or scientific representativeness.

The v1 scope is deliberately limited to:
- deterministic mass/volume unit scaling using a versioned unit table;
- as-purchased mass to edible-portion mass via an explicit factor;
- mass yield across a declared food-state transition;
- volume to mass via source-pinned density;
- conservative interval propagation for positive factors.

Household measures such as “one cup of beans” need a food-, fill-method-, and locale-specific mass reference; they are not silently treated as universal unit conversions. Nutrient retention, recipe composition, moisture transfer, discarded cooking water, oil uptake, and nutrient-per-100-g recomputation require separate typed transformations and are intentionally not implemented in v1.

## Research basis

- FAO/INFOODS publishes separate [Guidelines for Food Matching](https://www.fao.org/infoods/infoods/standards-guidelines/en/) and [Guidelines for Converting Units, Denominators and Expressions](https://www.fao.org/infoods/infoods/standards-guidelines/en/). The separation is architectural: record identity does not itself prove common denominators or process states.
- FAO's [food composition material on composite prepared dishes](https://www.fao.org/4/y4705e/y4705e23.htm) describes edible-portion correction, cooking yield, nutrient-retention factors and final per-weight/volume/serving values as distinct stages.
- FAO notes that calculated prepared-food compositions are rough estimates when preparation conditions vary, and that empirical/analytical and calculated values should not be conflated. [Food Composition Data](https://www.fao.org/4/y4705e/y4705e06.htm).
- USDA Foundation Foods reports values on an edible-material basis and provides food-data-type-specific portion weights; the same assumed portion weight should not be blindly reused across records. [Foundation Foods documentation](https://fdc.nal.usda.gov/Foundation_Foods_Documentation/).
- FAO/INFOODS standards should guide the later transformation vocabulary, but the v1 contract is not a claim of complete compliance with those guidelines.

## Required benchmark tracks

### 1. Synthetic conformance (first gate)

- 100% agreement between Rust implementation and independently enumerated expected results.
- Zero acceptance of broken step continuity, unsupported units, cross-dimensional unit scaling, missing source versions/timestamps, malformed or inverted intervals, and output values that disagree with recomputation.
- Zero qualified outcomes when the input or any transformation factor is synthetic, unreviewed, or merely reviewed.
- Exact listed metric conversions within 1e-9 relative-or-absolute tolerance.
- Deterministic results regardless of JSON key order; preserve the distinction between factor point estimates and lower/upper bounds.

### 2. Empirical food-process qualification

Build a rights-reviewed corpus spanning regions, staple foods, food parts, processing states, cooking methods, and equipment. Record raw mass, post-trim edible mass, post-process mass, water/fat additions and removals, replicate identity, instrument resolution, operator, temperature/time, and protocol version. Do not collapse these measurements into a single universal yield factor.

Use at least three independent preparation replicates per food × method × protocol cell as a starting design, increasing replicates from observed variance and power analysis. Split train/calibration/test by food concept, source release, and protocol—not by repeated rows from the same batch. Have an independent reviewer adjudicate outliers and protocol departures.

Report:
- mean absolute relative error and median absolute relative error for the point yield estimate;
- empirical coverage of nominal 90% and 95% prediction intervals;
- interval width as well as coverage, to prevent vacuous intervals from winning;
- bias and calibration by food class, region, household/industrial method, and state transition;
- fraction requiring review or lacking a defensible factor;
- uncertainty sensitivity to input mass, instrument precision, and factor uncertainty;
- replicability across operators and batches.

Provisional engineering targets, to be assessed on a locked held-out set rather than asserted as current performance:
- median absolute relative error ≤5% for narrowly specified, repeatable protocols;
- 95th-percentile absolute relative error ≤15% for that same protocol stratum;
- nominal 95% prediction-interval coverage between 92% and 98%, reported with confidence intervals and interval width;
- zero silently applied factors when food/process/denominator scope does not match the evidence;
- 100% source/version/time traceability for factors included in a qualified output.

Those numeric targets are hypotheses for test design, not promises for all food/processes. A failed target should lead to narrower qualified scope, wider calibrated bounds, or abstention—not to weakened acceptance rules.

### 3. Global and local robustness

Publish every metric by locale/language, staple-food class, food state, preparation method, equipment, as-purchased/edible/drained/dry basis, source type, and household vs industrial process. Report stratum sample counts and uncertainty intervals. Do not let strong performance on one well-measured food market hide low data coverage elsewhere.

Keep local names and measures separate from physical conversion evidence. Country code is not evidence of a person's culture, food preferences, or the density of a local household measure.

### 4. Adversarial / metamorphic tests

- Tamper with an output while retaining the source chain; validation must fail.
- Reorder JSON object keys; result must be unchanged.
- Change a unit factor but not its table-derived ratio; fail.
- Disconnect any step's unit, basis or process state from its predecessor; fail.
- Mark synthetic or unreviewed empirical factors as qualified; fail.
- Convert a volume to mass without a density step; fail.
- Attempt to use household units as globally fixed units; abstain/reject.
- Widen/narrow factor bounds and verify monotonic output-envelope changes.
- Use empty, future-dated, stale, missing, or rights-ineligible source evidence in the later source-registry integration; refuse qualification.
- Test numerical extremes, zero quantity, very small factors, overflow, and near-boundary intervals.

## Interoperability and ownership

- Food matching decides identity context; a match does not validate process factors.
- Food quantity conversion checks typed transformations and math.
- NUTRI-SOURCE-001 (#7481) owns canonical source rights, versions and freshness; this module must not fork a source registry.
- Regional capability packs (#8043) declare which local source/method combinations are qualified.
- A future food-composition calculator combines ingredient quantities, retained nutrients, added/lost ingredients, and final yield only once those transformations are modeled explicitly.

## Qualification limits

The current fixture has no real nutrition values and no real yield-factor evidence. The Rust API does not resolve external references, verify that a source was licensed, prove independent-factor assumptions, calibrate prediction intervals, or establish food safety/clinical suitability. Rust compilation and test execution have not been verified in the authoring environment; do not describe authored tests or static inspection as test passes.
