# SYM-RSI-SEM-001 — Empirical Semantic-Retrieval Calibration

Status: **PREREGISTERED — NO MEASUREMENT CLAIM YET**

## Question

Does a dimension-stratified empirical null make Symthaea's HDC semantic-retrieval **activation threshold more portable across context dimensions** without materially discarding genuinely nearby contexts?

This experiment does **not** test whether empirical calibration improves nearest-neighbor ranking within a fixed dimension. `SemanticNullAssessment::retrieval_specificity` is monotone in raw Hamming similarity inside each dimensional stratum, so claiming an intrinsic within-stratum ranking gain would be mathematically misleading.

## Epistemic boundary

SEM-001 concerns retrieval/search heuristics only.

A SEM-001 result must not be interpreted as:

- empirical truth of a retrieved memory;
- validation of generated/counterfactual content;
- epistemic confidence;
- permission to promote confidence;
- evidence for consciousness;
- evidence for general recursive self-improvement;
- evidence about the protected SYM-RSI-001 / 001D measurement partitions.

No SEM-001 benchmark path may execute or consume the protected 201–204, 301–304, 401–404, or 1201–1204 partitions.

## Frozen implementation subjects

Parent semantic-calibration implementation:

`273b71d710b9f9b77f006c2634dd8246fc4927d9`

SEM-001 must be evaluated only after the benchmark/preregistration implementation itself receives executable qualification. A later code repair requires a new exact subject identity; results from distinct subjects must not be mixed into one evidence lineage.

## Representation

The benchmark uses the existing deterministic `SemanticContextEncoder`:

- `BinaryHV` dimension: 16,384 bits;
- default semantic levels: 64;
- default absolute input clamp: 1.0;
- exact context identity: BLAKE3 over little-endian `f32` bytes;
- approximate retrieval identity: deterministic HDC encoding;
- raw similarity: matching-bit fraction in `[0,1]`.

The empirical null is learned independently per numeric-context dimension from exact-distinct reference contexts.

## Dataset generation

SEM-001 owns its own deterministic synthetic context generator. It shares no protected SYM-RSI seed or fixture partition.

Frozen context dimensions:

`D = {4, 8, 16}`

For each dimension the generator creates three disjoint sets by integer index range:

- **reference/null**: indices `0..48`;
- **held-out evaluation anchors**: indices `100..124`;
- **OOD anchors**: indices `200..216`.

Base scalar values are deterministic functions of `(dimension, example_index, coordinate_index)` and are constrained to the encoder's nominal `[-1,1]` range for reference and held-out anchors.

No runtime RNG is permitted.

### Held-out related queries

For every held-out anchor, create deterministic small perturbations of magnitude `0.04` and `0.08`, alternating coordinate/sign by index.

These cases are labelled **related by construction** because they are local perturbations of the same source anchor. The label does not come from HDC similarity.

### Held-out unrelated queries

Pair every held-out anchor with a separately generated anchor from a disjoint index offset. These are labelled **unrelated by construction**. Pairing must not use HDC similarity or calibration output.

### OOD clamp-saturation stress

For each OOD anchor create a query by scaling and shifting selected coordinates so at least one scalar lies outside `[-1,1]` while the exact BLAKE3 identity remains different.

This regime is diagnostic. Because the current encoder clamps inputs, OOD values can collapse onto the same semantic level. SEM-001 must report this behavior explicitly rather than interpret high HDC similarity as trustworthy support.

## Calibration corpus

Calibration is built **only** from the reference/null contexts.

Frozen calibration configuration:

- `max_reference_contexts_per_dimension = 48`;
- `min_null_pairs_per_dimension = 128`.

With 48 exact-distinct references, each dimensional stratum can contain up to `48*47/2 = 1128` null pairs.

The calibration identity must commit to encoder configuration, exact reference-context digests, and resulting null distributions.

Held-out and OOD contexts must not contribute to the calibration identity or null distribution.

## Two compared activation rules

### A — Global raw-similarity threshold

Construct one dimension-agnostic raw threshold from the calibration corpus as the arithmetic mean of the three dimensional null `p95_similarity` summaries.

A held-out pair activates when:

`raw_similarity >= global_raw_p95_threshold`

This deliberately models the operational problem of using one raw HDC threshold across heterogeneous context widths.

### B — Dimension-calibrated specificity threshold

For each pair, assess raw similarity against the calibration stratum matching that pair's numeric context dimension.

A held-out pair activates when:

`retrieval_specificity >= 0.95`

where specificity is `1 - (exceedances+1)/(n+1)` under the frozen empirical null.

This is an empirical rank threshold, **not a p-value**.

## Primary endpoint

### Unrelated false-activation dispersion across dimensions

For each method compute the false activation rate among held-out unrelated cases separately for dimensions 4, 8 and 16.

Define dispersion as:

`max(false_activation_rate_D) - min(false_activation_rate_D)`

Primary success requires:

`calibrated_dispersion < raw_dispersion`

If raw dispersion is already exactly zero, SEM-001 cannot claim improvement on this endpoint; it may report parity only.

## Safety/utility guards

Calibration must not buy threshold stability by discarding most useful near-context retrievals.

### Related retention guard

Compute activation rate for held-out related cases under both methods.

Require:

`calibrated_related_retention >= raw_related_retention - 0.10`

A loss greater than 10 percentage points fails the utility guard.

### Worst-dimension false activation guard

Require:

`max_calibrated_unrelated_false_activation <= max_raw_unrelated_false_activation`

This prevents a lower dispersion achieved merely by making a previously good dimension worse.

## Secondary descriptive metrics

Report without upgrading them into primary claims:

- per-dimension raw null median and p95;
- per-dimension held-out related/unrelated mean similarity;
- per-dimension mean retrieval specificity;
- raw and calibrated activation rates;
- related/unrelated activation margins;
- reference count and null-pair count per stratum;
- exact calibration identity;
- OOD saturation similarity and calibrated specificity distributions;
- number/fraction of OOD cases whose exact identity differs while HDC similarity remains above the global raw threshold.

No post-hoc metric becomes a success criterion.

## Interpretation matrix

### Primary PASS + guards PASS

Establishes only:

> For the frozen SEM-001 synthetic benchmark and encoder, dimension-stratified empirical calibration made a single semantic-activation criterion more portable across context widths without the preregistered retention or worst-dimension regression.

It does **not** establish improved semantic ranking, factual correctness, or confidence calibration.

### Primary PASS + any guard FAIL

Result is **tradeoff / not established**. Do not promote calibrated retrieval as the default live path.

### Primary FAIL

Keep empirical calibration diagnostic-only. The raw HDC retrieval path remains the behavioral baseline.

### OOD saturation warning

Regardless of primary result, strong activation under clamp-saturated OOD cases must be reported as an encoder-support limitation. It cannot be relabelled as successful semantic generalization.

## Promotion rule

Even after a SEM-001 PASS, calibration may initially only **attenuate** generated semantic search strength:

`calibrated_search_strength <= pre_calibration_search_strength`

Allowing empirical calibration to increase search strength requires a separate preregistered experiment demonstrating held-out benefit without worsened false activation/OOD behavior.

## Reproducibility

A SEM-001 receipt must bind at least:

- exact Git subject SHA;
- benchmark schema/version;
- semantic encoder configuration;
- deterministic generator version;
- calibration identity;
- reference/held-out/OOD context digests;
- raw global threshold;
- all preregistered endpoint values;
- claim disposition.

No result should be called qualified until the exact implementation subject has passed the focused executable qualification gate.