# SYM-RSI-SEM-002 — Ambiguity-Aware Semantic Admission

Status: **PREREGISTERED — NO MEASUREMENT CLAIM YET**

## Question

Does adding a preregistered top-1/top-2 HDC similarity-margin guard to empirically calibrated semantic retrieval reduce activation of ambiguous analogies without materially reducing correct held-out near-context retrieval?

SEM-002 tests **admission / abstention quality**, not nearest-neighbor ranking quality. The top-1/top-2 margin is a deterministic function of the same candidate similarities and does not reorder those candidates.

## Motivation

The current semantic retrieval stack deliberately separates several claims:

- raw HDC similarity — geometric resemblance;
- empirical-null specificity — how unusual that resemblance is under the frozen encoded-context null;
- encoder support — whether the numeric context is represented inside the encoder's nominal clamp region;
- top-1/top-2 margin — how strongly the best analogy is separated from its nearest rival;
- evidence authority — whether reality-grounded evidence may change belief confidence.

Absolute similarity and ambiguity are different geometric signals. A query may have a high best similarity while two candidate memories remain nearly tied. SEM-002 asks whether explicitly measuring that competition improves a semantic **abstention** decision.

Recent uncertainty-aware HDC work also treats prototype distance and the top-1/top-2 similarity gap as complementary nonconformity signals rather than interchangeable confidence values. SEM-002 adopts only the geometric distinction; it does not import any external confidence or conformal-coverage claim.

## Epistemic boundary

SEM-002 concerns generated/retrieval search heuristics only.

A SEM-002 result must not be interpreted as:

- empirical truth of a retrieved memory;
- validation of generated/counterfactual content;
- epistemic confidence;
- permission to promote confidence;
- evidence for consciousness;
- evidence for general recursive self-improvement;
- evidence about the protected SYM-RSI-001 / 001D measurement partitions.

No SEM-002 benchmark path may execute or consume the protected 201–204, 301–304, 401–404, or 1201–1204 partitions.

## Frozen implementation lineage

Parent ambiguity-diagnostic subject:

`0c42b0927fe959a575c05e840c8084f352fc2828`

This preregistration must be committed before the SEM-002 measurement harness is added. The later harness implementation receives its own exact subject identity and must pass executable qualification before any benchmark receipt is called qualified.

## Representation

Use the existing deterministic `SemanticContextEncoder`:

- binary HDC dimension: 16,384 bits;
- semantic levels: 64;
- absolute scalar clamp: 1.0;
- exact context identity: BLAKE3 over little-endian `f32` bytes;
- approximate identity: deterministic HDC encoding;
- raw similarity: matching-bit fraction in `[0,1]`.

The existing `SemanticNullCalibration` provides dimension-stratified empirical specificity.

The existing `SemanticRetrievalAmbiguity` defines:

- `best_similarity`;
- `runner_up_similarity`;
- `top_two_margin = best_similarity - runner_up_similarity` when at least two candidates exist.

The margin is an ambiguity diagnostic only. It is not probability, evidence, or confidence.

## Frozen dimensions

`D = {4, 8, 16}`

All calibration and evaluation is dimension-stratified where required. Cross-dimensional candidate comparison is forbidden.

## Deterministic data partitions

SEM-002 owns its own deterministic synthetic generator and shares no protected SYM-RSI seed or fixture partition.

For every context dimension use disjoint integer-index ranges:

- **null/reference memory bank**: indices `0..48`;
- **null margin queries**: indices `60..84`;
- **evaluation memory bank**: indices `100..124`;
- **clean related query sources**: derived from evaluation-memory anchors;
- **ambiguous query pairs**: deterministic pairs from evaluation-memory anchors;
- **unrelated evaluation queries**: indices `1000..1024`;
- **OOD anchors**: indices `2000..2016`.

No runtime RNG is permitted.

Reference, null-query, evaluation-memory, unrelated-query, and OOD anchor sets must be exact-identity disjoint.

## Calibration phase

Calibration must use **only** the null/reference partition and null margin queries.

### Similarity calibration

Build `SemanticNullCalibration` from the 48 reference contexts per dimension using:

- `max_reference_contexts_per_dimension = 48`;
- `min_null_pairs_per_dimension = 128`.

This produces the dimension-stratified empirical specificity used by SEM-001/SEM-002.

### Margin calibration

For each dimension:

1. encode the 48 reference-memory contexts;
2. for each null-margin query (`60..84`), compute similarity to every reference-memory context;
3. sort similarities descending;
4. compute `top1 - top2`;
5. collect the 24 exact null margins;
6. set the dimension-specific **margin admission threshold** to the empirical 95th percentile of those null margins using the deterministic nearest-rank rule:
   - sort ascending;
   - rank = `ceil(0.95 * n)` with one-based ranks;
   - select `rank - 1` in zero-based indexing.

No held-out/evaluation query may influence a margin threshold.

The margin calibration receipt must commit to the exact reference-memory and null-query identities plus the derived per-dimension thresholds.

## Evaluation memory bank

For each dimension, store exactly 24 evaluation contexts generated from indices `100..124`.

Each stored context has an exact `ContextIdentity`. No generated score, HDC similarity, specificity, or margin may alter which contexts enter the bank.

## Evaluation regimes

### A — Clean related

For each evaluation-memory anchor create two deterministic local perturbations with magnitudes:

- `0.04`;
- `0.08`.

Each clean-related query has a preregistered intended target: the exact anchor from which it was perturbed.

A **correct activation** requires both:

1. the admission rule activates;
2. top-1 semantic retrieval identity equals the intended target identity.

This prevents a gate from receiving credit for activating on the wrong near neighbor.

### B — Ambiguous-by-construction

Pair evaluation anchors deterministically as `(0,1), (2,3), ..., (22,23)`.

For each pair create two ambiguity queries:

1. exact numeric midpoint `0.5*A + 0.5*B`;
2. slightly biased midpoint `0.52*A + 0.48*B`.

All values are clipped to nominal `[-1,1]` only if arithmetic roundoff would exceed the support bound.

These are labelled **ambiguous by construction** because two distinct source memories jointly define each query. The label is fixed before HDC encoding and is not based on observed similarity or margin.

No single candidate is considered a uniquely correct answer for this regime. An activation is treated as an **ambiguous activation**.

### C — Unrelated

For each evaluation memory position `j`, generate one query from disjoint index `1000 + j`.

These are labelled unrelated by generator construction. The label must not depend on HDC output.

An activation is a **false semantic activation**.

### D — OOD clamp saturation

For each OOD anchor, deterministically push one coordinate outside `[-1,1]` while preserving a different exact BLAKE3 identity.

OOD cases are diagnostic only. If the strict operational support layer rejects them before semantic admission, record that rejection explicitly. If a raw diagnostic path is also evaluated, high raw similarity under clamp saturation must be reported as an encoder-support limitation, not successful generalization.

## Four frozen admission rules

All rules operate on the same candidate similarities and same evaluation memory bank.

### R — Raw similarity only

Activate when:

`best_similarity >= global_raw_similarity_threshold`

The global raw threshold is the arithmetic mean of the three dimension-specific null `p95_similarity` summaries, matching SEM-001's dimension-agnostic raw baseline.

### S — Empirical specificity only

Assess the top-1 raw similarity against the matching dimensional `SemanticNullCalibration` stratum.

Activate when:

`retrieval_specificity >= 0.95`

### M — Margin only

Activate when:

`top_two_margin >= dimension_specific_margin_p95`

A query with fewer than two candidates cannot satisfy M.

### SM — Specificity plus margin

Activate only when both:

`retrieval_specificity >= 0.95`

and

`top_two_margin >= dimension_specific_margin_p95`

SM is the proposed ambiguity-aware semantic admission rule.

## Primary endpoint

### Ambiguous activation reduction relative to specificity-only

Compute the total activation rate on ambiguous-by-construction cases under S and SM.

Primary success requires an absolute reduction of at least 0.10:

`ambiguous_activation_S - ambiguous_activation_SM >= 0.10`

This criterion is frozen before measurement. A smaller reduction is reported as not established even if numerically favorable.

## Utility / safety guards

### 1. Correct clean-related retention

Compute, for S and SM:

`correct_activation_rate = (# clean-related queries activated with intended target top-1) / (# clean-related queries)`

Require:

`correct_activation_SM >= correct_activation_S - 0.10`

A loss greater than ten percentage points fails the utility guard.

### 2. Worst-dimension clean-related guard

For dimensions 4, 8 and 16 separately compute correct activation under S and SM.

Require in every dimension:

`correct_activation_SM_D >= correct_activation_S_D - 0.15`

This prevents aggregate retention from hiding a severe regression in one context width.

### 3. Unrelated false-activation guard

Require:

`unrelated_activation_SM <= unrelated_activation_S`

SM must not reduce ambiguity by increasing unrelated activation.

### 4. No ranking claim

Top-1 retrieval accuracy before admission is identical across R, S, M and SM because all use the same candidate similarities. SEM-002 must report this once and must not claim ranking improvement from margin gating.

## Disposition

### PASS

Requires:

- primary ambiguous activation reduction >= 0.10;
- aggregate clean-related retention guard passes;
- all per-dimension clean-related guards pass;
- unrelated false-activation guard passes.

Establishes only:

> On the frozen SEM-002 synthetic benchmark, adding the preregistered top-two margin guard to empirical specificity improved semantic abstention on ambiguous queries without the preregistered clean-related or unrelated-query regressions.

### TRADEOFF

Primary ambiguous reduction passes but any utility/safety guard fails.

Do not promote SM as the default live path.

### NOT ESTABLISHED

Primary ambiguous reduction fails.

Keep margin diagnostic-only.

## Secondary descriptive metrics

Report without promoting them into success criteria:

- per-dimension raw null p95 similarity;
- per-dimension margin-null median and p95;
- candidate-bank size;
- clean-related top-1 accuracy before admission;
- clean-related activation/correct-activation rates for R, S, M and SM;
- ambiguous activation rates for R, S, M and SM;
- unrelated activation rates for R, S, M and SM;
- mean/median top-1 similarity by regime;
- mean/median top-two margin by regime;
- mean retrieval specificity by regime;
- number of exact ties;
- number of single-candidate cases, expected to be zero under the frozen bank size;
- OOD support rejection counts;
- raw OOD similarity/margin distributions when measured diagnostically.

No post-hoc secondary metric becomes a success criterion.

## Anti-leakage requirements

Before measurement the harness must assert:

- all deterministic source partitions are exact-identity disjoint;
- clean-related queries differ in exact identity from their source anchors;
- ambiguous queries differ in exact identity from both source anchors;
- unrelated queries are not evaluation-memory identities;
- calibration query identities never appear in the evaluation memory/query sets;
- no protected SYM-RSI fixture seed or partition is referenced.

Any anti-leakage failure invalidates the benchmark before endpoint computation.

## Receipt binding

A SEM-002 receipt must bind at least:

- exact Git subject SHA;
- benchmark schema/version;
- deterministic generator version;
- encoder configuration;
- reference-memory corpus digest;
- margin-null-query corpus digest;
- evaluation-memory corpus digest;
- clean-related query digest;
- ambiguous query digest;
- unrelated query digest;
- OOD query digest;
- `SemanticNullCalibration` identity;
- per-dimension margin thresholds and their calibration digest;
- all four frozen admission thresholds/rules;
- all primary and guard metrics;
- final disposition.

The receipt digest is integrity/provenance only. It does not authenticate the result unless frozen in an independent evidence lineage.

## Promotion rule

Even a SEM-002 PASS may only justify adding the margin as an **abstention / attenuation guard** for generated semantic search.

It may not:

- increase dream-derived search strength above its pre-margin value;
- promote confidence;
- turn generated content into empirical evidence;
- bypass query/candidate support checks;
- bypass semantic-null calibration;
- authorize protected measurement execution.

Any use of margin to increase action-prior strength requires a separate preregistered experiment.

## Reproducibility and qualification

The preregistration commit is frozen before the harness implementation commit.

No SEM-002 result is called qualified until the exact harness subject passes focused executable qualification. If implementation repair changes the exact subject, a new evidence lineage is required; results from distinct subjects must not be mixed.

## External methodological context

SEM-002's geometric motivation is consistent with uncertainty-aware HDC literature that separates distance-to-prototype and top-1/top-2 similarity gap and calibrates uncertainty on a held-out split. SEM-002 does not inherit any external dataset result, coverage theorem, or performance claim; its own preregistered synthetic benchmark must establish its own narrow result.
