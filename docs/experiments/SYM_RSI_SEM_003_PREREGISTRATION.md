# SYM-RSI-SEM-003 — Set-Valued Semantic Retrieval

Status: **PREREGISTERED — NO MEASUREMENT CLAIM YET**

## Question

Can Symthaea replace forced single-analogy retrieval with a calibrated **candidate set** that preserves the correct source memory on clean near-context queries, represents genuine ambiguity with multiple candidates, rejects unrelated queries with an empty set, and refuses clamp-saturated/OOD queries before set construction?

SEM-003 tests retrieval-set behavior only. It does not test factual truth, belief confidence, consciousness, or general recursive self-improvement.

## Motivation

SEM-001 tests empirical calibration of absolute semantic similarity.
SEM-002 tests ambiguity-aware abstention using top-1/top-2 margin.

SEM-003 asks a different question:

> When semantic memory is ambiguous, should Symthaea return a bounded set of plausible analogies instead of forcing a winner or collapsing immediately to a scalar confidence?

This follows the uncertainty-aware design principle that distance/specificity and ambiguity are complementary signals, while preserving Symthaea's stronger epistemic boundary: a retrieval candidate is not evidence that its contents are true.

## Epistemic boundary

A SEM-003 candidate set is a **search object only**.

A returned set must not be interpreted as:

- a probability distribution;
- belief confidence;
- empirical validation;
- permission to promote confidence;
- evidence of factual correctness;
- evidence for consciousness;
- evidence that generated/counterfactual content is true.

Any downstream empirical claim must resolve exact provenance independently through the evidence system.

No SEM-003 path may execute or consume protected SYM-RSI partitions 201–204, 301–304, 401–404, or 1201–1204.

## Frozen predecessor subjects

SEM-003 begins only after these semantic-memory subjects:

- SEM-002 complete receipt-integrity binding: `69f3a9e77323ee8cd81017cab2c154c15e8ae29b`
- SEM-002 preregistration: `97e47ae4b95844ceeae857e9500de15871859422`

The SEM-003 preregistration commit itself must be frozen before any SEM-003 harness implementation exists.

## Representation

Use the existing deterministic `SemanticContextEncoder`:

- binary HDC dimension: 16,384 bits;
- semantic levels: 64;
- nominal scalar support: `[-1, 1]`;
- exact identity: BLAKE3 over canonical little-endian `f32` bytes;
- approximate similarity: HDC matching-bit fraction in `[0,1]`.

All candidate memory items are exact-distinct `ContextIdentity` values.

## Set-valued retrieval rule

For a query `q` and candidate memory item `c`, define:

`nonconformity(q,c) = 1 - similarity(q,c)`

Smaller values mean stronger semantic agreement.

For each numeric context dimension `D`, learn a threshold `Q_D` from a **disjoint calibration split** containing labelled `(query, true_source)` pairs.

The calibration score for each pair is:

`a_i = 1 - similarity(query_i, true_source_i)`

Use the conservative split-calibration quantile index:

`k = ceil((n + 1) * (1 - alpha))`

with `alpha = 0.10`, clamped to the available sorted calibration scores.

Then:

`candidate_set(q) = { c : dimension(c)=dimension(q) and 1-similarity(q,c) <= Q_D }`

Exact identity is used for membership/provenance; HDC similarity is used only for admission to the search set.

## No formal conformal-coverage claim

The SEM-003 benchmark is deterministic synthetic data. Therefore SEM-003 may describe its threshold construction as **split-conformal-style calibration**, but it must not claim a distribution-free finite-sample coverage guarantee.

The preregistered claim is limited to **empirical held-out target coverage on the frozen synthetic benchmark**.

A formal conformal guarantee would require a separately justified exchangeability model and a separate protocol.

## Support boundary

Before candidate-set construction, the query must pass nominal encoder-support validation.

If any query coordinate is outside the configured clamp support, the operational result is:

`UnsupportedQuery`

and **no candidate set is returned**.

SEM-003 may still compute raw HDC diagnostics for OOD stress analysis, but those values are not operational set outputs.

Candidate-source contexts used in operational sets must also have support metadata captured at indexing time and be nominally supported.

## Dataset generation

SEM-003 owns a deterministic synthetic generator separate from protected SYM-RSI fixtures.

Frozen dimensions:

`D = {4, 8, 16}`

For each dimension create exact-disjoint partitions:

- candidate memory bank: 64 contexts;
- calibration queries: 32 labelled near-context queries;
- clean held-out queries: 32 labelled near-context queries;
- ambiguous held-out queries: 24 midpoint/mixed queries with two preregistered plausible parents;
- unrelated held-out queries: 24 queries generated from a disjoint index family;
- OOD held-out queries: 16 clamp-saturated queries.

No runtime RNG is permitted.

All partitions must be exact-disjoint by BLAKE3 `ContextIdentity` before measurement begins.

## Clean calibration queries

Each calibration query is a deterministic small perturbation of one known candidate-memory source.

The true source label is determined by construction, never by HDC similarity.

Calibration queries and their true sources are used only to estimate `Q_D`. They are not included in held-out metrics.

## Clean held-out queries

Each clean test query is a deterministic local perturbation of one candidate-memory item, with its true source fixed by construction.

Primary clean metrics:

- target coverage: fraction where the true source is present in the candidate set;
- singleton-correct rate: fraction where the set contains exactly the true source;
- mean candidate-set size;
- p95 candidate-set size.

## Ambiguous held-out queries

Each ambiguous query is constructed from two preregistered parent candidate contexts from the same dimension.

The benchmark must record both parents before semantic encoding.

Primary ambiguity metrics:

- dual-parent inclusion: fraction where both intended parents are in the candidate set;
- at-least-one-parent inclusion;
- singleton forced-choice rate;
- mean candidate-set size.

A multi-candidate result is not counted as failure merely because it does not force a winner.

## Unrelated held-out queries

Unrelated queries have no designated candidate-memory source.

Primary unrelated metric:

- non-empty set rate.

Lower is better.

## OOD stress

Every OOD query must contain at least one coordinate outside nominal support while retaining a distinct exact identity.

Primary operational requirement:

`OOD support rejection rate = 1.0`

Raw similarity/set-size diagnostics may be reported descriptively only after marking the operational query unsupported.

## Primary success criteria

SEM-003 establishes useful set-valued retrieval only if **all** of the following hold:

1. aggregate clean target coverage `>= 0.90`;
2. every dimension clean target coverage `>= 0.85`;
3. aggregate clean mean set size `<= 2.0`;
4. ambiguous at-least-one-parent inclusion `>= 0.90`;
5. ambiguous dual-parent inclusion `>= 0.70`;
6. ambiguous singleton forced-choice rate `<= 0.20`;
7. unrelated non-empty set rate `<= 0.10`;
8. OOD support rejection rate `= 1.0`.

These thresholds are frozen before the SEM-003 implementation exists.

## Interpretation

### All primary criteria pass

Establishes only:

> On the frozen SEM-003 synthetic benchmark, a dimension-calibrated set-valued semantic retrieval rule preserved the designated source on clean near-context queries, represented constructed ambiguity with bounded multi-candidate sets, rarely activated on unrelated queries, and rejected clamp-saturated queries operationally.

It does not establish factual correctness or belief confidence.

### Coverage passes but set-size/unrelated guards fail

Disposition: **Tradeoff / not established**.

Do not promote set-valued retrieval to the live operational default.

### Coverage fails

Disposition: **Not established**.

Keep the mechanism experimental.

### OOD rejection fails

Disposition: **Fail closed** regardless of all other metrics.

## Secondary descriptive metrics

Report without promoting them to success criteria:

- per-dimension calibration threshold `Q_D`;
- calibration score count;
- clean set-size histogram;
- ambiguous set-size histogram;
- unrelated set-size histogram;
- top-1/top-2 similarity margin for each regime;
- empirical-null retrieval specificity for each regime;
- exact tie counts;
- candidate-bank/context/query corpus digests;
- raw OOD similarities after support rejection.

## Promotion rule

Even after a SEM-003 PASS, candidate sets remain search-only.

A live integration may:

- present multiple analogies;
- branch search across candidates;
- reduce action-prior strength when sets are large;
- abstain from semantic guidance when the set is empty.

It may not:

- convert set membership into confidence authority;
- treat set size as a probability;
- promote counterfactual/generated content to empirical evidence;
- bypass exact provenance resolution.

## Reproducibility receipt

A future SEM-003 receipt must bind at least:

- SEM-003 preregistration SHA;
- exact implementation subject SHA;
- benchmark schema/generator version;
- encoder configuration;
- candidate-bank digest;
- calibration-query digest;
- clean-query digest;
- ambiguous-query digest;
- unrelated-query digest;
- OOD-query digest;
- per-dimension calibration thresholds;
- all primary endpoint values;
- disposition;
- complete receipt-integrity digest.

No SEM-003 result is qualified until its exact implementation subject passes focused executable qualification.
