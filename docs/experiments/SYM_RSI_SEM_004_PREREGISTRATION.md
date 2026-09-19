# SYM-RSI-SEM-004 — Gated Relative Set Retrieval

Status: **PREREGISTERED — NO SEM-004 IMPLEMENTATION OR MEASUREMENT CLAIM YET**

## Question

Can Symthaea separate **query admission** from **within-memory ambiguity** so that semantic memory:

- preserves a designated near-context source on clean held-out queries;
- returns both plausible parents for constructed ambiguous queries;
- rarely activates on unrelated queries;
- keeps candidate sets bounded;
- and rejects clamp-saturated/OOD queries before semantic guidance?

SEM-004 tests a new retrieval rule. It does **not** rewrite, rescue, or reinterpret SEM-003.

## Motivation from the frozen predecessor

SEM-003 calibrated one absolute nonconformity radius from labelled clean calibration pairs and then used that same radius both to decide whether a query was semantically related at all and which candidate memories belonged in its set.

Independent cross-implementation behavior canaries frozen after the SEM-003 protocol predict that this rule is too narrow for its held-out perturbation regime and does not represent the constructed midpoint ambiguity:

- SEM-003 behavior-canary subject: `1c7f5d475546da7b5d5de553089e8a1437d3773a`;
- replacement SEM-003 implementation candidate: `dd533164f8aff636f06e49a161fa75b4e70359d1`.

Those canaries predict `Sem3Disposition::NotEstablished`. SEM-004 treats that negative result as motivation for a distinct hypothesis; SEM-003's code, thresholds, criteria, corpus, and disposition remain frozen.

## Epistemic boundary

SEM-004 candidate sets are **search objects only**.

No SEM-004 quantity may be interpreted as:

- probability;
- belief confidence;
- empirical validation;
- factual correctness;
- confidence-promotion authority;
- evidence for consciousness;
- evidence for general recursive self-improvement.

Exact provenance remains separate from semantic retrieval. Protected SYM-RSI partitions `201–204`, `301–304`, `401–404`, and `1201–1204` are out of scope and must not be read or executed by SEM-004.

## Frozen predecessor identities

SEM-004 is downstream of:

- SEM-003 preregistration: `af8162162e13a2b3c95fc70a0af9b62a3183ac4f`;
- SEM-003 descriptive amendment: `418f057387eacf2747edba6e8efaffca05d4b04c`;
- SEM-003 failed implementation candidate retained in history: `df3eedea80cb21ffc7f4896b64ff6480ea8d2680`;
- SEM-003 replacement implementation candidate: `dd533164f8aff636f06e49a161fa75b4e70359d1`;
- independent SEM-003 corpus canaries: `22e246a0a9bcc0260c3b77bc384b9bc5cf17b0c9`;
- independent SEM-003 behavior canaries: `1c7f5d475546da7b5d5de553089e8a1437d3773a`.

SEM-004 must use a new generator/domain and must not reuse SEM-003 candidate, calibration, held-out, unrelated, or OOD exact identities.

## Representation

Use the existing deterministic semantic representation unchanged:

- `SemanticContextEncoder::default()`;
- 16,384-bit `BinaryHV`;
- 64 scalar levels;
- nominal scalar support `[-1, 1]`;
- exact BLAKE3 context identity;
- Hamming similarity in `[0,1]`.

Changing the encoder is not part of SEM-004.

## Core hypothesis

SEM-004 separates two decisions that SEM-003 coupled.

### Decision A — absolute query admission

Ask:

> Is the query sufficiently close to *some* candidate memory to permit semantic guidance at all?

For each context dimension `D`, learn an absolute top-1 similarity gate `G_D` using a disjoint **unrelated calibration-control** split.

Let `u_i` be the top-1 similarity of unrelated calibration query `i` against the candidate bank of the same dimension.

Freeze:

`G_D = q95({u_i})`

using deterministic nearest-rank indexing:

`rank = ceil(N * 0.95)`

clamped to `[1,N]` and returning `sorted[rank - 1]`.

Operational query admission requires:

`top1_similarity(q) > G_D`

The comparison is deliberately strict (`>`, not `>=`).

This gate is a synthetic false-activation control only. It is not a probability or evidence threshold.

### Decision B — relative candidate inclusion

Conditional on query admission, ask:

> Which candidates are close enough to the best available analogy that forcing one winner would be unjustified?

For each labelled clean calibration query with known true source, define:

`gap_i = top1_similarity(query_i) - similarity(query_i, true_source_i)`

where `gap_i >= 0`.

For each dimension `D`, learn:

`R_D = q95({gap_i})`

using the same deterministic nearest-rank rule.

For an admitted query with top-1 similarity `s*`, include candidate `c` when:

`s* - similarity(q,c) <= R_D`

and candidate `c` is same-dimensional and was indexed inside nominal encoder support.

This is a **relative ambiguity band**, not a confidence interval and not a conformal-coverage claim.

## Bounded-set rule

Freeze maximum operational set size:

`MAX_SET_SIZE = 4`

After applying the relative band:

- set size `0`: integrity error, because the top-1 candidate must lie at gap `0` after admission;
- set size `1..=4`: return the candidate set;
- set size `>4`: return `AmbiguityOverload` and no operational candidate set.

Do **not** truncate to four candidates. Truncation would silently manufacture a forced ordering inside an acknowledged ambiguity region.

`AmbiguityOverload` is an abstention/search-overload outcome, not a negative factual judgment.

## Support boundary

Before any semantic similarity can become operational:

- query must be inside nominal encoder support;
- candidate source support must have been captured at indexing time;
- candidate source must be inside nominal encoder support;
- candidate and query must have equal numeric context dimension.

A clamp-saturated query returns `UnsupportedQuery` before admission testing.

Raw diagnostics may be computed afterward for analysis, but they are not operational retrieval output.

## SEM-004 corpus

SEM-004 owns a new deterministic synthetic corpus with a new generator/domain string. It must be exact-disjoint from all SEM-003 corpus canaries.

Frozen dimensions:

`D = {4, 8, 16}`

Per dimension generate:

- candidate memory bank: `96` contexts;
- labelled clean calibration queries: `32`;
- unrelated calibration-control queries: `32`;
- clean held-out queries: `32`;
- ambiguous held-out queries: `24`;
- unrelated held-out queries: `24`;
- OOD held-out queries: `16`.

No runtime RNG is permitted.

The candidate bank must contain deterministic sibling families suitable for constructed ambiguity, but the SEM-004 generator must use a fresh domain/salt family so no candidate or query exact identity equals a SEM-003 identity.

Before measurement, exact BLAKE3 identity disjointness must be checked across:

- all SEM-004 partitions;
- the frozen SEM-003 candidate/query corpus canaries.

Any collision/overlap is an integrity failure, not a benchmark outcome.

## Calibration isolation

Only these partitions may calibrate retrieval parameters:

- labelled clean calibration queries → `R_D`;
- unrelated calibration-control queries → `G_D`.

Held-out clean, ambiguous, unrelated, and OOD queries must not affect either parameter.

The two calibration partitions must be exact-disjoint from one another and from all held-out partitions.

## Clean held-out regime

Each clean held-out query is a deterministic perturbation of a preregistered candidate source that is not used as the source of a clean calibration query.

Report:

- admission rate;
- designated-source coverage;
- singleton-correct rate;
- ambiguity-overload rate;
- mean returned set size among non-overload admitted queries;
- p95 returned set size.

A clean query rejected by the absolute gate counts as missing the designated source.

## Ambiguous held-out regime

Each ambiguous query is deterministically constructed from two preregistered sibling candidates from the same dimension. Both parent identities are frozen before semantic encoding/evaluation.

Report:

- admission rate;
- at-least-one-parent inclusion;
- dual-parent inclusion;
- singleton forced-choice rate;
- ambiguity-overload rate;
- mean returned set size among non-overload admitted queries.

A multi-candidate set is not a failure merely because it does not choose one winner.

## Unrelated held-out regime

Unrelated held-out queries have no designated candidate source and come from a generator family disjoint from both candidate memory and unrelated calibration controls.

Report:

- admission / non-empty operational-guidance rate;
- ambiguity-overload rate;
- top-1 similarity distribution relative to `G_D`.

Lower admission is better.

## OOD regime

Every OOD query contains at least one coordinate outside nominal encoder support while retaining a unique exact identity.

Primary operational requirement:

`OOD support rejection rate = 1.0`

No OOD query may reach `G_D`, `R_D`, or candidate-set construction operationally.

## Preregistered primary criteria

SEM-004 establishes useful gated relative set retrieval only if **all** criteria hold:

1. aggregate clean designated-source coverage `>= 0.90`;
2. every dimension clean designated-source coverage `>= 0.85`;
3. aggregate clean admission rate `>= 0.90`;
4. clean ambiguity-overload rate `<= 0.05`;
5. aggregate clean mean returned set size `<= 2.0`;
6. ambiguous admission rate `>= 0.90`;
7. ambiguous at-least-one-parent inclusion `>= 0.90`;
8. ambiguous dual-parent inclusion `>= 0.70`;
9. ambiguous singleton forced-choice rate `<= 0.20`;
10. ambiguous ambiguity-overload rate `<= 0.10`;
11. unrelated operational-guidance rate `<= 0.10`;
12. unrelated ambiguity-overload rate `<= 0.05`;
13. OOD support rejection rate `= 1.0`.

No primary threshold may be changed after SEM-004 held-out results are observed.

## Disposition

### All thirteen primary criteria pass

Disposition: `Pass`.

Establishes only:

> On the frozen SEM-004 synthetic benchmark, separating absolute query admission from a calibrated relative-to-best candidate band preserved designated near-context sources, represented constructed ambiguity with bounded sets, rarely activated on unrelated queries, and rejected clamp-saturated inputs operationally.

### OOD rejection fails

Disposition: `FailClosed` regardless of every other result.

### Clean coverage passes but ambiguity/unrelated/set guards fail

Disposition: `Tradeoff`.

Do not promote SEM-004 as the live default.

### Clean coverage fails

Disposition: `NotEstablished`.

## Secondary descriptive diagnostics

Report without promoting them to success criteria:

- per-dimension `G_D`;
- per-dimension `R_D`;
- clean and ambiguous set-size histograms;
- top-1/top-2 margins;
- empirical-null retrieval specificity;
- exact-tie counts;
- distance from top-1 for every returned candidate;
- calibration-control false-admission rate;
- clean-calibration true-source coverage under the learned band;
- corpus identities/digests;
- raw OOD similarity after support rejection.

These quantities cannot rescue a failed primary disposition.

## Controls

SEM-004 must report frozen controls on the same held-out queries:

1. **Top-1 forced choice** — always return the best same-dimensional candidate after support check, with no unrelated gate.
2. **SEM-003-style absolute true-source radius** — recalibrated only from SEM-004 clean calibration data, not copied from SEM-003.
3. **Absolute gate only** — if admitted, return top-1 only.
4. **Relative band only** — omit `G_D`, retaining the same `R_D`, to measure how much the absolute gate controls unrelated activation.

Controls are descriptive comparisons. SEM-004's preregistered disposition is determined only by the gated-relative rule and the thirteen primary criteria.

## No formal statistical guarantee

The corpus is deterministic synthetic data. `G_D` and `R_D` are empirical calibration rules, not distribution-free confidence bounds.

SEM-004 may report deterministic held-out performance only. Formal conformal/selective-prediction guarantees require a separate protocol with explicit distributional/exchangeability assumptions.

## Promotion boundary

Even after a SEM-004 `Pass`, live integration may only use the result for semantic **search policy**:

- branch exploration across multiple analogies;
- abstain when unrelated or overloaded;
- attenuate generated action priors when ambiguity is high;
- expose multiple candidates to downstream exact-provenance resolution.

It may not:

- treat membership as truth;
- treat `G_D`, `R_D`, set size, margin, or specificity as belief confidence;
- promote generated/counterfactual content to empirical evidence;
- bypass exact provenance;
- raise confidence merely because retrieval is selective.

## Reproducibility receipt

A future SEM-004 receipt must bind at least:

- this preregistration SHA;
- exact implementation subject SHA;
- generator/schema version;
- encoder configuration;
- SEM-003 disjointness-canary identity used by the overlap check;
- candidate-bank digest;
- both calibration-partition digests;
- all four held-out partition digests;
- per-dimension `G_D` and `R_D` bit patterns;
- all thirteen primary endpoint values/guards;
- all set-overload counts;
- control results;
- disposition;
- complete receipt-integrity digest.

No SEM-004 result is qualified until its exact implementation subject passes focused executable qualification.
