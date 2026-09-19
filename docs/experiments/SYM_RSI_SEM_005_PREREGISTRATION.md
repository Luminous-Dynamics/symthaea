# SYM-RSI-SEM-005 — XOR-Compositional Semantic Encoder Ablation

Status: **PREREGISTERED — NO SEM-005 IMPLEMENTATION OR MEASUREMENT CLAIM YET**

## Question

Can Symthaea preserve local numeric semantic similarity while removing the low-dimensional same-width similarity inflation observed under majority-bundle context composition?

SEM-005 compares two encoders on one new deterministic corpus:

- **BUNDLE** — the existing `SemanticContextEncoder::default()` composition;
- **XOR** — a preregistered alternative that keeps the same scalar-level encoding, dimension roles, and length identity but composes all components by XOR.

This is an encoder ablation only. Exact context identity, evidence authority, support semantics, and confidence boundaries remain unchanged.

## Motivation

Independent SEM-004 behavior canaries frozen at:

`3c9f77be463ca635d8f92d37de2751cbdb36b8d3`

predict that the gated-relative retrieval rule works well in 8-D and 16-D but fails in 4-D because the unrelated top-1 similarity ceiling approaches the clean/ambiguous region.

The current BUNDLE encoder forms each context from:

- `D` role-bound scalar-level hypervectors; plus
- one length hypervector shared by every context of that numeric width.

For even `D`, the shared extra vector resolves ties in majority composition. Low `D` has the highest tie frequency, so same-width contexts may inherit a strong common component.

SEM-005 tests that mechanism explicitly. It does not declare it causal in advance.

## Epistemic boundary

All SEM-005 representations and retrieval outputs are search geometry only.

Neither encoder output, similarity, separation gap, candidate-set membership, admission gate, relative band, set size, nor ablation difference grants:

- factual correctness;
- empirical validation;
- belief confidence;
- confidence-promotion authority;
- evidence of consciousness;
- evidence of general recursive self-improvement.

Protected SYM-RSI partitions `201–204`, `301–304`, `401–404`, and `1201–1204` are out of scope.

## Frozen predecessors

SEM-005 is downstream of:

- SEM-003 behavior canaries: `1c7f5d475546da7b5d5de553089e8a1437d3773a`;
- SEM-004 preregistration: `2a11924821831137a58253ec6013ba40a63f0f3d`;
- SEM-004 ambiguity amendment: `bdd779afedc7603978009278e4a661b9d568b443`;
- SEM-004 generator amendment: `2182c2bb8d6f3c6d3bd72bf6c187253397f49183`;
- SEM-004 behavior canaries: `3c9f77be463ca635d8f92d37de2751cbdb36b8d3`.

SEM-005 owns a fresh synthetic corpus and must not reuse exact identities from SEM-003 or SEM-004.

## Shared HDC primitives

Both arms use exactly:

- `BinaryHV` width: 16,384 bits;
- deterministic `BinaryHV::random(seed)` from BLAKE3 XOF over little-endian seed bytes;
- binding: XOR;
- Hamming similarity: matching-bit fraction in `[0,1]`;
- scalar levels: `64`;
- nominal scalar support: `[-1,1]`;
- current `mix64` implementation;
- current role and level salts;
- exact context identity: BLAKE3 over canonical little-endian `f32` bytes.

Frozen salts inherited unchanged:

- role salt: `0x5345_4d41_4e54_4943`;
- low-level seed: `0x4c45_5645_4c5f_4c4f`;
- high-level seed: `0x4c45_5645_4c5f_4849`;
- level-rank salt: `0x5448_4552_4d4f_4d45`;
- length salt: `0x434f_4e54_4558_544c`.

## BUNDLE arm

The baseline arm is byte-for-byte the existing `SemanticContextEncoder::default().encode(context)`.

No baseline behavior may be reimplemented differently merely for this experiment.

## XOR arm

The XOR encoder uses the same validation and same scalar `level_hv(value)` function as the BUNDLE encoder.

For context `x` of length `D`:

1. initialize

   `acc = BinaryHV::random(mix64(LENGTH_SALT ^ D))`

2. for each coordinate index `i`:

   `role_i = BinaryHV::random(mix64(ROLE_SALT ^ i))`

   `level_i = level_hv(x_i)`

   `component_i = role_i.bind(level_i)`

   `acc = acc.bind(component_i)`

3. return `acc`.

No bundling or majority vote occurs in the XOR arm.

For same-dimensional pair comparison, the identical length vector algebraically cancels in XOR distance while still distinguishing encodings across different dimensions.

## Why XOR is a plausible locality-preserving composition

If two same-dimensional contexts differ in one scalar coordinate, every identical role-bound component cancels under pairwise XOR except the changed coordinate's two level vectors. Therefore the resulting context similarity is directly governed by the frozen locality-preserving scalar-level encoding rather than by a shared majority tie-breaker.

This is a preregistered mechanism hypothesis, not a result.

## Fresh SEM-005 corpus

Generator/version:

`semantic-xor-ablation-generator.v1`

Dimensions:

`D = {4,8,16}`

Per dimension:

- candidate bank: `96`;
- labelled clean calibration queries: `32`;
- ambiguous calibration queries: `16`;
- unrelated calibration controls: `32`;
- clean held-out queries: `32`;
- ambiguous held-out queries: `24`;
- unrelated held-out queries: `24`;
- OOD held-out queries: `16`.

The exact same generated contexts and labels are evaluated by both encoder arms.

## Candidate generator

For candidate index `i in 0..96`:

`family = i / 2`

`side = i % 2`

For coordinate `j`:

`mixed = 0x6D2B`
`      + D * 229`
`      + family * 83`
`      + j * 47`
`      + j * j * 19`

using wrapping integer arithmetic.

`center_j = (mixed % 1501) / 1000.0 - 0.750`

Sibling-separation coordinate:

`k = (family * 11 + D * 5) % D`

Apply:

- side 0: `context[k] -= 0.090`;
- side 1: `context[k] += 0.090`.

All arithmetic is `f32` after integer-to-float conversion.

Every candidate must remain inside nominal support.

## Clean calibration queries

Use families `0..32`, even sibling source:

`source_index = 2 * family`.

Perturbation:

`m = 0.020 + 0.005 * (family % 9)`

`k = (source_index * 17 + 23) % D`

Direction is positive when `(family + D + 23) % 2 == 0`, negative otherwise.

Clamp perturbed coordinate to `[-0.95,0.95]`.

## Ambiguous calibration queries

Use sibling families `0..16`.

Each query is the exact coordinate midpoint of its two sibling candidate parents.

Both parent identities are frozen before encoding.

## Unrelated calibration controls

For index `i in 0..32`, coordinate `j`:

`mixed = 0xB7E1`
`      + D * 269`
`      + i * 127`
`      + j * 71`
`      + j * j * 23`

`query_j = (mixed % 1501) / 1000.0 - 0.750`.

## Clean held-out queries

Use families `0..32`, odd sibling source:

`source_index = 2 * family + 1`.

Perturbation:

`m = 0.0225 + 0.005 * ((family * 5 + 2) % 9)`

`k = (source_index * 19 + 31) % D`

Direction is positive when `(family + D + 31) % 2 == 0`, negative otherwise.

Clamp perturbed coordinate to `[-0.95,0.95]`.

## Ambiguous held-out queries

Use sibling families `24..48`.

Each query is the exact coordinate midpoint of its two intended sibling parents.

## Unrelated held-out queries

Use the unrelated equation above with indices `64..88`.

## OOD held-out queries

Use candidate sources `80..96`.

For local offset `r`:

`k = (r * 7 + D) % D`

Replace source coordinate with:

- non-negative source: `1.40 + abs(source[k])`;
- negative source: `-1.40 - abs(source[k])`.

Every OOD query must be support-rejected before operational retrieval.

## Corpus integrity

Before any calibration:

- every SEM-005 partition must be exact-internally unique;
- partitions must be pairwise exact-disjoint;
- all SEM-005 identities must be disjoint from frozen SEM-003 and SEM-004 corpora.

Any overlap is an integrity failure, not a result.

## Retrieval rule used for downstream utility

Each encoder arm independently calibrates the already-preregistered gated-relative rule using only its own encoding of the shared calibration corpus.

Per dimension:

`G_D = q95(unrelated-calibration top-1 similarity)`

`C_D = q95(clean-calibration true-source gap)`

`A_D = q95(ambiguous-calibration weaker-parent gap)`

`R_D = max(C_D, A_D)`.

Nearest-rank `q95` is exactly:

`rank = ceil(N * 0.95)`

clamped to `[1,N]`.

Operational admission requires strictly:

`top1_similarity > G_D`.

For an admitted query, include same-dimensional supported candidate `c` when:

`top1_similarity - similarity(q,c) <= R_D`.

`MAX_SET_SIZE = 4`; larger sets produce `AmbiguityOverload` with no operational set.

Each encoder arm gets independently calibrated `G_D/C_D/A_D/R_D`; no threshold learned under one encoder may be reused under the other.

## Representation-level primary metrics

For each arm and dimension report:

- unrelated calibration top-1 q95;
- clean held-out designated-source top-1 rate;
- clean true-source similarity distribution;
- unrelated held-out top-1 similarity distribution;
- clean-vs-unrelated separation:

  `median(clean true-source similarity) - q95(unrelated held-out top1)`.

Also report the same statistics aggregated across dimensions.

## Retrieval-level primary metrics

For each arm report the gated-relative held-out metrics:

- per-dimension and aggregate clean admission;
- per-dimension and aggregate clean designated-source coverage;
- clean overload rate and mean returned set size;
- per-dimension and aggregate ambiguous admission;
- ambiguous at-least-one and dual-parent inclusion;
- ambiguous singleton forced-choice rate;
- ambiguous overload rate;
- per-dimension and aggregate unrelated operational-guidance rate;
- unrelated overload rate;
- OOD support rejection.

## SEM-005 XOR success criteria

The XOR encoder establishes a useful replacement candidate only if **all** of the following hold on the frozen SEM-005 corpus:

1. 4-D unrelated held-out top-1 q95 `<= 0.95`;
2. every dimension clean designated-source top-1 rate `>= 0.90`;
3. every dimension clean designated-source coverage under gated-relative retrieval `>= 0.85`;
4. aggregate clean designated-source coverage `>= 0.90`;
5. aggregate clean admission `>= 0.90`;
6. clean overload rate `<= 0.05`;
7. aggregate clean mean returned set size `<= 2.0`;
8. every dimension ambiguous at-least-one-parent inclusion `>= 0.85`;
9. aggregate ambiguous at-least-one-parent inclusion `>= 0.90`;
10. aggregate ambiguous dual-parent inclusion `>= 0.70`;
11. ambiguous singleton forced-choice rate `<= 0.20`;
12. ambiguous overload rate `<= 0.10`;
13. every dimension unrelated operational-guidance rate `<= 0.10`;
14. aggregate unrelated operational-guidance rate `<= 0.05`;
15. unrelated overload rate `<= 0.05`;
16. OOD support rejection rate `= 1.0`;
17. median clean true-source similarity `>= 0.90` in every dimension;
18. clean-vs-unrelated separation is positive in every dimension.

All eighteen must pass for XOR to receive disposition `Pass`.

## Baseline comparison guards

Even if XOR passes the absolute criteria, report these preregistered regression comparisons against BUNDLE on the same corpus:

- XOR clean designated-source top-1 rate may not be more than `0.05` below BUNDLE in 8-D or 16-D;
- XOR clean designated-source coverage may not be more than `0.05` below BUNDLE in 8-D or 16-D;
- XOR aggregate unrelated operational guidance may not exceed BUNDLE by more than `0.05`.

If XOR meets all eighteen absolute criteria but violates a baseline regression guard, disposition is `Tradeoff`, not `Pass`.

BUNDLE is not required to fail for XOR to pass.

## Mechanism diagnostic

Report the per-dimension rate of BUNDLE component-vote ties before the shared length vector is added.

This diagnostic is intended to test the motivating mechanism:

> lower numeric context width should exhibit a larger fraction of tie-resolved bits under majority composition.

Also report correlation across dimensions between tie rate and unrelated top-1 q95 descriptively. With only three dimensions this correlation is not an inferential statistic and cannot determine disposition.

## Disposition

- `Pass`: all eighteen XOR absolute criteria and all three baseline regression guards pass.
- `Tradeoff`: all XOR absolute criteria pass but at least one baseline regression guard fails.
- `NotEstablished`: any non-OOD XOR absolute criterion fails.
- `FailClosed`: OOD rejection fails or corpus/support integrity fails.

No threshold may be altered after SEM-005 held-out results are observed.

## Promotion boundary

A SEM-005 `Pass` would authorize only further qualification of XOR composition as a **semantic retrieval representation candidate**.

It would not by itself authorize replacing the live encoder. A separate migration protocol must establish:

- persistence/versioning behavior;
- semantic-memory rebuild strategy;
- backward compatibility;
- performance cost;
- downstream retrieval calibration;
- no change to evidence/confidence authority.

## Reproducibility receipt

A future SEM-005 receipt must bind:

- this preregistration SHA;
- exact implementation subject SHA;
- generator version;
- every partition digest;
- cross-experiment disjointness evidence;
- encoder constants and salts;
- BUNDLE and XOR encoder identities;
- per-arm/per-dimension calibration thresholds;
- all eighteen XOR primary guard values;
- all three baseline regression guards;
- mechanism tie-rate diagnostics;
- disposition;
- complete receipt-integrity digest.

No SEM-005 result is qualified until its exact Rust subject passes focused executable qualification.
