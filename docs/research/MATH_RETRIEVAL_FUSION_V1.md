# MATH-RET-001B — Mathematical Retrieval Fusion Contract v1

Status: preregistration / qualification contract for combining syntax and exact-normal-form retrieval.

Authority: `MeasurementOnly`.

## Why this contract exists

A fusion arm can appear budget-equal while still receiving a hidden information advantage.

For example, if a single-channel arm retrieves top-8 items but fusion retrieves top-8 syntax plus top-8 normal-form candidates and only then chooses eight outputs, fusion inspected twice as many candidate identities.

v1 forbids that.

## Core law

```text
same final output size != same retrieval opportunity

syntax quota + normal-form quota
    <= global item ceiling

syntax byte quota + normal-form byte quota
    <= global byte ceiling
```

Fusion is an intervention in how a fixed opportunity budget is allocated, not permission to double the candidate pool.

## Inputs

v1 requires exactly two unique channels:

```text
Syntax
ExactNormalForm
```

Each input binds:

- exact retrieval-index manifest digest;
- maximum input items;
- maximum input bytes;
- rank weight.

v1 fixes:

`rank_weight = 1`

for both channels.

Weighted fusion belongs in a later lineage. Equal weights avoid post-hoc tuning and avoid making channel weighting a second simultaneous intervention.

## No adaptive quota refill

v1 freezes:

```text
channel_underfill_policy = NoReallocation
unused_quota_reallocation = false
```

If one channel returns fewer candidates than its quota—or if duplicates collapse the union—the fusion arm returns fewer effective items / consumes fewer bytes.

It may not compensate by querying deeper into the other channel.

This preserves the meaning of the frozen per-channel opportunity allocation.

## Rank-based fusion only

v1 permits:

```text
ReciprocalRankFusion
DeterministicInterleave
```

It deliberately does **not** permit raw-score weighted fusion.

HDC similarity, cosine similarity, BM25, Jaccard and other retrieval scores generally live on different scales and distributions. Combining them directly would require a separately qualified score-calibration intervention.

Rank-based fusion avoids making unqualified cross-metric calibration part of the initial causal experiment.

### Reciprocal Rank Fusion

RRF requires a preregistered positive integer `rrf_k`.

The exact RRF parameter is therefore part of the frozen policy.

### Deterministic interleave

Interleave has no `rrf_k`.

Channel order/tie resolution remains deterministic under the source-object digest law.

## Source-object deduplication

Fusion deduplicates by:

`SourceObjectDigest`

A source object retrieved in both channels appears once in final output.

The duplicate merge policy is:

`OneOutputUnionChannelProvenance`

Thus a duplicate may accumulate evidence/rank contribution from both channels but consumes only one final output slot.

## Deterministic tie handling

v1 fixes:

`tie_break = SourceObjectDigestAscending`

This prevents map iteration, insertion order, index build order, or concurrency timing from becoming an unregistered fusion intervention.

## Global packing budget

The fusion policy carries one global output budget:

```text
max_output_items
max_output_bytes
max_output_item_bytes
```

The semantic validator requires:

```text
sum(channel max_input_items) <= max_output_items
sum(channel max_input_bytes) <= max_output_bytes
```

The output packer uses:

```text
byte_accounting = CanonicalUtf8Bytes
partial_item_policy = RejectWholeItem
packing_policy = FusedRankOrderWholeItems
```

The final candidate is never partially sliced to make an arm appear budget-compliant.

## Canonical downstream payload

A critical separation is:

```text
retrieval representation
        !=
downstream context payload
```

The syntax/HDC/normal-form representation is used to rank source objects.

The downstream search/prover context receives:

`payload_kind = CanonicalSourceObject`

under one frozen payload-serialization digest.

This avoids conflating “which source object was retrieved?” with “which transformed representation was handed to the solver/model?”

If a later experiment wants normalized-form context as a second intervention, that requires its own contract and arm.

## Provenance

Every fused output must preserve:

- source-object provenance;
- contributing channel provenance.

A source object appearing in both channels records both channel contributions even though it counts once against the output item ceiling.

This makes later manipulation checks possible.

## Relationship to MATH-RET-001A

MATH-RET-001A freezes each input index independently:

```text
candidate universe
representation
scoring/index mechanics
ranked output
```

MATH-RET-001B freezes only the combination step:

```text
syntax ranked list -----+
                        +--> fixed-budget rank fusion --> canonical source payload
normal ranked list -----+
```

Before execution, the two input index manifests must also pass a cross-index fairness gate confirming the same candidate universe.

## Relationship to MATH-EXP-001 v2

MATH-EXP-001 v2 freezes one shared experiment-level resource ceiling.

A fusion-policy instance must fit inside that root ceiling.

The intended chain is:

```text
MATH-EXP shared budget
        |
MATH-RET-001A syntax index
MATH-RET-001A normal-form index
        |
MATH-RET-001B fusion policy
        |
canonical source-object context
```

A later cross-contract validator should verify:

```text
fusion.max_output_items <= experiment.retrieved_items_max
fusion.max_output_bytes <= experiment.retrieval_context_bytes_max
fusion.max_output_item_bytes <= experiment.retrieved_item_bytes_max
```

and should verify that the exact input index digests match the arm's frozen fusion policy.

## Adversarial validator tests

The stdlib validator rejects:

- channel quotas whose item sum exceeds the global output ceiling;
- unequal/tuned v1 rank weights;
- adaptive channel refill;
- insertion-order ties;
- channel-specific downstream payloads;
- duplicate source objects consuming multiple output slots;
- partial final-item truncation.

It positively accepts both RRF and deterministic-interleave fixtures.

## Nonclaims

A valid fusion policy does not establish that:

- fusion improves retrieval;
- RRF is better than interleave;
- HDC contributes useful complementary neighbors;
- normal-form retrieval contributes useful complementary neighbors;
- fused retrieval improves proof search;
- any retrieved item is mathematically relevant;
- any theorem is true or novel.

It freezes the combination mechanics needed to interpret those later measurements.
