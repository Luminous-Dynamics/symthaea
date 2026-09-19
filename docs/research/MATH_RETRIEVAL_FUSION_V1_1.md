# MATH-RET-001B2 — Retrieval Fusion v1.1

Status: versioned repair of MATH-RET-001B v1 before runtime trace qualification.

Authority: `MeasurementOnly`.

## Why v1.1 exists

MATH-RET-001B v1 correctly froze fixed-budget, rank-based fusion, but a runtime-trace audit found two semantics that were not fully executable from the manifest alone:

1. `DeterministicInterleave` did not freeze which channel starts or what happens when a duplicate or exhausted channel is encountered.
2. Reciprocal Rank Fusion froze `rrf_k` but did not explicitly freeze exact-rational versus floating-point ordering.

No retrieval result has been qualified under these ambiguous semantics. v1 remains preserved as historical contract evidence.

v1.1 changes only the method-level execution semantics required to make fusion rankings independently recomputable.

## Shared inherited invariants

All v1 invariants remain unchanged:

- exactly Syntax + ExactNormalForm inputs;
- equal `rank_weight = 1`;
- per-channel item/byte quotas bounded by one global ceiling;
- no adaptive quota refill;
- source-object deduplication;
- canonical source-object output;
- deterministic source-digest tie breaking;
- whole-item output packing;
- `MeasurementOnly` authority.

The v1.1 semantic validator projects the new method fields away and delegates all inherited checks to the exact sibling v1 validator.

## Input-rank interpretation

Both methods now require:

```text
input_rank_interpretation = OneBasedAscendingIndexRank
```

The first item from each index is rank 1.

Fusion operates only on the already-frozen per-channel ranked lists. It may not request deeper candidates after fusion starts.

## Reciprocal Rank Fusion

v1.1 freezes:

```text
method = ReciprocalRankFusion
rrf_k >= 1
rrf_arithmetic = ExactRational
rrf_formula = SumOneOverKPlusRank
```

For a source object `x`:

```text
score(x) = Σ_channel 1 / (rrf_k + rank_channel(x))
```

A missing source contributes zero for that channel.

The sum and comparisons are mathematical rationals, not `f32`/`f64` approximations.

Final order is:

```text
exact RRF score descending
then SourceObjectDigest ascending
```

This avoids platform- or implementation-dependent floating-point tie behavior.

Interleave-only fields are forbidden on RRF policies.

## Deterministic interleave

v1.1 freezes an explicit state machine:

```text
method = DeterministicInterleave
interleave_start_channel = Syntax | ExactNormalForm
interleave_step_policy = StrictAlternatingByChannel
duplicate_turn_policy = DuplicateConsumesTurnNoBackfill
exhausted_channel_policy = ContinueOtherWithinFrozenInputQuota
```

### State machine

1. Begin with `interleave_start_channel`.
2. Consume exactly the next ranked item from that channel.
3. If the source object has not appeared before, emit it.
4. If it has already appeared through the other channel, merge channel provenance but emit no new output item.
5. The turn is consumed either way. Do not pull another item to replace a duplicate.
6. Alternate to the other channel.
7. If one channel has exhausted its already-frozen input list, continue consuming the remaining channel's already-frozen list.
8. Never request deeper candidates beyond either channel's declared input quota.
9. Stop when both frozen input lists are exhausted or downstream output packing stops.

Thus duplicate collisions or channel underfill can reduce effective output size. They do not create an opportunity to fish deeper.

RRF-only fields are forbidden on interleave policies.

## Graph-validator profile

MATH-RET-001D remains the graph-fairness implementation.

v1.1 adds a thin profile:

```text
.github/scripts/validate-math-retrieval-graph-v1.1.py
```

It loads the existing graph validator and changes only:

```text
FusionPolicy delegate
  from validate-math-retrieval-fusion.py
  to   validate-math-retrieval-fusion-v1.1.py
```

Candidate-universe, normalizer, binding, packer, reachability and budget checks therefore remain one implementation rather than being copied into another validator.

The successful graph report format remains `math-retrieval-graph-validation-report-v1` because the cross-graph invariants and report fields are unchanged; the bundle's exact fusion-policy digest records which fusion contract instance was qualified.

## Intended qualification

Method-level adversarial self-test:

```bash
python3 .github/scripts/validate-math-retrieval-fusion-v1.1.py --self-test
```

Graph-profile self-test:

```bash
python3 .github/scripts/validate-math-retrieval-graph-v1.1.py --self-test
```

Concrete graph qualification:

```bash
python3 .github/scripts/validate-math-retrieval-graph-v1.1.py \
  path/to/retrieval-graph-bundle.json \
  --repo-root . \
  --report retrieval-graph-report.json
```

## Self-test state

The exact v1.1 method-semantics source passed its pure adversarial self-test before commit.

Canaries reject:

- floating-point RRF arithmetic;
- missing RRF formula identity;
- interleave fields smuggled into an RRF policy;
- missing interleave start channel;
- duplicate backfill behavior;
- RRF fields smuggled into interleave.

Full delegated v1 validation and graph-profile integration still require focused repository execution.

## Runtime trace consequence

MATH-RET-TRACE-001 may now independently recompute fusion ordering from evidence-side channel rankings:

- RRF using exact rational arithmetic;
- interleave using the frozen state machine above.

A runtime trace whose fused order differs from the recomputed order must fail trace qualification even if its final selected items fit the byte/item budget.

## Deliberate nonclaims

Fusion v1.1 does not establish that:

- either fusion method improves retrieval;
- RRF is better than interleave;
- HDC and normal-form channels are complementary;
- retrieved neighbors are mathematically relevant;
- proof search improves;
- any theorem is true or novel.

It only makes the fusion transformation sufficiently explicit to be replayed and audited.

## Next gate

With fusion ordering now executable, MATH-RET-TRACE-001 should freeze a per-query runtime evidence receipt and validator that binds:

- qualified graph/bundle/report identity;
- exact arm/binding/index/fusion/packer identities;
- query/source-object digest;
- per-channel input rankings where applicable;
- recomputed fused ranking;
- final packed source-object sequence;
- item/byte/query/compute accounting;
- control seed/artifact identity;
- zero theorem-truth authority.
