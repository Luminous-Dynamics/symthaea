# MATH-RET-TRACE-001A — Runtime Retrieval Evidence Receipt v1

Status: contract and replay validator for per-query mathematical retrieval execution.

Authority: `MeasurementOnly`.

## Purpose

MATH-EXP-001 v2.1 and MATH-RET-001A/B2/C/D can qualify a content-addressed retrieval **plan**. This tranche records what a real retrieval query actually did and checks that execution against the qualified graph.

The governing law is:

```text
valid retrieval trace
    != relevant neighbor
    != mathematical equivalence
    != theorem truth
    != novelty
```

The trace is runtime evidence about retrieval mechanics and resource use only.

## Required graph identity

Every trace binds:

```text
bundle_sha256
graph_report_sha256
experiment_sha256
arm_id
retrieval_binding_sha256
candidate_set_sha256
candidate_count
context_packer_sha256
```

The validator re-runs the qualified graph from the bundle and rejects disagreement with any of these identities.

The graph-report digest is computed from the canonical compact JSON report emitted by the MATH-RET-001D validator:

```text
json.dumps(report, sort_keys=true, separators=(",", ":")) + "\n"
```

This prevents a runtime receipt from claiming a different candidate universe or packer than the graph that was actually qualified.

## Seed and query identity

Each trace freezes:

```text
experiment_seed
query_id
query_source_object_sha256
```

The seed must be one of the experiment's preregistered seeds.

The query source object is forbidden from appearing in the returned ranked candidates under the existing candidate-universe self-exclusion law.

## Why scores are absent

The v1 receipt records ordered source-object identities, not raw similarity scores.

That is deliberate.

```text
BM25 score
cosine similarity
Hamming/HDC similarity
custom conventional score
```

are not assumed to share a calibrated numerical scale. For qualification, what must be replayable is the **ordered retrieval result**, the exact artifact that produced it, and any deterministic fusion transformation.

Score-distribution research can use a separate diagnostic evidence stream; it must not become an implicit cross-arm authority signal.

## Single-index execution

A `SingleIndex` trace records:

```text
index_manifest_sha256
index_artifact_sha256
requested_k
ranked_source_object_digests
```

The validator requires:

- the exact index referenced by the arm's retrieval binding;
- exact index-artifact identity;
- `requested_k` not above the experiment root retrieval item ceiling;
- `requested_k` not above the index's declared `top_k_supported`;
- no more returned items than requested;
- unique ranked source-object identities;
- exactly one retrieval query consumed.

The ranked list may be shorter than `requested_k`; underfill is evidence, not an error by itself.

## Fusion execution

A `Fusion` trace records two channel results:

```text
Syntax
ExactNormalForm
```

Each channel records:

```text
index_manifest_sha256
index_artifact_sha256
requested_k
input_bytes_used
ranked_source_object_digests
```

The validator checks each channel against its exact binding, index identity and frozen MATH-RET-001B2 input quotas.

Exactly two retrieval queries are accounted for in v1 fusion execution.

## Independent fusion replay

The trace includes the reported final fused ranking, but the validator does not trust it.

It independently recomputes the ranking from the two recorded input lists and the exact v1.1 fusion policy.

### Reciprocal Rank Fusion

Replay uses Python `Fraction` to implement the contract's mathematical rational score:

```text
score(x) = Σ_channel 1 / (rrf_k + rank_channel(x))
```

Final ordering is score descending, then source-object digest ascending.

No floating-point score comparison is used.

### Deterministic interleave

Replay implements the MATH-RET-001B2 state machine:

- start on the frozen channel;
- strict channel alternation;
- duplicate consumes a turn and emits no replacement;
- duplicates union provenance conceptually but count once;
- once one frozen list is exhausted, continue only through the remainder of the other already-frozen list;
- never query deeper to backfill.

The reported fused ranking must equal the recomputed ranking exactly.

## Control evidence

Every participating index has exactly one `control_bindings` entry.

For non-control indices:

```text
control_transform = None
```

and no control seed/artifact may be present.

For:

```text
RandomRetrieval
ShuffledHdcVectors
PermutedChallengeAssociations
```

the trace must reproduce the exact seed and content-addressed control artifact frozen in the index manifest.

Thus a run cannot silently change the randomization/control realization.

## Context packing replay

The trace records the exact ranked source-object sequence supplied to the context packer plus the final delivered items:

```text
rank
source_object_sha256
canonical_payload_bytes
```

The delivered sequence must be an exact prefix of the ranked input.

The validator enforces the frozen `StopBeforeFirstNonFittingItem` semantics and accepts only these terminal states:

### Empty

No ranked candidates and no output.

### RankedCandidatesExhausted

Every ranked candidate was delivered.

### ItemLimitReached

The exact item ceiling was reached while ranked candidates remained.

### FirstNonFittingItem

The next ranked item is recorded explicitly and its reported canonical payload byte size must violate either:

- `max_output_item_bytes`, or
- the remaining `max_output_bytes`.

The packer may not skip that source and continue to smaller lower-ranked objects.

## Resource accounting

Every trace records:

```text
retrieval_queries_used
wall_time_ms_used
normalized_compute_units_used_decimal
```

Normalized compute is represented as a canonical nonnegative decimal string rather than a JSON floating-point value.

The validator rejects resource usage above the experiment root ceilings.

Single-index mode requires one retrieval query; fusion requires two.

## Important byte-accounting limitation

MATH-RET-TRACE-001A validates the **reported** `canonical_payload_bytes` values and checks that their arithmetic obeys the qualified packer budget.

It does not yet independently fetch canonical source-object bytes and recompute their serialized byte lengths.

Therefore:

```text
trace packing accounting is internally consistent
```

does not yet imply:

```text
canonical source payload byte measurement was independently reconstructed
```

That stronger claim should be established by a later source-payload audit gate that binds or materializes the canonical source-object bytes under the packer's exact source-fetch and serialization policies.

This limitation is deliberate and must remain visible in evidence interpretation.

## Validation report

Successful validation emits:

```text
math-retrieval-trace-validation-report-v1
```

with:

- trace digest;
- graph-report digest;
- experiment digest;
- arm ID;
- preregistered seed;
- packed item count;
- packed reported byte count;
- `all_checks_passed = true`;
- `authority = MeasurementOnly`.

The validator report schema is:

`.github/schemas/math-retrieval-trace-validation-report-v1.schema.json`

Exact Git head, toolchain, workflow/job/operator identity and runtime binary identity remain external execution-lineage evidence.

## Pure self-test

```bash
python3 .github/scripts/validate-math-retrieval-trace.py --self-test
```

The pre-commit pure self-test covers:

- exact-rational RRF replay;
- deterministic interleave replay;
- correct prefix packing / first-nonfitting accounting;
- rejection of a non-prefix packing attack.

The exact validator source used for this tranche passed that pure replay self-test before commit.

## Full qualification command

Once a concrete qualified graph bundle and runtime trace exist:

```bash
python3 .github/scripts/validate-math-retrieval-trace.py \
  path/to/query-trace.json \
  path/to/retrieval-graph-bundle.json \
  --repo-root . \
  --report query-trace-validation-report.json
```

This full path also reruns all content-addressed graph validation and delegated predecessor validators.

## Nonclaims

A valid trace does not establish that:

- the index artifact itself was constructed correctly from the corpus;
- an approximate index met its declared recall floor at runtime;
- the reported canonical payload byte lengths were independently reconstructed;
- retrieved material was relevant;
- two retrieved problems are mathematically equivalent;
- a retrieval arm improves proof search;
- HDC, normal-form retrieval or fusion provides an advantage;
- a theorem is true, proved or novel.

It establishes that the recorded retrieval execution is consistent with the qualified retrieval graph and its frozen resource/fusion/packing rules.

## Next gates

Two follow-on tranches are now cleanly separable:

1. **MATH-RET-PAYLOAD-001** — independently reconstruct canonical source-object payload bytes and verify packing byte accounting.
2. **MATH-RET-TRACE-001B runtime emitter** — integrate trace production into the actual retrieval service so receipts are emitted automatically rather than assembled after execution.

Only after both are qualified should the held-out Q0/Q1 retrieval experiments consume these receipts as runtime evidence.
