# MATH-RET-CANDIDATE-001A — Auditable Candidate Universe

Status: contract-first hardening discovered during MATH-RET-RUNTIME-001 static audit.

## Problem

The retrieval graph already freezes:

```text
candidate_set_sha256
candidate_count
candidate eligibility policy
corpus snapshot
knowledge boundary
```

and the runtime trace already proves that returned rankings are unique, exclude
the current query source, and contain no more entries than the frozen count.

That is not sufficient to prove membership.

Without materializing the frozen set, a backend could return three arbitrary
source identities while still satisfying:

```text
len(returned) <= candidate_count
```

A digest/count commitment alone therefore cannot establish:

```text
returned source object ∈ frozen candidate universe
```

## Candidate-set artifact

`math-retrieval-candidate-set-v1` makes the universe inspectable.

It freezes:

- corpus snapshot digest;
- knowledge-boundary digest;
- candidate-eligibility-policy digest;
- `source_identity_kind = SourceObjectDigest`;
- `canonical_order = SourceObjectDigestAscending`;
- exact candidate count;
- exact sorted unique source-object digests.

The artifact deliberately contains **no self digest**. Its scientific identity
is SHA-256 of the exact UTF-8 JSON file bytes supplied to qualification.

The index manifest's existing `candidate_set_sha256` is expected to equal that
exact file digest.

## Static universe versus query-specific exclusion

The candidate set is the frozen **eligible source universe before per-query
exclusion**.

This distinction is necessary because one shared index can serve multiple
queries. A single static set cannot simultaneously remove a different query
source for every query.

The existing runtime law remains separate:

```text
candidate is in eligible frozen universe
                 AND
current query source is excluded from returned ranking
```

The trace validator already checks the second law. MATH-RET-CANDIDATE-001A
closes the first.

## Membership qualification

`validate-math-retrieval-candidate-membership.py` accepts:

```text
candidate-set file
runtime trace
qualified graph bundle
```

Before checking membership it revalidates the trace against the graph.

It then requires:

1. candidate-set file SHA-256 == `trace.graph.candidate_set_sha256`;
2. candidate-set count == `trace.graph.candidate_count`;
3. corpus, knowledge boundary, eligibility policy, source identity, canonical
   ordering and count agree with every qualified index that binds the set;
4. every raw single-index or fusion-channel result is a member;
5. every fused result is a member;
6. every packer input identity is a member.

Checking the raw fusion inputs matters: deduplication/fusion must not hide an
out-of-universe source that appeared in one channel but disappeared from final
output.

## Authority

`MeasurementOnly`.

A valid membership report means only:

```text
these returned source identities belonged to this exact frozen eligible set
```

It does not mean the sources were relevant, mathematically equivalent, useful,
correct, uncontaminated at corpus construction time, or theorem authority.

## Runtime hardening consequence

This contract exposed a limitation in the first MATH-RET-RUNTIME-001A seam:
it binds candidate-set digest/count but does not yet require a pre-materializing
membership oracle.

Therefore the current #4316/#4324 drafts must not be treated as sufficient for
production integration even if their queued compile/adapter workflows pass.

The next runtime hardening should introduce a `QualifiedCandidateUniverse`
boundary that verifies this exact artifact **before materialization and evidence
commit**, so an out-of-universe backend result becomes non-qualifying before it
can reach downstream cognition.

The post-execution membership validator remains valuable independent evidence
that the runtime guard actually behaved as declared.

## Required canaries

Candidate-set validator rejects:

- noncanonical ordering;
- duplicate candidates;
- count/list disagreement;
- a payload digest masquerading as source identity;
- insertion-order semantics.

Membership qualification rejects:

- candidate file hash differing from the graph commitment;
- count drift;
- corpus/knowledge/eligibility metadata drift;
- an out-of-universe source in a raw channel ranking;
- an out-of-universe source surviving into final/packing ranking.

## Next gate

1. incorporate an actual candidate-set file into the MATH-RET-RUNTIME-001B
   generated fixture and run this membership validator;
2. add a pre-materialization `QualifiedCandidateUniverse` guard to the Rust
   execution path;
3. repeat the adapter qualification with an injected out-of-universe backend
   result and require rejection before materializer invocation;
4. only then connect real structural retrieval backends.
