# MATH-RET-RUNTIME-001D — Guarded S-Arm End-to-End Adapter

Status: draft qualification tranche stacked on MATH-RET-RUNTIME-001C.

## Purpose

This tranche closes the difference between:

```text
"the graph names a candidate_set_sha256"
```

and:

```text
"that digest identifies an actual canonical candidate artifact,
and every runtime source returned by the backend is a member of it
before any source bytes are materialized"
```

It intentionally changes one causal dimension relative to the predecessor #4324 adapter: candidate-universe identity is now materialized and enforced.

## Evidence chain

```text
MATH-RET-CANDIDATE-001A candidate-set bytes
        ↓ exact SHA-256 + exact count
all retrieval-index manifests
        ↓
MATH-RET graph v1.1 qualification
        ↓
MembershipGuardBackend<FixtureBackend>
        ↓
raw S ranking membership check
        ↓
RetrievalExecutor
        ↓
canonical source materialization / prefix packing
        ↓
MATH-RET-TRACE-001A
        ↓
MATH-RET-CANDIDATE membership report
        ↓
MATH-RET-PAYLOAD-001A actual-byte audit
```

## Candidate artifact

The fixture writes one canonical candidate-set JSON object before it constructs any retrieval index. The candidate list contains exactly 32 unique sorted `SourceObjectDigest` identities, including the three sources returned by the positive S-arm fixture.

The SHA-256 of the **exact candidate-set file bytes** is substituted for the predecessor fixture's single candidate-set placeholder. From that point onward the existing #4324 graph builder computes index, binding, experiment and bundle identities normally.

The candidate file itself is not added as a new graph-bundle artifact kind. The existing retrieval-index contract already contains the commitment edge:

```text
index.candidate_universe.candidate_set_sha256
```

The candidate validator proves what that digest resolves to; graph qualification proves all arms share the digest; runtime guard + membership validation prove returned sources belong to it.

## Positive S execution

The positive backend returns the exact eligible identities corresponding to fixture values `0xa1`, `0xa2`, and `0xa3`.

It is wrapped in `MembershipGuardBackend`, then passed unchanged into the same `RetrievalExecutor` introduced by MATH-RET-RUNTIME-001A.

After execution, the tranche runs the existing validators unchanged:

- graph v1.1;
- runtime trace v1;
- candidate membership v1;
- payload audit v1.

The final evidence summary carries the exact candidate-set, graph, trace, membership-report and payload-audit digests.

## Negative canary

A second execution changes only one ranked source:

```text
legal 0xa2 → illegal 0xee
```

`0xee` is explicitly absent from the frozen candidate artifact.

The binary requires this execution to return `RetrievalError::Candidate` from the guard and proves:

```text
canonical materializer calls == 0
evidence commits == 0
staged evidence == 0
runtime evidence directory does not exist
```

This is stronger than merely validating a bad trace after the fact: the invalid candidate is forbidden from reaching the payload/evidence plane at all.

## No change to mathematical authority

A PASS proves only runtime membership/provenance interoperability. It does not show that any returned source is relevant, equivalent, useful for proof search, produced by HDC advantage, or mathematically true.

Authority remains `MeasurementOnly`.

## Next one-variable extension

After guarded S qualifies, use the **same exact candidate set, context packer, source materializer and budgets** for arm F. Only the frozen fusion retrieval path should change. The F qualification must check both raw channel rankings before deterministic fusion and then replay/validate the fused ranking through the existing trace contract.
