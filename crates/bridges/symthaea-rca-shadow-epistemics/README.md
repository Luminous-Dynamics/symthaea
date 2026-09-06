# Symthaea RCA Shadow Epistemics

This crate currently implements one narrow RCA shadow policy: **current-runtime relevance** for an already-derived `InstrumentedRuntimeEvidenceCandidateV1`.

It answers only:

```text
Does this authentic historical runtime observation describe
this exact cognitive execution closely enough in cycle-space
for the caller's declared use policy?
```

It does **not** answer whether the claim is true in general, whether it supports a proposition, whether it is independent evidence, whether Symthaea should believe it, or whether any action is authorized.

## Exact identity, not numeric epochs

RCA runtime provenance is content-addressed. The policy therefore compares these values exactly:

- source-generation digest;
- execution-lineage digest;
- adapter profile;
- adapter-contract digest.

It does not truncate/hash these identities into `u64` generations to fit a different currentness model.

## Cycle scope

Cycle indices are meaningful only inside one execution lineage.

```text
same execution lineage
        -> cycle comparison is meaningful

different execution lineage
        -> cycle lag is undefined
```

The implementation therefore never reports a stale/future-cycle defect across different lineages.

## Explicit lag policy

`max_cycle_lag` belongs to the use context, not to the evidence object. A caller must explicitly choose the maximum acceptable lag for the proposition/use under evaluation.

A zero-lag policy requires the observation to come from exactly the current cycle.

## Policy identity

`RUNTIME_RELEVANCE_CONTRACT_V1` defines the exact comparison semantics. Its BLAKE3 digest participates in the context commitment and is copied into every issued assessment.

Changing the meaning of relevance therefore requires a new policy/profile identity rather than silently changing v1.

## Issued result, not persisted capability

`RuntimeRelevanceAssessmentV1` has private fields and no `Deserialize` implementation.

Only:

```text
assess_current_runtime_relevance(candidate, validated_context)
```

can issue one.

The assessment may be serialized for audit, but deserializing bytes cannot recreate a trusted relevance result. A consumer must reload/revalidate the candidate and context and recompute relevance.

This prevents persisted or hand-written JSON from manufacturing:

```text
defects = []
```

and thereby manufacturing `is_relevant() == true`.

## Authority boundary

The current chain is:

```text
instrumented runtime observation
        -> candidate evidence
        -> exact current-runtime relevance
```

It still does not cross:

```text
relevance
    -X-> canonical evidence authority
    -X-> proposition support
    -X-> belief/workspace admission
    -X-> action authority
    -X-> self-improvement promotion
```

A future RCA-003 policy may consume a freshly issued relevance assessment as one input to a shadow epistemic disposition, but that disposition must remain data-only and causally inert until separately qualified.