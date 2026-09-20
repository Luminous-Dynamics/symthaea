# KAMA-DIALOGUE-001N4 — Privacy-Bound Index Build + Snapshot Receipts

Status: source-design candidate
Issue: #5029
Parent: KAMA-DIALOGUE-001N3 / #5023
Authority: index-build/snapshot provenance only; **no consent or physical authority**

## Purpose

KAMA-DIALOGUE-001N3 proves which candidate IDs a privacy-governed index claims to have produced. It still accepts the named index snapshot/build reference as caller-supplied evidence.

The missing theorem is:

```text
candidate receipt names snapshot X
!=
snapshot X is the current governed index state
```

001N4 makes the current reflective-memory index snapshot evidence-bearing and continuously privacy-bound.

## Index-build receipt

`ReflectiveMemoryIndexBuildReceiptV1` binds metadata only:

- exact KAMA-PRIV `EmbeddingOrIndex` artifact ID;
- exact index schema;
- backend implementation identity + commitment;
- embedding model identity + commitment;
- tokenizer/encoder identity + commitment;
- build environment ref + commitment;
- explicit clock domain and start/end times;
- exact privacy source-artifact lineage;
- one privacy-safe source-state commitment per source artifact;
- canonical typed build parameters;
- optional exact-enumeration evidence;
- resulting snapshot artifact ref + commitment;
- deterministic build-receipt commitment.

No raw intimate text or embedding vectors are present.

## Source-lineage identity

The build receipt derives a domain-separated source-lineage commitment from:

```text
privacy index artifact ID
+ exact ordered source-artifact IDs
+ exact privacy-safe source-state commitments
```

The receipt remains live only while the KAMA-PRIV index artifact exists, is retrievable/dependency-complete, and has the exact same source lineage.

Privacy retraction that deletes or narrows the index source basis therefore makes the old build receipt stale immediately.

## Current-snapshot registry

`CurrentReflectiveMemoryIndexRegistryV1` maps each governed privacy index artifact to one current build receipt.

Initial registration requires a live build receipt. Advancing the same index requires a fresh snapshot commitment. A previously known snapshot commitment cannot be reused.

Therefore:

```text
rebuild
-> fresh build receipt
-> fresh snapshot identity
```

not in-place semantic mutation under the previous snapshot hash.

## Integration with 001N3

`validate_candidate_source_against_current_index_v1` first revalidates the N3 candidate source against KAMA-PRIV and then requires it to match the registry's current build receipt on:

- privacy index artifact;
- index schema;
- backend identity + commitment;
- exact current snapshot commitment;
- exact build receipt reference;
- exact recorded source lineage.

A successful validation returns a separate domain-separated binding commitment over the current build receipt and N3 candidate-source receipt.

## Exact enumeration

001N3 can represent:

```text
ExactEnumeratedPartition
BoundedSearch
UnknownCompleteness
```

001N4 strengthens the first case. `ExactEnumeratedPartition` is accepted as current high-assurance provenance only when the current index-build receipt contains matching enumeration evidence for the same partition descriptor and exact count.

Thus:

```text
caller chooses ExactEnumeratedPartition
+ candidate count matches
!= exhaustive enumeration proven
```

Bounded and unknown-completeness searches do not require enumeration evidence because they make no exhaustiveness claim.

## Retraction and rebuild

If privacy retraction changes the source basis:

1. the prior build receipt fails live validation;
2. the prior current snapshot cannot authorize new candidate generation;
3. if the privacy graph retains the index on an independent basis, a new build receipt must be created against the narrowed lineage;
4. registry advancement requires a new snapshot commitment;
5. old candidate-source receipts no longer match current index state.

## Tests

Integration tests cover:

- deterministic build receipt/current-candidate binding;
- snapshot substitution rejection;
- privacy retraction immediately staling an old snapshot;
- source-lineage narrowing followed by a fresh rebuild;
- old candidate receipts failing after registry advancement;
- exact enumeration failing without index-layer evidence;
- exact enumeration succeeding with matching evidence.

## Nonclaims

001N4 proves index-build and current-snapshot provenance only. It does not prove semantic relevance, embedding quality, memory truth, exhaustive recall except for an explicitly evidenced enumeration protocol, consent, current desire, personhood, or authority for physical behavior.

## Qualification boundary

No format/compile/test/Clippy PASS is established until an exact-head qualifier executes against the frozen KAMA-DIALOGUE-001N4 subject.
