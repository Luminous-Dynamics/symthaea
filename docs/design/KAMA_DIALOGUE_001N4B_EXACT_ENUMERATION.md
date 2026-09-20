# KAMA-DIALOGUE-001N4B — Exact Enumeration Candidate-Set Binding

Status: source-design candidate
Issue: #5060
Parent: KAMA-DIALOGUE-001N4 / #5054
Authority: exhaustive-candidate provenance only; **no consent or physical authority**

## Purpose

KAMA-DIALOGUE-001N4 establishes a current privacy-bound index build and requires matching exact-enumeration evidence by partition descriptor and count. That is necessary but not sufficient: two different candidate sets can have the same size.

The missing theorem is:

```text
same partition
+ same count
!= same enumeration result
```

001N4B binds the exact canonical candidate-ID set.

## Exact enumeration result commitment

`exact_enumeration_result_commitment_v1` commits to:

- schema/version;
- exact partition descriptor;
- exact canonical sorted unique candidate-memory IDs;
- exact count;
- domain separator.

Candidate input ordering is intentionally not significant. Duplicate IDs fail closed rather than being silently deduplicated.

Therefore:

```text
[memory-b, memory-a]
== [memory-a, memory-b]
```

for evidence identity, while:

```text
[memory-a]
!= [memory-b]
```

even though both sets have count 1.

## Strict current-index validation

`validate_candidate_source_against_current_index_strict_v1` first executes the existing 001N4 current-index validation. It then strengthens `ExactEnumeratedPartition` by:

1. requiring the candidate count to match its exact-enumeration declaration;
2. finding matching current index-layer enumeration evidence for the same partition;
3. requiring the index-layer enumerated count to match;
4. recomputing the exact canonical candidate-set commitment;
5. requiring exact equality with `enumeration_result_commitment`;
6. binding the exact-enumeration result into a new strict current-candidate commitment.

Bounded and unknown-completeness candidate searches remain non-exhaustive and are not upgraded by this layer.

## Evidence chain

```text
privacy-governed source lineage
        ↓
current index-build receipt
        ↓
current snapshot
        ↓
index-layer exact enumeration evidence
        ↓
N3 candidate-source receipt
        ↓
canonical candidate-set commitment
        ↓
strict current-candidate binding
        ↓
N2 eligibility firewall / later ranking
```

## Tests

Integration tests cover:

- exact set admitted when the build receipt commits to the same set;
- same-count different-set substitution rejected;
- candidate order canonicalization;
- duplicate candidate IDs rejected;
- partition descriptor affecting commitment identity;
- bounded search remaining non-exact without enumeration evidence.

## Nonclaims

001N4B proves only that an exact-enumeration claim is bound to the exact candidate-ID set evidenced by the current index layer. It does not prove semantic relevance, memory truth, embedding quality, consent, current desire, personhood, or physical authority.

## Qualification boundary

No format/compile/test/Clippy PASS is established until an exact-head qualifier executes against the frozen KAMA-DIALOGUE-001N4B subject.
