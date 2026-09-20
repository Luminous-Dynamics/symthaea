# KAMA-DIALOGUE-001N3 — Candidate-Set Provenance

Status: source-design candidate
Issue: #4981
Parent: KAMA-DIALOGUE-001N2 / #4975
Authority: candidate-source evidence only; **no consent or physical authority**

## Purpose

KAMA-DIALOGUE-001N2 correctly establishes:

```text
semantic similarity
!= retrieval authority
```

but it accepts a candidate-ID universe supplied by an upstream search/index backend. 001N3 proves which privacy-governed index artifact produced that universe and makes candidate-set completeness explicit.

The missing theorem is:

```text
candidate IDs supplied
!=
candidate universe complete/current/governed
```

## Privacy-bound candidate source

`ReflectiveMemoryCandidateSourceReceiptV1` binds:

- retrieval query/context identity;
- retrieval-policy identity;
- exact KAMA-PRIV `EmbeddingOrIndex` artifact ID;
- index schema identity;
- backend implementation identity + commitment;
- index snapshot commitment;
- index-build receipt reference;
- exact privacy-graph source lineage recorded on the index artifact;
- explicit candidate-set completeness semantics;
- bounded candidate-generation parameters;
- canonical candidate-memory ID set;
- domain-separated receipt commitment.

No raw intimate content or embedding vectors are included.

## Completeness semantics

Candidate generation must declare one of:

```text
ExactEnumeratedPartition
BoundedSearch
UnknownCompleteness
```

An exact enumeration additionally binds a partition descriptor and expected count. The count must equal the canonical candidate set.

A bounded search binds a non-zero limit and search-profile reference; returned candidate count may not exceed the declared limit. It remains explicitly non-exhaustive.

`UnknownCompleteness` exists so lack of recall guarantees is represented directly rather than upgraded into an exhaustive claim.

## Search parameters

Candidate-generation parameters are bounded typed metadata:

- exact unsigned integer;
- boolean;
- finite IEEE-754 bit representation;
- semantic token;
- opaque reference.

Parameters are canonicalized by parameter ID. Changing a recall-affecting parameter changes the receipt commitment.

## Backend ordering is not authority

Candidate IDs are canonicalized into a sorted unique set before commitment.

Therefore:

```text
backend result order
!= retrieval priority
```

The semantic/vector ranker operating after 001N2 receives only the eligibility-approved partition and establishes its own ranked order under the retrieval policy.

## Continuous privacy binding

The candidate receipt is valid only while the exact KAMA-PRIV index artifact remains:

- active;
- retrievable;
- dependency-complete;
- `EmbeddingOrIndex` kind;
- non-export retention;
- bound to the exact recorded source lineage.

If privacy retraction deletes the index or narrows its source lineage, the old candidate-source receipt fails closed.

## Chained retrieval

001N3 wraps the existing 001N2 partition:

```text
live privacy-bound index
        ↓
candidate-source receipt
        ↓
001N2 pre-rank eligibility partition
        ↓
partition-chain commitment
        ↓
semantic ranking inside eligible set
        ↓
live source + privacy + partition revalidation
        ↓
001N2 retrieval receipt
        ↓
provenanced retrieval receipt
```

Before ranked IDs are admitted, both the candidate-source receipt and the complete 001N2 partition are recomputed against current privacy state. Stale index evidence or stale memory eligibility therefore cannot be resurrected by a cached vector result.

## Tests

Integration tests cover:

- end-to-end candidate-source → 001N2 partition → retrieval receipt chaining;
- deleted index invalidation before ranking;
- narrowed index lineage invalidating a prior receipt;
- query/context substitution;
- exact enumeration count mismatch;
- bounded search remaining non-exhaustive;
- canonicalization of candidate/parameter order;
- search-parameter mutation invalidating receipt identity.

## Nonclaims

001N3 does not prove semantic relevance, memory truth, current desire, consent, exhaustive recall for bounded/unknown searches, personhood, or authority for physical behavior.
