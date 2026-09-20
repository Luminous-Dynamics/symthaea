# KAMA-DIALOGUE-001N2 — Reflective-Memory Retrieval Firewall

Status: source-design candidate
Issue: #4940
Parent: KAMA-DIALOGUE-001N / #4935
Authority: retrieval eligibility only; **no consent, truth, or physical authority**

## Core theorem

```text
semantic similarity
!= retrieval authority
```

A semantic/vector ranker never decides which reflective memories it is allowed to see.

## Partition-first architecture

The first retrieval path is:

```text
metadata candidate IDs
        ↓
live KAMA-PRIV + KAMA-DIALOGUE-001N validation
        ↓
reality / namespace / perspective / sensitivity / retention scope
        ↓
committed eligible partition
        ↓
semantic/vector ranking inside partition only
        ↓
full live revalidation
        ↓
redacted retrieval receipt
```

Candidate lookup is ID/metadata work, not semantic ranking. The ranker receives only `eligible_memory_ids()` from the committed partition.

## Typed retrieval scope

`ReflectiveMemoryRetrievalScopeV1` binds:

- exact real-world or named fantasy reality namespace;
- allowed memory namespaces;
- allowed perspectives;
- maximum sensitivity class;
- optional minimum confidence;
- allowed retention classes;
- query/session context identity;
- retrieval-policy/version identity.

External-export retention is not accepted as reflective-memory retrieval state.

Scope commitments canonicalize set-like fields before hashing, so caller list ordering does not change scope identity.

## Live privacy gate

Every candidate ID is resolved through:

`ReflectiveIntimacyMemoryIndexV1::current(memory_id, privacy)`

before it becomes rankable.

That means a candidate is excluded when:

- its privacy artifact was deleted;
- its exact source lineage changed;
- its privacy dependency became non-retrievable;
- it was superseded;
- it belongs to the wrong reality namespace;
- its perspective is not allowed;
- its sensitivity exceeds the scope;
- its namespace/retention/confidence is outside scope.

A stale embedding/index entry therefore does not make a stale memory retrievable.

## Partition identity

The partition commitment binds:

```text
scope commitment
+ canonical source candidate IDs
+ canonical eligible memory IDs
```

The broad candidate list is included so two metadata backends that supplied different candidate universes do not silently produce the same partition identity merely because their currently eligible subset happened to match.

## Post-ranking revalidation

Before ranked IDs are admitted, the firewall recomputes the partition from the same candidate universe against the **current** memory/privacy state.

If the partition changed:

```text
old partition
!= current partition
        ↓
StalePartition
```

The ranked result is rejected rather than filtered into a newly invented result.

Every ranked ID must also be a member of the current eligible partition. Ranking score, similarity, or model confidence cannot widen eligibility.

## Receipt

A successful retrieval receipt contains only:

- query context ID;
- retrieval policy ID;
- scope commitment;
- partition commitment;
- admitted memory IDs in rank order;
- domain-separated receipt commitment.

It contains no raw intimate content, fantasy text, preference value, embedding, or reconstructive summary.

## Reality separation

A real-world retrieval scope cannot admit a fantasy-world memory, even if that fantasy memory would have the highest semantic similarity.

Likewise, a fantasy scope is bound to one exact `world_id`; one fantasy world cannot silently contaminate another.

## Sensitivity and perspective

Sensitivity and perspective are eligibility properties, not ranking features.

```text
very high similarity
+
wrong sensitivity or perspective
=
not rankable
```

## Qualification tests

The first integration campaign covers:

- cross-reality nearest-neighbor trap;
- sensitivity exclusion before ranking;
- perspective mismatch before ranking;
- stale partition after privacy retraction;
- superseded-memory exclusion;
- empty eligible partition and empty receipt;
- candidate-order independence;
- scope set-order independence;
- deterministic scope/partition commitments.

## Public API posture

Like KAMA-DIALOGUE-001N and the privacy line beneath it, this source candidate is exercised through explicit integration-path inclusion before public crate API admission. Public API widening should remain a separate qualified tranche.

## Nonclaims

This firewall does not define an embedding model, prove remembered content true, infer current desire, grant consent, establish personhood, authorize physical behavior, or prove deletion outside the governed privacy domain.
