# KAMA-DIALOGUE-001N — Reflective Perspective-Aware Intimacy Memory

Status: source-design candidate
Issue: #4772
Parent: KAMA-PRIV-001C / #4916
Authority: memory metadata/retrieval only; **no consent, clinical, content-generation, somatic, or physical authority**

## Purpose

Give long-running intimate dialogue a memory substrate that can preserve perspective, reality namespace, confidence, privacy lineage, and supersession without turning a transcript archive into a permanent undifferentiated memory store.

## Core theorem

```text
similar text
!= same memory
!= same speaker
!= same reality namespace
!= same confidence
!= same retention permission
```

Every memory must therefore remain attributable.

## Metadata-only contract

`ReflectiveIntimacyMemoryItemV1` stores:

- one-use memory identity;
- governing KAMA-PRIV artifact identity;
- opaque local content reference;
- domain-separated content commitment;
- semantic memory namespace;
- real-world vs named fantasy-world namespace;
- participant / Symthaea / shared perspective;
- source provenance;
- sensitivity;
- confidence;
- retention class;
- exact source-artifact lineage;
- creation/update time;
- optional superseded memory identity.

The contract does **not** contain raw intimate dialogue, fantasy text, preference values, embeddings, or reconstructive summaries.

## Semantic lanes

The first vocabulary separates:

- participant facts;
- Symthaea persona/history;
- shared relationship summaries;
- fantasy-world history;
- explicit participant preferences;
- inferred participant preferences;
- explicit boundaries;
- shared/system reflection.

Important source rules are structural:

```text
Boundary
-> ExplicitParticipantStatement only

ExplicitPreference
-> ExplicitParticipantStatement | ValidatedSelfReport

InferredPreference
-> BehavioralInference only

SymthaeaPersona
-> SymthaeaAuthored only
```

An inferred preference therefore cannot silently become an explicit preference or boundary.

## Reality separation

Fantasy-world memories carry an explicit named fantasy namespace. Supersession requires the same exact reality namespace.

```text
Fantasy(world-A)
X
RealWorld supersession
```

The system may later support explicit, separately governed promotion/copying between namespaces, but this tranche contains no implicit promotion operation.

## Privacy-bound retrieval

A reflective item is admitted only when its exact KAMA-PRIV artifact:

- exists and is active;
- has complete/retrievable dependency metadata;
- has the same retention class;
- has the exact same source-artifact set;
- has the artifact kind expected for the semantic memory lane.

Retrieval repeats those checks. This is deliberately stricter than checking only whether the privacy artifact still exists.

If a KAMA-PRIV fan-in artifact survives source retraction under `RetainIndependentBasis`, its source set changes. The old reflective memory then fails closed because its recorded lineage is stale. A fresh memory identity must be issued for the newly supported claim.

## Supersession

Supersession does not rewrite history. The old memory identity remains in the index lineage but is no longer current.

A replacement must:

- use a fresh memory identity;
- supersede a current known item;
- stay in the same semantic lane and perspective;
- stay in the same exact reality namespace;
- not move backward in time;
- independently satisfy the current privacy graph.

## Content commitment

Payload bytes may be held by a separate governed local store. This layer stores only:

```text
content_ref
+
blake3(domain || length || payload)
```

The commitment can detect content substitution without placing the payload into the metadata/audit index.

## Tests

Source + integration tests cover:

- privacy retraction immediately removing retrieval eligibility;
- fantasy memory not superseding real-world memory;
- inferred evidence being unable to establish a boundary;
- supersession retaining old identity while exposing only the new item as current;
- privacy source-lineage mutation making an old memory stale;
- deterministic content commitments;
- unknown task/source semantics failing closed through KAMA-PRIV.

## API admission

Like KAMA-PRIV-001C, this candidate is tested by explicit integration-path inclusion before public crate API admission. A later qualification tranche may export the privacy/memory modules through `symthaea-communication` once their exact subjects have execution evidence.

## Nonclaims

This does not prove that a remembered proposition is objectively true, grant consent, infer current desire, diagnose psychology, establish personhood, authorize physical behavior, or prove third-party deletion. It establishes only a privacy-bound, perspective-aware memory metadata and retrieval contract.
