# MACE Relation Closure and Assistance Binding V0.1

## Purpose

This crate adds a narrow contract layer for the GroupAssistBench question exposed by issue #14182: a memory system may retrieve a fact, yet still fail to integrate its relations or apply it correctly in the answer.

It is not another memory store, evidence registry, identity service, ACL engine, or answer evaluator. The production host must reuse and inject:
- #13264 verified transitive source lineage;
- #13967 use-time revalidation;
- #13973 exact provider-context exposure links;
- #14071 evidence-set authority and must/cannot-link constraints;
- Mycelix source and audience authority verification.

## Two distinct operations

### Relation closure

Given initial evidence IDs, an allow-listed set of typed relation kinds, a valid-time/known-time snapshot, scope, purpose, audience and policy digest, the kernel performs deterministic bounded BFS.

Every traversed edge must:
- be in the declared relation-kind allowlist;
- match scope and purpose;
- be valid and known at the requested bitemporal snapshot;
- carry source-relation, lineage and authority receipt commitments;
- pass the injected RelationEdgeVerifierV1.

The closure is bound to the exact audience, scope, purpose, policy and bitemporal snapshot later used by the response. It is bounded by explicit maximum hops and nodes. If the node limit is exceeded, it returns no partial receipt. Contradictory edges are preserved in the receipt as conflicts; the kernel never resolves them by relevance score, recency, majority, or LLM preference. A constraint sourced from a node in a recorded contradiction cannot be emitted as an unqualified assertion.

### Response constraint binding

A response constraint links one proposition to:
- its subject and (where applicable) speaker;
- whether the proposition is a quote, speaker assertion, observation, group agreement, inference or contested claim;
- a permitted action: may assert, must attribute, must qualify, must not assert, or must abstain;
- exact source span, evidence-use, #13973 exposure-link and source-lineage digests;
- the relation-closure digest and exact scope, purpose, audience and bitemporal validity.

The verifier checks that every constraint is current and matches the request's scope/audience/purpose; that the trusted host accepts its evidence receipts; and that every declared response claim use is linked to the exact constraint and response byte span.

A required attribution constraint cannot be used to state a speaker's claim as an unqualified world fact. Direct quotations and speaker assertions cannot be represented as unqualified MayAssert constraints; derived inferences and contested claims require qualification, prohibition or abstention. A required qualification constraint needs the matching qualification digest. A must-not-assert constraint cannot be used as supporting authority. A must-abstain constraint requires abstention rather than either an answer or a clarification. Required answer constraints cannot silently disappear from the structured claim-use set. Multiple independently qualified sources may bind to the same exact response span; ambiguous partial-overlap spans are rejected.

## What is not proven

- The digest is not a signature, source authentication, proof of truth, or external witness.
- The trait implementation must independently authenticate the issuer, authority, rights, lineage, revocation and exact #13973 use receipt.
- Byte-span binding proves where the response says a claim was placed, not that the natural-language span semantically entails the proposition digest.
- This version does not independently segment arbitrary prose into all factual claims. A production integration must make the structured claim inventory complete or add an independently qualified semantic claim extractor.
- The relation closure is a proposal over typed edges; it does not prove the graph contains every relevant relation or that relation labels are correct.
- The public fixture verifier is synthetic test scaffolding and must never be used as production trust.

## Tests

The 18 deterministic Rust tests cover bounded closure, cycle/deduplication, contradiction preservation, scope and authority rejection, stale/future-known relations, node-budget failure, closure/audience replay, exact byte spans, UTF-8 boundary enforcement, speaker attribution, provenance-sensitive assertion rules, required qualification, evidence membership in the closure, multi-source exact-span support, missing required constraints, must-abstain, must-not-assert, audience isolation, stale constraints and non-answer dispositions.

No GroupAssistBench score, causal effect, production integration or external source authentication is claimed by this contract.
