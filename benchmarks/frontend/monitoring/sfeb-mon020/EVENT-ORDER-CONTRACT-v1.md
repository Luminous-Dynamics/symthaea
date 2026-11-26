# MON-020 — Provenance Event-Order Assurance Contract v1

**Status:** research contract and semantic prototype; real source adapters and independent qualification pending.  
**Relationship:** complements MON-019 causal graphs by qualifying event order within and across source sequence domains.

## 1. Research question

A directed acyclic graph is necessary but insufficient for a valid history. A graph can be acyclic while claiming that an entity was used before it was generated. W3C PROV Constraints explicitly requires generation to precede use, and defines event-order constraints among activity starts/ends, entity generation/use/invalidation and agent attribution.

MON-020 asks:

> Can an independent verifier establish the required order between generation, use, invalidation, activity start/end and cross-source handoffs without trusting wall-clock timestamps or the producer's own sequence claims?

The goal is an evidence-qualified partial order, not a global timestamp sort.

## 2. Paradigm shift: from timestamps to proof-carrying event order

Wall clocks can drift, be reset, be maliciously altered or have uncertainty intervals that overlap. A timestamp is not a causal proof.

MON-020 uses:
- authenticated source-local monotonic sequence numbers where both events belong to the same verified sequence domain;
- explicit independently witnessed order proofs where events belong to different sequence domains;
- a partial-order graph that must be acyclic;
- mandatory PROV-style constraints over generation, use, invalidation and activity boundaries.

No cross-source ordering is inferred from timestamps alone.

## 3. Event model

Each event contains:

- stable event ID;
- event kind: entity generated, entity used, entity invalidated, activity started or activity ended;
- subject entity/activity ID;
- activity ID for events inside an activity;
- authenticated source identity;
- source-local sequence number;
- exact event-record digest and payload/profile context.

Each source identity contains:
- source ID;
- control-root identifier;
- identity-record digest;
- independently pinned verifier report.

Each event record must be bound to an exact digest and a pinned verifier result. A self-reported sequence number or a caller-supplied source name is not evidence.

## 4. Same-source sequence order

If two events belong to the same authenticated source ID, MON-020 may infer order from their strictly monotonic sequence numbers, but only when:
- each event record's signature/attestation is independently verified;
- the sequence value is included in the exact signed/attested event record;
- source identity and verifier implementation are pinned;
- source sequence values are unique within the captured input set.

A repeated sequence number assigned to different event IDs is divergence. Same sequence domain with equal/reversed ordering cannot satisfy a required precedence relation.

This establishes order within the declared source domain. It does not establish that the source omitted no events; completeness remains with MON-016/017.

## 5. Cross-source ordering

Two distinct sources do not share a globally comparable sequence counter. Their event sequence numbers must not be compared numerically.

A cross-source required order therefore needs a separate signed/witnessed handoff record that binds:
- exact predecessor event ID;
- exact successor event ID;
- witness source ID and control root;
- exact evidence digest and profile;
- pinned verifier report.

For the v0.1 profile, the witness source's declared control root must differ from both event-source control roots. If it shares a control root with either endpoint source, the order is not independently established and the result is indeterminate.

A future profile may support stronger message-level handoff proofs, distributed transaction records or consensus log proofs. Those must be separately versioned and qualified; do not silently treat an ordinary traceparent or timestamp as a handoff proof.

## 6. Mandatory event-order constraints

The v0.1 oracle enforces:

1. An entity generation event precedes every use event of that entity.
2. Entity generation precedes its invalidation event.
3. Every use precedes entity invalidation.
4. An activity start precedes each generation/use/invalidation event associated with that activity.
5. Every internal activity event precedes the activity end.
6. Activity start precedes activity end.
7. The event-order proof graph is acyclic.
8. Source-local sequence reuse across different event IDs is divergence.
9. Conflicting witness evidence or order cycles are divergence, not silently resolved.
10. Missing cross-source ordering evidence is indeterminate, not guessed from time proximity.

The initial package focuses on these event kinds; PROV derivation-generation, agent lifetime, delegation ordering and invalidation lineage remain subsequent profile extensions.

## 7. Decision model

| Outcome | Meaning |
|---|---|
| qualified | All required order constraints have verified evidence and an acyclic partial order |
| denied | A mandatory ordering relation is proven false (for example, use precedes generation) |
| indeterminate | Required event, sequence or independent cross-source witness evidence is missing/stale/unverified |
| divergent | Admissible evidence contradicts itself, sequence IDs equivocate or the order graph contains a cycle |

No aggregate confidence score can compensate for a failed required ordering constraint.

## 8. Relationship to MON-019 and PROV

MON-019 verifies graph structure, edge typing, exact evidence references, bounded capture manifests and verifier reports. MON-020 verifies the event semantics *behind* the graph edges.

W3C PROV Constraints specifies that an entity must not be used before its generation event and defines activity/agent ordering constraints. MON-020 does not replace PROV. It offers an independently verifiable order-evidence profile that can be attached to PROV-like provenance statements.

## 9. Threat model

Assume event sources may be faulty or malicious. Required attacks include:
- event sequence reset or reuse;
- source identity substitution;
- same source ID with conflicting event keys;
- cross-source order inferred from numeric sequence values;
- use-before-generation with a still-acyclic graph;
- invalidation before a use event;
- activity end before an internal event;
- witness signed by an administrator shared with either endpoint source;
- conflicting witnesses that create order cycles;
- event timestamp backdating while sequence evidence says the opposite;
- omitted source events (which this package cannot detect without independent completeness evidence).

## 10. Benchmark programme

- 100% of finite known-answer event-order cases receive the expected classification.
- Zero false-qualified use-before-generation, use-after-invalidation or activity-boundary cases.
- Zero accepted sequence collisions or order cycles.
- Zero cross-source order claims qualified without a verified independent witness.
- 500+ independently authored order holdouts.
- Measure event count, witness count, graph shape, report lookups, p50/p95/p99, memory and allocation cost.
- Compare same-source sequence checking separately from cross-source witness validation.
- Do not compare this semantic oracle to trace ingestion throughput without matching verification and witness costs.

Targets are prospective, not achieved results. Finite-case zero failures are not a universal security guarantee.

## 11. Claim ceiling

MON-020 does not prove event sources are truthful, complete, or correct beyond the declared capture boundary. A signed source can sign false content. A sequence can be internally consistent while the source omitted events. Cross-source witness independence is only as strong as the verified control-root inventory. No cryptography is implemented by this reference package; it consumes normalized verifier reports. No destination or physical effect is established.
