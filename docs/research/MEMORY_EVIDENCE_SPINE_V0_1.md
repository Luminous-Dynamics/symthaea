# Bitemporal evidence spine v0.1 for Symthaea memory

**Date:** 2026-11-24  
**Track:** memory lifecycle / MACE evidence qualification  
**Status:** additive Rust contract + schema + unit tests authored; execution pending

## Why this increment exists

Symthaea already has HDC semantic memory, hippocampal episodic memory, working-memory graduation, and MACE's decision-scoped evidence qualification work. The missing integration contract is a common way to answer four distinct questions about a stored claim:

1. **What evidence supports it?** Source identity, source revision and source/payload commitments.
2. **When was it true?** World-validity time.
3. **When did this system know it?** Knowledge/ingestion time.
4. **May it still be used for this decision?** Purpose, authority, consent, retention, supersession, retraction and erasure state at the requested time.

A timestamp alone cannot answer all four. In particular, backdated evidence may arrive late, an old preference may remain historically true after a correction, and a consent revocation can be known now but become effective at a declared future time.

## Architecture decision

This is an **evidence envelope and projection gate**, not another memory store.

- Keep HDC vectors and similarity indexes as candidate generators.
- Keep existing semantic, episodic and temporal memory as storage and retrieval machinery.
- Attach memory values to an EvidenceEvent ID at the read/use boundary through EvidenceBound<T>.
- Ask the evidence ledger for eligibility before a candidate value crosses a decision/action boundary.
- Continue to use the MACE decision-scoped qualification service (#13410) for external qualification receipts, query/candidate binding and capability-style use tokens.
- Continue to use pinned trust profiles (#13436) for external signature/issuer trust and anti-rollback.
- Keep candidate evidence-DAG construction (#13386) responsible for source lineage and derivation receipts.
- Keep poisoning, repair and erasure lifecycle qualification under #13319 and decision sufficiency under #13284.

The new code does not duplicate those services. It adds a memory-tier temporal and rights projection contract they can consume.

## Event model

The Rust module is at crates/domains/symthaea-memory/src/evidence.rs; the strict event schema is at data/schemas/memory_evidence_event_v1.schema.json.

### Bitemporal fields

- valid_from_millis is inclusive.
- valid_until_millis is exclusive; null means no declared end.
- known_at_millis is the knowledge/ingestion time.
- The ledger requires non-decreasing knowledge time in append order.
- Queries carry both valid_at_millis and known_at_millis.

That enables two distinct questions: “What was true on date X, using everything known today?” and “What did the system believe it knew on date X about the state at date Y?”

### Append-only transition types

| Event | Meaning |
|---|---|
| Assertion | A source-backed claim is added |
| Correction | A new claim supersedes one or more earlier claims with the same claim key |
| Retraction | An earlier claim is withdrawn |
| ConsentGranted | A consent grant with explicit purpose and validity scope |
| ConsentRevoked | The grant issuer revokes the exact grant event |
| ErasureTombstone | A trusted erasure transition invalidates targeted records |

The event log is append-only at the API level: append validates the full candidate ledger before mutating it. Corrections do not overwrite their targets. Erasure leaves a tombstone in the event history so a reconstructed ledger continues to reject the erased event. This in-memory module does not itself provide durable append-only storage.

### Use-time gates

An assertion/correction is returned as admissible only when all applicable checks pass:

- the event was known by the requested knowledge-time cutoff;
- the world-validity interval contains the requested valid time;
- declared purpose and required authority match exactly;
- retention has not expired;
- no in-scope correction/retraction or trusted erasure has invalidated it;
- any required consent grant exists, was known, applies to the requested purpose/time, has not been revoked by its granting issuer, and has not itself been erased;
- every declared derivation parent remains admissible.

Unknown, forward, duplicate or self references are rejected. A correction/retraction can invalidate a claim only within its own valid-time interval and declared purpose/authority scopes. Erasure is purpose-independent, but its append path must be gated by an external verifier and the event must carry the explicit memory-erasure authority scope.

## Integration without breaking existing memory APIs

The new generic EvidenceBound<T> wrapper lets existing entries participate without adding required public fields to SemanticEntry, MemoryTrace or GraduationEvent (which would break many struct-literal constructors).

Illustrative integration:

~~~rust
let qualified: EvidenceBound<SemanticEntry> =
    ledger.bind(entry, evidence_event_id, &query)?;

let semantic_candidate = qualified.value();
// Preserve qualified.evidence_event_id() alongside downstream use.
~~~

A caller should require the wrapper at a decision/action boundary and carry its event ID into the decision receipt. The wrapper is not serializable/deserializable; it must be minted through EvidenceLedger::bind at the actual use boundary. It is a query-specific admissibility snapshot, not a reusable authorization token: re-evaluate before each later use because consent, retention, validity or erasure state can change. A caller can still deliberately unwrap the value, so this is an API-level discipline rather than a sandbox. The authoritative enforcement remains in the consumer and external qualification gate.

## Security and claim boundaries

The validator checks declared structure, event ordering, temporal semantics, reference integrity and local rights policy. It does **not**:

- calculate or recompute the declared SHA-256/BLAKE3 digests;
- authenticate source identity or issuer signatures;
- prove an issuer is authorized in the real world;
- prove the claim itself is true;
- provide a trusted clock or monotonic external witness;
- guarantee append-only persistence against a privileged store operator;
- provide backup/restore erasure guarantees;
- verify rights or authority receipts from #13410/#13436.

A digest is not a signature. A source label is not authentication. A correction is not automatically more trustworthy than the original claim. External qualification must occur before privileged control events—especially erasure tombstones—are appended.

The in-memory reconstruction test proves only that a tombstone remains effective when the same event sequence is reconstructed. It does not prove storage durability, T9/T10 backup resistance or production erasure.

## Deterministic tests authored

The module contains tests for:

1. knowledge-time versus valid-time separation;
2. correction effective-time boundaries;
3. correction target identity and backward-only references;
4. derived-claim invalidation after a required parent is retracted;
5. consent revocation at use time;
6. revocation authority matching the grant issuer;
7. exact purpose/authority fail-closed behavior;
8. erasure tombstone behavior after ledger reconstruction;
9. append monotonicity and atomic rejection;
10. digest shape and nonempty scope validation;
11. invalid validity intervals;
12. unknown/invalid control references;
13. refusing to bind an ineligible value;
14. future-valid correction not erasing present truth;
15. non-decreasing knowledge time;
16. erasure of a consent grant invalidating dependent claims;
17. rejection of source observations after the system's knowledge time;
18. rejection of retractions carrying an invalid consent identifier;
19. valid ledger serialization/deserialization round-trip;
20. rejection of a serialized correction with a missing target;
21. binding an existing SemanticEntry without changing its public layout.

Ledger deserialization uses a custom validating implementation: decoded snapshots pass the same cross-event checks as EvidenceLedger::new. EvidenceBound<T> is deliberately not deserializable. Two codec tests cover valid ledger round-trip and rejection of a serialized correction with a missing target. These 21 tests are authored but have not been compiled or executed in this environment.

## Benchmark plan and comparison

Use external benchmarks for their original purpose, then test the additional guarantees here:

- **Agent Memory Leaderboard Cycle 2:** textual, coding and multimodal Add/Search reproduction. Cycle 2 results are released; its rules require deployed participant-hosted Add/Search APIs, not just source code or a Docker image.
- **Agent Zero Memory:** reproduce the event timeline + entity/event graph + citation-locked documentary memory composition on LongMemEval/LoCoMo before claiming competitive retrieval. Its paper reports 95.60% LongMemEval and 93.60% LoCoMo under its stated protocol; these are paper-reported results, not MACE results.
- **AMA-Bench:** evaluate causal/objective memory over agent-environment trajectories, not only conversation QA.
- **MemSecBench and MemoryDojo:** preserve the Write–Execute–Forget and poisoned-information lifecycle protocols, while adding bitemporal correction and consent/erasure controls.
- **MemoryAgentBench:** retain the four component capabilities, particularly selective forgetting and long-range understanding.
- **MACE extension:** pair current-state and as-of-history queries with late evidence, future-effective corrections, retractions, consent revocation, parent invalidation, erasure after compaction/restart and restore-from-backup tests.

Prospective hard-gate targets: zero stale-current-state uses after an effective correction/retraction; zero use beyond consent/purpose/retention scope; zero erased-parent descendants admitted; zero tombstone resurrection under qualified storage/restore tests. These are qualification targets, not measured results.

## Next integration steps

1. Bind SemanticEntry and MemoryTrace to evidence IDs at ingestion without changing existing public struct literals.
2. Store the append-only event ledger in a durable sidecar with an independent witness/monotonic sequence.
3. Recompute and validate the complete derivation closure at use time against #13386/#13410.
4. Connect current-use eligibility to HDC retrieval so similarity only ranks already-admissible candidates.
5. Add independent signature/issuer verification, purpose/consent administration and erasure/restore witnesses.
6. Run the external benchmark protocols and paired lifecycle extensions under frozen versions, model, reader, budgets and evaluator.

## Qualification boundary

This is a new reusable Rust contract, an initial schema and authored unit tests. No Rust compile, unit-test pass, durable store, production memory integration, external signature verification, human study, privacy certification or benchmark result is claimed.

## Source register

- Agent Memory Leaderboard Cycle 2 official results, released 19 November 2026: https://agentmemoryleaderboard.ai/results/cycle-2/
- Agent Memory Challenge Cycle 2 participation rules and Add/Search requirements: https://agentmemories.ai/rules
- Agent Zero Memory, Provenance-Aware Long-Term Memory for LLM Agents (2026): https://arxiv.org/abs/2608.29606
- AMA-Bench, Proceedings of Machine Learning Research 306 (ICML 2026): https://proceedings.mlr.press/v306/zhao26bs.html
- MemoryDojo (November 2026): https://arxiv.org/abs/2611.09721
- MemSecBench (2026): https://arxiv.org/abs/2607.27080
- MACE decision-scoped evidence qualification gate: PR #13410
- MACE pinned Ed25519 qualification trust profiles: PR #13436
- MACE candidate-side source-lineage evidence graph: PR #13386