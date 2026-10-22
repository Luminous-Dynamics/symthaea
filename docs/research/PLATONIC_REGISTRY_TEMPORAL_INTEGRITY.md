# Temporal Registry Integrity: Replay, Revocation, and Rollback Benchmark

**Status:** deterministic event-trace harness authored; exact-head validation pending  
**Implementation:** `crates/core/symthaea-core/examples/platonic_registry_event_integrity.rs`  
**Runner:** `.github/workflows/levin-pattern-experiments.yml`  
**Related work:** [PR #8316 — hostile enrollment / fixed-budget replacement](https://github.com/Luminous-Dynamics/symthaea/pull/8316); [Issue #8317 — temporal enrollment, revocation and replay](https://github.com/Luminous-Dynamics/symthaea/issues/8317)

## Research question

Can the registry maintain a coherent identity state when authorized and unauthorized operations interleave with duplicate event IDs, stale generations, revocation, rollback, and re-enrollment? Does it preserve legitimate recovery while preventing replayed or stale events from restoring a revoked identity?

This benchmark introduces a **temporal state-machine contract** beside the existing geometric retrieval and enrollment-poisoning experiments. It evaluates a small in-memory reference model, not the production identity-evidence API or persistence layer. Passing its tests does not prove that a deployed registry implements these rules.

## Model

Every event contains a monotonically sequenced envelope, an event ID, a synthetic authority label, and an operation. The model holds an ordered identity map with generation, vector digest, active/revoked status, known digest history, and revocation floor.

Operations:
- `Enroll`: new identities must start at generation 1 under an enrollment authority. An already active identity cannot be silently enrolled again. After revocation, explicit recovery authority and a generation greater than the tombstone are required.
- `Replace`: requires registrar authority, an active record, the matching expected generation, and exactly one forward generation step.
- `Revoke`: requires revoker authority, the matching expected generation, and exactly one forward step. The record becomes a tombstone rather than disappearing.
- `Rollback`: applies only to an active identity, requires recovery authority, a previously observed digest, and a forward-moving generation. Rollback cannot reactivate a tombstoned identity; revival is a separate recovery enrollment.

Each correctly sequenced event envelope consumes its sequence and event ID even if the operation is rejected. This models at-most-once handling for a uniquely identified attempted operation. A duplicate event ID with a new sequence is rejected before its operation executes; a regressing or skipped sequence is rejected without advancing the sequence watermark. A caller must submit any corrected attempt as a fresh event ID at the next valid sequence.

Authority values are deterministic fixture labels. **They are not signed credentials, cryptographic proof, or verified principals.** Digest fields are sample identifiers, not claims that the benchmark authenticates vector bytes. The canonical state and event encodings use explicit length prefixes to avoid delimiter-boundary ambiguity in the benchmark's own digests.

## Frozen trace matrix

The runner includes deterministic positive and negative event traces covering:
- valid enrollment → replacement → revocation → stale re-enrollment → rejected rollback → explicitly authorized forward-generation recovery;
- unauthorized initial enrollment;
- duplicated event IDs carrying a different operation;
- stale and skipped replacement generations;
- authorized rollback to a previously known digest while active;
- rollback to an unknown digest;
- unauthorized revocation and replacement;
- sequence regression followed by a valid next-sequence revocation;
- missing identities, duplicate active enrollment, and recovery without recovery authority.

Embedded tests validate every expected outcome in the trace table and directly check state invariants, including the revoked tombstone after a failed rollback, retained state after unauthorized replacement, at-most-once event IDs, deterministic digests, delimiter-safe canonical encoding and generation-overflow rejection.

The harness emits one row per event and an end receipt containing case/step totals plus BLAKE3 digests of the frozen fixtures and observed execution. If any expected decision differs, the executable exits nonzero. This is a useful regression oracle, but it is not an independent implementation: the expected outcomes and model are authored in the same codebase. A later phase should cross-check the implementation against an independently written reference oracle or a bounded TLA+/state-space model.

## Security properties and negative controls

| Property | Expected behaviour |
|---|---|
| Event ID replay | A repeated event ID never executes a second operation |
| Sequence monotonicity | Regressing or skipped event sequence is rejected |
| Generation monotonicity | Replace/revoke/rollback must move forward exactly one generation |
| Revocation finality | Rollback cannot reactivate a revoked tombstone |
| Re-enrollment after revocation | Requires recovery authority and a generation greater than the tombstone |
| Stale writes | Expected-generation mismatch leaves identity content unchanged |
| Recovery utility | Authorized recovery can reactivate a revoked identity using a new generation |
| Active rollback | Recovery may select a known historic digest only under a new generation |
| Overflow | `u64` generation overflow fails closed instead of repeating the maximum generation |

This reference model uses centralized ordering and in-memory state. It does not model concurrent writers, crash consistency, disk transactions, malicious storage, cross-replica divergence, key rotation, expiry, network partitions, remote authorization, or a cryptographic event log. Those are distinct future validation axes, not silently implied by passing a local trace.

## Paradigm shift

Treat registry integrity as a **temporal property of every state transition**, not just a signature check at enrollment or a high similarity score at query time. The decision must bind the expected generation and current revocation state, and the registry must preserve an explicit tombstone and auditable event order. Recovery should be a separately authorized forward transition rather than an implicit restore of old state.

For a production path, the next stages are:
1. Connect the policy to the production identity-evidence API and its actual durable registry adapter.
2. Execute process-level fault injection around write, journal commit, crash and recovery boundaries.
3. Model concurrent compare-and-swap generation conflicts and multi-replica stale views.
4. Bind a signed, versioned audit receipt to source revision, policy, registry epoch, and an append-only log checkpoint.
5. Compare the state model to an independent oracle and bounded model checker, including mutations that intentionally disable each guard.

RFC 9162's Certificate Transparency design is a useful precedent for append-only event evidence and consistency proofs: https://www.rfc-editor.org/rfc/rfc9162.html. But transparency logging alone does not prevent misissuance; it makes certain incorrect operations detectable by monitors and clients. Symthaea would likewise need both transition authorization and independent auditability, not a log hash alone.

## Proposed acceptance gates (targets, not achieved results)

- 100% agreement with the frozen positive/negative trace oracle on the declared suite.
- Zero accepted stale replays, duplicate-ID operations, unauthorized replacements, or rollback attempts against a revoked identity in the declared model.
- All legitimate controls pass: initial enrollment, authorized replacement, revocation, active rollback to a known digest under forward generation, and explicitly authorized recovery.
- Mutation controls must show each property test failing when its corresponding guard is deliberately removed. Compile failure or skipped execution does not count as a mutation kill.
- Exact source SHA, fixture digest, policy version and executed case count must be bound into an execution receipt.
- Later process tests should exercise interruption at every durable write boundary, replay after restart, stale replicas, and concurrent writers; no production security claim until those results exist.

These are contract targets. They are not measured performance, cryptographic guarantees, deployment security, or SOTA claims.

## Industry comparison

The relevant comparison is not vector-database QPS. It is whether identity lifecycle transitions remain correct under faults and adversarial orderings, with recovery availability measured alongside rejection.

The HDC/VSA retrieval track still needs fair task-level comparison against the standardized HDC and UCI benchmark sets proposed by Vergés et al. (2025), including conventional SVM/LDA/Ridge baselines: https://doi.org/10.1007/s10462-025-11181-2. Keep those task metrics separate from the registry's event-integrity metrics. A classifier can have high accuracy and still be unsafe if a revoked or unauthorized identity can re-enter the registry.

## Claim ceiling

An exact-head pass would establish only that this deterministic reference model produces the frozen outcomes. It would not prove production integration, durable crash safety, distributed consistency, real cryptographic identity, secure authorization, or resistance to all registry attacks.
