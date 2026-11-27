# MON-036 — Recovery Evidence Continuity Contract v1

**Profile:** \`mon036-recovery-evidence-continuity-input-v1\`  
**Status:** offline event-ledger verifier and selective claim-disposition evaluator implemented; signed change-feed adapters pending.

## Research question

Can a recovery system tell whether an already-qualified claim remains usable after its supporting environment changes, even when the claim's own expiration time has not elapsed?

Examples include changes to PostgreSQL cluster/system identity, timeline history, replication slot, publication/filter, Debezium connector configuration or image, Kafka topic configuration, backup repository/retention policy, restore verifier, signer authority, application dependency inventory or recovery policy.

MON-024 checks evidence freshness and shared recovery boundaries. Freshness alone is insufficient: an unexpired receipt can depend on a connector, timeline, verifier or authority that has changed since the receipt was produced.

MON-036 operationalizes a recovery-specific slice of the CEB Evidence Continuity and Claim-Support Closure designs. It does not create a competing general repository graph, signer or receipt system.

## Model

Each claim binds a stable claim/evidence ID, runtime-independent evidence grade, complete support closure, event-ledger watermark sequence and digest, observation/expiry time, and exact dependency IDs, revisions and SHA-256 digests.

The event ledger is a hash-chained, ordered append-only sequence. Each event includes a sequence, unique event ID, kind, optional target dependency, before/after revisions and digests where applicable, occurrence time, previous digest and its own digest. The ledger declares a completeness watermark, head digest, capture time and expiry.

Current dependency states are supplied separately. For every dependency supporting a claim, the verifier checks that the event tail after the claim watermark explains the current state. A changed snapshot with no matching event is unresolved; a chain contradicting the supplied state is also unresolved or invalid according to the exact failure.

## Dispositions

| Disposition | Rule | Consequential use |
|---|---|---|
| \`retain\` | No post-watermark event affects a support dependency; dependencies match current state; ledger is complete and fresh | Existing claim may remain usable within its original scope and expiry |
| \`revalidate\` | A supported dependency/verifier changed, or claim evidence expired, and history coherently explains the state | Do not use old claim as current; perform a fresh check and issue new evidence lineage |
| \`invalidate\` | Authority revoked, scope changed, contradiction recorded, event chain/digest malformed, impossible predecessor or ambiguous duplicate claim identity | Withdraw from current use; a new claim is required |
| \`unresolved\` | Missing/incomplete ledger, missing state, unknown-impact event, untracked drift, watermark gap, expired ledger or incomplete support closure | Abstain and reconcile before any mutation or new claim |

Precedence: invalidate, then unresolved, then revalidate, then retain.

A dependency change does not invalidate every claim. Only claims whose support closure includes that dependency are revalidated. This is the selective-invalidation property.

## Event semantics

- \`dependency_changed\`: configuration/runtime identity changed; matching claims require revalidation.
- \`verification_changed\`: verifier/evaluator/method revision changed; claims depending on that method require revalidation.
- \`authority_revoked\`: targeted authority is revoked; supporting claims are invalidated.
- \`scope_changed\`: scope of a premise changed; affected claims are invalidated.
- \`contradiction\`: material contradiction is recorded; affected claims are invalidated.
- \`unknown_change\`: impact is unknown for the targeted dependency or globally when no dependency is named; affected claims are unresolved.

Events after the claim's \`observed_through_sequence\` are considered. A nonzero watermark must identify the exact event digest and that event must strictly precede the claim observation time. Sequence zero requires an empty digest.

## Integrity and freshness

- V1 requires full event history from sequence 1; no checkpoint/truncation or suffix-only input is accepted.
- Sequences are contiguous and strictly increasing.
- Each \`previous_digest\` matches the prior event; each event digest is recalculated over the v1 payload with UTC RFC3339Nano time.
- \`through_sequence\` and \`head_digest\` match the final event (or sequence 0 / empty digest for genesis).
- Event IDs are unique. Events cannot be future-dated or occur after declared ledger capture.
- Claim watermark matches its exact event. Ledger capture cannot predate claim observation.
- Ledger snapshot must still be current at evaluation time.
- Claims require runtime-independent grade and complete support closure; structural/synthetic claims are unresolved.
- Current dependency state IDs, revisions and digests must be unambiguous. Duplicate claim IDs invalidate every ambiguous record.

Hash chaining proves internal consistency, not authenticity. Change feeds need independent signatures/custody and authenticated source adapters. MON-026 remains the custody authority; MON-036 does not self-authorize the ledger.

## Resource limits

100,000 events; 10,000 claims; 250,000 current states; 2,000 support bindings per claim; 256-byte identifiers; 32 MiB maximum CLI input.

The evaluator indexes events by dependency, so unrelated event history is not rescanned for every claim. It remains an in-memory evaluator, not a persistent event database.

## CLI

\`\`\`sh
cd benchmarks/frontend/monitoring/sfeb-mon036/continuity
go run ./cmd/qualify --input INPUT-example.json
bash CHECKS.sh
\`\`\`

The CLI prints a structured result and exits 0 when the input is syntactically evaluated. Exit 0 does not mean every claim is retained or that recovery is qualified. Invalid JSON exits 65; missing \`--input\` exits 64.

## Adversarial controls

Tests cover unrelated changes, dependency/verifier updates, revocation, scope change, contradiction, targeted/global unknowns, incomplete ledger, sequence gaps, digest corruption, wrong claim watermark, event after capture, impossible watermark chronology, untracked state drift, state/event-tail mismatch, claim expiry, missing state, incomplete closure, non-runtime evidence and ambiguous duplicate claim IDs.

## Limits

No live PostgreSQL, Debezium, Kafka, pgBackRest, cloud control-plane or signer feed is connected. This package does not perform revalidation; it determines whether old evidence may be reused unchanged. The hash chain is not a signature, transparency log or proof of feed completeness. V1 requires full ledger history from genesis; trusted checkpoint anchoring is future work. The evaluator cannot prove that a support closure or dependency inventory is complete. No production recovery claim follows from \`retain\` alone; MON-024, MON-026 and domain verifiers remain required.
