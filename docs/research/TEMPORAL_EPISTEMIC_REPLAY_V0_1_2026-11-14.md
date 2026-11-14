# EGDDM temporal epistemic replay v0.1 — design and implementation note

**Status:** implementation prototype and synthetic known-answer fixtures authored; not compiled, executed, or independently qualified.

## 1. Problem statement

A repository timeline is not the same thing as a decision-maker's knowledge timeline. A commit can have an effective date, a source may have been observed later, a particular actor may have received the observation at another time, and the event may only have been ingested into the engineering ledger afterwards.

EGDDM D2 therefore needs four distinct clocks:

1. **Effective / valid time:** when a fact or change applies to the subject system.
2. **Source observation time:** when the originating system or observer observed it.
3. **Actor availability time:** when a specific actor could have known it, with an explicit proof reference.
4. **Ledger recording time:** when the EGDDM ledger recorded the event.

Collapsing any two of these clocks creates retrospective leakage. In particular, a test result that is backdated to 09:00 but captured by the ledger at 11:30 must not automatically become part of the 10:00 ledger snapshot. It may be included in a retrospective reconstruction of the actor's knowledge at 10:00 only if a separately grounded actor-availability record says the actor could access it by then.

## 2. Query semantics

A replay query pins:

- repository identity;
- configuration-world digest;
- actor identity;
- knowledge cutoff;
- effective-time point to evaluate; and
- optionally, a ledger-recorded cutoff.

The last field distinguishes two different questions:

- **Actor-knowledge reconstruction:** what evidence was available to this actor by time T, including evidence captured into the ledger later, when its actor-availability claim is supported.
- **Ledger-as-recorded reconstruction:** what the ledger itself contained by time T, additionally excluding events recorded afterwards.
- **Retrospective restatement:** what the current event history says was effective at an earlier valid-time point, using only events available to the actor by the later knowledge cutoff.

A historical decision audit should normally retain both the original decision-time reconstruction and a later restated view. The latter must not overwrite the former.

## 3. Event contract

The v0.1 Rust model supports four append-only operations:

- `assert`: introduces a fact with explicit validity completeness and optional effective bounds.
- `invalidate`: marks one earlier assertion stale from an explicit effective time.
- `supersede`: marks one earlier assertion superseded from an explicit effective time.
- `propose`: records a proposal without mutating observed facts.

Each event carries a stable event ID, contiguous sequence, previous hash, event hash, source observation time, ledger recording time, per-actor availability records, exact repository revision/tree pins, configuration-world digest, optional validity/effective bounds, optional target event, and a JSON payload.

Transitions must target an earlier assertion for the same fact and configuration world. An actor receiving a transition must also have an availability record for the target assertion, and the target cannot become available to that actor after the transition. An assertion may receive only one terminal invalidation/supersession transition. Corrections are new events; history is not edited in place.

## 4. Integrity and authority boundaries

The event SHA-256 chain binds the schema version, ledger identity, repository identity, event identity, sequence, previous hash, and event body. A serializable replay checkpoint carries a separate digest over its cursor, query, processed history and projection state.

These digests detect accidental or unsophisticated content mutation when the digest itself is retained. They do **not** authenticate who supplied an event, whether an actor actually received it, whether a Git commit/tree/blob exists, whether a timestamp is trustworthy, or whether the ledger is complete. An attacker who can rewrite the entire unsigned ledger or checkpoint can recompute its hashes. Production use requires independently verified source receipts, signed or externally witnessed checkpoints, clock/availability provenance, exact Git-object authentication and an independently administered append-only custody boundary.

The v0.1 replay projection reports *temporal eligibility*, not proof of decision authority or source truth. It must not promote evidence to `ci_passed` or `independently_qualified`.

## 5. Deterministic replay and checkpoints

The crate exposes:

- `validate_ledger(&EventLedger)`;
- `append_event(ledger, event_id, body)` for atomic, validated in-memory appends with rollback on failure;
- `start_replay_cursor(prefix, query)`;
- `resume_replay_cursor(cursor, suffix)`;
- `serialize_replay_cursor(cursor)` and `deserialize_replay_cursor(bytes)`;
- `replay_ledger(ledger, query)` as a clean-from-genesis reference; and
- `replay_incremental(prefix, suffix, query)` for a validated prefix followed by an append-only suffix.

A cursor fixes its query at creation. Resuming with a changed actor, world or cutoff requires a new cursor. The cursor validates sequence continuity, chain linkage, event identity, append timestamps and correction target rules before applying a suffix. The intended equivalence property is byte-identical canonical projection output between a resumed cursor and clean replay for the same frozen ledger and query.

The current prototype retains the prefix event index in a serializable cursor. It is a correctness prototype, not yet a compact production checkpoint or asymptotically optimized storage engine. Cursor hashes are not a substitute for external signatures.

## 6. CLI

From the standalone crate directory:

    cargo run -- temporal-ledger ../../docs/development-map/fixtures/temporal-event-ledger-v0.1.json

    cargo run -- temporal-replay \
      ../../docs/development-map/fixtures/temporal-event-ledger-v0.1.json \
      ../../docs/development-map/fixtures/temporal-replay-actor-knowledge-v0.1.json

The JSON contracts are:

- [Temporal event-ledger schema](temporal-event-ledger-v0.1.schema.json)
- [Temporal replay-query schema](temporal-replay-query-v0.1.schema.json)

Synthetic known-answer inputs live in `docs/development-map/fixtures/`:

- `temporal-event-ledger-v0.1.json`
- `temporal-replay-actor-knowledge-v0.1.json`
- `temporal-replay-ledger-as-of-v0.1.json`
- `temporal-replay-restated-v0.1.json`

The fixture intentionally includes: a decision-relevant assertion, a retroactive invalidation observed and made available after the original knowledge cutoff, a late-ingested event with an earlier actor-availability claim, and an unexecuted proposal. It is synthetic and does not represent a real repository, Mycelix authorization, CI event or test outcome.

## 7. Adversarial properties to qualify

The authored unit tests cover:
- atomic event append, deterministic sequence/hash assignment and rollback of invalid candidates;

- event hash tampering and event reordering;
- event sequence gaps and duplicate IDs;
- repository and configuration-world mismatch;
- late capture versus actor knowledge versus ledger-recorded cutoff;
- correction visibility changing only after actor availability;
- retroactive effective-time correction without rewriting an earlier knowledge view;
- missing transition targets, cross-fact/world mismatch and duplicate terminal transitions;
- target-assertion availability preceding transition availability;
- inclusive validity boundaries and incomplete validity as indeterminate;
- proposal/observation noninterference;
- serializable checkpoint corruption rejection; and
- suffix replay equivalence with a clean rebuild.

These are authored test definitions, not test results. No compile, test execution, CI pass, benchmark score or cryptographic source qualification is claimed.

The next adversarial increments should add event-stream omission detection using external high-watermarks, signed/witnessed checkpoints, concurrent append conflict tests, idempotent duplicate-delivery semantics, clock-skew/uncertainty intervals, actor-specific revocation, multi-actor partial-order replay, cross-revision ancestry verification, and randomized/property-based event-sequence generation.

## 8. DynaMapBench D2 evaluation

D2 must be evaluated separately from graph structure and retrieval:

| Metric | Definition |
|---|---|
| As-of state accuracy | Correct fact lifecycle and payload at frozen valid-time / knowledge-time queries |
| Actor-knowledge leakage | Events accepted before actor availability |
| Ledger hindsight leakage | Late-ingested events accepted in ledger-as-recorded queries |
| False-current rate | Invalidated/superseded evidence remaining eligible |
| History retention | Earlier event and decision views remain reconstructible after correction |
| Correction latency | Time between source correction availability and updated projection |
| Replay determinism | Byte-identical output for repeated identical frozen inputs |
| Incremental equivalence | Resumed checkpoint output equals clean replay |
| Scope isolation | Evidence from another repository/configuration world is excluded |
| Unknown handling | Missing clocks, incomplete applicability or unavailable authority stays indeterminate/unknown |

Prospective gate targets:

- 100% rejection of planted future-availability and tampered-chain controls;
- 100% correct lifecycle outcomes on frozen deterministic known-answer cases;
- 100% incremental/clean replay equivalence across the frozen fixture suite;
- zero false-current outcomes on explicit invalidation/supersession controls;
- at least 95% precision and recall on independently adjudicated real temporal-evidence cases, with confidence intervals and per-repository breakdowns.

These targets are research goals, not achieved measurements or industry standards. The 95% target requires an independently curated corpus and at least two-person adjudication. Synthetic fixture success alone does not support a performance claim.

## 9. Next work after v0.1

1. **Authenticated capture:** obtain commit-to-tree and tree-to-blob bindings from a separately controlled Git collector; verify signatures/receipts and prove non-truncation.
2. **Causal correction semantics:** distinguish source correction, effective-time change, decision invalidation and policy revocation as separate typed event families.
3. **Partial-order ledger:** support multiple actors, event causality and concurrent appenders without relying on one globally trusted wall clock.
4. **Clock uncertainty:** use bounded time intervals or monotonic sequence evidence when clock synchronization is not trustworthy.
5. **Independent D2 oracle:** implement expected-state generation separately from the candidate reducer and keep gold custody outside the candidate context.
6. **Outcome loop:** link each decision to predicted impacts, selected evidence, action/defer result, independently accepted capability change, regressions and reviewer effort.

The scientific success criterion is not that a replay engine can reproduce its own input. It is that an independent oracle can establish what was knowable at a particular decision point, and that recommendations based on that reconstruction improve measured engineering outcomes without unauthorized effects.
