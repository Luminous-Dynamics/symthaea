# Multi-turn audio-edit session ledger (MMAE-TRACE-001)

A dependency-free Rust evaluator for append-only, content-addressed edit sessions.

## Why a session ledger?

Single-case edit scoring does not establish that an agent maintained every earlier user instruction across a conversation. A later turn can regress an earlier edit, act on a stale output, or implicitly discard a preservation requirement.

The ledger treats each turn as an append-only state transition. Constraints persist until an explicit same-role supersession event replaces them.

## Events

- SessionStarted binds initial input manifest, frozen context and initial output.
- UserTurnOpened records a contiguous user turn and instruction-batch digest.
- ConstraintAdded binds a stable instruction/preservation ID to its scope and rubric profile.
- ConstraintSuperseded explicitly replaces one active constraint with another active constraint of the same role.
- EditExecuted binds action/system receipts, exact parent output and new output.
- ExecutionAttemptFailed retains failed/unsupported/abstained attempts.
- ObservationAttached binds a decision to one active constraint and exact current output.
- SessionFinalized binds the final output, active-ledger digest and exact sorted active-constraint ID set; replay checks that set against derived state.

Events are append-only, indexed contiguously and chained through previous-event digests. This crate validates digest syntax, uniqueness and chain continuity; canonical serialization and cryptographic digest recomputation belong to the event producer.

## Final qualification

Finalization requires current-output evidence for every active constraint. A final session is:
- **QUALIFIED** only when every active instruction and preservation decision is PASS;
- **NONCOMPLIANT** when at least one active constraint is FAIL;
- **INDETERMINATE** when at least one active constraint is INDETERMINATE and none is FAIL.

Missing final evidence rejects finalization. A superseded constraint is excluded from the active set only when its explicit replacement event is valid. Failed attempts remain in the report even when later retries succeed.

## Integration

The ledger consumes existing VMEA rubric decisions. It does not invent new PER, F0-CORR, SIM or MOS metrics. For vocal editing, use the existing VELB locality, VIA interval and VCCR trajectory receipts; attach their qualified decisions to active constraint IDs.

The current v1 event model is a validation/evidence layer, not a conversational agent, tool router or audio editor.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/audio-edit-session-assurance/Cargo.toml
cargo +1.96.0 test --manifest-path tools/audio-edit-session-assurance/Cargo.toml
~~~

## Claim boundary

Synthetic tests qualify ledger replay, event order, output chaining, supersession and final constraints only. They do not establish that user intent was parsed correctly, that upstream evaluators are accurate, or that any audio operation sounds good.
