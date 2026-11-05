# Melothaea action-effect receipt ledger v1

This isolated Rust crate implements the deterministic conformance boundary proposed by issue 10382.

It consumes already-committed decision and downstream receipts. It does not run the turn-taking classifier, mutate MuseStream, own the output epoch clock, execute a CPAL callback, read Studio storage, authenticate consent receipts, or control audio hardware.

## Action semantics

- WAIT is a first-class decision attached to an exact input chunk. It carries no response plan, forecast, effect window, or output evidence. Turn finalization rejects missing chunks; absence is never silently interpreted as WAIT.
- RESPOND must bind the exact authorized plan digest, commitment/forecast, output epoch, expiry, and musical response window. Its downstream artifact digest is then carried unchanged through evidence stages.
- CONTINUE requires the active commitment and an exact before/after state equality check. Cursor, queued notes, pending application frames, and output epoch cannot change under a Continue receipt.
- YIELD advances the application generation epoch once, preserves the already-emitted note cursor, and accounts exactly for discarded unconsumed notes and pending application-owned frames.
- VETO is unavailable through the ordinary reducer entry point. It can be applied only with an explicit `VetoAuthorityVerifier`, which receives the exact session, turn, participant, control sequence and opaque signed-receipt reference. The reducer does not implement signature/authentication itself; the production verifier must bind this to the trusted identity/consent system.

## Evidence ladder

A response artifact can accumulate separate observations:

1. Applied at note queue.
2. Enqueued to application output with the exact number of application frames appended.
3. Observed/admitted by a CPAL callback with the exact number of application frames consumed.
4. CPAL predicted playback (requires a clock-bridge digest).
5. Independently loopback-observed audio (requires a separate evidence digest).

CPAL callback observation is not predicted DAC time. Predicted playback is not independent acoustic measurement. Loopback is a separate clock domain; timestamps from different domains are never subtracted without a qualified mapping.

The ledger updates pending application-frame counts only from exact enqueue and callback receipts; callback consumption cannot exceed the pending queue, and rejected clock/frame receipts do not partially mutate state. The ledger does not infer physical silence from YIELD or VETO. It reports only queue/frame invalidation receipts supplied to it.

## Run

From this directory, run cargo fmt --check and cargo test --locked.

The repository workflow stages this crate outside the root Cargo workspace and runs formatting and tests against the exact PR head.

## Integration order

1. Adapt the streaming-turntaking prediction from issue 10309 to DecisionReceipt while preserving the exact observed-through frame.
2. Feed actual MuseStream immediate-update discard counts into YieldReceipt; do not synthesize these counts.
3. Feed the real output-epoch transition and application-queue stale-frame counters into YieldReceipt or VetoReceipt.
4. Bind RespondApplied to the committed plan, forecast, and output artifact digest produced by MEL-013A; independently read and re-hash durable bytes there.
5. Feed application enqueue and callback receipts from the non-real-time adapter; never log, allocate, or block inside the CPAL callback.
6. Attach runner-to-CPAL clock-fit evidence only when a verified fit bundle exists.
7. Attach independent loopback captures only when actually acquired and independently digested.

This crate is a reference reducer and synthetic conformance target. It does not prove production integration, model correctness, musical value, performer agency, or hardware behavior.
