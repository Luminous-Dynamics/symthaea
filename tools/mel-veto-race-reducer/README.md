# MEL veto/pending-output race reducer

Dependency-free Rust reference reducer for replaying the control-plane and callback-admission boundary in MEL-RECOVERY-001.

## Run

The workflow stages this crate outside the repository Cargo workspace because workspace metadata is independently broken by an unrelated missing member manifest.

```sh
cargo fmt --check --manifest-path tools/mel-veto-race-reducer/Cargo.toml
cargo test --locked --manifest-path tools/mel-veto-race-reducer/Cargo.toml
```

## Event model

- `Authorize`: binds session/turn, decision, response-plan digest, current output epoch, control sequence, authorization time and expiry.
- `ProducerStarted`: captures the authorization, plan digest, epoch and runner-monotonic start time **before rendering begins**. A producer cannot start before authorization, at/after expiry, under a revoked authorization or with a superseded control sequence.
- `RenderCompleted`: binds completed frame range and completion time to the producer's captured authorization and epoch. A producer that genuinely started before veto/expiry may finish later, but it cannot retag output to the new epoch.
- `Enqueue`: records application queue admission and its runner-clock time independently from render completion. Expired, non-revoked authorizations cannot enqueue; post-veto late enqueue remains visible as stale work for explicit callback discard/accounting.
- `Veto`: validates participant/turn authority, revokes authorization and advances the epoch exactly once.
- `ExpireAuthorization`: invalidates the same authorization at or after its declared expiry and advances the epoch. Queued old-epoch frames are stale on subsequent callback checks.
- `CallbackEpochCheck`: classifies an enqueued frame range as callback-admitted or stale-discarded using the epoch observed at the admission check.

The reducer additionally:
- preserves pre-veto/pre-expiry callback admissions as admitted/in-flight and never retroactively calls them cancelled;
- uses half-open frame ranges and rejects overlapping output intervals so the frame denominator cannot be double-counted;
- reports producer start/completion/in-flight, rendered-but-not-enqueued, queued, callback-admitted, stale-discarded and pending frame totals separately;
- applies exact duplicate events idempotently and rejects conflicting event-ID reuse;
- rejects out-of-order traces, cross-turn/unauthorized vetoes, plan/epoch substitution, invalid producer/enqueue time, premature expiry events, duplicate producer completion, duplicate enqueue, duplicate callback terminal disposition and epoch wraparound.

## Verification boundary

This is a deterministic **reference reducer/model**, not a test of the production CPAL callback or its atomic memory ordering. Trace order represents the linearization order of modeled producer-start, veto/expiry, render-completion, enqueue and callback-epoch-observation events. Production integration must persist/reconstruct this order (or use callback-observed epochs as the authority for classifying frames).

All `*_at_ns` values in one reducer trace are assumed to be normalized to the same runner-monotonic clock domain and origin bound by the upstream decision/grounding receipts. The reducer does not perform cross-clock conversion or authenticate the origin; the live adapter must reject mismatched domain/origin receipts before constructing this trace.

The crate does not prove physical silence, DAC cancellation, acoustic latency, human musical usefulness or artistic quality. Those require separate device, calibrated loopback and musician cohorts. It has no audio callback, thread scheduler, I/O or timing-performance claim.
