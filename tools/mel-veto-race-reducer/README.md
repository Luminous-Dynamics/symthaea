# MEL veto/pending-output race reducer

Dependency-free Rust reference reducer for replaying the control-plane and callback-admission boundary in MEL-RECOVERY-001.

## Run

The workflow stages this crate outside the repository Cargo workspace because workspace metadata is independently broken by an unrelated missing member manifest.

```sh
cargo fmt --check --manifest-path tools/mel-veto-race-reducer/Cargo.toml
cargo test --locked --manifest-path tools/mel-veto-race-reducer/Cargo.toml
```

## Event model

- `Authorize`: binds session/turn, decision, response-plan digest, current output epoch, control sequence and expiry.
- `ProducerStarted`: captures the authorization, plan digest and epoch **before rendering begins**. A new producer cannot start from a revoked authorization or a stale epoch.
- `RenderCompleted`: binds completed frame range to the producer's captured authorization and epoch. A producer that genuinely started before veto may finish after veto, but it cannot retag its output to the new epoch.
- `Enqueue`: records application queue admission independently from render completion.
- `Veto`: validates participant/turn authority, revokes authorization and advances the epoch exactly once.
- `CallbackEpochCheck`: classifies an enqueued frame range as callback-admitted or stale-discarded using the epoch observed at the admission check.

The reducer additionally:
- preserves pre-veto callback admissions as admitted/in-flight and never retroactively calls them cancelled;
- uses half-open frame ranges and rejects overlapping output intervals so the frame denominator cannot be double-counted;
- reports producers started/completed/in-flight, rendered-but-not-enqueued, queued, callback-admitted, stale-discarded and pending frame totals separately;
- applies exact duplicate events idempotently and rejects conflicting event-ID reuse;
- rejects out-of-order traces, cross-turn/unauthorized vetoes, authorization/plan/epoch substitution, duplicate producer completion, duplicate enqueue, duplicate callback terminal disposition and epoch wraparound.

## Verification boundary

This is a deterministic **reference reducer/model**, not a test of the production CPAL callback or its atomic memory ordering. Trace order represents the linearization order of modeled producer-start, veto, render-completion, enqueue and callback-epoch-observation events. Production integration must persist/reconstruct this order (or use callback-observed epochs as the authority for classifying frames).

The crate does not prove physical silence, DAC cancellation, acoustic latency, human musical usefulness or artistic quality. Those require the separate device, calibrated loopback and musician cohorts. It has no audio callback, no thread scheduler, no I/O and no timing-performance claim.
