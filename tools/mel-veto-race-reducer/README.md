# MEL veto/pending-output race reducer

Dependency-free Rust model for replaying the control-plane and callback-admission boundary in MEL-RECOVERY-001.

## Run

```sh
cargo fmt --check --manifest-path tools/mel-veto-race-reducer/Cargo.toml
cargo test --locked --manifest-path tools/mel-veto-race-reducer/Cargo.toml
```

## What it does

- Replays an ordered trace of authorization, render, enqueue, veto and callback epoch-observation events.
- Binds rendered frames to the authorization, response-plan digest and epoch captured before rendering.
- Allows a producer already in flight to finish/enqueue under its original (now revoked) epoch; it does not silently retag this work.
- Classifies each enqueued frame range exactly once as callback-admitted or stale-discarded using the epoch actually observed by the callback.
- Preserves frames admitted before a later veto as admitted/in-flight. It never retroactively calls them cancelled.
- Applies exact duplicate events idempotently and rejects conflicting event-ID reuse.
- Rejects out-of-order traces, cross-turn vetoes, invalid authority, epoch regressions, unauthorized plan substitution and duplicate terminal frame dispositions.

## Important limits

This is a deterministic **reference reducer/model**, not a test of the production CPAL callback or its atomic memory ordering. Trace order represents the linearization order of the modeled control updates and callback epoch reads. Production integration must capture enough evidence to reconstruct that order (or use the callback's observed epoch as the authority for classifying a frame). This crate does not prove physical silence, DAC cancellation, acoustic latency, or artistic quality. Those require the separate device, loopback and musician cohorts.

The model is intentionally dependency-free. It has no audio callback, no thread scheduling, no I/O, and no timing-performance claim.
