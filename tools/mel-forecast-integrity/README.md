# MEL Forecast Integrity — reference evaluator

This standalone Rust crate is an executable integrity slice for predictive transparency described by issue #10190.

## Run

From repository root:

    cargo run --manifest-path tools/mel-forecast-integrity/Cargo.toml -- --seal tools/mel-forecast-integrity/fixtures/precommitted-visible-forecast-unsealed-v1.json /tmp/mel-forecast-sealed.json
    cargo run --manifest-path tools/mel-forecast-integrity/Cargo.toml -- /tmp/mel-forecast-sealed.json
    cargo test --manifest-path tools/mel-forecast-integrity/Cargo.toml

## Checks

- input observation precedes a forecast and the forecast binds the exact input event;
- forecast issue precedes its target window;
- preview display occurs after issue and before target execution;
- collaborator disposition requires a prior display;
- forecast supersession and invalidation retain the old forecast but block its later realization;
- actual realization is bound to a still-pending forecast and falls within its target and expiry windows;
- event IDs, sequence, monotonic timestamps and SHA-256 chain are replay checked;
- report separates input-to-issue, preview lead time and input-to-realization latency;
- an undisplayed forecast is never counted as human-visible even if it is later realized.

## Claim boundary

This evaluator validates the internal ordering and digest references in a supplied trace. It does not open prediction/score/audio bytes, compare predicted musical content with the realization, authenticate an actor, verify a signed display receipt, prove a display was visible or understood, or measure human anticipation. It is not a live scheduler or CPAL adapter. These are explicit next integration and human-study responsibilities.
