# Irregular Event Contract (v0.1)

This candidate-side Rust crate preserves native time-series event semantics for
irregular, missing, delayed and asynchronous datasets.

It distinguishes:
- `event_time_us`: when the phenomenon happened;
- `available_at_us`: when the candidate could first access the record;
- `source_id` and `source_sequence`: source-local ordering and identity;
- `modality_id` and `entity_id`;
- observed, missing and censored measurements;
- fixed-point value scale, unit code and immutable source digest.

An observation with `available_at_us < event_time_us` is rejected. Prepublished
planned information must use the separate `KnownFutureCovariate` type, which
is available only when it was published no later than the forecast origin and
its validity interval covers the target timestamp.

`observations_available_at` filters by availability time and preserves source
vector order. It does not sort by event time, interpolate, resample, forward
fill, zero-impute or regularize. Those transformations must be separately
versioned and charged by the adapter.

The crate contains no future target or evaluator label. It does not define
benchmark-specific train/validation/test splits: those must remain pinned to
each benchmark's official protocol.

Fifteen Rust test functions are authored for delayed-observation non-leakage,
out-of-order event times, missingness preservation, censored intervals,
source-local sequence uniqueness, legitimate future-covariate availability,
and input validation. No Cargo build, rustfmt, Clippy, tests or dataset adapter
has run for this draft.
