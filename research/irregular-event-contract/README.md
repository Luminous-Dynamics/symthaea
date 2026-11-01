# Irregular Event Contract (v0.1)

This candidate-side Rust crate preserves native time-series event semantics for
irregular, missing, delayed and asynchronous datasets.

It distinguishes:
- `event_time_us`: when the phenomenon happened;
- `available_at_us`: when the candidate could first access the record;
- `source_id` and `source_sequence`: source-local ordering and identity;
- `modality_id` and `entity_id`;
- observed, missing and censored measurements;
- fixed-point value scale, unit code and immutable source digest. Text, audio, image and embedding bytes can remain in an external immutable object store and be referenced by payload digest, byte length and encoding ID.

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

Thirty-five Rust test functions are authored (16 event contract, 11 bitemporal revision, 8 purged-split tests) for delayed-observation non-leakage,
out-of-order event times, missingness preservation, censored intervals,
source-local sequence uniqueness, legitimate future-covariate availability,
and input validation. No Cargo build, rustfmt, Clippy, tests or dataset adapter
has run for this draft.


## Microsecond bitemporal source revisions

The same candidate-side crate includes an append-only event revision ledger.
Event time is the valid/world-time coordinate; `recorded_at_us` is the
transaction/knowledge-time coordinate. Each event has a contiguous revision
sequence. Corrections append `Supersede` records and removals append
`Retract` records; old revisions remain in the digest-bound history.

Transactions are atomic and share one recorded-at timestamp; committed
transaction timestamps strictly increase. A record cannot be committed before
its declared availability. Corrections preserve entity/modality/source identity
and must target the latest active revision. As-known queries apply both a
valid-time cutoff and a knowledge-time cutoff, so a later correction/retraction
cannot rewrite an earlier decision view.

The microsecond ledger is a precision extension for streaming sensor events; it
does not replace Broca's existing day-granularity bitemporal source ledger
(`crates/domains/symthaea-broca/src/memory_bitemporal.rs`, PR #9109) or its
alternative-justification composition (PR #9292). This adapter layer only
records source-event history and does not maintain derived proofs.

The 11 additional bitemporal test functions cover late correction and
retraction visibility, independent valid/knowledge clocks, source-identity
preservation, transaction atomicity, revision sequencing, target rules,
availability-before-commit rejection, source sequence collision rejection,
and deterministic history digests.


## Leakage-controlled chronological splits

The `splits` module supplies `purged_chronological_split`. It partitions
forecast windows per entity by distinct forecast origin (tied origins stay in
one split), then purges any training target interval crossing into validation
and any validation target interval crossing into test. An optional embargo is
applied at both boundaries. Historical input windows may overlap; target labels
may not cross the next partition boundary.

This is intended to complement, not silently replace, the official benchmark
split during reproduction. Report both official-path results and the
leakage-controlled split when the official harness normalizes using full-record
statistics or has overlapping forecast targets across split boundaries.
