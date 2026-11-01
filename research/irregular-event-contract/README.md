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

Forty-four Rust test functions are authored (16 event contract, 11 bitemporal revision, 8 purged-split, 9 train-only-scaler tests) for delayed-observation non-leakage,
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


## Train-only standardization for clean comparisons

The `scaler` module fits per-entity/per-feature means and sample standard
deviations from values attached only to the generated split plan's training
windows. It uses deterministic Welford moments. Raw observation IDs deduplicate
observations reused by overlapping training histories; conflicting values for
the same identity fail closed. Validation, test and purged-window numeric
values are skipped before the scaler reads them. The scaler also revalidates
the plan's per-entity boundaries and target-purge assignments, so a caller
cannot simply relabel a later window as training.

Missing values must be omitted by the adapter (or represented separately); no
imputation is performed. A zero or one-sample standard deviation is treated as
zero and transform subtracts the mean without division. Fitted counts, means
and standard deviations are available for audit.

This clean scaler is for the leakage-controlled comparison. The official
IMM-TSF reproduction must preserve and disclose the released full-record
normalization path; do not call clean-split results official leaderboard
reproductions.


## Time-IMM CSV adapter

The `time_imm` module reads one per-entity `time_series.csv` or `text.csv`
and emits the event contract above.

- Requires exact feature-column mapping and stable modality/unit/decimal-scale
  IDs. Unexpected or unconfigured columns are rejected, not silently ignored.
- Binds exact source-file bytes with a file SHA-256; each emitted event also
  binds a canonical parsed-row digest. Text payload bytes are separately
  content-addressed without trimming their content.
- Supports RFC3339 timestamps directly. Naive timestamps require an explicit
  fixed offset or IANA timezone. DST-ambiguous and nonexistent local times fail
  closed. The timezone configuration receives its own digest.
- Requires an explicit availability policy: synchronous-at-event-time
  assumption or a declared fixed delay. The policy receives its own digest.
  Neither option is evidence of physical acquisition or arrival time.
- Numeric decimal strings are converted to configured fixed-point scale in
  [-9, 0], rounding nearest with ties away from zero. A half-unit quantization
  error bound is returned for every configured feature.
- Numeric empty/NA/NaN cells remain Missing records; text payloads remain
  opaque digest/length/encoding references.

The concrete Time-IMM EPA-Air/Bexar release has the documented headers
`date_time,record_id,temp,pm2_5,aqi,ozone` for numeric observations and
`date_time,record_id,text` for text. The adapter requires the caller to supply
the actual timezone and availability policy; it does not infer either from
these CSV columns.

Nine source-authored adapter tests cover timezone and DST, missing numeric
cells, explicit availability delay, text payload hashing/whitespace,
fixed-point quantization, file/row provenance, record identity and schema
failures. The crate now has 53 source-authored test functions total. No build or
test has run.
