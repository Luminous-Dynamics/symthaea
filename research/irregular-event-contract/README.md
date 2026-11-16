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

Eighty-one Rust test functions are authored (28 event/bitemporal contract, 8 purged-split, 10 train-only-scaler, 10 Time-IMM adapter, 25 source-watermark tests) for delayed-observation non-leakage,
out-of-order event times, missingness preservation, censored intervals,
source-local sequence uniqueness, legitimate future-covariate availability,
and input validation. The standalone crate was first exercised by a pinned Rust 1.96 workflow: dependency resolution and changed-module formatting succeeded, then whole-crate compilation exposed four real type errors. Those have been corrected. The latest exact-head test and Clippy results remain pending; no test pass or dataset score is claimed.


## Microsecond bitemporal source revisions

The same candidate-side crate includes an append-only event revision ledger.
Event time is the valid/world-time coordinate; `recorded_at_us` is the
transaction/knowledge-time coordinate. Each event has a contiguous revision
sequence. Corrections append `Supersede` records and removals append
`Retract` records; old revisions remain in the digest-bound history.

Transactions are atomic and share one recorded-at timestamp; committed
transaction timestamps strictly increase. A record cannot be committed before
its declared availability. Corrections preserve entity/modality/source/source-sequence identity
and must target the latest active revision. The ledger maintains revision-ID,
per-event-chain, latest-revision, active-source-sequence and entity-to-event
indexes. Single appends are preflighted without cloning the full log; atomic
batches clone once. As-known queries binary-search each event's transaction-time
ordered revision chain, and an explicit consistency audit rebuilds the indexes. As-known queries apply both a
valid-time cutoff and a knowledge-time cutoff, so a later correction/retraction
cannot rewrite an earlier decision view.

The microsecond ledger is a precision extension for streaming sensor events; it
does not replace Broca's existing day-granularity bitemporal source ledger
(`crates/domains/symthaea-broca/src/memory_bitemporal.rs`, PR #9109) or its
alternative-justification composition (PR #9292). This adapter layer only
records source-event history and does not maintain derived proofs.

The 12 additional bitemporal test functions cover late correction and
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
  A canonical adapter-config digest also binds dataset/entity identity and the
  feature-sorted mapping of numeric columns to modality, unit and scale.
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

Ten source-authored adapter tests cover timezone and DST, missing numeric
cells, explicit availability delay, text payload hashing/whitespace,
fixed-point quantization, file/row provenance, record identity and schema
failures. This contributes to the 80-test crate inventory above; hosted compile/test qualification is pending.



## Source progress watermarks and qualified completeness

The `watermarks` module keeps source completeness distinct from event arrival
and sensor missingness. Each source progress revision carries entity/modality/
source identity, explicit stream generation and progress sequence, an exclusive
event-time watermark, transaction time and a nonzero policy digest. The
watermark is supplied by the source or named adapter policy; it is never
inferred from the maximum event timestamp observed.

Completeness over a half-open interval is Complete only when every required
source has advanced its visible watermark to the exclusive interval end, or
has explicitly declared end-of-stream. Missing progress remains Incomplete.
A query uses a knowledge-time cutoff, so later source progress cannot leak
backward into historical completeness decisions.

An event with event_time strictly below a previously recorded exclusive
watermark is preserved as a late-arrival breach, not silently dropped. An
event exactly at the watermark is not late. The module records all event
arrival classifications idempotently, so an event first accepted as on-time
cannot be reclassified after a later watermark. Progress transactions and
arrival transactions share one strictly increasing knowledge-time clock; a
watermark cannot be backdated behind an already observed event. Multiple
same-time arrivals can be committed as one atomic batch. A late breach contests only
new completeness queries after its receipt time; earlier as-known results
remain unchanged. The caller must also append the raw event to the bitemporal
event ledger; the watermark module records the breach receipt, not the
authoritative event itself.

Twenty-five source-authored watermark tests cover empty progress, multi-source
minimum progress, exclusive boundaries, as-known progress, late-breach
reopening, terminal streams, generation and monotonicity, atomic batches,
idempotent event arrivals, evidence-bound completeness and breach receipts,
unqualified arrivals without a visible watermark, historical index pruning of future-known receipts, and deterministic rebuild/digest parity. A completeness or late-breach result now returns the
exact supporting progress identity, generation, sequence, transaction time,
and policy digest. Events received before any progress claim are recorded as
`NoPriorProgress`, not `OnTime`; an absent watermark cannot certify timeliness.
Hosted execution previously reached whole-crate compilation and exposed errors that have since been corrected. The latest exact-head test result is pending; no pass is claimed.


The source watermark adversarial evaluation fixture is
`crates/domains/symthaea-capability-closure/fixtures/temporal_source_watermark_completeness_v0.1.json`.
It pins 20 deterministic scenario families and separates false-complete rate
(the primary safety metric) from false-incomplete rate/completeness lag. All
qualification thresholds are targets, not measured results.


The completeness hot path uses the ordered per-source progress chain and a secondary index keyed by (source, event_time_us) for late breaches. It no longer rebuilds the entire append-only ledger on every read. The `completeness_verified` method performs an explicit full-history replay/index audit before returning the same report. Both paths are checked for exact report parity. This separates per-query work from audit cost; throughput and p50/p95/p99 latency remain to be measured.


The deterministic multi-source differential regression generates 48 event arrivals under three distinct source watermarks, then compares indexed and fully audited reports across six required-source sets, five half-open intervals and six knowledge-time cutoffs (180 exact report comparisons). This exercises late-event indexing, source scoping and historical visibility without nondeterministic fixtures.


## Reproducible source-watermark scaling benchmark

`src/bin/watermark_scaling.rs` builds deterministic ledgers with 4,096, 16,384 and 65,536 late-arrival
records. It crosses source counts 1/4/16, half-open interval widths 1/16/256
microseconds, and early/full knowledge cutoffs (54 workload combinations).
Event-time keys are permuted relative to four receipt-time batches. Each case
compares 51 indexed queries with 15 `completeness_verified` queries that replay
and validate the full history. CSV output includes source count, interval width,
knowledge cutoff, event-time buckets visited, receipt candidates examined,
matching breaches and indexed p50/p95/p99 and verified median/worst-observed latency. Exact equality of
indexed and verified reports is required before timing. The benchmark is diagnostic, not a physical
energy measurement; runner CPU model, toolchain, release profile and exact
source/fixture hashes must accompany each artifact. The latest run has not
completed; no latency advantage is asserted yet.


## Current foundation-model comparator snapshot

The versioned comparator observations are stored at
`crates/domains/symthaea-capability-closure/fixtures/tsfm_comparator_snapshot_v0.1.json`.
The snapshot keeps FEV-Bench win rate distinct from MASE skill, GIFT-Eval CRPS
and MASE distinct from BOOM telemetry CRPS, and PMLR energy-study observations
separate from leaderboard accuracy. It preserves the H2O site / GitHub README
FEV-Bench disagreement and the 15 Nov 2026 TSFM.ai MASE-skill snapshot.
The comparator registry hashes this fixture in the exact-head workflow.
No model checkpoint or external baseline has been reproduced.


The breach index is keyed by source identity, event time and receipt/knowledge time. A historical `known_at` query prunes future receipts inside the ordered range lookup instead of visiting them and filtering afterward. A regression checks zero candidates before the first breach, one candidate at the first receipt and three after all receipts, with full replay/index parity.
