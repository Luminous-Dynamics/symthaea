# Irregular Event/Availability Contract (v0.1)

**Tracking:** NEURO-SILICON-011 / #9700  
**Implementation:** `research/irregular-event-contract/`  
**Status:** candidate-side typed contract authored; no external data adapter or test execution

## 1. Why two clocks?

Irregular datasets frequently contain delayed observations, asynchronous
channels and different source clocks. Sorting everything by event time and
then filtering at the forecast origin can leak observations that were not
actually available to the system yet.

The contract distinguishes:
- event time: when the underlying event/measurement occurred;
- availability time: when the candidate could first observe it.

For forecast origin (t), observations are visible only when
`available_at_us <= t`. If an event occurred before (t) but arrived after
(t), it remains hidden until arrival. The candidate view preserves source
order; no silent event-time sorting takes place.

## 2. Observation schema

Every event contains nonzero event/entity/modality/source identifiers,
source-local sequence index, event time, availability time, unit code,
measurement status/value and a source digest.

Payloads are typed as:
- observed fixed-point numeric values, with explicit decimal scale;
- missing values with an explicit reason (not measured, sensor offline,
  below detection, withheld, unknown);
- censored intervals with open or closed bounds;
- content-addressed opaque payloads (for example text, audio, images or
  precomputed embeddings), binding a nonzero payload digest, byte length and
  encoding ID while keeping the payload bytes outside the event envelope.

Fixed point avoids floating-point equality in the transport contract. Dataset
adapters must preserve native values/units and document their conversion to
integer scale; no lossy conversion is allowed without an explicit error bound.

An observation record cannot claim availability before its event time. Planned
or prepublished context must use the separate future-covariate type.

## 3. Known-future covariates

`KnownFutureCovariate` is deliberately not an `EventRecord`. It is visible
for a target timestamp only when:
- the entity matches;
- `published_at_us <= forecast_origin_us`;
- the target lies inside its declared validity interval.

This distinction is essential for contextual-forecast benchmarks. Future-derived
labels or annotations reconstructed from the realized target trajectory do not
become candidate-visible merely because they are attached to a benchmark sample.

## 4. Source identity and duplicates

Source sequence uniqueness is scoped to
`(entity_id, modality_id, source_id, source_sequence)`. Different sensors or
upstream sources may each have sequence zero. Event IDs and known-future
covariate IDs are globally unique within the supplied batch. The batch validator
does not rewrite, reorder or regularize records.

## 5. Explicitly out of scope

- Benchmark-specific chronological/entity split generation.
- Resampling, interpolation, imputation, forward fill and zero fill.
- Physical source-digest recomputation from raw source payload bytes.
- Unit conversion beyond explicit fixed-point scale metadata.
- Future-label/evaluator target storage in the candidate package.

Those operations belong in pinned adapters/evaluator code with separate
provenance and preprocessing resource accounting.

## 6. Test coverage and limits

Sixteen source-authored Rust test functions cover irregular and asynchronous
records, delayed arrival, non-retroactive visibility, out-of-order event time,
missingness/censoring, opaque multimodal payload validation, duplicates, source identity, fixed-point validation,
known-future covariate publication/validity, entity filtering and tie handling.

No Cargo build, rustfmt, Clippy, tests or real dataset adapter has run. This
contract is a candidate-side causality guard, not proof of correct ingestion
from an external benchmark.


## 7. Microsecond bitemporal revision history

The crate includes an append-only `BitemporalEventLedger` for streaming
observations. `event_time_us` is valid/world time; `recorded_at_us` is
transaction/knowledge time. Each event has contiguous revision numbering.
Corrections append Supersede operations and retractions append Retract
operations; previous revisions are never mutated.

Queries use both coordinates: a revision is considered only if recorded by the
knowledge-time cutoff, its latest event revision is not retracted, its event
time is no later than the valid-time cutoff, and the record was available by
the knowledge time. This makes "what happened by V, according to what we knew
at T?" independently testable at microsecond resolution.

Transactions require one shared timestamp and strictly increasing timestamps
between commits. A failed multi-event batch leaves the original ledger
unchanged. Active source sequence uniqueness is checked at commit. The history
has a domain-separated SHA-256 digest.

This is a precision extension to, not a replacement for, the existing
Broca source ledger in `crates/domains/symthaea-broca/src/memory_bitemporal.rs`
(PR #9109) and combined bitemporal/alternative-proof runtime (PR #9292).
Broca's source valid-time uses `NaiveDate`; this streaming extension uses
microsecond timestamps and does not yet join the derived justification graph.

The crate now has 27 source-authored test functions: 16 event contract tests
and 11 bitemporal source-revision tests. No Rust build or test has run.


## 8. Purged chronological window splits

The `splits` module partitions per entity using distinct forecast origins.
Windows sharing an origin cannot be divided between train/validation/test.
Training windows are retained only when the exclusive end of their target
interval is no later than the validation forecast-origin boundary (minus an
optional embargo). Validation windows obey the analogous test boundary.
Windows that cross a boundary are counted as purged rather than silently
relabelled. Historical input overlap is allowed; future target-label overlap is
not.

Eight source-authored tests cover overlapping forecast windows, exact boundary
semantics, embargoes, tied origins, input-order invariance, per-entity
boundaries, invalid intervals/ratios, duplicate IDs and insufficient/empty
partitions. The package now has 35 source-authored test functions total.


## 9. Train-only feature standardization

The `scaler` module binds standardization to a `PurgedSplitPlan`. Only raw
observations attached to eligible training windows contribute to the moments.
Validation, test and purged-window values are not read. Repeated observations
from overlapping training windows are deduplicated by entity/feature/raw
observation ID; conflicting duplicate values are rejected.

Statistics use deterministic Welford accumulation and sample standard
deviation (ddof=1). A zero standard deviation is handled by mean subtraction
without division. Fitted observation counts, means and standard deviations
are queryable for provenance. The scaler validates that each window is in the
partition required by the split boundaries and the target-horizon purge; it
rejects forged or inconsistent split plans.

The scaler does not impute missing data or compute a physical arrival time.
Adapters must supply only legitimate training observations and preserve
missingness separately. This is the clean comparison path; the official
benchmark code path must be separately reproduced and labeled when it uses
different normalization.


## 10. Time-IMM CSV adapter

The `time_imm` module parses one entity's numeric and text CSV files.

It requires an exact numeric feature mapping: column name, modality ID, unit
code and fixed-point exponent. Unknown, missing or duplicate columns fail
closed. Source-file SHA-256 binds exact input bytes; event source digests bind
canonical parsed rows. Text is content-addressed without whitespace trimming.

RFC3339 timestamps are unambiguous. Naive timestamps require an explicit fixed
UTC offset or IANA timezone; DST-ambiguous/nonexistent local times are rejected
rather than silently shifted. The timestamp-zone configuration is digest-bound. A separate canonical adapter-config SHA-256 binds dataset/entity identity, availability policy, and a feature-sorted mapping of every numeric column to modality, unit and decimal scale; feature declaration order does not change this digest.

The adapter also requires a caller-selected availability policy (synchronous
at event time as an explicit assumption, or a fixed delay). Its digest is
reported independently. The Time-IMM CSV does not contain a separate observed
system-arrival timestamp, so the adapter makes no claim that either policy
reconstructs actual acquisition latency.

Numeric values are converted to per-feature fixed-point scales in [-9, 0].
Rounding is nearest, ties away from zero, and the maximum half-unit error bound
is returned. Empty/NA/NaN measurements become Missing records; text is stored
as an opaque payload reference. No interpolation, imputation or resampling is
performed.

The parser is a dataset adapter, not a timezone/availability oracle. Dataset
maintainers must verify the correct IANA timezone and the benchmark's intended
availability semantics before using a configuration for claims.


The parsed table returns dataset ID, record ID, stable entity ID, exact source
file digest, timestamp-zone digest, availability-policy digest and the
canonical adapter-configuration digest. These values should be copied into the
benchmark run manifest alongside the split and train-only scaler digests.
