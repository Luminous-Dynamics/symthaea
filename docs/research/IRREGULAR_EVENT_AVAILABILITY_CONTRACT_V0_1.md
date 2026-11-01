# Irregular Event, Bitemporal, and Time-Series Adapter Contract (v0.1)

**Tracking:** NEURO-SILICON-011 / #9700  
**Implementation:** `research/irregular-event-contract/`  
**Status:** source and test functions authored; no Rust build or test execution

## 1. Event-time and availability-time contract

Every observation keeps distinct:
- event time: when the phenomenon or measurement occurred;
- availability time: when the candidate could first access the record.

At forecast origin t, an observation is eligible only when
`available_at_us <= t`. A delayed observation whose event time is before t
but availability time is after t remains hidden. Source-local sequence numbers
are preserved independently of event-time ordering.

Observations cannot claim availability before event time. Pre-published planned
information uses the separate `KnownFutureCovariate` type and is usable only
when it was published by the forecast origin and its validity interval covers
the target timestamp.

## 2. Measurement and provenance envelope

The event contract carries nonzero event/entity/modality/source IDs, source
sequence, event/availability timestamps, unit code and a source SHA-256.
Measurements are typed as observed fixed-point values, missing with explicit
reason, censored interval, or opaque content-addressed payload. Text/audio/
image/embedding bytes remain outside the event envelope and are referenced by
digest, byte length and encoding ID.

The base validation rejects duplicate event IDs and duplicate active source
sequences within (entity, modality, source). It preserves caller order and
never silently sorts, resamples, interpolates, forward-fills, zero-imputes or
regularizes data.

## 3. Microsecond bitemporal source-event revisions

`BitemporalEventLedger` is an append-only event revision store. It models
point-event validity at microsecond resolution:
- world-time coordinate: `event_time_us`;
- knowledge/transaction-time coordinate: `recorded_at_us`.

Operations are Assert, Supersede and Retract. Each event has a contiguous
revision sequence. Corrections/retractions append new records; previous
revisions remain in the ordered SHA-256 history. A record must have been
available by the transaction that records it. Corrections preserve entity,
modality, source and source-sequence identity and must target the latest active
revision.

Queries use (valid_at_us, known_at_us): choose the latest revision committed by
the knowledge-time cutoff and then apply the world-time cutoff. Later
corrections or retractions cannot rewrite an earlier as-known view.

The ledger maintains indexes for revision ID, event-to-revision chains, latest
revision per event, active source-sequence uniqueness and entity-to-event
membership. A single append preflights then commits without cloning the full
log. A multi-event transaction clones once and commits only after every
revision validates. As-known queries binary-search each entity event's
transaction-time-ordered revision chain. The explicit consistency audit
reconstructs and compares the indexes from the append-only log.

This is a point-event extension, not a replacement for Broca's day-granularity
valid-interval ledger in PR #9109 or the proof-composition runtime in PR #9292.
Durable WAL persistence, source authentication, authorization and derived-proof
integration are not implemented here.

## 4. Purged chronological forecast splits

The `splits` module partitions windows by entity and distinct forecast origin,
keeping tied origins together. It purges training windows whose target interval
crosses validation and validation windows whose targets cross test; optional
embargoes shrink both boundaries. Historical input windows may overlap, but
target labels cannot cross the next partition boundary.

This is a clean comparison split, not an automatic replacement for official
Time-IMM / IMM-TSF splits. Report official-path reproduction and the
leakage-controlled split separately.

## 5. Train-only standardization

The `scaler` module fits per-entity/per-feature Welford moments using only
training-window values from the generated split plan. It deduplicates raw
observations reused by overlapping history windows. Conflicting duplicate
values fail closed. Validation, test and purged-window values are skipped
before their numeric values are read. The scaler revalidates per-entity
boundaries and purge assignments. No imputation is performed.

For fair reproduction, preserve and disclose IMM-TSF's released full-record
normalization path separately; do not call the clean train-only result an
official benchmark reproduction.

## 6. Time-IMM per-entity CSV adapter

The canonical `time_imm` module parses the repository's numeric
`time_series.csv` and unstructured `text.csv` formats.

- Exact feature-column-to-modality/unit/scale configuration; unknown columns
  and duplicate feature mappings are rejected.
- Exact source-file SHA-256 and canonical parsed-row digests.
- Numeric values use configurable fixed-point scales in [-9, 0], round
  nearest with ties away from zero, and return a half-unit error bound.
- Empty/NA/NaN numeric cells remain Missing; text whitespace is preserved and
  text payloads are content-addressed.
- RFC3339 timestamps are accepted directly. Naive timestamps require an
  explicit fixed UTC offset or IANA timezone. Ambiguous/nonexistent DST local
  times fail closed.
- Availability policy is explicit and digest-bound: synchronous-at-event-time
  assumption or fixed delay. Neither is represented as observed physical
  arrival time.
- Adapter configuration digest binds dataset/entity identity and a
  feature-sorted mapping. Timezone and availability policies have separate
  digests.

The sample manifest identifies EPA-Air/Bexar schemas as
`date_time,record_id,temp,pm2_5,aqi,ozone` for numeric data and
`date_time,record_id,text` for text. The source repository's Git blob OIDs
are object identifiers, not SHA-256 file digests; the adapter recomputes
SHA-256 over the exact input bytes.

## 7. Authored test inventory and limits

The current source contains 55 authored Rust test functions:
- 16 event-contract tests in `lib.rs`;
- 12 bitemporal ledger tests in `lib.rs`;
- 8 purged-split tests;
- 9 train-only-scaler tests;
- 10 Time-IMM adapter tests.

No Cargo build, rustfmt, Clippy, unit test, dataset adapter execution or
benchmark has run in this environment. The source and fixture inventory is
not evidence of passing tests. The Time-IMM sample metadata records the
repository's released file shape, but the Rust parser has not been executed
against the files. No real-data performance result or SOTA claim is made.
