# Irregular Event, Bitemporal, and Time-Series Adapter Contract (v0.1)

**Tracking:** NEURO-SILICON-011 / #9700  
**Implementation:** `research/irregular-event-contract/`  
**Status:** 81 Rust test functions authored. An earlier pinned Rust 1.96 whole-crate compile exposed and led to fixes for four type errors. The latest exact-head format/test/benchmark/Clippy workflow is pending; no test or benchmark pass is claimed.

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

The current source contains 81 authored Rust test functions:
- 16 event-contract tests in `lib.rs`;
- 12 bitemporal ledger tests in `lib.rs`;
- 8 purged-split tests;
- 10 train-only-scaler tests;
- 10 Time-IMM adapter tests;
- 25 source-watermark/completeness tests.

No Cargo build, rustfmt, Clippy, unit test, dataset adapter execution or
benchmark has run in this environment. The source and fixture inventory is
not evidence of passing tests. The Time-IMM sample metadata records the
repository's released file shape, but the Rust parser has not been executed
against the files. No real-data performance result or SOTA claim is made.


## 9. Source progress watermarks and qualified absence

The `watermarks` module extends asynchronous source semantics with explicit
progress. Each source progress revision binds source identity, stream
generation, monotonic sequence, exclusive event-time watermark, transaction
time and a nonzero progress-policy digest. The maximum event time observed is
not sufficient to infer a watermark; the source or an explicitly named
adapter policy must make that progress claim.

For query interval [start,end), every required source must have a visible
watermark >= end, or explicitly declare end-of-stream, before the interval is
Complete. Missing progress is Incomplete, not evidence that no event occurred.
The query uses known_at, preserving the historical completeness view.

A late event with event_time < an already recorded exclusive watermark is
retained as a breach receipt and changes subsequent completeness to Contested
for affected intervals. An event exactly at the watermark is not late. Every
event arrival classification is idempotently recorded. Crucially, an event
received before any progress claim is visible is `NoPriorProgress`, not
`OnTime`: lack of a watermark cannot certify that an event was timely. An
event at or beyond a visible exclusive watermark is classified OnTime relative
to that source claim, not as proof of global completeness. Historical queries
before the late receipt remain unchanged. The caller must append the raw event
to BitemporalEventLedger separately; the watermark ledger owns progress and
breach evidence, not the authoritative source event.

The source-completeness report separates Complete, Incomplete and Contested,
shows each required source's exact supporting progress record (ID, generation,
sequence, recorded-at knowledge time and policy digest), and returns the exact
blocked source and late-breach sets. Each late-breach receipt also binds the
specific progress record and policy digest that established the exclusive
boundary. Terminal/end-of-stream is stronger than a finite watermark and
closes intervals beyond its final event-time bound.

The normal completeness query uses the ordered source-progress chain and a
secondary index keyed by source identity, event time and receipt/knowledge time
to inspect only late breaches that could affect the requested half-open
interval. It does not replay the full append-only history on every query.
`completeness_verified` is the explicit audit path: it rebuilds progress and
arrival indexes from the history, verifies replay/digest equality, and returns
a distinct `VerifiedCompletenessReport` with private fields, the exact ledger
SHA-256, and progress/arrival/breach counts. Its `.report()` contains the same
completeness result as the fast path. The fast and audited paths are tested for
exact report equality. The type distinction prevents a caller from passing an
index-only `CompletenessReport` where replay verification is required. It
certifies only the integrity of this in-memory ledger history and derived
indexes; it is not source authentication or proof that a source's progress
claim is truthful.

The breach index is nested by source identity and event time, with a
receipt/knowledge-time sub-index inside every event-time bucket. Historical
`known_at` queries prune future receipts within each bucket. The report counts
event-time buckets visited separately from receipt candidates examined; this
makes the remaining range-scan cost visible. A regression with distinct event
times checks zero visible candidates before the first receipt, one at the first
receipt and three after all receipts, then verifies replay/index parity. This is a query
complexity improvement by construction, not yet a measured throughput result;
the benchmark records p50/p95/p99 query latency, late-breach candidates visited,
and full-audit rebuild latency separately.

Twenty-five source-authored tests cover missing progress, multi-source minimum,
exclusive boundary behavior, as-known watermarks, late arrivals, terminal
sources, monotonicity/generation, policy identity, atomic batches, duplicate
arrivals, unqualified no-progress arrivals, evidence-bound completeness and
breach receipts, and replay/index digest parity.

This is a source-provided progress contract, not proof that a sensor's claim is
authentic or physically true. Signature verification, source authorization,
durable WAL and a trusted progress-policy registry remain out of scope.


## 10. Source watermark benchmark fixture

The machine-readable fixture
`crates/domains/symthaea-capability-closure/fixtures/temporal_source_watermark_completeness_v0.1.json`
pins twenty-one adversarial scenarios and required metrics. False-complete rate is
the primary safety metric; false-incomplete rate and completeness lag are
reported separately so an always-incomplete strategy cannot pass. Late-arrival
detection recall/latency, source watermark skew, and historical as-known
mutation count are also recorded. The zero-false-complete and exact-replay gates
are preregistered targets, not benchmark results.


## 11. Indexed watermark scaling experiment

The standalone executable
`research/irregular-event-contract/src/bin/watermark_scaling.rs` constructs
deterministic ledgers at 4,096, 16,384 and 65,536 event records. It varies
required-source counts 1/4/16, half-open interval widths 1/16/256 microseconds,
late-event densities 0/100/5,000/10,000 basis points (0%, 1%, 50%, 100%), and
two historical knowledge cutoffs, for 216 workload configurations. Event-time
keys are permuted relative to four receipt-time batches.

Each workload first requires exact equality between the fast indexed report and
the report inside a distinct, non-forgeable `VerifiedCompletenessReport`.
The verified receipt binds the full-history replay to the ledger SHA-256 and
progress/arrival/breach counts; this is in-memory ledger consistency, not source
authentication. It additionally requires the receipt digest to match the
current ledger snapshot before timing. Each case records 51 indexed query
samples (nearest-rank p50/p95/p99) and 15 verified replay samples (median and
observed maximum), plus event-time index buckets visited, receipt candidates
examined and matching breaches.

The hosted workflow retains CSV output with the exact commit, Rust toolchain,
dependency lock, and fixture/source digests. This measures in-memory release
build CPU query and audit cost only; it does not establish durable-WAL,
distributed-stream, throughput or physical-energy performance. No latency
advantage is claimed until the latest exact-head run yields an artifact.

## 12. Clock-domain calibration and conservative time mapping

A timestamp unit does not establish a shared clock. The event-time and
watermark fields in this v0.1 ledger are scalar microseconds and therefore
must already be expressed in one declared reference coordinate before they
are compared. A source-local device timestamp must not be aggregated with
another device's timestamp merely because both are measured in microseconds.

The new `research/irregular-event-contract/src/clock.rs` module defines
`ClockCalibrationV1`: source and reference domain IDs, source-time validity,
anchor coordinates, a bounded offset interval, maximum absolute drift in
parts per billion, calibration knowledge time, and a policy digest. Mapping
returns a conservative time interval:

[
I_r(t_s) = [
 t_a^r + \Delta + \delta^- - \lceil |\Delta|\rho / 10^9 \rceil,
 t_a^r + \Delta + \delta^+ + \lceil |\Delta|\rho / 10^9 \rceil
]
]

where `Delta = t_s - t_a^s`, `[delta-, delta+]` is the anchor offset
uncertainty and `rho` is the maximum absolute drift in ppb. Arithmetic is
checked and uncertainty is rounded outward. A calibration recorded after the
requested knowledge-time cutoff, outside its validity range, or for a
different source clock is rejected.

For an exclusive source watermark, only the earliest possible mapped endpoint
is safe for absence/completeness decisions. When combining sources, all mapped
watermarks must share the same reference domain and the system takes the
minimum earliest endpoint. An optimistic upper bound, midpoint, or uncalibrated
raw clock value cannot close an interval.

The mapped-time and conservative-watermark types have private fields to
prevent fabricated result structs. Fourteen clock-specific tests cover offset
uncertainty, drift growth and outward rounding, historical knowledge-time,
validity expiry, domain mismatch, common-reference aggregation, empty source
sets, invalid calibration, arithmetic overflow, as-known revision selection,
atomic batches, duplicate calibration IDs, transaction ordering and
replay/digest parity.

The append-only `ClockCalibrationLedger` indexes revisions by source/reference
domain pair and enforces unique calibration IDs plus monotonic knowledge-time
transactions. Historical lookups select the newest revision that was known at
the decision cutoff and whose source-time validity covers the requested tick.
Same-time multi-source calibration updates are atomic; a full history replay
rebuilds its index and state digest. This selection boundary prevents a caller
from accidentally applying a future calibration or a revision outside its
validity range.

**Integration boundary:** this module is not yet automatically wired into the
v0.1 `SourceWatermarkLedger`. That ledger still expects callers to supply
timestamps already normalized to one reference coordinate. The policy digest
and content hash bind parameters but do not authenticate a calibration
authority. A governed, append-only calibration registry, signed calibration
receipts, key revocation/rotation and end-to-end clock provenance are future
qualification steps. This contract does not claim physical clock truth.

## 13. Current external comparator snapshot

The model-to-benchmark observations are pinned separately in
`crates/domains/symthaea-capability-closure/fixtures/tsfm_comparator_snapshot_v0.1.json`
and referenced by the real-irregular registry. The fixture keeps FEV-Bench
win-rate snapshots, MASE skill, GIFT-Eval CRPS/MASE, BOOM telemetry CRPS and
accuracy-energy study findings distinct. It preserves conflicting H2O public
surfaces and identifies vendor claims versus leaderboard mirrors. It is a
dated comparator manifest, not a set of reproduced scores.


The scaling benchmark has an explicit measurement boundary: deterministic
in-memory release-mode CPU query latency only. It is not a durable-WAL or
distributed-stream benchmark, and must not be presented as physical-energy
measurement.