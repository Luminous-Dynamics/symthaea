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
