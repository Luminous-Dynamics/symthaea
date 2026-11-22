# MEL Musical Timing Evaluator v2

**Status:** standalone Rust evaluator with a synthetic known-answer panel. It consumes expected-onset data and provenance from the canonical musical time mapper; no independent beat-to-time conversion is implemented here. No live model, musician, MIDI hardware or audio loopback was measured.

## Integration boundary

- #9477 MUSE-TEMPO owns canonical notation tempo and beat-to-performance-time mapping semantics.
- #10263 owns the exact symbolic score-timing oracle.
- #10262 MEL-TIME owns temporal-controllability event conformance and its TimingBench comparison.
- #10307 MEL-RTI owns streaming WAIT/RESPOND/CONTINUE/YIELD policy and sample-frame causal tracing.

This evaluator is a downstream measurement consumer. Each reference target includes:
- canonical mapper ID, version and kind (notation_tempo or performance_time);
- a digest binding the exact mapping receipt;
- expected onset timestamp already computed by the canonical mapper;
- local beat period supplied by that mapper; and
- score beat/subdivision identity for drift analysis.

The evaluator never recomputes beat index or subdivision into seconds. Any canonical time-map schema change must be resolved at the mapper boundary; do not add another local mapping implementation here.

## Phase metrics

For each observed target:

    signed_phase_error_beats =
      (observed_onset_ns - canonical_expected_onset_ns) / canonical_local_beat_period_ns

The report includes signed and absolute phase error, p50/p95/p99/max distributions, millisecond error, signed bias, threshold rates at 0.05/0.10/0.25/0.50 beat, descriptive phase drift and per-instrument metrics.

Every expected target must explicitly be observed or marked missed. Unmatched extra onsets are separately counted. Misses and extras prevent qualification even when the observed events have low phase error.

## Tempo adaptation

Tempo changes are separate observed events. Their reference local period is supplied by the canonical mapper. The estimator records:
- change ID;
- monotonic time the change became observable;
- reference period;
- estimate observation timestamp;
- evidence-window end timestamp;
- estimated period, confidence and provenance digest.

An estimate is rejected if it uses evidence beyond its observation time or predates the change becoming observable. Estimate timestamps must be strictly increasing and meet a preregistered minimum spacing. Recovery occurs only after K consecutive estimates fall inside the relative-period tolerance; one lucky estimate cannot qualify.

## Clock and evidence rules

All timestamps use the declared monotonic clock domain. A clock-calibration receipt binds source and output clock alignment. The mapping, onset-capture and calibration receipt digests are shape-checked but not cryptographically authenticated by this reference CLI.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-musical-timing-audit -- score \
      tools/mel-judge-grounding-audit/fixtures/valid-musical-timing-v2.json \
      report.json

    cargo +1.96.0 test --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Claim ceiling

The committed fixture uses invented MIDI-style onsets, synthetic receipts and mocked canonical mapper outputs. The validator does not authenticate receipt contents, measure hardware loopback, infer missing notes or establish human musical quality. It checks expected-onset measurements against supplied canonical values and validates evidence-window chronology. Drift is descriptive, not inferential. Real performance claims require the actual #9477 mapper, held-out musicians/compositions and independently captured MIDI/audio onsets. MIDI callback time must not be presented as audible acoustic onset.
