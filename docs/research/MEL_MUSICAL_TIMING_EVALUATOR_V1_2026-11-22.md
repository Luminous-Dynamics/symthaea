# MEL Musical Timing Evaluator v1

**Status:** standalone Rust evaluator with a synthetic known-answer panel. No live model, musician, MIDI hardware or audio loopback was measured.

## Purpose

This evaluator complements the interaction trace auditor. It scores whether observed musical events land at the intended beat/subdivision, whether signed phase error drifts over a phrase, and how rapidly a beat-period estimator recovers after a tempo change.

## Timing contract

All timestamps use the declared monotonic clock domain. The panel binds:
- a reference-grid receipt;
- an onset-capture receipt;
- a clock-calibration receipt;
- piecewise tempo segments with a phase-zero timestamp, start beat index, beat period and time at which the change became observable;
- target events with rational subdivisions and an explicit observed/missed outcome;
- observed onset times with per-event capture-receipt digests;
- tempo estimates with observation time, evidence-window end, estimated period, confidence and provenance digest; and
- unmatched extra onsets, which cannot be hidden by a good average phase score.

For a target at beat index b with segment origin b0 and subdivision p/q, expected onset is:

    phase_zero_ns + (b - b0) * beat_period_ns + p * beat_period_ns / q

For each observed target, signed phase error is:

    (observed_onset_ns - expected_onset_ns) / beat_period_ns

The report also gives absolute error in milliseconds. Threshold rates are reported at 0.05, 0.10, 0.25 and 0.50 beat. Per-instrument distributions prevent a strong transient instrument from hiding sustained/bass timing errors.

## Tempo adaptation

A tempo estimate is eligible only after its declared evidence window has ended and after the corresponding reference tempo change became observable. Recovery is the first point at which K consecutive estimates fall within the preregistered relative period-error tolerance. A single lucky estimate is not recovery.

Adaptation is reported in nanoseconds and beats. If the required stable sequence never appears, the segment remains unrecovered and the overall timing contract is not qualified.

## Drift and statistics

Phase drift is a descriptive least-squares slope of signed phase error against beat index. It is not an inferential confidence interval. Any human-study comparison must cluster by session, musician and composition; note events are not IID samples. Publish all per-instrument/task-family slices and sample counts.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-musical-timing-audit -- score \
      tools/mel-judge-grounding-audit/fixtures/valid-musical-timing-v1.json \
      report.json

    cargo +1.96.0 test --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Claim ceiling

The committed fixture contains invented MIDI-style onset times and synthetic receipt digests. The validator does not authenticate receipt contents, measure hardware loopback, infer missing notes, or establish human musical quality. It checks declared timing arithmetic and causal evidence-window bounds. Real performance claims require held-out musicians/compositions and independently captured MIDI/audio onset times, with calibration between clock domains. MIDI callback time must not be presented as audible acoustic onset.
