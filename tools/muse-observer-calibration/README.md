# MUSE Observer Calibration

Rust-native calibration and selective-risk evaluation for provenance-bound acoustic observer features.

## Purpose

The current pitch observer emits a raw fundamental/harmonic salience value in dB. This package treats it as a feature, not a confidence probability. It fits a monotonic isotonic mapping on one source-disjoint split, selects a positive-support threshold on a second split under a false-support constraint, then evaluates once on a third held-out split.

Rows with no measurable evidence always abstain.

## Commands

Evaluate a sealed dataset against a calibration contract:

```bash
cargo run --manifest-path tools/muse-observer-calibration/Cargo.toml -- evaluate DATASET.json CONTRACT.json
```

Replay a calibration result:

```bash
cargo run --manifest-path tools/muse-observer-calibration/Cargo.toml -- verify DATASET.json CONTRACT.json BUNDLE.json
```

## Evidence boundary

- Ingestion must first replay upstream audio/MIDI observation receipts against their exact WAV/MIDI bytes (currently #11617).
- The dataset digest is an unkeyed content commitment, not a signature or rights grant.
- The calibrator does not authenticate human annotations or prove source identity by itself.
- Fit, threshold-selection and held-out splits must be source-group disjoint. Exact WAV/MIDI digests cannot be re-used across source groups.
- Confidence intervals resample complete source groups, never individual notes.
- No-safe-threshold means no support decision is emitted; all cases abstain.
- Held-out results report false support by negative family (e.g. octave collision vs cross-part contamination), not only an aggregate rate.
- The calibration bundle can support an acoustic claim; it cannot replace symbolic correctness, renderer execution assurance, score-to-audio conformance or human acceptance.

See `docs/research/MUSE_OBSERVER_CALIBRATION_V1.md` and issue #12998 for the full protocol and claim limits.

## Qualification

The dedicated exact-head workflow runs Rust formatting, tests, Clippy and JSON Schema Draft 2020-12 checks. Source tests are authored; do not treat them as executed until the workflow reports a successful result on the exact PR head.
