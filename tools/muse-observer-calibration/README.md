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
- Held-out results report false support by negative family (e.g. octave collision vs cross-part contamination), including independent source-group counts, failing-group counts and a one-sided nominal 95% Wilson score upper bound. This is marginal per family, not a simultaneous family-wise bound. Underpowered families report insufficient evidence.
- The calibration bundle can support an acoustic claim; it cannot replace symbolic correctness, renderer execution assurance, score-to-audio conformance or human acceptance.

See `docs/research/MUSE_OBSERVER_CALIBRATION_V1.md` and issue #12998 for the full protocol and claim limits.

## Qualification

The source-group family-risk implementation passed the dedicated exact-head workflow at code commit `280c5e6524f44f9d01319e16f7ae53d3bf2da31d`:
- Rust formatting: passed.
- Library, CLI and adversarial tests: 18 passed.
- Clippy with `-D warnings`: passed.
- CLI help smoke: passed.
- Draft 2020-12 schema/fixture checks: passed.

This qualifies the implementation and synthetic fixture, not real-audio calibration, annotation authenticity, rights or instrument/renderer generalization. See the research note for the exact-head boundary and current evidence gaps.
