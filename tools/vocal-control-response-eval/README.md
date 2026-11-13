# Vocal control-response evaluator (VCRB-001)

A dependency-free Rust evaluator for testing whether a vocal-performance control produces its intended acoustic response without unintentionally moving other measured properties.

## Why this exists

A control API accepting a parameter is not evidence that the rendered audio obeys it. A renderer can increase vibrato depth while also shifting mean pitch, changing phrase energy, moving note onsets, or altering neighboring phone timing.

VCRB-001 turns the existing VCEB-001 one-factor-at-a-time idea into an executable contract:

1. Request one control perturbation on a frozen performance.
2. Measure the target acoustic property and a declared set of collateral properties.
3. Compare matched baseline/intervention repetitions.
4. Require the target response to land in a predeclared signed range.
5. Require every collateral property to stay within its independently calibrated tolerance.
6. Report each response row separately; do not hide failures inside a composite quality score.

## Input contract

The caller supplies acoustic measurements, not audio samples. An independent analyzer must produce the observations.

- One ControlSpec declares the requested input delta/unit, target metric, signed expected output-delta interval, minimum in-range fraction, minimum number of repetitions, and collateral limits.
- One ControlTrial supplies matched baseline/intervention pairs for that control.
- Every control row in one matrix must use the same source-case ID, identical ordered pair IDs, and the same monitored metric IDs and units.
- Every observation must be finite, unique, expected, and unit-consistent. Missing or extra metrics fail closed.
- The expected target interval must exclude zero; the evaluator will not accept an interval that permits a no-op.
- At least two matched pairs are required so response variation can be reported.

The tool does not render audio, extract F0/energy/phones, align audio, infer target windows, or decide whether a voice sounds natural. Those responsibilities remain with separately versioned analyzers and human evaluation.

## Metrics

Each acoustic metric reports:

- mean signed intervention-minus-baseline delta;
- sample standard deviation across matched pairs;
- minimum and maximum paired delta;
- maximum absolute paired delta.

Each control row additionally reports:

- the target-response interval and fraction of pairs inside it;
- empirical target response gain (observed target delta divided by requested control delta);
- per-metric collateral tolerance, observed maximum absolute delta, tolerance utilization where defined, and pass/fail.

A matrix passes only when every control's target response and every collateral guard pass. A target success cannot compensate for collateral damage, and preservation cannot compensate for a no-op target.

## Example control dimensions

| Requested control | Target acoustic metric | Useful collateral metrics |
|---|---|---|
| Increase vibrato depth | Vibrato depth (cents) | Mean F0, vibrato rate, onset, energy |
| Delay vibrato onset | Vibrato onset (ms) | Note onset, mean F0, vibrato depth |
| Change one note by two semitones | Stable-note F0 (cents) | Neighbor-note F0, vibrato, onset |
| Soften one phrase | Aligned phrase energy (declared dB domain) | Phone timing, F0, breathiness |
| Increase phoneme-level energy | Phone-aligned energy | Note velocity response, neighboring phone energy, F0 |
| Extend a syllable into melisma | Vowel-span duration and note coverage | Consonant identity/timing, rests, phrase boundary |

Metric definitions and acoustic-analysis versions must be frozen per benchmark release. Do not compare dBFS, LUFS, normalized amplitude, or backend-internal energy without an explicit calibration.

## Reproducibility and threshold calibration

Use the same source performance, score, pronunciation plan, voice/model artifact, renderer build, analysis pipeline, and sample-rate/channel policy across matched pairs. Pair by the same seed or other controlled stochastic condition where possible. Store input and output audio digests and analyzer versions outside this scorer.

Calibrate collateral tolerances on a separate no-op/repeatability split before the confirmatory split. Do not select tolerances after looking at test results. For deterministic paths, a zero tolerance can demand exact preservation of a metric. For nondeterministic neural paths, a non-zero tolerance must be estimated and justified independently.

The reported sample standard deviation is descriptive. This version does not claim statistical significance, estimate population confidence intervals, or calibrate thresholds. Publish the number of source cases, singers, songs, repetitions, and held-out split policy with any aggregate result.

## Integration boundaries

- VCEB-001 defines the broader vocal-control and quality tracks.
- VELB-001 measures waveform target-versus-protected-region locality for edits; VCRB measures acoustic control response and cross-coupling. They answer different questions.
- Issue #11357 defines the separate phoneme-energy control plan and ASRU 2025 reproduction track. VCRB can evaluate its acoustic response once the renderer consumes that control.
- The existing backend capability manifest remains authoritative for native/approximated/conditional/unsupported controls. Unsupported controls must be reported as unsupported, never inserted as passing rows.

## Run locally

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-control-response-eval/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-control-response-eval/Cargo.toml
~~~

The tests are synthetic scorer-contract controls only. They do not establish renderer behavior, singing quality, human preference, or a state-of-the-art result.
