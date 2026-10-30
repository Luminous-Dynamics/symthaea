# MEL-MR-EVAL-018 — Pairwise Musical Interaction Attribution

**Status:** implementation draft stacked on MEL-MR-EVAL-017 / PR #9128.  
**Claim ceiling:** four-coalition artifact-bound diagnostics and synthetic unit fixtures. No live FluidSynth four-way render, CI execution, renderer-specific calibration, real-audio benchmark or SOTA improvement is claimed.

## Research question

When two musical events interact through overlap, phase interference, shared effects, voice scheduling or nonlinear mastering, how much of their joint acoustic effect cannot be explained by measuring each event independently?

MEL-MR-EVAL-017 estimates one event's paired counterfactual effect. MEL-MR-EVAL-018 adds a 2×2 factorial experiment over two source-event removals.

## Paradigm shift: measure the second-order intervention

For source events A and B and a fixed renderer context C, produce four exact renders:

- (Y_{00}=R(S,C))
- (Y_{10}=R(S\setminus A,C))
- (Y_{01}=R(S\setminus B,C))
- (Y_{11}=R(S\setminus\{A,B\},C))

For a frozen feature transform \(\Phi\), estimate:

\[
J_{AB}=\Phi(Y_{00})-\Phi(Y_{10})-\Phi(Y_{01})+\Phi(Y_{11})
\]

This is a second-order finite difference in the selected representation. It does not infer exclusive sample ownership.

A non-zero PCM residual can indicate an interaction under a matched black-box renderer experiment. A non-zero spectral-magnitude residual is not, by itself, proof of a nonlinear renderer: magnitude is a nonlinear feature transform, so phase interference can produce a spectral interaction even when the underlying PCM render is additive. Keep these evidence channels separate.

## Implemented in this draft

### Four-coalition evidence contract

The public input requires four opaque `VerifiedRendererAudioBinding` tokens created by MEL-MR-EVAL-017's receipt-replay verifier:

1. baseline;
2. remove A;
3. remove B;
4. remove A and B.

Every coalition must have:
- the same exact source Score digest;
- the same verified renderer identity and normalized execution profile;
- the same sample rate, channel count and frame count;
- a PCM view whose digest matches its verified token;
- a unique input MIDI digest and a unique performed-event-map receipt/artifact digest.

The expected source-event sets are exact: all events in the baseline, all except A in remove-A, all except B in remove-B, and all except A and B in remove-both.

All non-removed event descendants must preserve source identity, descendant ordinal, track/channel, pitch, onset/end time and projected frames. MIDI note-on ordinals are deliberately ignored in that comparison because removing earlier events may shift them without changing the retained event.

### Multi-event exclusion prerequisite

The previous event-map verifier only admitted one explicit exclusion. This branch adds `verify_for_score_excluding_many` and `verify_for_score_excluding_many_and_pitches`.

The set is explicit, nonempty, unique, in range and bounded by the Score event count. Every non-excluded event remains required; unknown, unresolved, duplicate, non-contiguous or reintroduced excluded events reject. A typed-Score wrapper derives retained pitch obligations from the actual Score.

This does not weaken the complete-map rule for baseline or other coalitions.

### Separate measurement channels

Per window and channel, the implementation calculates:

- signed PCM second-order residual RMS;
- PCM residual peak;
- mean of the two first-order PCM effect RMS values;
- PCM interaction-to-main-effect ratio;
- signed second-order STFT magnitude residual RMS;
- spectral interaction-to-main-effect ratio.

The interaction report binds all four renderer receipts, schedule/MIDI digests, event-map artifact and parser-receipt digests, PCM digests, renderer context, feature configuration and per-window results. A receipt hash and a replay verifier bind the computed report to its exact input evidence.

There is deliberately no PASS/FAIL “nonlinear renderer” classification in v0.1. Such a decision requires a calibrated factorial null distribution and appropriate seed-matched controls.

## Critical scientific controls

### Additive controls

- A and B synthesized independently and summed through a fixed linear PCM path: PCM second-order residual should be within floating-point tolerance.
- Shared linear reverb: a fixed linear time-invariant reverb remains additive in PCM; do not incorrectly classify it as nonlinear.
- Identical deterministic render contexts and frozen feature-extractor settings.

### Feature-space controls

- Same-frequency phase-shifted sinusoids: PCM residual can be zero while spectral-magnitude residual is non-zero.
- Report this as feature-space non-additivity, not renderer nonlinearity.
- Include phase-sensitive waveform, complex-STFT or magnitude-STFT channels as distinct representations.

### Nonlinear positive controls

- Shared compressor, limiter and saturation.
- Nonlinear distortion and gain staging.
- Modulated/time-varying effects with state or seed controls.
- Instrument or bus interactions that can be shown to change PCM under a matched four-way intervention.

### Musical timing and overlap strata

- no temporal overlap;
- partial overlap;
- repeated same-pitch events;
- chord tones and cross-voice overlap;
- sustained notes and long releases;
- dense polyphony, voice crossing and multi-descendant source events.

### Renderer and stochastic controls

- same executable, SoundFont/assets, arguments, environment, sample rate, effects and stable render profile;
- exact MIDI bytes bound per coalition;
- common random numbers / matched seeds for stochastic renderers where supported;
- repeated factorial blocks to estimate the four-way interaction null distribution;
- independent-seed blocks only as a separate uncertainty study;
- refuse calibrated claims when execution profiles or random-stream controls differ.

## Benchmark metrics

- PCM interaction residual RMS/peak per window and channel;
- time-frequency interaction residual, reported independently;
- interaction-to-main-effect ratio with explicit denominator guards;
- per-window support localization and temporal spill;
- false interaction rate on held-out additive controls;
- detection power on calibrated nonlinear positive controls;
- uncertainty calibration and abstention;
- protected-event lineage/pitch/timing leakage;
- extra renders, wall-clock cost and memory.

## Proposed research gates (not results)

1. Exact coalition identity and protected-event equality on deterministic fixtures.
2. PCM interaction residual below a frozen floating-point tolerance for deterministic linear additive fixtures.
3. Explicit separation of feature-space interaction from PCM non-additivity.
4. At least 95% sensitivity on held-out calibrated nonlinear positive controls while false-positive rate stays at or below 1% on held-out additive controls.
5. No additive/nonlinear verdict unless a frozen renderer-specific null distribution and seed-matching protocol are available.
6. Abstain whenever any coalition, renderer identity/configuration, source event, event map or PCM artifact is missing or mismatched.

The proposed sensitivity and false-positive gates are research targets, not present measurements.

## Industry comparisons and benchmark strategy

- **SMEBench** ([arXiv:2610.36715](https://arxiv.org/abs/2610.36715), published October 27, 2026) provides 3,600 paired synthetic examples, 12 musical edit operations and a real-audio complement. Use it for edit target agreement and source preservation. Its paired targets do not constitute ground truth for per-note factorial renderer interactions.
- **MMAE / ICASSP 2027 Audio Editing Challenge** ([official site](https://audio-editing-challenge.github.io/)) provides a public 2,000-example / 17,741-rubric test set before the hidden challenge release. The official leaderboard is scheduled to open November 10, 2026, with 500 hidden examples per track. The stated registration deadline was October 8, 2026, and the Agent Track public model/weight cutoff is November 1, 2026. Its IFR, CR and EMR measure instruction following and content consistency, not event-level causal interactions.
- **DDSP** is established prior art for interpretable differentiable synthesis components. Use it as an optional computational path or native backend, not as proof that black-box paired-render measurement is unnecessary.

There is no directly matched public benchmark for per-source-event pairwise counterfactual interaction maps. Publish the synthetic factorial microbenchmark and baselines before making superiority claims.

## Next increments

1. Execute four real FluidSynth renders from the MEL-MR-EVAL-016 process-observation adapter.
2. Add a renderer-specific repeated factorial-block calibration and common-random-number record.
3. Add a deterministic linear-convolution reverb control and a saturating nonlinearity control.
4. Add a synthetic event-pair generator with ground-truth coalition schedules and event maps.
5. Add higher-order (k)-event analysis only after pairwise calibration and cost scaling are established.
6. Compose interaction diagnostics with MEL-MR-EVAL-006 preservation and MEL-MR-EVAL-017 influence reports; keep musical-quality and human listening outcomes separate.

## Implementation and validation status

- Module: `src/counterfactual_interaction.rs`.
- Multi-event exact exclusion/pitch verifier: `src/performed_event_map.rs`.
- Receipt-to-token set-exclusion adapter: `src/counterfactual_influence.rs`.
- Source-level tests cover PCM additivity, PCM interaction, phase/magnitude non-additivity, malformed audio/config, exact multi-event exclusion sets, four-coalition replay, wrong coalition and renderer-profile substitution.
- Rust compilation, unit tests, rustfmt, Clippy, actual FluidSynth four-way renders, factorial null calibration, and public benchmark runs have **not** been executed in this environment.

## Related work

- [PR #9128 — MEL-MR-EVAL-017](https://github.com/Luminous-Dynamics/symthaea/pull/9128)
- [Issue #9093 — Counterfactual performed-event influence maps](https://github.com/Luminous-Dynamics/symthaea/issues/9093)
- [Issue #9132 — Pairwise musical interaction attribution](https://github.com/Luminous-Dynamics/symthaea/issues/9132)
- [PR #9090 — MEL-MR-EVAL-016 FluidSynth process observation](https://github.com/Luminous-Dynamics/symthaea/pull/9090)
