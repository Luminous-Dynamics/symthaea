# MEL-MR-EVAL-017 — Counterfactual Performed-Event Acoustic Influence

**Status:** implementation draft, stacked on MEL-MR-EVAL-016 / PR #9090.  
**Claim ceiling:** deterministic fixture and local artifact replay primitives only. No Rust build, test execution, real FluidSynth integration, acoustic calibration, human listening study, or SOTA improvement is claimed.

## Research thesis

A performed-event MIDI map establishes what the serialized schedule says was performed. An onset detector estimates where acoustic attacks occur. Neither establishes the complete acoustic consequences of a note after instrument release, shared effects, resampling, mastering, compression, or nonlinear interactions.

This work tests the following bounded proposition:

> Given two exact, receipt-verified renders that differ by one declared MIDI event removal, can Muse measure the observed PCM and local spectral difference above the measured repeat-render floor, while proving that every non-target source event retained its identity, pitch, and scheduled timing?

It does not infer exclusive sample ownership. A measured influence interval is the support of an observed paired intervention under a declared renderer/configuration, not a claim that all samples in that interval were generated only by the removed note.

## Paradigm shift: intervene on the renderer input

For source event e, compare:

- baseline schedule S and output y = R(S, C);
- schedule S_without_e and output y_without_e = R(S_without_e, C).

The measured feature difference is:

I_e = Phi(y) - Phi(y_without_e)

Phi is deliberately split into independent channels in v0.1:

- phase-sensitive waveform RMS difference;
- peak absolute sample difference;
- phase-insensitive short-time spectral-magnitude distance.

Do not compress these into one perceptual-quality score. These features measure output difference, not subjective quality or exclusive source ownership.

A black-box paired-render strategy is the reference design. Differentiable DSP may later accelerate sensitivity estimates for native paths, but its gradients are not a substitute for measured output from FluidSynth or another black-box renderer.

## Implemented in this draft

### Replay-verifiable renderer/audio binding

- The raw token constructor is crate-private.
- The token is created only after replaying the exact `RendererExecutionReceipt` against its `RendererReceiptInputs`.
- The exact output PCM view must match receipt sample rate, channel count, frame count, and canonical decoded little-endian f32 PCM digest.
- The exact MIDI bytes in the receipt are reparsed through `PerformedEventMap::from_midi_bytes`; a caller-supplied map must equal the independently reconstructed map.
- The parser-derived map receipt and the hash of the serialized map artifact are kept as separate commitments.
- The FluidSynth path is the only supported renderer token in v0.1. NativeMuse is deliberately rejected until its exact in-synth schedule boundary has a parser/replay contract.

### Source-event and pitch correctness

- A normal baseline map must have complete score-event coverage and exact primary MIDI-pitch agreement.
- The counterfactual map uses a separate `verify_for_score_excluding_and_pitches` gate: every expected source event must remain except the one explicitly removed; unknown, unresolved, duplicate, non-contiguous, or reintroduced excluded events reject.
- The production-facing `verify_renderer_audio_binding_for_score` wrapper (feature `theory`) serializes the actual `symthaea_music_theory::Score`, requires those exact bytes to equal the receipt's Score input, and derives expected MIDI pitches directly from `Score::events()`.
- The low-level pitch-inventory verifier is crate-private; callers cannot mint public evidence tokens from arbitrary pitch arrays.
- For v0.1, only `EventIntervention::Remove` is admitted. Transpose, onset, duration and velocity variants remain reserved and fail closed until each has an operation-specific symbolic verifier.

### Paired-render repeatability calibration

- Calibration is tied to the exact source Score, complete baseline map artifact, parser-derived map receipt, baseline MIDI bytes, expected primary-pitch inventory, renderer identity/configuration, anchor PCM, and no-op receipts.
- A minimum of three no-op render receipt tokens is required.
- Reusing the same no-op receipt digest multiple times does not count as multiple controls.
- Every no-op token must bind the same PCM layout, Score, baseline MIDI bytes, map artifact/receipt, pitch inventory, renderer identity and configuration.
- Per-window/per-channel thresholds use the maximum observed no-op difference multiplied by a declared factor, bounded below by absolute configured thresholds.

This calibration is a conservative empirical floor over the supplied controls. It is not a population quantile, confidence interval, proof of independent execution, or universal acoustic tolerance. Different receipt IDs can still be produced by an untrusted local host; the receipt chain remains local artifact evidence, not attestation.

### Replayable influence report

The report binds:
- source-event ID and intervention kind;
- Score and baseline/counterfactual MIDI schedule digests;
- baseline/counterfactual map artifact and parser-receipt digests;
- exact renderer receipts, identity/configuration digests;
- baseline/counterfactual PCM digests and format metadata;
- repeatability calibration digest;
- analysis configuration, per-window/channel measurements, measured support intervals and disposition.

Independent replay re-runs the measurements over the supplied exact artifacts and PCM buffers and requires the report digest and fields to match.

## Critical controls in v0.1

- A complete two-event baseline MIDI fixture with explicit sequencer-specific source markers.
- A counterfactual MIDI fixture removing event 0 while event 1 retains its source ID, MIDI pitch, onset, duration and frame projection despite a shifted note-on ordinal.
- A protected-pitch mutation: removal of event 0 while event 1 is transposed must fail the exact-score pitch verifier before influence evidence is minted.
- Distinct receipt IDs for three no-op controls; duplicate receipt reuse rejects.
- Stale baseline PCM, different renderer configuration, different baseline schedule, different score/map context, detached PCM, altered receipt/report, malformed/non-finite PCM and byte-identical schedule reject.
- The counterfactual map must omit exactly one source event; full-score completeness is not weakened or silently reinterpreted.

## Known limitations and next increments

1. **Not a real renderer run:** the deterministic test fixtures construct receipt-shaped local artifacts; they do not execute FluidSynth. The first integration should use #9090's actual observed process runner after exact-head validation.
2. **No proof of independent execution:** distinct receipt digests prevent accidental reuse of one receipt reference, but unkeyed local receipts cannot prove that multiple processes genuinely ran independently.
3. **No loaded-page attestation:** hashes before/after and child-process observation cannot exclude a change-and-restore race or a privileged malicious host.
4. **No acoustic timing calibration yet:** the event map still uses MIDI tick/tempo-to-frame projections. Instrument delay, sample attack, resampler latency, and effect tails require measured renderer-specific calibration.
5. **No nonlinear attribution theorem:** shared reverb, chorus, compression, saturation, limiting, and overlapping voices can cause interaction. Pairwise/multi-event interventions and interaction residuals must be measured separately.
6. **Limited metric channels:** RMS and spectral magnitude are engineering diagnostics, not perceptual equivalence. Add perceptual or source-aware metrics only as separate independently calibrated channels.
7. **No native schedule support:** NativeMuse needs a canonical performed-schedule serialization captured at the actual synthesizer boundary, not a recomputed schedule.
8. **No confidence-calibrated threshold:** max over three no-op controls is a prototype floor. Held-out repeated renders, tail quantiles or robust uncertainty intervals are required before any probabilistic reliability claim.

## Benchmark programme

### Matched public comparisons

- **SMEBench** (arXiv:2610.36715, October 27, 2026): 3,600 paired synthetic test examples, 12 editing operations and a real-audio complement. Use it for target agreement, source preservation and text compatibility separately. Do not present the benchmark as a ground-truth per-note attribution suite.
- **MMAE / ICASSP 2027 Audio Editing Challenge:** 2,000 public examples and 17,741 rubric checks. Report IFR, CR and EMR separately; EMR requires all required rubrics to pass. The challenge uses a frozen Qwen3-Omni judge with three shuffled evaluations and majority voting. The official hidden test/SDK and leaderboard are scheduled for November 10, 2026; each track has 500 hidden examples. The Agent Track model release cutoff is November 1, 2026, and its registration deadline was extended only through October 8, 2026. Eligibility must be checked with the organizers; do not assume late entry.
- **MuseCPBench** (arXiv:2512.14629): complementary musical-context preservation across musical facets. Keep melody, harmony, rhythm, structure and timbre separate; do not allow good preservation on one facet to compensate for failure on another.
- **Differentiable DSP / DDSP:** use as an architectural comparator or optional acceleration for the native renderer. Differentiable sensitivity is not a substitute for black-box paired-render evidence.

### Proposed ablation matrix

1. MIDI schedule interval projection only.
2. Existing RMS/spectral/onset preservation detector.
3. Event lineage + local render receipt.
4. Event lineage + renderer receipt + measured repeat-render floor.
5. Above plus exact counterfactual event map and protected pitch/timing checks.
6. Above plus calibrated acoustic timing and multi-resolution effect-tail checks.
7. Above plus pairwise interaction measurements and human listening.

### Proposed research gates (not achieved results)

- 100% expected source-event identity and pitch agreement on deterministic supported fixtures.
- Zero false accepts across at least 384 predeclared critical-negative cases, with a Wilson interval and threat-class breakdown. This is an initial test-distribution gate, not a universal guarantee.
- At least 95% held-out support containment on calibrated synthetic fixtures, while reporting precision and mask width to prevent an always-broad mask from gaming containment.
- No reported effect below the measured repeat-render floor.
- Explicit abstention when no-op controls are insufficient, receipts are absent, the renderer/config differs, source identity is incomplete, or measurement is below its floor.
- Paired p50/p95 localization error, protected-region leakage, false attribution, abstention/risk-coverage, renderer cost, extra-render count, and memory per event.
- Human listening study with randomized/blinded trials; preference and perceived preservation remain distinct from raw signal difference.

### Industry-facing stretch targets

These are proposed targets, not Muse results:
- MMAE full 2,000-example scope: reproduce the Agent baseline before comparison (44.06% IFR / 74.63% CR / 7.45% EMR as published in the project benchmark materials); aim for >=12% EMR with paired 95% confidence interval above the reproduced matched baseline, IFR no more than 2 points lower and CR at least 2 points higher.
- SMEBench: statistically supported per-operation improvement against reproduced open baselines on the same synthetic and real-audio strata; report target agreement, source preservation and textual compatibility independently.
- No SOTA claim without the same dataset split, evaluator, model/tool budget, render budget, and failure denominator.

## Implementation and validation status

- Source module: `src/counterfactual_influence.rs`.
- Score-event exception verifier: `src/performed_event_map.rs`.
- Renderer receipt source: `src/renderer_execution_receipt.rs`.
- Source-level test functions are authored, including exact MIDI lineage/receipt replay, no-op receipt reuse rejection, event removal with protected-event equality, protected-pitch rejection, renderer/schedule substitution, and malformed PCM/config rejection.
- Rust tests, Cargo compilation, rustfmt, Clippy, real FluidSynth integration, calibration study and public benchmark runs have **not** been executed in this environment. GitHub workflow states must not be reported as passes unless they actually succeed at the exact head.

## Related work

- [PR #9090 — MEL-MR-EVAL-016 FluidSynth process observation](https://github.com/Luminous-Dynamics/symthaea/pull/9090)
- [Issue #9093 — MEL-MR-EVAL-017 counterfactual acoustic influence](https://github.com/Luminous-Dynamics/symthaea/issues/9093)
- [Issue #9026 — MEL-MR-EVAL-013 renderer execution and performed-time mapping](https://github.com/Luminous-Dynamics/symthaea/issues/9026)
- [PR #8864 — MEL-MR-EVAL-006 localized audio preservation](https://github.com/Luminous-Dynamics/symthaea/pull/8864)
- [PR #8944 — MEL-MR-EVAL-008 symbolic/rendered-audio binding](https://github.com/Luminous-Dynamics/symthaea/pull/8944)
