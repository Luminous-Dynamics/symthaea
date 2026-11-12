# MEL-MR-EVAL-017 — Counterfactual Performed-Event Acoustic Influence

**Status:** first deterministic measurement primitive authored on 12 November 2026. It is not yet integrated with live FluidSynth/NativeMuse render orchestration, and its eight unit tests are authored but not reported as executed.

**Parent issue:** [#9093 — Build counterfactual performed-event acoustic influence maps](https://github.com/Luminous-Dynamics/symthaea/issues/9093)

**Evidence chain:** [#9037 — Performed event map](https://github.com/Luminous-Dynamics/symthaea/pull/9037) → [#9064 — Local render receipt](https://github.com/Luminous-Dynamics/symthaea/pull/9064) → [#9090 — FluidSynth process observation](https://github.com/Luminous-Dynamics/symthaea/pull/9090) → this measurement layer.

## Why this is a separate layer

A performed-event map records which score event produced which MIDI note and when it was scheduled. It does not say which output samples the note acoustically influenced. Release envelopes, reverb, chorus, overlapping voices, shared buses, resampling, compression and limiting can spread or alter an event's audible effect well beyond the scheduled note interval.

The existing audio_preservation verifier compares protected regions between two outputs. It is a preservation decision primitive, not an event-to-output influence estimator. MEL-MR-EVAL-017 measures that missing relationship using a controlled counterfactual pair.

## Paradigm

For a baseline schedule S, a declared intervention on event e, renderer profile C, and renderer R:

- Baseline render: y = R(S, C)
- Same-schedule repeat renders: y'_j = R(S, C) for j in 1..N
- Counterfactual render: y_e = R(intervene(S, e), C)

The estimator measures per-window/channel differences between y and y_e, then estimates a local nondeterminism floor from y versus each y'_j.

This is a conditional intervention effect in one exact context. It is **not** exclusive sample ownership, a source-separation mask, proof of perceptual importance, or a claim that event effects are additive. A nonzero interaction can arise from phase interference even in an otherwise additive signal path; shared effects and nonlinear mastering can create further interactions.

## Implemented first tranche

The new src/counterfactual_acoustic_influence.rs module provides:

- Typed interventions: remove, transpose, shift onset, change duration and change velocity. Zero-delta edits reject.
- Exact context commitments: source Score, source event ID, baseline/counterfactual schedule, renderer profile, baseline/repeat/counterfactual render receipt digests.
- Mandatory same-schedule repeat renders. Missing repeat audio or mismatched repeat-receipt counts reject.
- Strict audio preflight: finite samples, nonzero sample rate/channels, matching format and frame count, and bounded analysis-window sizes.
- Per-channel paired measurements over overlapping windows:
  - sample-domain RMS of the counterfactual difference;
  - normalized one-sided spectral-magnitude L1 distance;
  - maximum observed same-schedule repeat noise for both measurements;
  - thresholds computed as max(absolute floor, noise × multiplier).
- Channel-specific support intervals formed by merging detected overlapping windows.
- Three-way result: Detected, NoDetectableInfluence, or Abstain. No calibration ID always forces abstention while retaining exploratory measurements.
- Domain-separated SHA-256 report receipt and a replay verifier that recomputes the full measurement map from exact supplied buffers.

The lower-level fixture API binds caller-provided renderer receipt digests but does not validate those renderer receipts itself. A new preferred entry point, `estimate_from_verified_renderer_evidence`, accepts each receipt together with its exact `RendererReceiptInputs`, calls the receipt replay verifier, decodes the same WAV bytes, and rejects any output-byte substitution before estimating influence. It also checks that:
- baseline and repeat render receipts bind the exact same serialized Score and schedule digest;
- the counterfactual receipt binds the same Score but the separately declared changed schedule digest;
- all renders use the same backend, executable/SoundFont commitments, semantic render-config digest and output format;
- baseline, repeat and counterfactual receipts have distinct receipt digests;
- the renderer-profile digest is derived from those independently replay-verified receipt fields rather than accepted as an unchecked caller claim.

This still does not attest a malicious host or prove that an untrusted process loaded the claimed pages. It does remove the avoidable gap where the influence estimator could be handed a valid-looking receipt digest alongside different WAV bytes. The caller must still establish that repeat renders are independent executions and that their seed/effects policy is suitable for the experiment.

## Explicit non-claims and limits

- The support interval is window-resolution evidence, not sample-exact causal ownership. Its temporal resolution depends on frame size and hop.
- RMS and spectral magnitude answer different questions. Neither is a universal perceptual metric, and phase-sensitive changes may not be faithfully represented by magnitude distance alone.
- The current first tranche uses one baseline plus at least one repeat. Production experiments should use multiple independent repeats and report the distribution, not only the maximum, of the noise floor.
- The same-schedule repeats must be independent executions. Reusing one output buffer under multiple receipt IDs is invalid experimental practice even though this module cannot detect it.
- The module does not currently orchestrate MIDI rewriting, renderer invocations, native synthesis schedule capture or deterministic seed pairing. The verified entry point does decode exact receipt-bound WAV bytes into PCM; the fixture-level API continues to consume already-decoded audio views.
- A receipt digest establishes internal consistency, not trusted execution, source authenticity, or a malicious-host security guarantee.

## Initial deterministic tests

Eight tests are authored:

1. A source-event removal with a decaying release tail is detected against a same-schedule repeat.
2. A counterfactual deviation that matches repeat-render noise is not falsely classified as an intervention effect.
3. Missing calibration produces Abstain while retaining raw measurements.
4. Stereo support remains channel-specific.
5. Replay rejects changed audio and changed renderer-profile context.
6. Empty event IDs and zero-delta interventions reject.
7. Baseline, repeat and counterfactual NativeMuse fixtures are independently replay-verified before PCM influence analysis.
8. Replacing WAV bytes without changing the corresponding renderer receipt rejects.

These tests have not yet been run by the pinned Rust workflow. No empirical accuracy, false-attribution rate, calibration quality or SOTA result is claimed.

## Next implementation sequence

1. **Live adapter integration:** connect FluidSynth/NativeMuse candidate rendering to the verified-evidence entry point, using its exact WAV bytes and receipt inputs. The first verified NativeMuse fixture path is covered by tests; live Studio orchestration is not yet wired.
2. **Repeat-render campaign:** collect at least three same-schedule repeat renders per deterministic renderer/profile and estimate per-window noise-floor distributions. Test renderers that deliberately vary effects or random seeds.
3. **MIDI intervention compiler:** bind exact source event ID to one typed MIDI/schedule intervention. Verify every non-target event is unchanged before rendering.
4. **Effect-tail calibration:** use isolated notes, repeated same-pitch notes, chords, sustained notes, release tails, reverb/chorus and mastering settings. Establish support-containment, onset/offset error and false-attribution curves.
5. **Non-additivity:** extend to four-coalition paired trials for event pairs and then bounded higher-order coalitions. Keep sample-domain residuals distinct from spectral-domain residuals.
6. **Preservation composition:** combine measured event influence with audio_preservation only after the influence map and preservation report independently replay. Never let the influence estimator self-authorize an edit.

## Benchmark and industry positioning

SMEBench and the ICASSP 2027 MMAE challenge answer different questions from this primitive:

- SMEBench's paired source/target edits measure requested edit agreement and source preservation across controlled operations and a real multitrack complement. It is the right external comparator for edit quality, not ground truth for per-note renderer influence. Paper: https://arxiv.org/abs/2610.36715
- MMAE reports instruction-following, consistency and exact-match rates for broader editing tasks. It is a product-capability comparison, not an event provenance or renderer-influence benchmark. Official protocol: https://audio-editing-challenge.github.io/
- The ICASSP challenge permits agentic systems to combine planners and DSP/model tools. Melothaea's proposed differentiator is not tool count; it is capability-qualified execution and independent preservation evidence.

There is not yet a directly matched public benchmark for per-source-event paired-render acoustic influence maps. The first publishable contribution should therefore be a frozen protocol, datasets and matched baselines—not a superiority claim.

## Proposed evaluation targets — not measured results

| Dimension | Initial research target |
|---|---:|
| Event identity and symbolic intervention correctness on deterministic fixtures | 100% |
| Support containment on held-out synthetic intervention fixtures | ≥95% |
| Critical negative controls | 0 false accepts across ≥384 independent cases, with intervals |
| False attribution | Report per event family and renderer; target <1% |
| Timing support localization | Report p50/p95; target p95 ≤ one analysis hop after calibration |
| Noise-floor calibration | No accepted detection below measured repeat-render threshold |
| Abstention | Report risk–coverage curve; do not reward forced classifications |
| Cost | Report extra renders, wall time and peak memory per source event |

Targets must be frozen on development/calibration splits and evaluated on composition-lineage-held-out data. The 384-case gate is an initial statistical screen, not proof of universal safety.

## Required ablations

- MIDI schedule interval only.
- Existing onset/spectral preservation detector.
- Paired counterfactual without repeat-render noise floor.
- Paired counterfactual with one repeat.
- Paired counterfactual with multiple repeats.
- RMS-only support.
- Spectral-only support.
- Combined RMS + spectral support.
- Full map plus independent preservation verifier.

The final report must keep musical edit quality, protected-content preservation, influence-map calibration, provenance completeness, runtime and render cost as separate measurements.
