# MEL-MR-EVAL-006 — Localized Rendered-Audio Preservation Audit

**Status:** executable Rust measurement primitive; calibration and end-to-end evaluation remain pending  
**Tracking:** [MEL-MR-EVAL-005 protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-005-audio-preservation-protocol-20261029/crates/domains/symthaea-music-theory/MEL-MR-EVAL-005_RENDERED_AUDIO_PRESERVATION.md)  
**Implementation:** `crates/domains/symthaea-muse/src/audio_preservation.rs`

## Research change: from global scores to localized invariants

A global similarity score can hide a small but important corruption when most of a recording is unchanged. This module therefore evaluates complete short-time windows outside the authorized edit mask, per channel, and reports the worst values and every violating window. Time and channel are not averaged away. A channel-window violates the operational gate when either the absolute RMS-level change exceeds its configured dB tolerance or the normalized spectral-magnitude distance exceeds its configured threshold.

This is a deliberately narrow, deterministic detector suite—not a claim of perceptual equivalence. The module now has two independent checks: (1) window-local RMS level and spectral-magnitude distance; and (2) normalized positive spectral-flux onset matching with an explicit millisecond tolerance, per-channel one-to-one matching, and protected edit masks. The onset detector is intended to expose temporal shifts that an aggregate similarity score can obscure. It can still produce domain-specific false positives/negatives, does not model source attribution in dense mixes or arbitrary effect tails, and needs calibration plus blind listening validation.

## Measurement contract

- Inputs are interleaved `f32` buffers with explicit sample rate and channel count.
- Edit intervals use half-open audio-frame coordinates, not scalar sample-array offsets.
- Both inputs must have the same sample rate, channel count, and frame count. There is no automatic alignment, resampling, trimming, or loudness normalization.
- Frames intersecting the authorized mask expanded by a configured boundary guard are skipped. The guard is an explicit allowance, not a learned reverb-tail model.
- Every other complete window is measured independently per channel.
- The onset audit computes spectral-flux novelty per channel and matches detected onset events one-to-one within the configured timing tolerance. An event outside tolerance is conservatively recorded as a missing reference onset and an extra candidate onset.
- Onset events inside the authorized mask expanded by both the detector's STFT support (`frame_size / 2 + hop_size`) and the configured boundary/spillover guard are excluded. The effective guard is recorded in the receipt. The onset audit abstains when there is too little protected duration, no calibration ID, or no measurable protected reference onset.
- Threshold exceedance in any protected channel-window blocks acceptance. Non-finite audio, malformed masks, and incompatible inputs are errors.
- Passing additionally requires an externally maintained calibration ID and a minimum number of protected windows. An identifier is only recorded; the module cannot prove that calibration is real, appropriate, or frozen.
- Reports contain SHA-256 commitments over domain-separated sample bits and format metadata, plus a separate receipt SHA-256 computed from the canonical serialized report with the receipt field blank. The receipt digest covers the verdict, measurements, thresholds, content digests, format, and config; verification detects tampering but does not authenticate authorship, capture time, renderer identity, or calibration authenticity.
- FFT analysis windows are bounded to at most 1,048,576 frames. Spectral inputs are peak-scaled only when amplitudes exceed 1.0, preventing finite high-amplitude f32 inputs from overflowing the f32 FFT; the independent RMS-dB channel still detects gain changes.

## Compositional audio-evidence gate

The module now exposes `compose_audio_preservation_evidence`, which combines the acoustic and onset reports only when both child receipt digests verify and both reports carry identical reference/output content hashes, sample rate, channel count, and frame count.

The combination rule is deliberately non-compensatory:

- Any child `fail` makes the combined audio-only verdict `fail`.
- Otherwise, any child `abstain` makes it `abstain`.
- The combined audio-only verdict is `pass` only when both child reports pass and their content/format bindings match.
- A tampered child receipt returns an error; mismatched input bindings cannot be combined.
- The composite report includes both child receipt hashes and has its own receipt digest.

This is **not** the complete edit-contract verdict. The symbolic requested-change and invariant verifier must separately pass before the overall edit can be accepted. The API is intended to make the audio-evidence composition rule explicit and replayable, not to imply that the current two acoustic channels cover every audible or symbolic failure mode.

## Tests included

The Rust unit suite contains 20 tests covering identity preservation, one localized protected-region mutation, authorized-only changes, no-calibration abstention, stereo mutation, mismatched duration, non-finite samples, boundary guards, onset identity, delayed protected transients, onset masking, onset abstention, and abstention when no reference onsets can be measured, receipt tampering, oversized analysis settings, high-amplitude finite input, composite pass conditions, cross-input receipt rejection, tampered-child rejection, and abstention composition. The runner also verifies child receipt hashes as part of its conformance verdict; the four composite-gate regression cases live in the Rust unit suite. An independent JS numerical reference check using a direct DFT also checked the synthetic burst fixture: the baseline onset was detected at frame 992 and the delayed onset at frame 1504 (512-frame displacement). This is an algorithm sanity check, **not execution of the committed Rust code**. Rust tests, formatting, and clippy remain unexecuted in this environment; a passing CI result must be attached to the exact PR head before implementation validation is claimed.

## Calibration and benchmark plan

1. Pin renderer version, sample rate, channel layout, instrument bank/effects, effect-state resets, and latency handling. Record where exact deterministic replay is not available.
2. Generate matched source/output pairs with known truth: identity, authorized-only mutation, wrong-region edit, wrong-channel edit, gain shift, frequency substitution, onset displacement, clipping, phase-only variation, reverb-tail extension, and renderer-version mismatch.
3. Split by composition/motif family **and** renderer/configuration. Fit thresholds on a calibration partition; freeze them before held-out testing. Deduplicate source identities before splitting.
4. Measure sensitivity and false-accept rate per mutation family, not only overall. Publish counts and confidence intervals. An initial target is zero accepted critical mutations in at least 300 independently generated/annotated negatives, but this is a pilot target—not a safety proof.
5. Calibrate the spectral-flux onset threshold and timing tolerance separately from the RMS/spectral thresholds. Evaluate onset precision/recall, timing-error percentiles, missing/extra events, and false positives under sustain, vibrato, slow attacks, reverb, and phase/pitch-preserving shifts.
6. Add source-aware checks before expanding the detector's claim domain.
7. Run blinded listening trials to measure whether detector verdicts track audible collateral changes. Keep technical preservation, requested-edit success, naturalness, and preference as separate endpoints.

## Synthetic conformance runner

Run the fixture-only conformance suite with:

```bash
cargo run --locked -p symthaea-muse --example mel_mr_eval_006_conformance
```

The runner emits JSON for eleven predeclared cases and exits non-zero if any expected verdict is not reproduced:

- identical protected audio → pass;
- localized mutation outside the edit mask → fail;
- localized gain drop → fail;
- local 440 Hz to 880 Hz substitution → fail;
- mutation only inside the authorized mask → pass;
- identical transient timing → pass;
- 512-frame transient delay in protected content → fail;
- additional transient in protected content → fail;
- edit mask plus analysis guard hides all onset evidence → abstain;
- silence contains no measurable onset evidence → abstain;
- removal of a protected reference burst → fail.

The runner labels its threshold ID as synthetic-only. It is a quick regression/conformance suite, not a production calibration, a human preference study, or a leaderboard result. Each case has a mutation-family label and an explicit critical-mutation flag. The 11 fixtures are grouped into five families: identity controls (2), protected acoustic mutations (3), onset-event mutations (3), authorization-scope control (1), and evidence-coverage controls (2). The JSON reports per-family expected-outcome counts, critical false accepts, false rejects, abstentions, and two-sided 95% Wilson intervals with denominators. This makes small-panel uncertainty visible instead of relying on a bare percentage. It should be run on every code revision before collecting a larger calibration set.

An independent NumPy/FFT reference simulation was executed for the same eleven scenarios in the development environment and reproduced all eleven predeclared verdicts, including the gain-drop, frequency-swap, and extra-transient mutations. The fixture definitions contain six critical negative controls and three expected-pass controls. Even if this tiny fixture panel observes zero false accepts, the two-sided 95% Wilson upper bound for the false-accept rate is about 39.0% (0/6); that is far too weak for a reliability claim. Zero false accepts in 300 independent critical negatives would lower that same bound to about 1.26%, still not proof of zero population risk. These are statistical properties of the sample sizes, not measured performance of the Rust implementation. This remains independent reference evidence—not execution of the Rust runner, whose compile and test status remains unknown until a Rust toolchain is available.

## Independent reference oracle

A language-independent NumPy oracle is included as research/test tooling at `research/mel_mr_eval_006_reference.py`. It mirrors the documented synthetic signal definitions and metric semantics without importing or executing the Rust implementation. Its output includes per-case diagnostics, grouped mutation-family results, Wilson intervals, its own script hash, and the exact Rust source/runner blob identities supplied on the command line.

Reproduce the reference run with:

```bash
python3 crates/domains/symthaea-muse/research/mel_mr_eval_006_reference.py \
  --source-blob-sha 771b75ee53dcb42f4ee594a2589a2e241e9a8257 \
  --runner-blob-sha d43e960af838bd090187e1e8d66a285e4ec38dcf \
  --output /tmp/mel-mr-eval-006-reference.json
```

The source/runner SHAs above identify the code snapshot used by the preserved sanity artifact. They will need to be updated when rerunning against later blobs. The oracle source is committed for reproducibility, but **the committed script itself has not been executed in this environment**; the earlier equivalent reference simulation ran independently, and the Rust implementation remains unexecuted.

## Preserved reference artifact

The independently executed synthetic reference check is committed as `MEL-MR-EVAL-006_REFERENCE_SANITY_2026-10-29.json`. It pins the acoustic source and conformance-runner file blob hashes and records every case's verdict, representative measurements, onset frames, mutation family, and uncertainty intervals. It explicitly records that this was a Python/NumPy reference simulation—not execution of Rust.

The fixture panel observed 0 false accepts in 6 critical negatives and 0 false rejects in 3 positive controls. The 95% two-sided Wilson upper bounds are approximately 39.0% and 56.1%, respectively. This is why the panel is only a smoke test. The future reliability gate should target at least 300 independent critical negatives and report composition/renderer-family holdouts; the interval would still be non-zero even then.

## Positioning against public benchmarks

Treat the evaluator as one layer in a broader stack, not as a replacement for music or instruction-level benchmarks:

- **Known-truth mutations:** this module checks localized acoustic and temporal fault injection, where intended outcome is deterministic.
- **Symbolic edit locality:** MEL-MR-EVAL-004 / MEL-MR-EVAL-005 evaluate the requested symbolic change and protected symbolic properties.
- **Music context preservation:** [MuseCPBench](https://arxiv.org/abs/2512.14629) and the companion [MuseCPEval implementation](https://github.com/Yashvishe13/MuseCPEval) cover musical facets such as harmony, rhythm, structure, melody, and timbre. Use these scores as complementary measures; do not let a high global facet score compensate for a known critical local mutation.
- **General editing instructions:** [MMAE](https://arxiv.org/abs/2606.07229) reports instruction-following, consistency, and exact-match outcomes. A future guarded-editor ablation should report all three on a matched public split.

The research hypothesis is that complementary oracles reduce blind spots: exact symbolic assertions for symbolic content, localized signal measurements for acoustics and timing, musical-context metrics for higher-level facets, and blinded human ratings for perceptual validity. The hypothesis must be tested; combining more metrics does not by itself guarantee better decisions.

## Required reporting

Report raw denominators, calibration set ID, held-out composition and renderer families, thresholds, mutation-family confusion matrix, protected-window coverage, abstentions, p50/p95 runtime, Rust commit SHA, exact invocation, and input/output hashes. Never translate a deterministic fixture pass into a general claim that audio is perceptually preserved or that Symthaea beats a state-of-the-art editor.

## Benchmark gate (targets, not attained results)

| Gate | Target for a future calibrated study |
|---|---|
| Deterministic controls | 100% expected outcomes on the predeclared conformance suite |
| Critical mutation sensitivity | No critical mutation accepted in the test panel; publish confidence bounds |
| Requested edit | Report separately; the preservation detector alone cannot establish it |
| Human-audible collateral change | Blinded human agreement and detector sensitivity/specificity with intervals |
| Distribution shift | Report by unseen composition and unseen renderer strata |
| Reproducibility | Exact receipt recomputation and rejection of cross-input/tampered receipts |
| Efficiency | p50/p95 detector latency, including FFT work, separated from rendering time |

The first paradigm shift is architectural: use typed authorization masks plus independent localized detectors and content-bound receipts, then test the *incremental causal effect* of that guard against the same editor with matched inputs and budgets. This does not remove the need for stronger detectors or external baselines; it makes that comparison falsifiable.
