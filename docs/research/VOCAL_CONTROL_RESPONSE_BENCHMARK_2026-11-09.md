# VCRB-001 — Vocal Control-Response Benchmark

**Date:** 9 November 2026  
**Status:** executable feature-delta scorer and proposed audio protocol; no renderer has yet passed real-audio intervention testing.

## Why this benchmark

Most singing-synthesis evaluations measure output quality, melodic accuracy, lyric accuracy, speaker similarity or listener preference. Those are important, but they do not answer whether an exposed control does what it says.

A renderer can expose `breathiness`, `vibrato_depth_cents`, `portamento`, `energy` and `articulation` as parameters while:
- ignoring the control;
- changing the wrong acoustic property;
- changing several unrelated properties;
- satisfying a symbolic parameter change without changing the rendered sound;
- changing the sound but damaging lyrics, pitch or identity.

VCRB tests each control as a pre-registered intervention over matched renders.

## Paradigm

Treat each control as a testable causal hypothesis:

> Holding lyrics, pronunciation, score, voice identity, backend, model artifacts, renderer settings, random seed and post-processing fixed, changing control X should measurably change acoustic target Y while preserving guard metrics Z.

This is stronger than checking that a struct field changed. It compares measured render features from the baseline and intervention.

The evaluator under `tools/vocal-control-eval` implements this contract over already-measured feature vectors. It normalizes changes by pre-registered metric ranges, requires the primary target to move in the requested direction, and fails when collateral metrics exceed their guardrails. Missing, non-finite, or out-of-range measurements fail closed.

## Test protocol

1. Freeze a valid VocalPerformance and backend/model artifact manifest.
2. Render a baseline stem.
3. Clone the performance and change exactly one declared control.
4. Preserve all other input fields and renderer settings; use a matched seed where supported.
5. Render the treatment stem.
6. Extract features using at least two independent analysis paths for high-impact metrics.
7. Score the target effect and collateral guardrails with frozen ranges and thresholds.
8. Repeat over phrases, registers, singers, voices and backend variants; report per-case results and uncertainty.
9. Conduct a blind listener check that the requested perceptual effect is actually audible and that collateral damage is not perceptible.

For stochastic renderers, use repeated paired seeds and analyze within-seed treatment-minus-baseline effects. Do not pair unrelated random renders and call the difference a control effect.

## Initial control matrix

| Intervention | Primary acoustic target | Collateral guardrails |
|---|---|---|
| Increase breathiness | aspiration/noise-to-harmonic ratio; independently rated breathiness | F0 median, vibrato rate/depth, loudness, phone error, intelligibility |
| Increase vibrato depth | F0 modulation depth in cents | mean F0, rate, note accuracy, phone timing, loudness |
| Increase vibrato rate | modulation frequency in Hz | depth, mean F0, vibrato regularity, note onset |
| Delay vibrato onset | time from vowel/note onset to stable periodic modulation | pitch center, onset timing, phone identity |
| Increase portamento | glide duration and pitch-trajectory curvature | target note center, note boundaries, phone timing |
| Sharpen consonants | consonant-specific spectral/temporal cues and listener clarity | vowel formants, pitch, duration, identity |
| Increase vocal energy | calibrated RMS/loudness and perceived intensity | clipping, spectral tilt, pitch, breathiness |
| Change register | independent register classifier and spectral envelope | lyric intelligibility, pitch accuracy, identity continuity |
| Shift breath placement | detected breath onset and phrase coverage | word omissions, phrase timing, pitch continuity |

Each primary target needs an operational definition and measurement range. A predictor's own uncalibrated output is not sufficient. Acoustic measures should be paired with blinded listener judgments and independent analyzers.

## Proposed qualification goals

These are target gates for a pilot, not observed renderer performance:

- 100% of deterministic positive-control fixtures pass.
- 100% of wrong-direction and insufficient-effect fixtures fail.
- 100% of missing, NaN, infinite and out-of-range measurements fail closed.
- At least 90% per-control pass rate on repeated matched synthetic/render fixtures before promoting that control as dependable.
- No control passes if any predeclared critical guard (lyric identity, pitch center, voice identity, clipping) fails.
- Report effect size, guardrail failure rate, variance across seeds and confidence intervals; never report only a composite score.

For human confirmation, establish inter-rater agreement in a pilot and power the confirmatory study from pilot variance. Blind backend, condition, and treatment direction where feasible.

## Industry and SOTA comparisons

- **SoulX-Singer** is a strong external open baseline for score-conditioned and zero-shot singing. Its stated training scale exceeds 42,000 hours and it supports symbolic score/MIDI conditioning as well as F0 conditioning. It is the first priority for same-score quality/fidelity comparison, but its public benchmark does not replace a control-response study of every exposed parameter.
- **Artificial Analysis AA-Music-Vocal v1.1** uses blind pairwise human preference for prompt-to-song systems, with Bradley-Terry-derived Elo. That is a valuable whole-song production frontier but not a score-conditioned intervention benchmark.
- **MusicJudge** is a useful reference for joint lyric and music-block correctness, not a substitute for controlled parameter interventions.
- **SingMOS-Pro** and MOS predictors provide auxiliary quality measures, not ground truth for the acoustic mechanism of each control.

The competitive claim we should target is not "one parameter slider exists." It is a reproducible response surface: Symthaea knows how much each control changes its intended acoustic property, how strongly controls interact, where controllability breaks down, and how to compensate without violating lyrics, score or voice identity.

## Paradigm-shifting extensions

### 1. Learn a control response surface

Estimate the local Jacobian of measured audio features with respect to performance controls. This reveals:
- controls with no measurable effect;
- controls that are entangled;
- controls whose effect changes by register or vowel;
- control ranges with saturation or instability;
- combinations requiring compensation.

The first prototype can use finite differences on paired renders. More advanced systems may learn a differentiable surrogate, but all claims must be checked against held-out real renders.

### 2. Invert the response model

Instead of asking a user to guess parameter values, let them specify an acoustic/perceptual target ("more breathy, retain pitch center and keep consonants crisp"). Solve for the smallest control change that meets the target subject to hard guardrails.

### 3. Make controls uncertainty-aware

A requested effect should return target achieved, partially achieved, unsupported, or uncertain. The renderer must not claim a breathiness change if the feature estimator has insufficient support.

### 4. Localize edits

Change one phrase or phone while preserving unaffected regions. Measure target success and collateral change separately, including identity continuity and boundary artifacts.

### 5. Share motor controls across renderers

The same high-level motor plan should map to different renderer-specific controls. The response matrix tells us when two backends implement nominally similar controls with different acoustic consequences.

## Relationship to existing vocal work

- MEL-VOCAL-001: phone identity, timing and melisma alignment.
- VOCAL-STRICT-001: validated VocalPerformance and fail-closed phoneme/language handling.
- VCEB-001: six-track vocal control, identity, fidelity, local-edit and perceptual benchmark.
- VCRB-001: measured response of individual performance controls.

These are complementary. A breathiness intervention cannot pass if it damages phoneme identity; a voice identity similarity score cannot substitute for perceptual naturalness; and a good MOS score cannot establish score compliance.

## Sources

- SoulX-Singer paper: https://arxiv.org/abs/2602.07803
- SoulX-Singer implementation: https://github.com/Soul-AILab/SoulX-Singer
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/music
- Recent survey on music generation representations, control and evaluation: https://doi.org/10.1007/s10462-026-11582-x

## Claim boundary

The Rust scorer currently evaluates supplied measured feature vectors and has synthetic unit tests. It does not perform audio analysis, render a treatment, or establish that any Symthaea control works perceptually. Real-audio extraction, matched render receipts, external backend comparison and blinded listening are required before making a control-quality claim.
