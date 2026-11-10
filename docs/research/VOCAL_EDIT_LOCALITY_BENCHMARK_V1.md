# VELB-001 — Vocal edit locality benchmark

**Date:** 10 November 2026  
**Status:** standalone evaluation primitive and synthetic contract tests; no production localized-edit API or real-audio result is claimed.

## Research question

When a singer or editor asks to change one word, phoneme, note, melisma or phrase control, can the system make that change without damaging the rest of the performance?

This is a separate capability from full-clip naturalness or score compliance. A renderer may produce a convincing replacement word while silently altering timbre, timing, pitch or articulation in every other phrase. Whole-song preference and mean opinion scores do not expose this reliably.

VELB-001 measures two things independently:

1. **Target realization:** the declared target region changed by at least a predeclared amount.
2. **Protected-region preservation:** regions outside the requested edit remained unchanged, either bit-exactly or within a renderer-specific, calibrated acoustic tolerance.

It does not determine whether the new pronunciation/note is correct, or whether the edit sounds good. Those claims require the selected pronunciation/score plan, independent phoneme and pitch analysis, and blinded human review.

## Current implementation

The dependency-free Rust evaluator is in tools/vocal-edit-locality-eval/.

Its input contract is deliberately strict:
- mono f32 before/after stems on the exact same sample clock;
- identical sample count and sample rate;
- one explicit target interval;
- one or more protected intervals;
- a complete, contiguous and non-overlapping partition of the full audio;
- finite samples and explicit, finite caller-supplied thresholds.

The evaluator does not auto-align, time-warp, normalize, resample or denoise. Any fixed downmix/preprocessing must be applied identically and recorded by the caller. A changed sample clock, timing shift or global re-render is counted as collateral drift rather than hidden by internal alignment.

### Reported metrics

For each target/protected region:
- before and after RMS;
- absolute RMS sample delta;
- delta normalized by baseline-region RMS (using a 1e-12 numerical floor);
- Pearson correlation where both signals have non-zero variance;
- exact sample-match fraction;
- maximum absolute sample delta.

For the full edit:
- aggregate protected-region RMS delta;
- protected-to-target RMS-delta ratio;
- explicit target and protected-region thresholds;
- objective pass/fail.

Thresholds are expressed in the input amplitude units. No universal tolerance is asserted. Deterministic cached audio may require zero protected-region delta; nondeterministic neural renderers need a separately calibrated tolerance.

### Synthetic positive and negative controls

The seven unit tests currently cover:
- zero target-change thresholds and duplicate region labels are rejected;
- target-only modification with bit-exact protected regions (positive control);
- unchanged target (negative control);
- a collateral change in a protected phrase (negative control);
- gaps, overlaps and incomplete region partitions;
- sample-rate mismatch, unequal lengths and non-finite samples;
- collateral changes in silent protected regions, where correlation is undefined but absolute RMS delta remains measurable.

These tests validate the scorer's arithmetic and fail-closed input contract. They are not audio-model or listener results.

## How to connect it to VocalPerformance v3

The pronunciation/score alignment graph should be the authority for target and protected regions:

1. Start from a frozen, provenance-bound performance plan and rendered dry-vocal stem.
2. Apply one explicit edit operation, such as changing a selected word pronunciation or one note's pitch/control.
3. Derive the target interval from the v3 alignment graph and its dependent score events.
4. Derive protected intervals from all non-dependent phrases/score events. Keep rests and coarticulation windows explicit.
5. Re-render only the dependency cone when the backend supports localized rendering. Otherwise render the full performance and let the evaluator quantify collateral drift.
6. Preserve exact input plan digest, edit-patch digest, renderer/model artifact digest, runtime/config digest, source and output audio digests.
7. Run phoneme/score analyzers on the target and protected intervals. A low waveform delta is not proof that the corrected word is right.
8. Use blinded listeners to assess naturalness, intelligibility, identity continuity and musical coherence.

The first version supports exactly one target interval per report. A multi-region edit should be evaluated as separate target reports or by an explicitly designed multi-target extension; do not turn unrelated gaps into a large target region.

## Controlled experiment matrix

Use the same licensed voice, source performance, renderer artifact and frozen score/lyrics for all comparisons.

| Edit class | Target check | Protected-region check |
|---|---|---|
| One-word pronunciation correction | Selected candidate realized; word/phone sequence correct | Other words, phrase boundaries and voice identity preserved |
| One-phoneme correction | Intended phone identity/timing improved | Neighboring phones and unrelated phrases preserved |
| One-note pitch correction | Correct note/transition achieved | Other notes, vibrato and phrase-level expression preserved |
| One phrase dynamics change | Intended energy contour changed | Other phrase energy/timbre preserved |
| One breath-placement change | Breath/rest event moved or added as requested | Adjacent phrases and lyric timing preserved |
| One melisma correction | Designated vowel remains continuous over assigned notes | Consonants, rests and other phrases remain unchanged |

Record target success, protected drift, human acceptability and edit latency separately. Never hide a bad target result behind excellent preservation, or destructive collateral changes behind an excellent target result.

## Fair comparison

Compare only operations that each backend can actually express. Publish a backend capability manifest classifying each control as native, approximated, conditional or unsupported. An unsupported local-edit operation is not a failed waveform score; it is unavailable functionality and must be reported as such.

For deterministic renderers, test whether unaffected cached intervals are byte-identical. For stochastic renderers, freeze seeds where possible, estimate the no-edit self-variance distribution, and set protected thresholds using a calibration split before evaluating edit requests. Do not calibrate thresholds on the same cases used to claim improvement.

All before/after audio must use identical sample-rate, channel, alignment and preprocessing policies. Keep original dry WAVs; make listening-normalized copies separately and record their transformation.

## Industry comparison and opportunity

Whole-song arenas such as Artificial Analysis measure blind prompt-to-song preference. Score-conditioned systems such as SoulX-Singer and DiffSinger provide more direct musical-control comparisons. MusicJudge provides a useful independent reference for block-aligned lyric and pitch/rhythm assessment.

None of these by itself answers whether a localized correction preserves all unaffected audio regions. This can become a differentiator for Muse: not merely creating a convincing vocal, but allowing users to revise a musical-linguistic performance with bounded, inspectable side effects.

The competitive claim should be a Pareto profile across:
- target correctness;
- protected-region preservation;
- naturalness/intelligibility;
- identity and expressive continuity;
- supported control coverage;
- edit latency and compute;
- reproducibility and evidence completeness.

Do not claim state-of-the-art edit locality until at least two integrated backends and a held-out, human-reviewed edit corpus have been evaluated under the same protocol.

## Acceptance milestones

1. **Scorer correctness:** all synthetic positive/negative controls pass at the exact commit.
2. **Audio ingestion:** a provenance-bound WAV loader/downmix path is implemented with exact sample-clock metadata and digest receipts.
3. **Renderer integration:** at least one backend exposes an edit or deterministic localized rerender operation.
4. **Reference set:** licensed edits annotated with target/protected intervals and human acceptability labels.
5. **Cross-backend test:** same edit cases run through Kokoro, DiffSinger and the resolved-plan formant adapter only for controls each can honestly support.
6. **Confirmatory study:** blinded ratings with predeclared thresholds, rater-aware uncertainty and separate technical/perceptual results.

## Claim boundary

This PR provides a standalone locality-scoring primitive and deterministic synthetic tests. It does not add a production edit API, load or compare actual vocal WAV files, or demonstrate naturalness, human preference, identity preservation or SOTA performance.
