# Track A trained smoke test — result (2026-07-30)

Follow-up to `examples/track_a_smoke_test.rs` (untrained, WER=100%, naturalness below even a
known-bad baseline — that harness's own adversarial-review agent flagged the result as not a
fair capability ceiling, since the controller had never been trained). This document records
the actual, real result of training it before synthesizing, per that agent's explicit
recommendation.

## What was run

`examples/track_a_trained_smoke_test.rs` (committed alongside this doc):

1. Built a `FormantTarget` for all 39 canonical non-silence ARPAbet phonemes this crate
   recognizes, derived from `phonetics::arpabet_articulation`'s own articulatory metadata (not
   hand-picked) — the full accepted set, covering vowels, stops, fricatives, nasals, liquids,
   glides, and affricates.
2. Trained `VocalTractPipeline::controller` on that full table via the crate's own real,
   unit-tested `train_on_phoneme_targets` method: 40 epochs, default hyperparameters.
3. Synthesized 8 phonetically-balanced short phrases (up from the original 3) through the SAME
   audio backend as the untrained harness (`speech::vocoder::synthesize`).

Run via `cargo run --release -p symthaea-vocal-tract --example track_a_trained_smoke_test
--features hound` (release build required — a debug build ran for 30+ minutes without
finishing 40 epochs on this 39-phoneme table and was killed; release finished the same training
in well under a minute after a 9m14s compile).

## Result: training did NOT converge, and neither WER nor naturalness improved

**Training loss got WORSE, not better**: `first_epoch_loss=0.1470` →
`final_loss(after 40 epochs)=0.1596`. The harness's own honest convergence check (mirroring
`controller.rs::test_phoneme_training_reduces_loss`'s before/after pattern) correctly printed
`NO IMPROVEMENT (final >= first) -- training did not converge`.

**WER, scored with the exact faster_whisper method/venv used throughout the singing-voice
research arc** (`base` model, int8, CPU, normalized word-level Levenshtein against the literal
word each phrase is named after):

| Set | n | mean WER |
|---|---|---|
| Untrained (3 phrases) | 3 | 1.000 |
| Trained (8 phrases) | 8 | 1.000 |

No change — every render, trained or not, was fully unintelligible to the transcriber.

**Naturalness (DNSMOS + UTMOS, same method/venv as the singing-voice arc)**:

| Set | n | mean DNSMOS (ovr) | mean UTMOS |
|---|---|---|---|
| Untrained | 3 | 1.496 | 1.551 |
| Trained | 8 | 1.453 | 1.555 |

Both metrics are statistically flat between the two sets — DNSMOS nominally *lower* for the
trained set, UTMOS nominally higher by a negligible amount. Neither is a meaningful movement;
both are well within the noise floor already established as inherent to this whole family of
articulatory-synthesis smoke tests in this monorepo (see the singing-voice arc's own repeated
observation that DNSMOS/UTMOS sometimes disagree in direction on small changes).

## Epoch-budget sweep (2026-07-30 follow-up) — the under-training hypothesis does NOT hold up

The interpretation below originally guessed under-training as the likely explanation. That
guess was tested directly with `examples/track_a_training_epoch_sweep.rs`: train a FRESH
controller (same genesis/init each time — not a continuation) for exactly N epochs in one
call, for N in {1, 10, 40, 100}, and see whether loss trends down with more budget.

**Important mechanism this harness accounts for**: `train_on_phoneme_targets_configured`'s
learning rate follows a cosine schedule *within* one call (`lr_peak` at epoch 0 down to
`lr_min` at the final epoch, scaled to that call's `epochs` argument) — so calling it
repeatedly in small chunks would restart the anneal at high LR each time, not continue a
single decay. The sweep instead trains one fresh controller per budget, an apples-to-apples
comparison of "one full anneal at budget N."

**Real result** (bit-exact reproduced across two independent runs):

| epochs | loss | trend |
|---|---|---|
| 1 | 0.1470 | baseline |
| 10 | 0.1607 | worse |
| 40 | 0.1596 | better than 10, still worse than 1 |
| 100 | 0.1594 | better than 40, still worse than 1 |

Loss got worse almost immediately (1→10 epochs), then plateaued around 0.159-0.160 through
100 epochs — never approaching, let alone beating, the 1-epoch reading. **The best loss of the
entire sweep was at epochs=1.** Budgets of 200-4000 were not run: at the ~12s/epoch rate
observed (151 cumulative epoch-equivalents took ~30 real minutes even in a `--release` build),
that would cost several more hours for a smoke test, not run.

## Interpretation — honest, not overclaimed, and corrected from an earlier guess

This is a genuine negative result, not a null/inconclusive one: neither the original 40-epoch
run nor the epoch-budget sweep found training helping, on any axis measured (WER, DNSMOS/
UTMOS, or the loss itself). **The under-training explanation floated immediately after the
first 40-epoch run does not hold up** — more epochs didn't help, they made things worse and
then plateaued at a level strictly above the 1-epoch reading. The more consistent read of the
sweep data is some mix of (a) the 1-epoch reading being a lucky/non-representative early
snapshot rather than a real optimum, and/or (b) genuine interference across the 39 diverse
phoneme targets in one shared output projection — plausible, neither confirmed. What IS now
fairly well-supported: simply "train it longer" is not a fix for this crate's current training
setup on a table this size, at least not without changing something structural (curriculum,
per-manner loss weighting, a lower peak LR, or more capacity) — none of which were tried.

**What this does NOT license claiming**: nothing here says anything about Track A's missing
physical/gesture synthesis layer (the actual "Series 23" renderer) — this smoke test only ever
exercised the existing legacy formant-cascade backend (`speech::vocoder::synthesize`).

## Not pursued this pass

Training-hyperparameter changes beyond epoch count (lower peak LR, curriculum ordering,
per-manner-class loss weighting, more network capacity) were not attempted — this document
reports what was tried and its real result, it does not chase the result to a positive
outcome. If this line is revisited, the next informative step is probably NOT more epochs
(already shown not to help) — it's isolating whether the 1-epoch reading is real (e.g. by
checking loss on a smaller, non-interfering subset of phonemes at higher epoch counts) before
assuming this crate's controller can ever fit the full 39-phoneme table well.
