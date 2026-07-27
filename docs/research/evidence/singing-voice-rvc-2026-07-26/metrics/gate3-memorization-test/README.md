# Gate 3: memorization/overfit test (2026-07-26)

An external review of Gate 2's finding ("bottleneck localized to the
acoustic model's mel prediction") correctly flagged that "under-trained"
was plausible but not yet *proven* — the same symptom could equally come
from a broken pipeline, bad alignment, feature-normalization mismatch, or
a model that can't learn even in principle. It proposed a full multi-axis
program (step ladder × data ladder × frontend/alignment ladder) as the
rigorous way to settle it. That's a multi-day undertaking; this gate is
the single highest-value piece of it, scoped down to a bounded, cheap
test matching this investigation's existing gate methodology: **can the
acoustic model learn to reproduce ONE clean real phrase to intelligibility
at all, given focused training on nothing else?**

If it can't: something is broken in the pipeline (alignment, feature
mismatch, capacity), and more data/steps on the full corpus won't help.
If it can, and does so quickly: the full-corpus run's failure (Gate 2) is
consistent with simple under-training (2000 steps spread across ~100
songs means very little gradient signal per song), not a structural
defect.

## Method

- **Dataset**: literally one real phrase — "won't you sing along with
  me" (en001a, 60.0-63.8813s, the same clip used throughout Gates 1-2),
  used as the ONLY training example (`build_overfit_dataset.py`). A
  second identical copy (`overfit01_val`) is marked as the validation set
  so the binarizer's train/valid split has a non-empty valid set —
  train==val by design, this is not a generalization test. A 0.4s real
  trailing-silence tail was included solely to satisfy the binarizer's
  phoneme-coverage check (needs at least one SP occurrence); this is real
  audio content, not synthetic.
- **Dictionary**: a minimal 14-phoneme dictionary (`overfit-01.txt`)
  covering only the phonemes this one phrase uses — the full 40-phoneme
  `csd-en.txt` dictionary requires every phoneme to appear at least once
  in training data, which one clip can't satisfy. This only changes the
  embedding table size, not what's being tested.
- **Config** (`overfit01_acoustic.yaml`): same architecture/hyperparameters
  as the main run, augmentation disabled (pitch-shifting would confound
  "can it memorize" with "can it memorize across synthetic pitch
  variance"), `max_updates: 6000` with checkpoints every 1000 steps.
- **Eval**: at each checkpoint, render the exact training phrase (same
  real ph_seq/ph_dur/F0 the model trained on — `build_overfit_ds_file.py`,
  reusing the same real-F0-extraction code as the rest of this bundle)
  through both the trained NSF-HiFiGAN vocoder and the Gate-2-style
  independent Griffin-Lim path (`gate3_eval_checkpoint.py`), then
  transcribe both with Whisper.

## Result

**Checkpoint at step 1000** (training stopped here — see below):

| Render path | Whisper transcript |
|---|---|
| Predicted mel → trained vocoder | "I want you to sing along with me" |
| Predicted mel → Griffin-Lim | "I want you to sing along with me" |

Both paths transcribe **identically to the real ground-truth audio's own
Whisper transcription** from Gate 2 ("I want you to sing along with me" —
the same "won't you"→"I want you to" ASR quirk Gate 2 already
established on the real recording, not a new error). This is a clean,
unambiguous positive: the model reproduced the training phrase
intelligibly, via both vocoder paths, after only 1000 training steps
focused entirely on this one example — half the step count of the full
2000-step/100-song run that produced Gate 2's unintelligible result.

Training loss (`loss-trajectory.csv`) shows real, noisy-but-converging
descent (single-sample batches make per-step loss noisy; `mel_loss` drops
from an initial ~0.14-0.7 range down to a ~0.02-0.06 floor by step
~1000-1500), consistent with genuine learning rather than a stuck/broken
optimizer.

**Training was stopped at step 1526** (mid-epoch, via SIGKILL after a
graceful SIGTERM didn't exit promptly) once the step-1000 checkpoint gave
an unambiguous answer — continuing to the full 6000-step ladder wasn't
needed to answer the bounded question this gate asked, per this
investigation's practice of stopping once a gate's question is answered
rather than running the full originally-scoped program regardless.

## Honest interpretation

**This directly demonstrates the model CAN learn to reproduce clean
content to intelligibility, and does so quickly on focused signal.**
Combined with Gate 2's finding, the most parsimonious explanation for the
Gate 2 acoustic model's unintelligible predictions is now **under-training
from signal dilution** — 2000 steps of gradient updates spread thin
across ~100 songs' worth of content gives each individual phrase far less
effective training than 1000 steps devoted entirely to it — rather than a
pipeline defect, broken alignment, or an architectural incapacity to
represent intelligible speech. This doesn't rule out every alternative
the external review raised (feature-normalization mismatch, alignment
quality on other phrases, generalization across unseen phrases) — this
gate tests memorization capacity specifically, not generalization, and
n=1 phrase — but it does rule out "the pipeline/architecture cannot
produce intelligible mel predictions at all," which was the live
competing hypothesis to "just needs more training."

**What this changes**: the natural next lever for improving DiffSinger
singing intelligibility is training scale (more steps and/or more varied,
better-covered data), not further vocoder swapping, RVC settings, or
duration-heuristic patching — all three of which Gates 0-2 already
excluded. A full generalization test (train on N songs, evaluate on
held-out phrases) remains the next real question if training-scale
investment is pursued, since this gate only proves memorization, not
generalization.

## Files

- `build_overfit_dataset.py` — single-phrase raw dataset builder
- `build_overfit_ds_file.py` — eval `.ds` builder (real ph_seq/ph_dur/F0)
- `overfit01_acoustic.yaml` / `overfit-01.txt` — training config and minimal phoneme dictionary
- `gate3_eval_checkpoint.py` — per-checkpoint render (trained vocoder + Griffin-Lim)
- `gate3_whisper_1000.py` / `gate3_whisper_1000.log` — transcription script and raw output for the step-1000 checkpoint
- `loss-trajectory.csv` — training loss at ~100-step intervals through step 1500
- Audio: `symthaea/audio_output/gate3_memorization_test_2026-07-26/step1000_{vocoder,griffinlim}.wav`
