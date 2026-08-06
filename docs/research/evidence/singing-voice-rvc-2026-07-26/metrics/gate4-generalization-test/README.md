# Gate 4: generalization test (2026-07-26)

Gate 3 proved the acoustic model can *memorize* one clean phrase to
intelligibility. The open question that leaves: can it *generalize* —
produce an intelligible phrase it never saw during training, by
recombining phonemes it learned in other contexts? This gate tests that
directly, across the full planned checkpoint ladder (1000/2000/3000/4000).

## Method

- **Training set**: ~57s of real, continuous en001a content (2.4s-59.2s
  — the whole song up to but excluding the target region), built as a
  single training utterance (`build_generalization_dataset.py`). Real
  ph_seq/ph_dur via the same heuristic used throughout this bundle.
- **Held-out target**: the exact "won't you sing along with me" phrase
  used in Gates 2-3 (60.0-63.8813s). **Zero overlap with training** —
  marked as `test_prefixes` so the binarizer routes it to the validation
  split only. Verified programmatically: every phoneme in the held-out
  phrase appears somewhere in the training set (30 phonemes covered), but
  this exact sequence of phonemes never does — a genuine compositional-
  generalization test, not a relabeled memorization test like Gate 3.
- **Dictionary**: `generalize-01.txt`, the 30 phonemes actually used in
  training.
- **Config**: `generalize01_acoustic.yaml`, same architecture as the main
  run, augmentation disabled, checkpoints every 1000 steps up to 4000.
- **Eval**: at each checkpoint, render the held-out phrase (`heldout01.ds`
  — real ph_seq/ph_dur/F0, `build_generalize_ds_file.py`) through both the
  trained vocoder and the Gate-2/3-style independent Griffin-Lim path,
  then transcribe with Whisper.

## Result: full checkpoint ladder

Ground truth: "won't you sing along with me".

| Step | Trained vocoder | Griffin-Lim | Held-out val_loss | Train mel_loss (nearby) |
|---|---|---|---|---|
| 1000 | "Won't you sing with me?" | "Won't you sing where you're with me?" | 1.071 | ~0.02 |
| 2000 | "Don't you sing with me?" | "Don't you sing when you're with me" | **0.333 (min)** | ~0.01 |
| 3000 | "Don't you sing with me" | "Don't you sing where you're with me" | 1.013 | ~0.01 |
| 4000 | "I feel sad when you're my knee" | *(empty — no transcribable speech)* | 0.859 | ~0.01 |

Full raw transcripts: `gate4_whisper_{1000,2000,3000,4000}.log`. Full loss
curve at ~200-step resolution: `loss-trajectory.csv`.

## Honest interpretation

**Two real, distinct findings, not one:**

1. **Steps 1000-3000 show genuine partial generalization.** All three
   checkpoints correctly recover the phrase's start ("Won't"/"Don't you
   sing") and end ("with me") for content never seen during training —
   qualitatively different from every total-hallucination failure
   elsewhere in this bundle (Gates 1, 2, the original full-corpus run).
   The consistent word "along" was never recovered, and a "Won't"→"Don't"
   confusion appears from step 2000 onward (plausibly an F0/rhythm
   similarity between the two words at this training scale, not
   investigated further).

2. **The model overfits its tiny single-utterance training set, and this
   is visible in BOTH the loss curve and perceptual quality — but they
   don't move together.** Held-out validation loss traces a real
   non-monotonic curve: 1.07 (step 1000) → **0.33, the minimum** (step
   2000) → 1.01 (step 3000) → 0.86 (step 4000) — classic overfitting
   past the step-2000 optimum, while training loss keeps falling smoothly
   throughout (~0.02 → ~0.01), confirming the model keeps fitting its 57s
   of training content tighter at the held-out phrase's expense.
   **Critically, perceptual quality does not track this loss curve
   cleanly**: steps 2000 and 3000 sound almost identical despite a 3x
   loss difference, and step 4000's loss partially *recovers* (0.86 <
   1.01) while its actual output *collapses* — the trained vocoder output
   becomes totally unrelated to the phrase, and the Griffin-Lim path
   produces no transcribable speech at all. Aggregate mel-loss is not a
   reliable proxy for perceptual/ASR-relevant intelligibility on this
   held-out example -- a loss-curve minimum and a perceptual-quality
   optimum are not guaranteed to coincide.

**Net read**: this gate demonstrates real, if partial and fragile,
compositional generalization — the model isn't purely memorizing, it
can recombine learned phonemes into an unseen sequence and get most of
it right — but that capability is not stable across training duration on
this small a dataset; it peaks somewhere around step 1000-2000 and then
degrades as the model overfits. Step 2000 is the best checkpoint by both
loss and (by inspection) transcription quality; it was not, however,
better than step 1000 in transcription terms despite a much lower loss,
reinforcing the loss/quality divergence point above.

**Scope honesty**: n=1 held-out phrase from a single training song is a
narrow test. Two claims need to stay separated: **supported** — "the
early checkpoints produced recognizable fragments at the correct
locations in an unseen phrase, consistent with limited phonetic
recombination"; **not supported by this bundle** — "the model can
generally synthesize unseen English lyrics intelligibly." Other real
limits on how far this generalizes: likely substantial phonetic overlap
between the held-out phrase and training content, one voice/song
context, no independent human listener transcription, and checkpoint
selection made after inspecting the outputs rather than pre-registered.
A broader test (more training songs, multiple held-out phrases spanning
phonetic difficulty, blinded transcription) is the natural next step if
training-scale investment in this pipeline continues, alongside proper
early-stopping/checkpoint-selection by held-out loss rather than fixed
step count.

## Files

- `build_generalization_dataset.py` — training/held-out dataset builder (verifies zero phoneme-sequence overlap)
- `build_generalize_ds_file.py` — held-out eval `.ds` builder (real ph_seq/ph_dur/F0)
- `generalize01_acoustic.yaml` / `generalize-01.txt` — training config and phoneme dictionary
- `gate4_eval_checkpoint.py` — per-checkpoint render (trained vocoder + Griffin-Lim)
- `gate4_whisper_{1000,2000,3000,4000}.py` / `.log` — transcription scripts and raw output per checkpoint
- `loss-trajectory.csv` — training/validation loss at ~200-step intervals through step 4000
- Audio: `symthaea/audio_output/gate4_generalization_test_2026-07-26/step{1000,2000,3000,4000}_{vocoder,griffinlim}.wav`
