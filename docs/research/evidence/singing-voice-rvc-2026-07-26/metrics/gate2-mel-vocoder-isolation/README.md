# Gate 2: mel-spectrogram-vs-vocoder isolation (2026-07-26)

Per the pre-agreed decision framework: Gate 0 found no near-zero durations
(real but non-catastrophic duration-heuristic weakness). Gate 1 found that
generous durations plus real, in-distribution pitch still produced total
unintelligibility. Gate 2 asks the remaining open question directly: is the
**acoustic model's own predicted mel-spectrogram** deficient, or is the
**NSF-HiFiGAN vocoder** (the neural mel→waveform step) where information is
lost?

## Method

For two phrases from the held-out `en001a` test utterance ("won't you sing
along with me" and "me" alone), extracted with `--mel` (DiffSinger's
built-in flag to dump the acoustic model's raw predicted mel instead of
running it through the vocoder):

1. **Predicted mel → trained NSF-HiFiGAN vocoder** (already have this —
   `gate1-intelligibility-ladder/`'s v2 renders).
2. **Predicted mel → Griffin-Lim** (`librosa.feature.inverse.mel_to_audio`,
   60 iterations) — a deterministic, non-learned, non-neural vocoder. If
   the predicted mel carries real phonetic structure, Griffin-Lim should
   recover at least some of it, independent of whatever NSF-HiFiGAN
   specifically does or doesn't do well.
3. **Real ground-truth mel → Griffin-Lim**: computed directly from the
   actual CSD singer recording for the identical words/timing (same STFT
   params as the acoustic model's own training config —
   `sr=44100, n_fft=2048, win=2048, hop=512, n_mels=128, fmin=40, fmax=16000`,
   `log(clip(x, 1e-5))` compression, matched exactly to
   `modules/nsf_hifigan/nvSTFT.py`). **This is the critical control**: it
   establishes how much Griffin-Lim itself degrades even a real, correct
   mel, so a bad predicted-mel-via-Griffin-Lim result can't be blamed on
   Griffin-Lim being inherently too lossy to be informative.
4. **Real ground-truth audio** (sanity check — should transcribe close to
   verbatim).

All four transcribed with the same Whisper (`small`, CPU) setup used
throughout this bundle.

## Results

| Rung | 1. Real audio | 2. Real mel → GriffinLim | 3. Predicted mel → GriffinLim | 4. Predicted mel → trained vocoder |
|---|---|---|---|---|
| "won't you sing along with me" | "I want you to sing along with me" | "I want you to sing along with me" | "Or might just be a lonely week" | "All my life's just in the moment with me" |
| "me" | "me." | "me." | "You know what I mean?" | "Yeah" |

**Columns 1 and 2 are identical in both rungs.** Griffin-Lim reconstruction
from the *real* mel reproduces exactly what Whisper hears in the *real*
audio — proof that Griffin-Lim itself is not meaningfully lossy for this
comparison; it faithfully carries through whatever structure is actually in
the mel it's given.

**Columns 3 and 4 both fail, in both rungs**, regardless of which vocoder
(Griffin-Lim or the trained neural one) converts the *predicted* mel to
audio.

## Structural comparison (raw energy stats, not ASR-dependent)

| Rung | Predicted mean/frame | Predicted std | Ground-truth mean/frame | Ground-truth std |
|---|---|---|---|---|
| 04 | 2.24 | 1.47 | 4.79 | 2.62 |
| 01 | 1.61 | 1.50 | 2.38 | 0.47 |

The predicted mel is consistently lower-energy and, in the "me" case,
higher-variance/less concentrated than the real mel — consistent with an
under-modulated, blurrier spectral prediction rather than a sharp,
formant-structured one. This is a coarse proxy, not a validated
speech-quality metric, but it's directionally consistent with the ASR
result.

## Honest interpretation

**Both cases point the same direction, with a clean isolating control**:
the trained vocoder is not the primary problem. When the *acoustic model's*
predicted mel is fed to a completely independent, non-learned vocoder
(Griffin-Lim), it still fails — while the exact same Griffin-Lim procedure
applied to a *real* mel (same words, same alignment approach) succeeds
perfectly, matching real-audio transcription word-for-word. This isolates
the bottleneck to **the acoustic model's own mel prediction**, not to
NSF-HiFiGAN specifically, and not to Griffin-Lim's general lossiness.

**Scope honesty**: n=2 phrases, both drawn from the same held-out test
utterance (`en001a`) used throughout this bundle. This is not a
broad-coverage study, but the design is well-controlled (a genuine positive
control that succeeds identically to real audio in both cases), and the
qualitative result — real mel intelligible, predicted mel not, regardless
of vocoder — replicates cleanly across a 5-word phrase and a 1-syllable
word. As always: Whisper is not validated for sung content specifically,
but here it is being used as a *relative* comparator across four render
paths from the same underlying text, not as an absolute score, which
somewhat mitigates that caveat.

**What this changes**: earlier addenda (Addendum 2 of `CLAIMS.md`) flagged
"the DiffSinger source's own articulation" as the likely primary
bottleneck without being able to separate acoustic-model-vs-vocoder. Gate 2
directly tests that separation and supports it: the deficiency traces to
the **acoustic model's mel prediction**, most plausibly a consequence of
this bundle's small (2000-step, single-song, English-CSD-only) training
run rather than an inherent limitation of the DiffSinger architecture or
the vocoder choice. A larger/more-varied training corpus, more training
steps, or an architecture/capacity change would be the natural next levers
— none of which this bundle has tested.

## Files

- `gate2_mel_isolation.py` — mel extraction, ground-truth-mel computation, Griffin-Lim reconstruction
- `gate2_transcribe.py` — Whisper transcription across all 4 render paths, both rungs
- `gate2_transcribe.log` — raw transcription output
- Audio: `symthaea/audio_output/gate2_mel_vocoder_isolation_2026-07-26/*.wav` (predicted/ground-truth slices and Griffin-Lim renders; the trained-vocoder renders are the same files already in `gate1_intelligibility_ladder_2026-07-26/v2_real_pitch/`)
