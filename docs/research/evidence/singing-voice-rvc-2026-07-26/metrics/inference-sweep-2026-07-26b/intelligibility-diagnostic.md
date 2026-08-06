# Exploratory Whisper intelligibility diagnostic (2026-07-26)

**Status: exploratory / warning signal, NOT a validated intelligibility
metric.** Whisper (faster-whisper "small", CPU) has no established
validity for scoring sung, spelled-out-letter content -- this test is
included because it surfaced a real, reproducible difference between
conditions, not because WER/CER against sung audio is a trustworthy
absolute intelligibility score. See caveats below before drawing
conclusions from the numbers.

Ground truth (en001a, confirmed literally the ABC song via CSD's own
`lyric/en001a.txt`):
> A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z Now I know my
> A B C Next time won't you sing with me [repeated with B-section variant
> ending "won't you sing along with me"]

## Results

| condition | transcript | WER | CER |
|---|---|---:|---:|
| source (DiffSinger, untouched) | "Won't you sing along with me?" | 92.6% | 87.3% |
| untuned RVC (defaults) | "A, T, E, C, D, E, F, C, H, H, I, V, G, N, F, N, Z, V, N, B, F, U, R, F, S, G, U, V," | 85.2% | 83.6% |
| tuned RVC (rms_mix_rate=1.0, index on) | "They DCFDFFG, these high attorney, D-E-M-M-P, Q-F-R-H-I-S, D-U-U-N-P, Born on Geese, FX, Y-H-H-C, Milo Han, all of ID-A-P-C, ..." (full hallucinated transcript in `measure2.log`) | 98.8% | 149.3% |

## Why these numbers must NOT be read at face value

1. **The ground truth is mostly single-letter "words"** (A B C D E...).
   Word-level WER against a reference full of one-character tokens
   behaves atypically -- untuned's numerically "better" 85.2% is not
   good evidence it's more intelligible than source; it may just be a
   scoring artifact of matching some single-letter slots by chance.
2. **Tuned's CER exceeds 100%** (149.3%), meaning the hypothesis is much
   *longer* than the reference -- that's Whisper hallucinating additional
   content, a sign of the model becoming unstable on that audio, not a
   simple "more words wrong" degradation.
3. **Whisper is not validated for sung, isolated-letter recognition.**
   This project's own prior investigation (SING-3 through SING-6a, see
   `../../../..` project history) found ~100% WER for a *different*
   Kokoro-based singing pipeline in the same style of test -- a known
   general weakness of Whisper on sung audio, not necessarily reflecting
   what a human listener perceives. Today's source-file WER (92.6%) is
   in a similar range to that prior finding, so it cannot be cleanly
   separated from "Whisper struggles with sung isolated letters in
   general" vs. "this specific source has an articulation problem."

## What this diagnostic DOES support (the qualitative, not numeric, reading)

Reading the actual transcript text (not the WER number), coherence
degrades monotonically: **source (one short, 100%-correct real phrase)
> untuned (garbled but letter-shaped output) > tuned (long, hallucinated,
largely incoherent output)**. This is evidence that:
- RVC conversion likely adds *some* further articulation degradation on
  top of whatever the source has -- direction, not yet magnitude.
- The tuned configuration (rms_mix_rate=1.0 + index on) causes worse ASR
  behavior than the untuned baseline, in the opposite direction from what
  the (confounded) gating metric suggested.

## What this diagnostic does NOT establish
- That the source alphabet section is seriously unintelligible *to a
  human listener*.
- That untuned RVC is definitively more intelligible than tuned RVC to a
  human listener (only that it produces more Whisper-parseable, if
  wrong, output).
- That Whisper WER/CER is a valid primary singing-intelligibility metric
  for this content type.
- Whether the degradation is mainly a phoneme-duration issue upstream in
  DiffSinger or an articulation issue introduced by RVC's conversion.

## Required next step (not yet done)
A blinded, loudness-matched, phrase-level human transcription test --
see `blinded-listening-test/` in this directory.
