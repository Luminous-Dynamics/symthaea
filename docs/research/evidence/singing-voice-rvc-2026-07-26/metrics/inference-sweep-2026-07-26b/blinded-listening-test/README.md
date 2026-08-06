# Blinded phrase-level intelligibility test — protocol

**The actual audio and answer key live at
`symthaea/audio_output/blinded_listening_test_2026-07-26/`** (this repo's
established convention — see `../../../samples/README.md` — is to keep
audio out of the docs tree; this file is the protocol, not the media).

**Do not open `ANSWER_KEY_do_not_open_until_after_transcribing.json`
until after you've transcribed all three clips.** Opening it first
defeats the entire point of the test.

## What this tests
Which of source / untuned RVC / tuned RVC is actually more intelligible
to a human listener — the real question the automated proxies
(gating/silence metric, Whisper WER) disagreed on and could not settle
(see `../intelligibility-diagnostic.md` and `../../../CLAIMS.md`'s
addendum for why neither automated metric is trustworthy enough here).

## The clips
- `Clip_A.wav`, `Clip_B.wav`, `Clip_C.wav` — the same final phrase
  ("...Now I know my ABC, won't you sing along with me"), trimmed
  identically (55.0s-63.9s of `en001a`) from all three pipeline stages,
  **loudness-matched to the same RMS** (0.0253, the quietest of the
  three untouched) so no clip has an unfair loudness advantage — this
  was the exact confound that invalidated the earlier automated gating
  comparison.
- Which label maps to which real condition (source / untuned / tuned)
  was assigned by genuine random shuffle at generation time, not a fixed
  or guessable order.

## Protocol
1. Listen to `Clip_A.wav`, `Clip_B.wav`, `Clip_C.wav` in any order you like.
2. For each one, **before checking the answer key or the real lyrics**,
   write down exactly what words/letters you can make out.
3. Only after all three are transcribed, open
   `ANSWER_KEY_do_not_open_until_after_transcribing.json` and the real
   lyrics (`A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z Now
   I know my A B C won't you sing along with me`) and score each clip's
   transcription for intelligibility (e.g., correct words identified /
   total words).
4. Optional, if you want finer resolution: repeat with the alphabet
   section instead of this closing phrase, split into shorter chunks
   (A-G, H-N, O-Z) — the same trim-and-match procedure applies, just
   with different start/end timestamps (see the CSD phoneme-timing
   printout referenced in this bundle's session history for exact
   letter boundaries, or re-derive from
   `../../../pipeline-configs/diffsinger/build_ds_file.py`'s CSD CSV
   parsing).

## Recording your result
There's no automated scoring here by design — write your transcription
and intelligibility judgment into a new file in this directory (e.g.
`human-listening-result.md`) once you've done the test, so it becomes
part of the permanent record alongside everything else in this bundle.
