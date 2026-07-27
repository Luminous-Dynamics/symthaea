# ACE-Step base-model verification (2026-07-27)

The recommended next verification from `VOCAL_APPRENTICE_IMPROVEMENT_PLAN.md`:
does ACE-Step's **base model** (no LoRA -- there is no publicly downloadable
Lyric2Vocal checkpoint, see that plan's correction) produce intelligible
English sung vocals at all, before investing in any adapter-architecture
design around it?

## Result: yes, substantially -- a real, positive, licensable finding

Three renders, `facebook`-style base model (`ACE-Step/ACE-Step-v1-3.5B`,
Apache-2.0), prompt `"acapella, clean female vocals, no instruments, pop, a
cappella"`, transcribed with `faster-whisper` (same tool used throughout
Gates 0-5 for consistency):

| Lyrics fed in | Whisper transcript | Verdict |
|---|---|---|
| "Won't you sing along with me" (Gate 3/4's exact target phrase) | "Won't you sing along with me? Won't you sing along with me?" | **Exact match** (phrase repeats across the 15s clip, both correct) |
| "Chirp chirp chirp" (Gate 5's trivial control) | "Chapp chapp chapp" / "Chapp chapp" | **Partial**: correct syllable count/rhythm/repetition, wrong vowel/final consonant. "Chirp" is an unusual, hard-to-sustain-melodically monosyllable -- plausibly a harder case, not necessarily representative |
| "The quick brown fox jumps over the lazy dog" (novel pangram, not from any prior gate) | "The quick brown fox jumps, jumps over the lazy door!" | **Near-exact**: 8/9 words verbatim, one duplicated word, one close final-word substitution |

Audio sanity-checked as real signal (RMS 0.15-0.25, peak 0.88-0.98 across
all three renders) -- not silence or a Whisper hallucination on empty
audio. Full transcripts: `whisper_transcripts.log`.

**This is a categorically different result than every DiffSinger-on-CSD
render across Gates 0-5**, which failed completely on held-out content
(Gate 5: 0/8 unseen-song phrases, including a trivial "chirp chirp chirp"
control that ALSO failed there -- here the same trivial control gets a
structurally-recognizable, if imperfect, result). Two of three phrases
here were never used anywhere in this investigation's prior gates
("novel_sentence" is a standard English pangram with no connection to
CSD/singing-voice work at all) -- this is real evidence of general
text-to-vocal capability, not overfitting to a specific evaluation set.

## What this means for the Vocal Apprentice plan

Updates `VOCAL_APPRENTICE_IMPROVEMENT_PLAN.md`'s open question: ACE-Step's
base model is now a **verified-promising** (not just licensing-attractive)
candidate foundation, without needing any LoRA at all for basic English
vocal intelligibility. The still-open question is unchanged and now more
important: whether it can accept Symthaea's explicit phoneme/duration/pitch
control interface (the "Symthaea supplies explicit control" design
philosophy from the DiffSinger path) rather than only free-text lyrics +
style tags. That's the real remaining architecture question, not "can it
sing at all" (yes, largely, per the above).

## Setup notes (for reproducing)

- Needed Python 3.11 (system default is 3.14, too new for the pinned
  `spacy==3.8.4` dependency -- no wheels exist yet); used
  `nix-build '<nixpkgs>' -A python311`.
- `bfloat16` (the default/documented dtype) fails on this host's RTX 2070
  (Turing, sm_75, no native bf16 tensor cores) with `RuntimeError: GET was
  unable to find an engine to execute this computation` (a cuDNN
  algorithm-selection failure, not a code bug). `float32` avoids that but
  OOMs on 8GB VRAM for the 3.5B model even with `cpu_offload=True`.
  **`float16` works** (Turing has full fp16 tensor-core support) -- forced
  via the `ACE_PIPELINE_DTYPE=float16` environment variable (the
  `dtype=` constructor argument only recognizes the literal string
  `"bfloat16"`, else silently falls back to `float32`, so `ACE_PIPELINE_DTYPE`
  is the only way to actually request float16).
- `torchaudio.save()` in this install (`torchaudio==2.11.0`) routes through
  a `torchcodec` backend requiring FFmpeg shared libraries not on
  `LD_LIBRARY_PATH` here (same class of issue as Step 1's alignment spike --
  torchaudio's I/O stack is in active flux around this version). Rather
  than chase the FFmpeg/torchcodec dependency, monkeypatched
  `torchaudio.save` to write via `soundfile` directly (`run_infer.py`) --
  a two-line workaround, not a real fix, fine for this bounded verification.
- Reusable venv: `/var/lib/symthaea/training-runs/ace-step/venv` (Python
  3.11.15), env fix in `env.sh`.

## Files

- `run_infer.py` -- first test (won't you sing along with me).
- `run_infer2.py` -- replication tests (chirp control + novel pangram).
- `whisper_transcripts.log` -- full transcripts + audio sanity-check.
- `*_input_params.json` -- exact generation parameters for each render
  (ACE-Step's own auto-saved metadata).
- `env.sh` -- NixOS env fix for the venv.
- Audio itself: `symthaea/audio_output/ace_step_verification_2026-07-27/`
  (gitignored per this bundle's existing convention -- see
  `../samples/README.md` -- not duplicated into tracked docs).

## What NOT to conclude from this

- This is 3 short renders with hand-picked vocal-forward tags, not a
  systematic benchmark -- it answers "is this worth investigating further"
  (yes), not "how good is it" (unmeasured: naturalness, sustained-note
  quality, melody control, longer-form coherence).
- The explicit-control-interface question (phoneme/duration/pitch vs.
  free-text lyrics) is completely untested here -- this only used
  free-text lyrics, ACE-Step's native interface.
- No comparison was made to the current DiffSinger pipeline's *melody
  fidelity* -- this check was scoped to intelligibility only, matching
  Gates 0-5's own evaluation focus.
