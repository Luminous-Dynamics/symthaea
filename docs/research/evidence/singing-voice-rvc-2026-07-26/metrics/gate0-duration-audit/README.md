# Gate 0: phoneme/duration table audit (2026-07-26)

Bounded, deterministic audit per the user-specified plan: export the
full word-syllable-phoneme-note timing table for two regions, flag
known failure patterns, and cross-check planned timing against the
actual rendered audio. No retraining. Not the beginning of the full
intelligibility ladder (Gate 1) or mel-vs-vocoder isolation (Gate 2).

## Regions audited
1. Closing phrase: "Now I know my ABC, won't you sing along with me"
2. Alphabet segment: "H I J K L M N O P" (includes user-flagged letters M, N)

Full raw table: `gate0_output_raw.txt`. Generator script (real, re-runnable):
`gate0_duration_audit.py`.

## What was checked and the result

| Check | Result |
|---|---|
| Consonants with near-zero (<30ms) or sub-frame duration | **None found.** Every consonant gets exactly 70ms (~6 frames). |
| Vowels consuming >90% of their note | None found. |
| Missing word-boundary pauses | Several words run together with no pause ("Now I know my"), but this **matches the real CSD ground-truth singer's timing** -- not a synthesis defect. |
| Phonemes outside note/syllable bounds | None found (heuristic keeps all phonemes within syllable start/end by construction). |
| Multiple phonemes compressed onto one short note | **Found**, real and repeated: "won't" (w+ou+n+t, 4 phonemes/note), "sing" (s+i+ng), "along" (l+ou+ng), "with" (w+i+dh) -- exactly the words flagged as unclear in earlier listening. |
| Final phonemes cut off by render boundary | None found -- last phoneme ends exactly at the audio's declared end time. |
| Questionable G2P for letters B, C, G, M, N, S, Z | None found -- all phonemicize correctly (B=b_ii, C=s_ii, G=j_ii, M=e_m, N=e_n, S=e_s, Z=z_ii). |

## The real finding: a uniform, phonetically-naive duration heuristic

`convert_csd.py`'s `split_syllable_duration()` assigns every consonant a
**flat, fixed 70ms** (`CONSONANT_DUR = 0.07`), regardless of phoneme
type, syllable position, note tempo, or stress. This is not "near-zero"
in absolute terms, but it's a crude approximation -- real speech doesn't
give stops, fricatives, and nasals identical durations regardless of
context. This heuristic directly produced the training AND inference
conditioning (durations control frame counts fed to the model at both
stages, per `train/train.py`'s `ph_acc` computation), so any weakness
here was baked into every render of this checkpoint, not introduced at
inference.

## Realized-audio cross-check (informal, not a validated forced aligner)

For each planned consonant window in the closing phrase, measured RMS
energy relative to surrounding context and high-frequency energy ratio
in the actual rendered `en001a-step2000-final.wav`. Real, re-runnable
script: `acoustic_crosscheck.py`.

**Most consonants show acoustically plausible signatures matching their
phonetic class**: fricatives (C's /s/, sing's /s/) show real elevated
high-frequency energy (HF ratio 0.66-0.92); won't's /t/ shows a genuine
stop-closure dip (RMS ratio 0.37, HF ratio 0.69 -- textbook burst
signature). **One clear anomaly**: "B"'s /b/ shows no closure dip at
all (RMS ratio 1.41 -- energy *increases* where a stop closure should
dip). This looks like an isolated defect for that specific instance,
not a systemic "consonants are acoustically absent" pattern.

## Conclusion and recommendation

This is **not** the "obvious starvation/truncation" case that would
warrant one surgical duration-heuristic fix and a quick closing-phrase
re-render. The evidence is more nuanced:
- The duration heuristic is real and worth eventually fixing (it's
  phonetically crude), but no phoneme was found sub-frame or
  near-zero.
- The acoustic model largely **does** differentiate consonant classes
  in its output, with at least one specific, isolated failure (/b/'s
  missing closure).

Per the pre-agreed decision framework: since no obvious defect was
found, **the recommended next step is Gate 1 (the intelligibility
ladder)**, not a quick patch. The ladder is specifically designed to
separate "rapid pacing / compressed multi-phoneme syllables" (a real,
demonstrated pattern here) from "fundamental acoustic-model or vocoder
limitation" (not yet distinguishable from this audit alone) -- exactly
the ambiguity this Gate 0 pass leaves open.
