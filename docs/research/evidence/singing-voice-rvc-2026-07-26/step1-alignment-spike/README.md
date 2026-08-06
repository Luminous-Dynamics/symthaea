# Step 1 (part A): CSD <-> espeak/IPA phoneme transducer (2026-07-27)

Per `SYMTHAEA_SINGING_VOICE_NEXT_STEPS_2026-07-27.md` Step 1 item 1: "inventory
both phoneme sets explicitly... build this as an explicit, versioned, *tested*
transducer, not a loose dictionary lookup." This is that transducer, plus the
data-driven process used to build it.

## Method

Rather than hand-guess a mapping between CSD's 40-symbol `csd-en.txt`
phoneme set and the `facebook/wav2vec2-lv-60-espeak-cv-ft` model's IPA
vocabulary, `build_transducer_from_corpus.py` mines the real co-occurrence
statistics from the CSD English corpus itself:

1. CSD's `txt/` files are phoneme sequences with underscores joining
   phonemes *within one note* and spaces separating notes -- **not**
   English word boundaries (a real, initially-surprising finding: e.g.
   "merrily" appears as two separate space-separated CSD tokens,
   `m_e_r l_ii`, because it spans two sung notes). The first version of
   this script assumed CSD's space token == one English word and got a
   ~15% match rate; fixed by aligning at the flattened whole-*line*
   phone sequence instead (both CSD and espeak's IPA output for that
   line, phoneme-for-phoneme), which raised matched pairs from 2,093 to
   16,363 words with clean phone-count agreement.
2. For each matched line, espeak-ng (`--ipa --sep=_`) provides the
   English phone sequence; where the total phone count matches CSD's for
   that line, position-wise pairs are tallied into a co-occurrence table
   (`csd_espeak_cooccurrence.json`, full distributions, not just the
   winner).
3. `phoneme_transducer.py` is the curated result: a canonical target
   phone per CSD symbol plus a documented set of real corpus variants
   (singing genuinely varies -- e.g. American-English intervocalic
   flapping realizes `/t/` as `[ɾ]`, unstressed vowels reduce toward
   schwa). Most consonants land at >=0.9 confidence; a handful of vowels
   are honestly ambiguous and documented as such (`eo` is CSD's generic
   reduced-vowel catch-all at 0.48 confidence; `oi`'s raw majority vote
   was overridden -- see below).

## Two real bugs/artifacts found and fixed while building this

- **CSD token granularity was not what the plan assumed.** Space
  separators in CSD's `txt/` files are per-*note*, not per-*word* --
  multi-syllable words split across notes. Caught by an 85% mismatch
  rate on the first (word-level) alignment attempt; fixed by aligning
  at the line level instead.
- **Raw `espeak-ng` CLI output contains phones the model itself can
  never produce.** The CLI emits tie-barred palatalization variants
  (`iːʲ`, `iʲ`, `ɪʲ`) for certain glide contexts that do not exist
  anywhere in the wav2vec2 model's own `vocab.json` (392 symbols) --
  it was trained via the `phonemizer` library, not this CLI directly.
  Caught by a test (`test_phoneme_transducer.py`) that checks every
  transducer entry against the real model vocabulary; fixed by
  excluding those three tokens from `ii`'s acceptable-variant set.
- **One manual override, documented in-table**: `oi`'s raw majority
  vote picked `ɪ` (n=24) over the phonologically correct `ɔɪ` (n=10)
  at low total sample size (36) -- almost certainly noise from
  melisma/held-note decomposition, not a real substitution. Overridden
  to `ɔɪ` with the raw counts left visible in the note field rather
  than silently corrected.

## Files

- `phoneme_transducer.py` -- the transducer itself (canonical target +
  documented variants + confidence per CSD symbol), `build_expected_sequence()`
  for turning a CSD phone list into an espeak-token sequence for forced
  alignment (drops AP/SP silence markers).
- `test_phoneme_transducer.py` -- validates full symbol coverage, that every
  mapped phone actually exists in the model vocabulary, silence handling, and
  a phonological spot-check against hand-known English mappings. All pass.
- `build_transducer_from_corpus.py` -- the data-mining script (re-runnable).
- `csd_espeak_cooccurrence.json` -- full raw co-occurrence distributions
  (not just the winning phone) for every CSD symbol.
- `env.sh` -- NixOS `libstdc++`/`LD_LIBRARY_PATH` fix for the reusable
  venv at `/var/lib/symthaea/training-runs/ctc-align/venv`
  (torch/torchaudio/ctc-forced-aligner/transformers/phonemizer installed).

## Step 1 item 2: forced alignment against real sung audio -- decisive negative result

`align_phrase.py` implements genuine *forced* alignment: build the expected
espeak-token sequence from CSD's own known phones (via the transducer
above), generate wav2vec2-espeak CTC emissions over the real audio slice,
and run `ctc_forced_aligner.forced_align()` (a generic, backend-agnostic
Viterbi routine reused as-is, not reimplemented) with those emissions and
targets.

Two phrases were tested from `en038a` ("row row row your boat", a
600ms-per-note held passage, and "gently", a fast syllabic run) --
deliberately different pacing to check whether any failure was specific to
sustained notes. **Every single non-blank phone in both phrases aligned to
exactly one 20ms frame** (`align_phrase_row_row_row.log`,
`align_phrase_gently.log`), regardless of the note's true duration (up to
600ms per CSD's ground truth) or whether it was a consonant or a vowel.
Order and relative onset position were plausible in both cases (each
phone's spike landed close to where it should, in the right sequence) --
but essentially all of each note's real singing duration was left
unaccounted for, reported by the alignment as CTC blank.

**Root-caused, not just observed**: `probe_blank_dominance.py` inspects the
model's *raw* per-frame log-probabilities directly, bypassing the aligner
entirely. Over a 600ms held note, the target vowel's best frames anywhere
in the slice sit around log-prob -6 to -9 (probability ~0.0002-0.0015) --
never becoming a real competitor to blank, which sits at ~log-prob 0.0
(probability ~1.0) across nearly the entire slice. **This is the acoustic
model's own emission behavior, not an artifact of the alignment
algorithm** -- no forced-alignment implementation can recover duration
information the model never assigned meaningful probability to in the
first place. Consistent with, and now confirming, Step 0's disclosed
caveat that `wav2vec2-lv-60-espeak-cv-ft` was trained on spoken (Common
Voice), not sung, audio.

Quantified: across the two tested phrases, CTC-recognized phonetic content
accounted for only ~12% (row-row-row) and ~27% (gently) of the time CSD's
ground truth confirms the singer was actually vocalizing -- the rest is
misreported as silence.

## Step 1 acceptance-gate verdict (per the plan's pre-declared criteria)

The plan's explicit gate: *"do not convert the training corpus unless CTC
alignment materially improves consonant boundaries, doesn't corrupt
sustained vowels... If it doesn't beat the heuristic on singing
specifically, keep it only as an auditor, not the primary alignment
source."*

**Verdict: fails decisively, primary-source path closed.** The corruption
found isn't a marginal or close-call result needing the full 20-30-clip
gold-set/3-way comparison to adjudicate (items 3-4 of the original plan) --
it's a ~75-90% duration data loss confirmed at the raw-probability level on
the very first two test phrases, on both a sustained-note and a fast
syllabic passage. Building the full gold set would not change this
conclusion, so this session stops here rather than mechanically completing
every originally-scoped sub-step on a foregone question. **The heuristic
70ms-per-consonant split (Gate 0's finding) is phonetically cruder but
strictly more usable than raw wav2vec2-espeak CTC output for this corpus.**

Residual possible value, *not yet validated*: onset locations (not
durations) looked plausible in both tests and might have some use as a
coarse consonant-order/rough-timing auditor layered on top of the existing
heuristic -- but this is speculative and would need its own validation
before any use, per the plan's "keep it only as an auditor" fallback.

## What this changes for the next-steps plan

- Do not pursue Step 2 (heuristic-vs-CTC training A/B) as originally
  scoped -- there is nothing to A/B; CTC durations are unusable as-is.
- The alignment-quality confound flagged before this spike (Gate 0's flat
  heuristic, never fixed) is **still open** -- this spike ruled out the
  specific fix that was planned (CTC-based sub-syllable realignment), it
  did not resolve whether alignment quality still matters for the
  DiffSinger generalization problem. A different fix approach (e.g. a
  phonetically-informed heuristic using known consonant-class durations
  instead of a flat 70ms, or a duration model fine-tuned on singing) would
  need separate justification before pursuing.
- The reusable venv, transducer, and forced-alignment harness
  (`align_phrase.py`) remain valid infrastructure if a *singing-adapted*
  phone-level acoustic model is ever identified/fine-tuned -- the failure
  here is specific to this one pretrained model's training domain, not to
  the forced-alignment approach in general.

## Files

- `phoneme_transducer.py` -- the transducer itself (canonical target +
  documented variants + confidence per CSD symbol), `build_expected_sequence()`
  for turning a CSD phone list into an espeak-token sequence for forced
  alignment (drops AP/SP silence markers).
- `test_phoneme_transducer.py` -- validates full symbol coverage, that every
  mapped phone actually exists in the model vocabulary, silence handling, and
  a phonological spot-check against hand-known English mappings. All pass.
- `build_transducer_from_corpus.py` -- the data-mining script (re-runnable).
- `csd_espeak_cooccurrence.json` -- full raw co-occurrence distributions
  (not just the winning phone) for every CSD symbol.
- `align_phrase.py` -- forced-alignment harness (known CSD phones -> espeak
  targets -> wav2vec2-espeak emissions -> `ctc_forced_aligner.forced_align()`
  -> phone boundaries). Re-runnable: `python3 align_phrase.py <csv> <wav>
  <start_row> <end_row>`.
- `align_phrase_row_row_row.log`, `align_phrase_gently.log` -- full
  transcripts of the two test runs, including per-frame blank/non-blank
  segment traces.
- `probe_blank_dominance.py` -- the root-cause diagnostic (raw per-frame
  log-probabilities, bypassing the aligner) that confirms blank-dominance
  is the model's own behavior, not an alignment-algorithm artifact.
- `last_alignment.json` -- machine-readable output of the most recent
  `align_phrase.py` run.
- `env.sh` -- NixOS `libstdc++`/`LD_LIBRARY_PATH` fix for the reusable
  venv at `/var/lib/symthaea/training-runs/ctc-align/venv`
  (torch/torchaudio/ctc-forced-aligner/transformers/phonemizer installed).

## Status

Step 1 closed with a decisive negative result. Item 3 (gold set) and item 4
(3-way comparison) were not built -- the acceptance gate in item 5 was
already answerable without them, and building a hand/visual-labeled gold
set now would cost real effort to confirm a conclusion the raw-probability
evidence already settles. See `SYMTHAEA_SINGING_VOICE_NEXT_STEPS_2026-07-27.md`
for how this changes Steps 2-4.
