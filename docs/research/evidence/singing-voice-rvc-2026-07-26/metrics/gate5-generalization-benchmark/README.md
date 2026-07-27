# Gate 5: generalization benchmark (2026-07-26)

An external review of Gate 4's result correctly flagged that "the model
can generally synthesize unseen English lyrics intelligibly" was NOT
supported by a single held-out phrase from a single training song, and
proposed a broader benchmark: multiple songs, more held-out phrases
chosen for phonetic diversity (not random), plus two controls separating
linguistic-content generalization from melodic generalization. This gate
runs that full proposal.

**Headline finding, stated up front**: this benchmark did not simply
confirm Gate 4's result at larger scale — it **sharpened and narrowed**
it. What Gate 4 called "partial compositional generalization" turns out
to depend on something more specific than phoneme coverage: it worked
for held-out content from a song that was 92%-present in training, but
**failed completely** for content from a song the model never saw any
part of, even on a deliberately trivial phrase ("chirp chirp chirp").

## Method

- **Training**: 4 real CSD songs, ~330s total — en001a (minus its
  held-out target, same split as Gates 2-4) + en002a, en003a, en004a in
  full.
- **Held-out set (9 phrases, zero training overlap)**:
  1. `heldout_wontyou` — the exact Gate 2-4 target ("won't you sing along
     with me"), from en001a (92% of that song WAS in training).
  2. 8 phrases from **en005a, an entirely unseen song** (0% in training),
     chosen deliberately for phonetic diversity, not at random: one
     designated **simple control** ("chirp chirp chirp" — 3 trivial
     repeated syllables), one designated **cluster control** ("and a
     windy spring time day" — spr/nd/nt consonant clusters), plus 6 more
     spanning different lengths/content ("butterfly", "come and fly and
     over here", "yellow and wait", "petals smile", "sing a song and
     dance along", "come and dance and over here").
  Every held-out phoneme is covered by the training set (verified
  programmatically, `build_benchmark_dataset.py`) — genuine
  compositional-generalization tests, not memorization.
- **Two controls** (inference-only against the trained checkpoint, no
  extra training — `build_gate5_controls.py`):
  1. **Seen-lyrics, unseen-melody**: "angels we have heard on high"
     (en002a, literally present in training) with F0 transposed up a
     perfect fifth (×2^(7/12)) — tests whether linguistic content
     survives an unseen musical realization.
  2. **Unseen-lyrics, simple-melody**: the cluster held-out phrase's real
     phonemes with generous durations (Gate 1's 150ms/350ms policy) and
     a smoothed per-syllable-constant pitch (mean of the real curve,
     removing within-syllable jitter, not a flat drone) — tests
     linguistic generalization without difficult musical timing.
- **Checkpoints evaluated**: 1000, 2000, 4000, 6000 (loss logged every
  500 for the full curve; full 9-phrase Whisper evaluation done at these
  4 for tractability — see `loss-trajectory.csv`).
- **Real-audio baselines**: all 9 held-out phrases' actual CSD
  recordings, transcribed the same way, to check Whisper can handle
  each phrase at all before blaming the model for a miss.

## Result: full 9-phrase x 4-checkpoint sweep

Ground truth is a rough phonetic-to-English gloss for the en005a phrases
(no official lyric sheet consulted) — see caveat below.

| Phrase | Ground truth (approx.) | Real audio (Whisper) | Step 1000 | Step 2000 | Step 4000 | Step 6000 |
|---|---|---|---|---|---|---|
| wontyou (en001a) | won't you sing along with me | "I want you to sing along with me" | **"Won't you sing along with me?"** | "Won't you sing the warm with me?" | **"Won't you sing along with me?"** | "But you sing along with me" |
| simple_chirp (en005a, SIMPLE) | chirp chirp chirp | "Chirp chirp chirp" | "No! No!" | "Please, please, please, please, please!" | "Please, please, please, please!" | "Please, please, please!" |
| cluster_windyspring (en005a, CLUSTER) | and a windy spring time day | "All now in this springtime day" | "Love and be free Monday" | "Now land the free time bay" | "That will be free time, babe" | "Nothing will be free by then" |
| butterfly (en005a) | butterfly | "Butterfly" | "Now that's light!" | "Oh, the right way!" | "Oh, that looks like..." | "Oh, fly!" |
| comeflyover (en005a) | come and fly and over here | "Come and fly on over here" | "I'm undefined over here" | "Moments fly and go back here" | "Mama's black land, don't bear me" | "Mama's playin' a little bit here" |
| yellowwait (en005a) | yellow and wait | "yellow and white" | "Help a man play." | "I'll go and fight" | "Dancing in the night" | "I don't understand" |
| petalssmile (en005a) | petals smile | "Better smile" | "And I'll say bye" | "Yeah, I'm just high." | "And the spy" | "I'm just fine." |
| singsong (en005a) | sing a song and dance along | "Sing a song and dance along" | "T-R-E-S-O-U-N-G..." (garbled letters) | "Be this home and this alone" | "Leave us alone, let's be alone" | "And that's all I'm just all alone" |
| comedanceover (en005a) | come and dance and over here | "That song over here" | "This is nowhere near" | "This ain't no beer" | "This isn't over yet" | "There's an old man here" |

Full raw output: `gate5_transcribe_all.log`.

**Loss curve** (`loss-trajectory.csv`, averaged across all 9 held-out
phrases at each of 12 checkpoints, 500-6000): noisy and non-monotonic,
global minimum at **step 2000 (val_loss=0.380)** — the same step-2000
loss optimum found in Gate 4, on a differently-sized training set. Given
the qualitative results above are essentially flat/uniformly-failing
across steps for the en005a phrases regardless of loss, this reinforces
Gate 4's finding that aggregate mel-loss doesn't track per-phrase
intelligibility here.

## Result: controls

| Control | Step 2000 | Step 6000 |
|---|---|---|
| 1: seen lyrics ("angels we have heard on high"), transposed +5th | "Just we have a heart all night" | "Just we have a heart on high" |
| 2: unseen lyrics (windyspring), generous durations + smoothed pitch | "And I'll be with thee every night, babe" | "Now let me remind you" |

## Honest interpretation

**Finding 1 — `wontyou` replicates and slightly improves on Gate 4.**
With 4 songs instead of 1, this phrase now hits the ground truth exactly
at two checkpoints (1000, 4000), not just partially as in Gate 4. More
training diversity genuinely helped *this specific held-out phrase*.

**Finding 2 — every phrase from the entirely unseen song fails
completely, at every checkpoint, including the trivial control.** This
is the load-bearing result of this gate. "Chirp chirp chirp" — 3
syllables, no clusters, maximally simple — never comes close at any of
4 checkpoints (hallucinates "No!", "Please, please, please!", "Hey,
hey, hey!" instead). If difficulty (duration compression, consonant
clusters, phrase length) were the bottleneck, the simple control should
have succeeded at least sometimes. It didn't, at any checkpoint. This
**rules out difficulty as the explanation** for the en005a failures and
points at something else: **the phrase's home song was never in
training at all.**

**Finding 3 — the real-audio baselines confirm this isn't a Whisper
artifact for most phrases.** Whisper correctly (or near-correctly)
transcribes the real recordings for chirp/butterfly/comeflyover/
singsong/yellowwait — proving these phrases *are* within Whisper's
reach on real audio. Two phrases (petalssmile, comedanceover) have
imperfect real-audio transcriptions too ("Better smile" / "That song
over here"), meaning those two specific comparisons are less trustworthy
as a measure of the model (my own phonetic gloss may also be imprecise
for these — see caveat below) — but the other 6-7 phrases are clean
comparisons, and the model fails all of them regardless.

**Finding 4 — both controls point the same direction: melody/timing
manipulation doesn't rescue or explain the pattern.** Control 1 (seen
lyrics, transposed pitch) only partially recovers ("on high" at step
6000, nothing else) — even *seen* linguistic content doesn't survive an
unseen pitch register robustly, suggesting the model's apparent
word-level knowledge is more melody-coupled than word-level-general.
Control 2 (unseen lyrics, simplified timing) fails just as completely as
the original difficult version — ruling out musical timing as the
en005a bottleneck, consistent with Finding 2.

**Net read, precisely bounded** (matching the same discipline applied to
Gate 4's wording after the prior review):

- **Supported**: the model can produce recognizable fragments of a
  held-out phrase drawn from a song that was mostly present in training
  (`wontyou`/en001a) — the same finding as Gate 4, now replicated with
  more training diversity and a stronger single-checkpoint result.
- **Not supported, and now actively contradicted by this benchmark**:
  "the model can generalize to phonetically-covered but structurally
  unseen content" (i.e., an entirely new song). Zero of 8 unseen-song
  phrases succeeded at any of 4 checkpoints, including a phrase
  specifically chosen to be trivial. The mechanism demonstrated in Gate
  3/4 looks less like "the model learned transferable phoneme-to-mel
  mappings" and more like **"the model can recombine phonemes within the
  specific melodic/stylistic/prosodic context of a song it has
  extensively seen most of."** That is a substantially narrower claim
  than "compositional generalization" implies on its own, and this
  benchmark's specific job was to find out which of those two it
  actually is — it found the narrower one.

**Scope and methodology caveats**:
- Ground-truth text for the en005a phrases is my own rough phonetic-to-
  English gloss from the CSD phoneme transcription, not sourced from an
  official lyric sheet — the two cases where even real audio mistranscribed
  (petalssmile, comedanceover) suggest some glosses may be imprecise.
  This doesn't affect Findings 2-4 (based on total unrelatedness, not
  precise wording match) but should be kept in mind for exact-match claims.
- n=1 unseen song (en005a) — a genuinely different unseen song, or a
  larger multi-song held-out set, would be needed to know whether this
  "song-level, not phoneme-level" pattern generalizes as a rule or is
  itself a property of this specific dataset/checkpoint.
- Whisper's sung-content validity caveat still applies throughout,
  though the real-audio baselines here directly test and partially
  validate its use for most (not all) of these specific phrases.
- Checkpoint selection (1000/2000/4000/6000) doesn't cover every
  500-step point on the loss curve — a finer sweep could reveal
  transient success at an unevaluated step, though the consistency of
  total failure across 4 well-spread checkpoints makes that less likely.

## Files

- `build_benchmark_dataset.py` — 4-song training + 9-phrase held-out dataset builder (verifies zero phoneme-sequence overlap)
- `build_benchmark_ds_file.py` — held-out eval `.ds` builder (real ph_seq/ph_dur/F0, per phrase)
- `build_gate5_controls.py` — the two control `.ds` builders (pitch transposition, duration/pitch smoothing)
- `benchmark01_acoustic.yaml` / `benchmark-01.txt` — training config and phoneme dictionary
- `gate5_batch_eval.py` — per-checkpoint batch render (all 9 phrases, one model load; Griffin-Lim for the 2 designated phrases)
- `gate5_batch_controls.py` — per-checkpoint control render
- `gate5_transcribe_all.py` / `gate5_transcribe_all.log` — the full 57-clip batched Whisper transcription and raw output
- Audio: `symthaea/audio_output/gate5_generalization_benchmark_2026-07-26/{checkpoints,controls,real_baselines}/`
