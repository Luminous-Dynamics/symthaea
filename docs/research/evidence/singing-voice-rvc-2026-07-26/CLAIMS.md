# Claims: demonstrated vs. not yet demonstrated

Scope: a single pipeline run (2026-07-25/26) — DiffSinger acoustic model
trained on CSD English (2000 steps) → RVC target-speaker model fine-tuned
on an `af_heart` (Kokoro-82M) speech corpus (200 epochs) → voice-converted
sung output. All specific numbers referenced here are in `metrics/`.

## Demonstrated (this bundle has direct evidence)

- The DiffSinger acoustic-model training run completed 2000 steps without
  crashing, after fixing two real bugs found during the run: a phantom
  outlier sample from augmentation baked into stale binarized data, and a
  VRAM-headroom OOM from too-large batch settings. See
  `pipeline-configs/diffsinger/csd_en_acoustic.yaml` inline comments and
  `metrics/diffsinger-training-loss.csv` for the full loss trajectory.
- The RVC target-speaker training run completed all 200 epochs without
  crashing, fine-tuning from RVC's own pretrained_v2 checkpoint (verified
  "All keys matched successfully" on load). Full per-epoch losses in
  `metrics/rvc-training-loss.csv`.
- Native DiffSinger inference (`scripts/infer.py acoustic`) on a genuinely
  held-out CSD test file (`en001a`, one of the config's declared
  `test_prefixes`, never seen in training) produced real, non-silent,
  correctly-durationed audio.
- RVC inference (`vc_single`) on that DiffSinger output, using three
  different checkpoints (epoch 50, 75, 200/final), produced real,
  non-silent audio each time, verified by direct peak/RMS inspection —
  not just "the process exited 0."
- **Melody/rhythm preservation, alignment-free measure**: F0 correlation
  between DiffSinger source and RVC output was 0.99+ on 12-second clips
  (epoch 50 and 75) and 0.868 on a 64-second clip (epoch 200) — see
  `metrics/methodology.md` for why clip length affects this specific
  metric, and why the 64s number is not directly comparable to the 12s
  numbers.
- **RVC output is measurably more aggressively gated than the source**:
  silence-below-(-50dB) fraction rose from ~27% to ~41% on the 12s clips,
  and ~25% to ~34% on the 64s clip, consistently across every checkpoint
  tested. This is real and repeatable, independent of alignment
  assumptions.
- **Diminishing returns by epoch ~75-94**: `loss_mel` in the RVC training
  log dropped sharply in the first ~50 epochs (34.6 → ~17) then flattened
  (14-16 range from epoch ~75 through epoch 188, the last logged value
  before completion). See `metrics/rvc-training-loss.csv`. This is a
  direct read of the actual training log, not an inference from listening.
- GPU was genuinely used throughout (not a CPU fallback): confirmed via
  `nvidia-smi --query-compute-apps` showing the training PID holding ~4GB
  of GPU memory with nonzero utilization and power draw, at multiple
  points during the ~18-hour training run.

## Not yet demonstrated (this bundle has no evidence either way)

- **Speaker-identity similarity to `af_heart`.** No reference `af_heart`
  speech (Kokoro output alone, unconverted) was compared against the RVC
  output in this bundle's own analysis. The corpus used to *train* the
  RVC model is `af_heart` Kokoro speech (see
  `manifests/corpus-manifest-af-heart.csv`), so the model was fine-tuned
  toward that identity — but no speaker-embedding or blinded-listening
  test confirms how close the *singing* output actually sounds to that
  identity versus, say, a generic RVC-converted timbre.
- **A distinct "Symthaea" identity independent of `af_heart`.** This
  pipeline currently produces an `af_heart`-derived singing voice, not a
  demonstrably novel one. No test in this bundle establishes
  distinctiveness from the base Kokoro identity.
- **Commercial clearance.** See `LICENSE_STATUS.md` — explicitly not
  established, and not a target of this bundle.
- **Generalization** across different songs, vocal registers, or
  languages beyond the English CSD subset and the three checkpoints
  tested here.
- **That epoch 200 is perceptually better than epoch 75** under any
  blinded listening protocol. The loss-curve plateau is real; whether a
  human listener can tell the two apart, or prefers one, has not been
  tested.
- **Whether corpus size or inference settings (`f0_up_key`, `protect`,
  `rms_mix_rate`, index/retrieval blending, filter radius) are the
  limiting factor on quality.** Only one fixed set of inference defaults
  was used throughout this bundle (see `commands/infer.sh`) — no sweep
  was run.
- **Root cause of the elevated gating/silence artifact.** Documented as a
  real, repeatable measurement; not yet diagnosed as a training-data
  issue, an RVC inference-setting issue, or a domain-mismatch issue
  (RVC's target corpus is *spoken* Kokoro audio; it is being asked to
  convert *sung* DiffSinger audio, which has different sustained-note and
  silence characteristics than the training distribution).

## Addendum (2026-07-26, later same day): inference-settings sweep, corrected

A follow-up inference-only sweep and an exploratory intelligibility check
were run against the completed epoch-200 checkpoint. Full detail,
scripts, and raw results:
`metrics/inference-sweep-2026-07-26b/`.

**What was initially claimed and then corrected**: the first pass reported
`rms_mix_rate=1.0` cutting the gating artifact by ~71% (12s clip) / ~71%
(64s clip). That number was **confounded by an ~8.9 dB loudness increase**
in the tuned output — `rms_mix_rate=1.0` mechanically pulls the output's
loudness envelope toward the source's (louder) envelope, so a fixed
-50dBFS silence threshold counted fewer "silent" frames largely from
gain alone, not from a real reduction in gating. After loudness-matching
the tuned output to the untuned output's RMS, the real reduction in
low-energy frames is **~0.5 percentage points, not ~7-13**. See
`metrics/inference-sweep-2026-07-26b/loudness-confound-reanalysis.csv`
for the exact numbers. One real, non-confounded effect did survive:
exact-zero-sample fraction dropped from 8.96% to 1.89%, suggesting the
tuned settings do fill in some hard digital gaps, separate from the
loudness-inflated headline number.

**Exploratory Whisper intelligibility check** (full detail and required
caveats in `metrics/inference-sweep-2026-07-26b/intelligibility-diagnostic.md`
— read the caveats before citing the numbers below):
- Transcript coherence, read qualitatively, degrades monotonically:
  source (one short, verbatim-correct phrase) > untuned RVC (garbled but
  letter-shaped) > tuned RVC (long, hallucinated, largely incoherent).
- This is a **warning signal, not a validated intelligibility score** —
  Whisper is not validated for sung, spelled-out-letter content, and this
  project's own prior investigation (SING-3 through SING-6a) already
  found ~100% WER for a different, unrelated singing pipeline in this
  exact style of test, a known general Whisper-vs-singing weakness.
- **What it does support**: the tuned configuration causes measurably
  worse ASR behavior (hallucination, CER >100%) than the untuned
  baseline — the opposite direction from the (confounded) gating result.
  The two automated proxies disagree with each other.

### Configuration status (revised again, 2026-07-26 later same day)

A label-independent acoustic comparison (STOI, ESTOI, phrase-envelope
and spectral-transient correlation, mel-spectrum similarity, 3-6kHz
band energy) was run against the loudness-matched final-phrase clips.
Full detail: `metrics/inference-sweep-2026-07-26b/blinded-listening-test/acoustic-comparison-2026-07-26.md`.
**Every metric agrees**: untuned preserves the source's articulation
more faithfully than tuned; tuned loses measurably more energy in the
3-6kHz consonant/fricative band. The intended human blind-listening
test has **still not been completed by anyone** — the project owner's
own attempt was contaminated (the answer key was auto-exposed by their
tooling before evaluation), and Claude has no audio-perception
capability and never performed or claimed a listening judgment.

| Configuration | Status |
|---|---|
| DiffSinger source | Reference |
| Untuned epoch 200 (RVC defaults: protect=0.33, rms_mix_rate=0.25, index off) | Current RVC baseline |
| Tuned epoch 200 (rms_mix_rate=1.0, index on) | **Rejected as default** — measurably worse articulation preservation on every acoustic metric tested; may still be preferred for smoothness/fullness, but no valid preference data exists |
| Canonical production configuration | **None yet** |

**Bigger-picture finding this addendum surfaced**: even the *better* of
the two RVC outputs (untuned) only reaches STOI ~0.53 against a source
that was itself only partially transcribable by Whisper. This points
toward **the DiffSinger source's own articulation as the primary
bottleneck**, not the RVC conversion settings.

## Addendum 2: Gate 0 phoneme/duration audit (2026-07-26, same day)

A bounded, deterministic audit of the exact planned phoneme/note timing
data fed to DiffSinger (both at training and inference time) — full
detail: `metrics/gate0-duration-audit/`. **Ruled out**: no near-zero or
sub-frame consonant durations (every consonant gets a uniform 70ms).
**Found real but non-catastrophic**: the duration heuristic
(`convert_csd.py`'s `CONSONANT_DUR=0.07` flat constant) is phonetically
naive — every consonant gets identical duration regardless of type or
context, most stressed in compressed multi-phoneme syllables ("won't",
"sing", "along", "with" — exactly the words flagged as unclear
earlier). An informal acoustic cross-check against the actual rendered
waveform found most consonants **are** acoustically differentiated in a
phonetically plausible way (fricatives show real high-frequency energy,
one stop shows a genuine closure-burst signature), with one isolated
anomaly (the /b/ in "B" shows no closure dip at all).

**This is not the "obvious defect, apply one surgical fix" case.**
Per the pre-agreed decision framework, the recommended next step is
**Gate 1 (the intelligibility ladder)** — it's designed specifically to
separate "rapid pacing / compressed syllables" (demonstrated here) from
"fundamental acoustic-model or vocoder limitation" (not yet
distinguishable). Gate 1 has not been started.

## Addendum 3: Gate 1 intelligibility ladder, confound found and corrected (2026-07-26, same day)

Full detail: `metrics/gate1-intelligibility-ladder/`. Built 7 phrases of
increasing length with GENEROUS, UNIFORM per-phoneme durations (150ms
consonants / 350ms vowels, vs. Gate 0's documented flat 70ms consonants),
same trained checkpoint, no RVC. First pass (v1) used a flat 220Hz pitch
contour and Whisper transcribed all 7 rungs as completely unrelated to
ground truth. **Before drawing any conclusion, a real design flaw was
flagged**: a flat pitch is out-of-distribution for this model (DiffSinger's
training data never has monotone pitch), potentially confounding the
duration variable this test was meant to isolate.

**Corrected rerun (v2)**: identical generous durations, but pitch replaced
with the real CSD ground-truth pitch per syllable (converted from the
actual MIDI note, all 7 rungs being verbatim sub-phrases of the same
training song — genuinely in-distribution, not invented). **Result:
intelligibility did not improve.** All 7 rungs still transcribe as
fluent-sounding but unrelated English, the same qualitative failure as v1,
including the single-syllable rung ("me"). Side-by-side table in
`metrics/gate1-intelligibility-ladder/README.md`.

**What this does and doesn't support**: fixing the pitch-flatness confound
did not restore intelligibility, so the flat pitch alone does not explain
v1's failure. Combined with Gate 0 (no near-zero durations; most
consonants acoustically plausible), this shifts weight away from "the
duration heuristic is the bottleneck" and toward the next item in the
pre-agreed framework — **Gate 2 (mel-spectrogram-vs-vocoder isolation)**,
not yet started. As always: Whisper is not validated for sung, especially
spelled-letter, content, so total failure here is a strong signal but not
a substitute for the still-outstanding human blind-listening test.

## Addendum 4: Gate 2 mel-vs-vocoder isolation -- bottleneck localized to the acoustic model (2026-07-26, same day)

Full detail: `metrics/gate2-mel-vocoder-isolation/`. Directly tests
whether the trained NSF-HiFiGAN vocoder or the acoustic model's own
predicted mel-spectrogram is the intelligibility bottleneck, using
DiffSinger's built-in `--mel` flag to dump the raw predicted mel and
converting it via an independent, non-neural vocoder (Griffin-Lim)
instead of the trained one -- with a genuine positive control: the same
Griffin-Lim procedure applied to a **real** ground-truth mel (computed
from the actual CSD singer recording, identical STFT params, same
words).

**Result, replicated on 2 phrases** ("won't you sing along with me" and
"me" alone): the real-mel-via-Griffin-Lim control transcribes **identically
to the real audio** in both cases (proving Griffin-Lim itself isn't the
limiting factor here) -- but the acoustic model's **predicted** mel fails
intelligibility via *both* Griffin-Lim and the trained vocoder, in both
cases. A coarse structural check (mean/variance of linear mel energy per
frame) also shows the predicted mel is consistently lower-energy and
less-concentrated than the real mel, consistent with an under-modulated
prediction rather than a sharp one.

**This localizes the bottleneck to the acoustic model's own mel
prediction, not the vocoder.** Revises the "DiffSinger source's own
articulation" hypothesis from Addendum 2 (RVC/vocoder-focused, unable to
separate acoustic-model-from-vocoder) into a directly-tested claim: the
deficiency is upstream of any vocoder choice, most plausibly a consequence
of this bundle's small training run (2000 steps, single song, English-CSD
only) rather than an architectural limit. n=2 phrases from the same
held-out utterance -- not a broad study, but a well-controlled,
cleanly-replicated result. Whisper's sung-content validity caveat still
applies, though here it's used as a *relative* comparator across 4 render
paths of the same text rather than an absolute score.

## Addendum 5: Gate 3 memorization test -- under-training confirmed as the driver, not a pipeline defect (2026-07-26, same day)

Full detail: `metrics/gate3-memorization-test/`. An external review of
Addendum 4 correctly flagged that "under-trained" was plausible but not
yet proven -- the same symptom could come from a broken pipeline, bad
alignment, or a feature mismatch instead. It proposed a full step/data/
alignment ladder program; this gate is the single highest-value piece of
it, scoped down to match this investigation's bounded-gate practice: can
the acoustic model memorize ONE clean real phrase to intelligibility at
all, given focused training on nothing else?

Built a single-phrase training set (the same "won't you sing along with
me" clip used throughout Gates 1-2, train==val by design -- a
memorization test, not a generalization test) and trained from scratch.
**At step 1000 -- half the step count of the full 100-song/2000-step
run that produced Gate 2's failure -- both the trained vocoder and an
independent Griffin-Lim path transcribe the output identically to real
audio's own Whisper transcription: "I want you to sing along with me."**
Training was stopped at step 1526 once this unambiguous answer landed
(no need to run the full 6000-step ladder to answer the bounded
question).

**This directly demonstrates the model CAN learn to reproduce clean
content to intelligibility, quickly, given focused signal.** Combined
with Gate 2, the most parsimonious explanation for the full-corpus run's
failure is under-training from signal dilution (2000 steps spread across
~100 songs gives each phrase far less effective training than 1000 steps
devoted to it alone) -- not a pipeline defect, broken alignment, or
architectural incapacity. This rules out the live competing hypothesis
("the pipeline/architecture cannot produce intelligible mel predictions
at all"), though it doesn't test generalization (n=1 phrase, memorized
not held-out) -- that remains the next real question if further
training-scale investment is pursued. Not tested further: the 2000/4000/
6000-step checkpoints, and the broader data/alignment ladders the
external review also proposed.

## Addendum 6: Gate 4 generalization test -- real partial generalization, then overfitting (2026-07-26, same day)

Full detail: `metrics/gate4-generalization-test/`. Directly follows Gate
3's open question: can the model *generalize*, not just memorize? Trained
on ~57s of real, continuous en001a content (everything except the target
region) and held out the exact "won't you sing along with me" phrase used
in Gates 2-3 -- verified zero phoneme-*sequence* overlap with training
(every individual phoneme is seen elsewhere, but never this combination).
Evaluated the full planned checkpoint ladder (1000/2000/3000/4000).

**Steps 1000-3000 show genuine partial generalization**: all three
correctly recover the phrase's start ("Won't"/"Don't you sing") and end
("with me") for content never trained on -- qualitatively different from
every total-hallucination failure elsewhere in this bundle. The word
"along" was never recovered; a "Won't"->"Don't" confusion appears from
step 2000 onward.

**The model then overfits its tiny training set, visible in both the
loss curve and output quality -- but the two don't move together.**
Held-out val_loss traces a real non-monotonic curve: 1.07 (step 1000) ->
0.33, the minimum (step 2000) -> 1.01 (step 3000) -> 0.86 (step 4000),
classic overfitting past step 2000, while training loss falls smoothly
throughout. But perceptual quality doesn't track this cleanly: steps
2000/3000 sound almost identical despite a 3x loss gap, and **step
4000's loss partially recovers while its actual output collapses** --
trained-vocoder output becomes totally unrelated ("I feel sad when
you're my knee"), and Griffin-Lim produces no transcribable speech at
all. Aggregate mel-loss is not a reliable proxy for perceptual/ASR-level
intelligibility here.

**Net read, precisely bounded**: what's supported is "the early
checkpoints produced recognizable fragments at the correct locations in
an unseen phrase, consistent with limited phonetic recombination" --
evidence of partial compositional generalization, fragile and peaking
around step 1000-2000 before overfitting degrades it. What's **not**
supported by this bundle: "the model can generally synthesize unseen
English lyrics intelligibly" -- that claim needs a broader held-out
suite (n=1 phrase, one song, likely substantial phonetic overlap with
training content, no independent human listener check, and checkpoint
selection made after inspecting the outputs, all real limits on how far
this generalizes). A broader test (more songs, multiple held-out
phrases spanning phonetic difficulty, proper early stopping by held-out
loss, and a blinded transcription pass) is the natural next step if
training-scale investment continues.

## Addendum 7: Gate 5 generalization benchmark -- the claim narrows, doesn't expand (2026-07-26, same day)

Full detail: `metrics/gate5-generalization-benchmark/`. Runs the
external review's proposed broader benchmark in full: 4 real songs
(~330s) for training, 9 held-out phrases (the Gate 2-4 target plus 8
phonetically-diverse phrases from an entirely unseen song, including a
deliberate simple control and a deliberate consonant-cluster control),
two inference-only controls (seen-lyrics/transposed-melody,
unseen-lyrics/simplified-melody), evaluated at 4 checkpoints.

**This did not confirm Gate 4's generalization claim at larger scale --
it sharpened and narrowed it.** The Gate 2-4 target phrase (`wontyou`,
from a song 92% present in training) replicates and slightly improves:
exact match "Won't you sing along with me?" at 2 of 4 checkpoints.
**But all 8 phrases from the entirely unseen song fail completely, at
every checkpoint, including the deliberately trivial simple control**
("chirp chirp chirp" never comes close in 4 tries). Real-audio
baselines confirm most of these phrases are within Whisper's reach on
the real recording, ruling out an ASR artifact. Both controls point the
same direction: transposing a *seen* phrase's melody mostly breaks it
(only fragment recovery), and simplifying an *unseen* phrase's timing
doesn't rescue it -- ruling out musical difficulty as the driver.

**Revised bound on the generalization claim**: what Gates 3-4 showed
looks less like "the model learned transferable phoneme-to-mel mappings"
and more like "the model can recombine phonemes within the specific
melodic/stylistic/prosodic context of a song it has extensively seen
most of." That's a materially narrower claim than "compositional
generalization" implies unqualified. Loss curve (min at step 2000,
val_loss=0.380, same optimum step as Gate 4 despite a very differently
sized training set) again doesn't track the qualitative pattern --
en005a phrases fail uniformly regardless of loss. Caveat: en005a's
"ground truth" text is this session's own phonetic-to-English gloss, not
an official lyric sheet (two phrases' real-audio transcriptions diverged
from the gloss too) -- doesn't affect the total-unrelatedness findings,
but exact-wording claims should be read with that in mind. n=1 unseen
song; whether "song-level, not phoneme-level" generalization is a
general rule or specific to this dataset/checkpoint needs a second
unseen song to know.
