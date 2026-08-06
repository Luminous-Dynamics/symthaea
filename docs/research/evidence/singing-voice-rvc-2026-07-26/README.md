# Evidence bundle: DiffSinger + RVC singing-voice pipeline (2026-07-25/26)

A real end-to-end run — `score/lyrics → DiffSinger performance → RVC
identity conversion → complete render` — captured as an auditable record
rather than left as scratch files in a persistent-but-untracked scratch
directory (`/var/lib/symthaea/training-runs/`, which itself was chosen to
fix an earlier session's loss of an entire training run to `/tmp`
scratch-cleanup).

**Start here, in this order:**
1. `CLAIMS.md` — what this run actually demonstrated vs. what it didn't.
   Read this before repeating any claim about this pipeline elsewhere.
2. `LICENSE_STATUS.md` — why every output here is research-only despite
   the voice-conversion layer itself being permissively licensed.
3. `metrics/methodology.md` — exactly how the quantitative comparisons
   were measured, including a documented discrepancy with an earlier,
   less-specified external review of the same audio.
4. `REPRODUCE.md` — how to redo this from scratch, and what's known not
   to reproduce bit-for-bit.

## One-paragraph summary
A DiffSinger acoustic model was trained for 2000 steps on the English
subset of CSD (a research-only, CC-BY-NC-SA-licensed singing dataset),
proving the training pipeline works end-to-end and produces real sung
audio from held-out test lyrics. Separately, an RVC target-speaker model
was fine-tuned for 200 epochs on a 20-minute corpus of `af_heart`
(Kokoro-82M, Apache 2.0 — a voice already used elsewhere in this project)
speech, generated from public-domain text. The DiffSinger output was then
run through the trained RVC model to convert its timbre toward `af_heart`
while preserving the source's sung pitch and rhythm. All three stages
completed without a hard failure, after finding and fixing several real
bugs along the way (see the inline comments in `pipeline-configs/` and the
`commands/*.sh` scripts for what those were). Quantitative comparison
(`metrics/`) confirms strong pitch preservation and a real, repeatable
increase in audio "gating"/silence in the converted output relative to
the source — a genuine open issue, not yet root-caused.

## Directory map
```
README.md              — this file
CLAIMS.md               — demonstrated / not-yet-demonstrated split
LICENSE_STATUS.md        — why outputs are research-only
REPRODUCE.md             — how to redo this run
environment/             — OS, GPU, pip freezes, pinned source revisions
pipeline-configs/
  diffsinger/             — exact training config + data-prep scripts
  rvc-training/            — exact RVC training config + corpus/filelist scripts
  rvc-inference/            — exact inference-driving script
manifests/
  corpus-manifest-csd.csv     — all 100 CSD files used, train/test split, license
  corpus-manifest-af-heart.csv — all 183 af_heart clips, source text, license
  checkpoints.sha256           — hashes of every checkpoint referenced anywhere
  outputs.sha256                — hashes of every sample .wav referenced
metrics/
  diffsinger-training-loss.csv  — real per-step losses, all 2000 steps
  rvc-training-loss.csv          — real per-logged-step losses, all 200 epochs
  methodology.md                  — how audio-comparison.json was produced
  analyze_audio.py                 — the actual (re-runnable) analysis script
  audio-comparison.json             — its output
commands/
  preprocess.sh, train.sh, infer.sh — exact historical invocations
samples/
  README.md              — points to the actual .wav files + what each is
```

## Addendum (2026-07-26, same day): sweep + intelligibility check, one claim corrected

A follow-up inference-settings sweep and an exploratory Whisper
intelligibility check were run after this bundle was first committed.
One of the sweep's headline findings (a "~71% gating reduction" from
`rms_mix_rate=1.0`) was **found to be substantially confounded by an
~8.9 dB loudness increase** and has been corrected in `CLAIMS.md`'s
addendum section — the real, loudness-controlled effect is ~0.5
percentage points, not ~7-13. A blinded, loudness-matched, phrase-level
human listening test (the actual tiebreaker, since the two automated
proxies disagree with each other) is prepared but not yet completed —
see `metrics/inference-sweep-2026-07-26b/blinded-listening-test/`.
**Read `CLAIMS.md`'s addendum before citing any "tuned settings improve
quality" claim from this bundle — it is explicitly not yet established.**

## Provenance of this document
Written after an external review (pasted into the session by the project
owner, methodology and tooling unspecified) raised reasonable concerns
about reproducibility and measurement rigor. Several of that review's
qualitative conclusions were independently corroborated by rerunning
analysis against the actual audio files (see `metrics/methodology.md`);
its specific numeric claims were not all reproducible and are flagged
explicitly where they diverge, rather than being copied in as fact.
