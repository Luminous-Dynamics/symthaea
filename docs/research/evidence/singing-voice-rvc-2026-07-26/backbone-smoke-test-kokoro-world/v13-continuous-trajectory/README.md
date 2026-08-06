# v13: continuous-trajectory WORLD + Vocos — hypothesis confirmed, architecture closed

Per the reviewer's plan (2026-07-29, after per-word Vocos was correctly rejected for
discarding cross-word temporal context): builds ONE continuous phrase-length WORLD
(f0, sp, ap) trajectory across every word — no per-word `pw.synthesize()`, no
artificial 60ms silence gap between words — overlap-adding each word's parameter
arrays onto the previous word's over a 30ms crossfade instead, then a single
`pw.synthesize()` call for the whole phrase, then one global Vocos pass. Reuses
Arm B's exact per-phoneme F0/vibrato/masking logic (imported, not reimplemented) so
the only architectural change under test is HOW words are joined.

3 phrases only (positive_control, fricative_heavy, long_sustained_vowels),
analytics-gated per the pre-registered decision rule: **"Only ask for listening
when the boundary measurements improve materially without a WER regression... If
not, close this architecture."**

## Infrastructure notes from this run (not the experiment itself)

- The pip venvs (`voice-conversion`, `ace-step`, `ace-step-1.5`, `ctc-align`) were
  found deleted mid-session — confirmed by the user to be intentional cleanup by
  another concurrent session on this shared host, not an incident. Recreated both
  needed venvs from the intact pip wheel cache (fast, no re-download).
- The nix-managed `voice-vocoder` devShell (added for v12's Vocos step) needed
  rebuilding from scratch twice more this session, for the same reason (nix-store
  GC eviction from unrelated disk-pressure activity). Found and fixed a real,
  separate build-time issue while doing so: nixpkgs' `python3Packages.torchaudio`
  runs its own full pytest suite as part of the build (observed 1h49m+ at 100%
  CPU before being killed) — a major, avoidable tax on every rebuild. Fixed via a
  `doCheck = false` overlay override, with a real gotcha along the way: the first
  attempt only partially worked (nixpkgs' top-level `python3Packages` doesn't
  automatically re-derive from an overridden `python3` in every code path, so two
  different torchaudio derivations were being built — one overridden, one still
  vanilla). Fixed by explicitly binding `python3Packages = final.python3.pkgs` in
  the same overlay. Verified: a clean rebuild after the fix completes in minutes.

## Result: silence/continuity metrics improve dramatically — WER regresses substantially

### Silence duration (total near-silent audio in the render)

| phrase | Arm B | v12 | v13 world-only | v13 + Vocos |
|---|---|---|---|---|
| positive_control | 0.48s | 0.54s | **0.18s** | 0.18s |
| fricative_heavy | 0.58s | 0.62s | **0.28s** | 0.24s |
| long_sustained_vowels | 0.44s | 0.50s | **0.24s** | 0.18s |

Consistent ~50-60% reduction across all 3 phrases — the artificial per-word gap is
genuinely gone, exactly as designed.

### RMS drop at word-join boundaries (median interior RMS − median boundary RMS)

| phrase | Arm B | v12 | v13 world-only | v13 + Vocos |
|---|---|---|---|---|
| positive_control | 210.3 dB | 60.9 dB | **0.26 dB** | 0.21 dB |
| fricative_heavy | 59.8 dB | 33.3 dB | **-1.96 dB** | -2.26 dB |
| long_sustained_vowels | 209.7 dB | 61.3 dB | **4.81 dB** | 3.43 dB |

(Arm B's 200+ dB figures are literal hard zero-padding — the boundary window sits
inside true digital silence.) v13's boundaries are **acoustically indistinguishable
from the interior of a word** by this measure — the single strongest, cleanest
result in the whole comparison.

### Spectral-envelope / aperiodicity discontinuity at joins (ratio to typical interior frame-to-frame change; 1.0 = normal, no special jump)

| phrase | metric | v13 world-only | v13 + Vocos |
|---|---|---|---|
| positive_control | sp ratio | 1.11 | 1.45 |
| positive_control | ap ratio | 0.74 | 1.64 |
| fricative_heavy | sp ratio | 1.09 | 1.32 |
| fricative_heavy | ap ratio | 0.00 | 0.00 |
| long_sustained_vowels | sp ratio | 1.17 | 1.37 |
| long_sustained_vowels | ap ratio | 0.84 | 0.88 |

World-only joins land close to 1.0 (ordinary phoneme-to-phoneme variation, not an
extra artifact) — the crossfade is doing its job before Vocos touches anything.
Vocos's own pass raises the sp ratio somewhat at those same points (1.3-1.45,
still far below Arm B's degenerate 0.0-from-true-silence numbers) — a real, if
secondary, side effect of the analysis-resynthesis step itself.

### Melody/pitch accuracy — unaffected or slightly better

Median cents error stays in the same 11-30 cent range as every other arm in this
whole research line (all comfortably resolved as "on pitch" by ear); v13 world-only
is comparable to or better than Arm B and clearly better than v12 on all 3 phrases.
The join mechanism does not cost pitch accuracy.

### WER — the decisive, disqualifying result

| | mean WER (3-phrase subset) |
|---|---|
| Arm B (v10) | 0.378 |
| v12 (Arm B + Vocos) | 0.500 |
| **v13 world-only** | **0.711** |
| **v13 + Vocos** | **0.833** |

A clear, substantial regression, worst on `fricative_heavy` (Arm B 0.333 → v13
world-only **1.333**, v13+Vocos **1.500**) with hypotheses showing real repetition/
garbling artifacts: `"He'll see, he'll see, he'll see, he'll see..."` (world-only),
`"See you all see, shall we? See you sure."` (+Vocos) — vs. the actual target
`"she sells seashells by the seashore"`.

## Root cause (diagnosed, not yet fixed)

The fixed 30ms parameter-domain crossfade is long relative to some consonants'
own duration, and blends `(f0, sp, ap)` across word boundaries **without regard
for whether the two sides are phonetically similar** — so on a word boundary
landing mid-fricative or between two different consonants, it's averaging two
genuinely different spectral shapes together rather than smoothing a natural
transition, plausibly producing the stutter/repetition character Whisper is
picking up as real content. `fricative_heavy` has proportionally the most
consonant-dense word boundaries of the 3 phrases, matching where the regression
is worst.

## Verdict: architecture closed, per the pre-registered rule

The boundary-continuity hypothesis was **real and is now confirmed** — silence and
RMS/discontinuity metrics all moved decisively in the predicted direction, the
strongest confirmation of any diagnostic in this whole arc. But the fixed-width,
content-blind crossfade that achieved it costs real intelligibility, worse than
either baseline. Per the pre-registered decision rule ("if not [obviously better],
close this architecture"), **no listening clips were produced** — WER regressed,
which was the explicit stop condition.

Not pursued further this pass, but a concrete, bounded next step if this line is
revisited: make the crossfade width phoneme-aware (short or skipped across a
voiceless-obstruent boundary, longer only across two voiced/sonorant sounds)
rather than a fixed 30ms blind average — a much smaller change than the whole
architecture, targeting the specific mechanism the WER regression points at.

## Files

- `22_continuous_trajectory_world_vocos.py` — stage 1 (WORLD-only continuous
  synthesis, pip venv).
- `23_v13_vocos_pass.py` — stage 2 (Vocos resynthesis, nix `voice-vocoder` devShell).
- `24_v13_boundary_analytics.py` — silence/RMS/envelope/aperiodicity/pitch gates.
- `25_v13_wer_evaluate.py` — WER on the 3-phrase subset, all 4 conditions.
- `v13_continuous_trajectory_results.json`, `v13_boundary_analytics_results.json`,
  `v13_wer_results.json` — raw results.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_continuous_trajectory/*.wav`
  (gitignored, not duplicated here; not sent for listening per the stop rule).
- Flake fix: `symthaea/flake.nix`'s `voice-vocoder` devShell, torchaudio
  `doCheck = false` + `python3Packages` binding fix.
