# Naturalness screening: DNSMOS + UTMOS confirm the human listening verdict quantitatively

Step 1 of `/home/tstoltz/.claude/plans/synthetic-tumbling-raccoon.md`. The
user listened to the v10 4-arm-matrix renders and reported: none of the
four arms sound good, and A/B are indistinguishable in overall quality.
This adds an automated, no-reference naturalness proxy (no perceptual-
quality metric existed anywhere in this repo before this) to (a) confirm
that verdict quantitatively and (b) give a cheap screening tool for future
experiments, before spending more of the user's own listening time.

## Method

- **DNSMOS** (`torchmetrics.functional.audio.dnsmos`, Microsoft's
  pretrained ONNX model, auto-downloaded on first use) — no-reference,
  reports SIG/BAK/OVR; OVR is the headline number used here.
- **UTMOS** (`torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`)
  — no-reference, SSL-based, correlates well with human MOS on speech/TTS.
- Both are **speech-trained, not singing-validated** — reported honestly
  as a screening proxy, not ground truth. (A SingMOS-specific model exists
  in the VoiceMOS 2024 literature but isn't packaged/available yet.)
- Scored: all 10 spoken references (`{phrase}_spoken.wav`, real unmodified
  Kokoro speech, the calibration anchor) and all 40 v10 4-arm-matrix sung
  renders (`{phrase}_sung_v10full_{a,b,c,d}.wav`).

## Result: quantitative confirmation of the ear test, in both directions

| | n | mean DNSMOS (OVR) | mean UTMOS |
|---|---|---|---|
| Spoken reference | 10 | **3.319** | **4.373** |
| Arm A (baseline) | 10 | 1.767 | 1.984 |
| Arm B | 10 | 1.784 | 1.881 |
| Arm C | 10 | 1.751 | 1.809 |
| Arm D | 10 | 1.708 | 1.792 |

**Every sung arm scores roughly HALF the spoken reference on both
metrics** (DNSMOS ~1.7-1.8 vs ~3.3; UTMOS ~1.8-2.0 vs ~4.4) -- a large,
consistent gap, not noise (spoken scores cluster tightly, 3.2-3.4 DNSMOS
across all 10 phrases). **The four arms are statistically indistinguishable
from each other** (DNSMOS range 1.708-1.784, a 0.076 spread dwarfed by the
~1.55-point gap to spoken) -- this is the same "A/B sound about the same"
verdict the user gave by ear, now quantified, and extended: C and D aren't
meaningfully different either.

This is a clean double confirmation: the automated proxy agrees with the
human ear in both the direction that matters (sung is much worse than
spoken) and the direction that could have been a surprise (the four
consonant-handling variants don't differ in overall naturalness) --
validating DNSMOS/UTMOS as a meaningful screening tool for this pipeline
specifically, despite neither being singing-validated in general.

## What this does and doesn't establish

**Does establish**: a real, large, quantified naturalness gap between the
sung output and the underlying spoken voice, and that none of the
consonant-boundary work (the entire v0-v10 arc) touches this gap in either
direction -- consistent with all four arms sharing the identical
WORLD-vocoder pitch-shifting core.

**Does NOT establish** which of the two competing hypotheses (pitch curve
too rigid vs. WORLD's own timbre being the ceiling) explains the gap, or
how much of it either could close. That's what Step 2/3 (pitch
micro-naturalization, then re-screening) tests next.

## Not yet done

- Step 2: implement note-onset scooping, slow pitch drift, and micro-
  jitter on top of the existing vibrato/glide.
- Step 3: re-run this exact screen on Step 2's output; report the delta
  honestly, including if it's null.
- Step 4 (contingent on a null Step 3 result): evaluate a drop-in neural
  vocoder (Vocos first).

## Files

- `16_naturalness_screen.py` -- the scoring script.
- `naturalness_screen_results.json` -- raw per-phrase, per-arm DNSMOS
  (p808_mos/sig/bak/ovr) and UTMOS scores, plus all 10 spoken references.

## Update: Step 2/3 -- pitch micro-naturalization tried, honest mixed result

Implemented Step 2 exactly as scoped: added note-onset scooping (-60
cents, 70ms ramp to target), slow pitch drift (3 summed random-phase
sinusoids, 0.4-1.0Hz band, 10 cents depth), and micro-jitter (4 cents,
frame-independent), layered on top of (not replacing) the existing
vibrato+glide in `15_hybrid_event_synthesis_matrix.py`'s `synthesize_word`
(toggle: `NATURALIZE`, deterministic seed `NATURALIZATION_SEED=20260729`).
Rendered as `v11_b_natural` on the same Arm B (the WER-winning arm) across
all 10 Gate-2 phrases.

**A real bug found and fixed before any result counted**: the first
drift implementation used a moving-average convolution with a kernel
length of `1/rate/FRAME_DT` frames (286 frames at 0.7Hz/5ms) -- longer
than several words' own frame count, causing `np.convolve(...,
mode="same")` to return an array sized to the kernel, not the word,
crashing on the multiply. Fixed by switching to summed random-phase
sinusoids (output length is exactly `len(t_abs)` by construction, no
convolution-length failure mode possible).

### WER regression check (required per the plan, not optional): a real cost

| | mean WER |
|---|---|
| Arm B baseline (no naturalization) | 0.284 |
| Arm B + naturalization | **0.365** |

This contradicts F8c's "pitch and WER are measured uncorrelated"
expectation -- for THIS specific stochastic perturbation (not the
deterministic exact-target singing F8c measured), there's a real,
non-trivial WER cost. Worst individual regressions: `positive_control`
0.000->0.500 ("Won't you sing along with me?" -> "What's using along
with me?"), `fricative_heavy` 0.333->0.500.

### Naturalness re-score: small positive movement, not close to closing the gap

| | mean DNSMOS (OVR) | mean UTMOS |
|---|---|---|
| Arm B baseline | 1.784 | 1.881 |
| Arm B + naturalization | **1.852** (+0.068) | **1.913** (+0.032) |
| Spoken reference (for scale) | 3.319 | 4.373 |

Both metrics move in the intended direction, but only marginally --
+0.068 DNSMOS closes about **4.5%** of the ~1.53-point gap to spoken
reference; +0.032 UTMOS closes about **1.3%** of the ~2.49-point gap.
Per-phrase results are genuinely mixed, not uniformly positive:
`positive_control` DNSMOS jumped +0.61 and `short_unstressed` +0.39, but
`fricative_heavy` DROPPED -0.10 and `long_sustained_vowels` DROPPED
-0.16.

### Honest verdict

This is neither a clean win nor a clean null -- a small, real naturalness
gain, at a real WER cost, that closes only a small fraction of the total
gap to natural speech. Per the plan's own decision framework, this result
is closer to "no meaningful movement" than "moves meaningfully": even a
genuine, correctly-implemented attempt at exactly the F8c-identified
mechanism (over-precise, unnaturally rigid pitch) barely moved the needle.
**This is real evidence that pitch curve shape is NOT the dominant
naturalness bottleneck** -- consistent with, and now empirically
supporting, the hypothesis that WORLD's own vocoder timbre (not the F0
trajectory riding on top of it) is the ceiling. This favors moving to
Step 4 (evaluate a drop-in neural vocoder, starting with Vocos) over
further tuning of naturalization parameters, whose expected payoff now
looks small relative to the effort.

Not pursued further this pass: tuning down jitter/scoop depth to recover
the WER cost while keeping the DNSMOS gain (a smaller, bounded follow-up
if this line of work is revisited) -- deprioritized given the effect
size is already small before accounting for the WER tradeoff.

## Files (updated)

- `15_hybrid_event_synthesis_matrix.py` -- now includes the
  `NATURALIZE` toggle and the onset-scoop/drift/jitter implementation.
- `naturalized_b_screen_results.json` -- per-phrase DNSMOS/UTMOS for the
  naturalized render.
- Audio: `symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v11_naturalized_b/*_sung_v11_b_natural.wav`.
