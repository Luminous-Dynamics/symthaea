# ACE-Step 1.5 base-intelligibility replication (2026-07-27)

Per the audit doc's recommended sequence: replicate v1's base-intelligibility
verification (same 3 phrases, same phrasing) on 1.5 before moving to its
cover mechanism. Per user direction: turbo first as a fast smoke test, then
base as the real comparison ("more useful as a research foundation even if
turbo produces prettier audio faster").

## Result: both 1.5 variants underperformed v1 on this identical test

| Phrase (target) | v1 (3.5B, 60 steps) | 1.5-turbo (2B, 8 steps) | 1.5-base (2B, 32 steps) |
|---|---|---|---|
| "Won't you sing along with me" | **"Won't you sing along with me? Won't you sing along with me?"** -- exact, twice | **"You"** -- essentially failed | **"Sing along with me"** -- partial, core content right but missing "Won't you" |
| "The quick brown fox jumps over the lazy dog" | **"The quick brown fox jumps, jumps over the lazy door!"** -- 8/9 words verbatim | **"The quick brown fog stumps out the lazy door"** -- degraded, multiple substitutions | **"A quick brownth jumps over the lazy dog"** -- mixed: back half ("jumps over the lazy dog") exact, front half garbled ("brown fox" -> "brownth") |
| "Chirp chirp chirp" | **"Chapp chapp chapp"** -- recognizable structure, wrong vowel | **"Qin-jie, Qin-jie, ..."** -- unrelated syllables | **"San-Zen San-Zen ..."** then an unrelated hallucinated phrase ("Take away what you want") appearing later in the clip |

**None of the three models correctly transcribed "chirp chirp chirp"** --
this specific word appears to be hard across the board, consistent with
it being an unusual, hard-to-sustain-melodically monosyllable (as already
noted in the v1 verification). But on the other two phrases, **v1 is
clearly, consistently ahead of both 1.5 variants**, and base is somewhat
better than turbo (matching the a priori expectation that turbo trades
quality for speed) but still behind v1.

## Important caveat, not yet resolved

**This is a single seed (111) per phrase per variant.** Gate A already
demonstrated substantial seed-to-seed variance exists in this exact
domain (F0 mean spanning nearly an octave across seeds on v1, for
example) -- a single-seed comparison cannot rule out that this specific
seed happened to land badly for 1.5 on these specific phrases. **This
finding should not be treated as a settled verdict on 1.5's
intelligibility ceiling** until re-tested across multiple seeds, matching
Gate A's own methodology. Also not controlled for: 1.5's CoT "thinking"
step auto-rewrote the caption and auto-selected bpm/key/time-signature
(visible in `infer3.log`: it expanded "acapella, clean female vocals..."
into a much longer generated caption and picked "E♭ major", 100bpm) --
v1's test used the caption verbatim with no such rewriting step. This is
an uncontrolled variable, not necessarily the cause, but a real
difference in test conditions between the two models that a fair
comparison should eventually pin down (e.g. disabling `thinking` or
supplying explicit bpm/key to remove this variable).

## Setup notes (real friction, for reproducibility)

- Needed the pinned `torch==2.10.0+cu128` (not the default pip
  resolution) per `requirements.txt`; a fresh Python 3.11 venv (system
  default 3.14 already ruled out for a different pinned dep in the v1
  investigation).
- `flash-attn` fails to build from source in this environment (no
  prebuilt wheel for this torch/CUDA/Python combo) -- removed from
  requirements before install; confirmed optional per the project's own
  docs ("Flash Attention is auto-detected and enabled when available").
- The repo's own GPU-tier auto-detection correctly identified this host
  as "Pre-Ampere CUDA" (Turing, sm_75) and **automatically switched to
  float16 + eager attention** on its own -- no manual dtype workaround
  needed this time, unlike v1 where I had to force `ACE_PIPELINE_DTYPE`
  manually.
- The DiT checkpoint auto-downloads to `PROJECT_ROOT/checkpoints`, but
  `LLMHandler.initialize()` does **not** auto-download the 5Hz LM
  checkpoint -- had to fetch `ACE-Step/acestep-5Hz-lm-0.6B` manually via
  `huggingface_hub.snapshot_download` into the matching path first.
- Reusable venv: `/var/lib/symthaea/training-runs/ace-step-1.5/venv`
  (Python 3.11.15), env fix in `env.sh`.

## Files

- `run_infer_15.py` -- turbo variant, 3-phrase test.
- `run_infer_15_base.py` -- base variant, same 3 phrases, 32 steps,
  guidance_scale=7.0 (base supports CFG, turbo doesn't).
- Audio: `symthaea/audio_output/ace_step_1.5_verification_2026-07-27/`
  (6 files: 3 phrases x {turbo, base}; v1's originals remain in
  `ace_step_verification_2026-07-27/` from the earlier pass).

## What this means for the audit

Doesn't rule out 1.5 as eventually the better foundation (its dedicated
cover/reference conditioning and official LoRA support remain real,
structural advantages over v1's generic `audio2audio` per the Gate B v1
closure) -- but it does mean **the base intelligibility comparison isn't
settled in 1.5's favor yet**, contrary to a natural assumption that a
newer, MIT-licensed model would simply be strictly better. Before
building anything on 1.5, this gap needs either: (a) a multi-seed rerun
to check if it's seed variance, not a real capability gap; (b) removing
the caption-rewriting confound; or (c) accepting that v1 may remain the
stronger base-intelligibility model even if 1.5 wins on controllability
(a plausible, not-yet-tested split outcome: better cover/reference
mechanism, worse raw diction).

## Next (not yet done)

1. Multi-seed rerun of this exact 3-phrase test on both 1.5 variants
   (5 seeds, matching Gate A's methodology) before drawing a final
   intelligibility verdict.
2. If the gap holds up across seeds, test 1.5's cover mechanism anyway
   (per the audit's real goal -- controllability, not raw diction) but
   flag the diction gap as a real, separate concern for any eventual
   foundation choice.
3. Try the 1.7B LM (vs. the 0.6B used here) -- the model card claims
   melody-copy fidelity scales with LM size; worth checking whether
   caption/lyric fidelity does too.
