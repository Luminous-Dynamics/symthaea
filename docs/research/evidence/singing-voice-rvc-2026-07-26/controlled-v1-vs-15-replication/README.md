# Controlled v1-vs-1.5 intelligibility replication (2026-07-27)

Per user direction, after the single-seed 1.5 verification showed both
1.5 variants underperforming v1 but left open whether that was a real
gap or a seed-variance/CoT-rewriting confound: a controlled 5-seed x
3-phrase comparison between v1 and `acestep-v15-base`, with the CoT
confound **removed directly** (not frozen-and-reused) --
`use_cot_caption=False`, `use_cot_metas=False`, `use_cot_language=False`,
with `bpm=100`, `keyscale="C Major"`, `timesignature="4"`,
`vocal_language="en"` all pinned explicitly. Same 5 seeds
(111/222/333/444/555, reused from Gate A for continuity), same 2 phrases
from the earlier test plus a new normal short lyric phrase ("I love the
summer breeze tonight") replacing the unusual "chirp chirp chirp" for
this specific comparison, per user direction.

## Headline result: the gap is real, not a confound -- and widens on reliability

### Valid-render rate

| Model | Attempted | Succeeded | Rate |
|---|---|---|---|
| v1 | 15 | 15 | **100%** |
| 1.5-base | 15 | 10 | **66.7%** |

**5/15 of 1.5-base's renders failed outright** with `Generation produced
NaN or Inf latents` -- a real float16-overflow bug on this pre-Ampere
(Turing, RTX 2070) GPU, not a v1-vs-1.5 fairness artifact (v1 never
failed once across 15 attempts under the same GPU/precision
constraints). **Correction (verified same day via a follow-up stability
gate, see `../ace-step-1.5-stability-gate/README.md`): this is NOT a
seed-specific bug.** The initial read -- "seed 222 failed on all 3
phrases, a consistently unstable seed" -- looked plausible from this
batch alone but was wrong: a direct repeat of the identical (seed=222,
phrase) pair 6 times in a fresh process succeeded 5/6 times. The failure
is genuinely **non-deterministic**, consistent with GPU-kernel-execution
numerical variance pushing marginal float16 values over the overflow
edge on some runs and not others, not a fixed property of any seed. The
project's own error message names this exact failure mode
("Float16 overflow on pre-Ampere GPU") and suggests `ACESTEP_DTYPE=float32`
as a fix -- **that env var doesn't actually exist anywhere in the
codebase except that one error string** (verified via direct grep of
the cloned repo), a real bug in their own error handling, not just an
untried fix. `initialize_service()` has no explicit dtype-override
parameter either; the dtype is auto-selected and baked in at model-load
time. Not pursued further in this pass given the added complexity/OOM
risk of chasing float32 on an 8GB card with base+LM both loaded.

### Lyric accuracy (valid renders only)

| Phrase (target) | v1 (15/15 valid) | 1.5-base (10/15 valid) |
|---|---|---|
| "Won't you sing along with me" | 4/5 exact/near-exact (1 substitution: "sing along"->"stay alone", seed333) | **0/3 exact** -- all 3 valid renders got the opening word wrong ("Won't"->"Can't"/"Don't"/"And"), though "sing along with me" was consistently preserved; seed111 additionally showed a bizarre stutter artifact ("...with me, me, me, me, me, me, me, me, me, me, me?") |
| "The quick brown fox jumps over the lazy dog" (pangram) | 1/5 fully exact (twice, seed333); most others near-exact with a recurring "dog"->"door"/"doll" substitution; seed444 messiest | 1/3 exact (seed111); 2/3 near-exact with a consistent "The"->"A" substitution (seed333, seed444) |
| "I love the summer breeze tonight" (normal phrase) | **5/5 exact** (or exact modulo punctuation) -- flawless across every seed | 1/4 exact (seed444); **3/4 notably garbled**, especially the ending: "tonight" came back as "to none of", "to naves", and "to night" (split) in the other three |

**The clearest single data point**: the "normal" short lyric phrase,
specifically chosen to be unremarkable (no unusual vocabulary, no
letter-names, no melodically-awkward monosyllables), which v1 nailed
**5/5 with zero errors**, 1.5-base got right only **1/4** of its valid
renders, with the other three showing real, specific degradation at the
phrase-final word.

## Verdict, per the pre-registered decision rule

**"If 1.5 base remains consistently behind v1 across seeds, keep v1 as
the primary lyric foundation and treat 1.5 as a specialized
controllability/editing candidate."** -- that's what this controlled
data shows. The CoT-rewriting confound is now ruled out (it was
disabled, not just frozen) and the gap persisted -- in fact widened,
because this pass also surfaced the float16 valid-render-rate problem
that the earlier uncontrolled single-seed test hadn't hit (that one
render, by chance, was seed 111, which succeeds on all three phrases in
this batch too).

**ACE-Step v1 remains the verified lyric-intelligibility baseline.**
ACE-Step 1.5 is not yet shown to match it on raw diction -- but this
doesn't settle 1.5's value for the audit's actual question
(controllability). The planned split-architecture outcome remains live:
v1 for highly intelligible unconstrained generation, 1.5's cover/base
mechanism for reference-controlled performances and editing where some
diction cost may be acceptable in exchange for structural control, with
a dispatcher choosing per task.

Turbo was not re-tested in this controlled pass, per the reasoning that
it doesn't need separate study once base's relationship to v1 is
established -- turbo's purpose is speed, not the quality ceiling.

## Files

- `gate_controlled_v1.py`, `gate_controlled_15base.py` -- generation
  scripts (re-runnable; 15base's skip-if-exists means a rerun only fills
  gaps).
- `controlled_v1_transcripts.log`, `controlled_base_transcripts.log` --
  full Whisper transcripts for all valid renders.
- `1.5_base_valid_render_log.txt` -- every attempt's OK/FAILED status and
  timing, including the 5 NaN failures.
- Audio: `symthaea/audio_output/controlled_v1_vs_15_2026-07-27/` (25
  files: 15 v1 + 10 valid 1.5-base renders; gitignored, not duplicated
  here).

## What NOT to conclude

- This doesn't test 1.5's cover/reference mechanism at all -- that
  remains the audit's actual open question and is unaffected by this
  diction finding either way.
- The float16 NaN-overflow bug is specific to this pre-Ampere GPU class;
  it says nothing about 1.5's reliability on Ampere+ hardware, where the
  project's own defaults (bfloat16) would apply instead.
- 5 seeds x 3 phrases is a real, controlled sample but still modest --
  the specific substitution patterns (dog->door/doll for v1, The->A and
  mangled phrase-endings for 1.5-base) are consistent enough across
  seeds within this run to look like real model tendencies rather than
  noise, but a larger sample would strengthen that reading.
