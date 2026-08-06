# Muse cognition-study apparatus — first end-to-end execution (2026-07-29)

**Status: mechanism check passed. This is not evidence about music.** Every
response below is synthetic. Nothing here says anything about whether Muse's
cognition improves what it composes — that is what the real listener study is
for, and it still has not been run.

Reproduce: `crates/domains/symthaea-muse/scripts/dryrun-cognition-study.sh`

## Why this was needed

The cognition-study apparatus is 51 of the crate's 150 `src/` modules — 25,809
of 76,332 source lines — behind a single 2,020-line binary exposing **120
subcommands**. Before today none of it had ever been executed:

- `data/cognition-study/` contains only `templates/*.example.json`; nothing has
  ever been committed beneath it.
- `scripts/check-cognition-v*.sh` (8 scripts) are `test -f` and `grep -q`
  assertions. They check that source files exist and contain certain strings.
- `scripts/verify_cognition_study_v*.py` (5 scripts) reimplement the
  sealing/digest logic **in Python** and self-test that reimplementation.
  Verified by grep on 2026-07-29: **zero of the five ever invoke the
  `cognitive_study` binary.**

So a third of the crate had a verification layer that never ran it.

## What was run

Two arms over **identical** fixtures, schedule, structural evidence, and
analysis plan. Only the listener preferences differ.

| | EFFECT arm | NULL arm |
|---|---|---|
| preference ordering | Symthaea > Heuristic > RandomValid > Fixed | independent of arm |
| noise | 20% of listeners randomised | fully random |

Chain: `validate-manifest` → `build-schedule` → `validate-schedule` →
`seal-evidence` → `compile-evidence` → `analyze`.

Inputs: the minimum legal study — 4 pilot + 24 confirmatory fixtures × 4 arms =
**112 blinded presentations**, 12 listeners, 336 response blocks per arm.
Fixture shape mirrors what `blinded_study.rs`'s own tests construct.

## Result

```
EFFECT success=True  analysis_gate=True  passing=['Preference']
       FixedSuperiority        effect=+0.7778 CI=[+0.7396,+0.8160] holm_p=0.00030
       RandomValidSuperiority  effect=+0.5046 CI=[+0.4699,+0.5405] holm_p=0.00030
       HeuristicNonInferiority effect=+0.2870 CI=[+0.2488,+0.3229] holm_p=0.00030

NULL   success=False analysis_gate=False passing=[]
       FixedSuperiority        effect=+0.0116 CI=[-0.0336,+0.0579] holm_p=1.00000
       RandomValidSuperiority  effect=-0.0532 CI=[-0.1204,+0.0116] holm_p=1.00000
       HeuristicNonInferiority effect=+0.0278 CI=[-0.0313,+0.0880] holm_p=0.21628
```

The apparatus **discriminates**: it accepts the injected ordering and rejects
random preference, from otherwise identical inputs.

Two details worth recording, because they are what make this more than a
smoke test:

1. **The effect sizes recover the injected structure quantitatively.** I placed
   the arms 3, 2, and 1 rank-positions from Symthaea; the recovered effects are
   0.778 / 0.505 / 0.287 — monotonic and roughly proportional. The blinding and
   codebook-unblinding path is therefore carrying real information, not just
   passing a threshold.
2. **`raw_one_sided_p = 9.999e-05 = 1/10001`** is exactly the floor of a
   10,000-replicate randomization test, which is the replicate count the plan
   declares. The p-value is being produced by the permutation machinery it
   claims to use.

## The validator has real teeth

The first analysis attempt was **rejected**, correctly:

```
TooFewRandomizationReplicates { found: 2000, required: 10000 }
TooFewListenersPerFixture     { planned: 8,  required: 12 }
```

Those floors are preregistered minimums enforced by `analyze`, not knobs. The
plan was raised to meet them rather than the floors lowered to meet the plan.

## Limits — what this does NOT establish

- **No claim about music.** Synthetic responses only.
- **Only the scientific core was exercised**: manifest → schedule → evidence →
  analysis. The artifact-bundle, runner-package, session-event, pilot,
  external-review, replication, stewardship, and publication subsystems — the
  bulk of the 120 subcommands and most of the 25.8K lines — remain unexecuted.
  They need real audio files and a real cohort on disk.
- **One seed per arm.** The manipulation check is deterministic, not a power
  analysis. It shows the pipeline can separate signal from noise at this effect
  size, not what effect size it could detect in general.
- The harness **fails closed**: if both arms passed, or both failed, it exits
  non-zero rather than printing a plausible summary. That is deliberate, per
  this project's standing lesson about experiment runners that silently
  substitute a fallback.

## What this changes

Scheduling the listener study is now a question of recruitment and timing, not
of engineering readiness — for the analysis core. Extending the harness to the
remaining subsystems is the obvious next step, and would very likely surface
more of what this pass found in the first hour of running code that had never
run.
