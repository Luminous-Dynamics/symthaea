# ARC benchmark-validity record

Per-consumer status for every benchmark that scores through
`arc_dataset::fair_distractor_grid`. Written because a 2-AFC score is only meaningful if the two
options are comparable, and for three of these six they are not.

Companion to `ARC_2AFC_DISTRACTOR_GEOMETRY_2026-07-31.md` (the mechanism) and
`CI_FAILURE_CATALOGUE_2026-07-30.md` Addendum 2 (the `ArcChain` case that started it).

**Nothing here changes a recorded number. It records which recorded numbers mean something.**

## The shortcut variable, stated once

All six benchmarks score `sim(pred, target) > sim(pred, distractor)`, chance 0.50.
`fair_distractor_grid` returns a **generic transform of the input** — in practice
`reflect_x(input)` — so the distractor's distance from the input is roughly **constant**.

`apply_rule` is near-identity when a rule fails to generalise: measured on `ArcChain`, the
prediction sat ~0.78 similar to the *input* but only ~0.55 to a 3–4 step target.

So the decisive quantity is not "did the model learn the rule" but:

> **is the target nearer the input than the distractor is?**

When yes, a do-nothing predictor wins automatically. When no, it loses automatically. Both are
independent of rule learning, and both are decided before any training pair is seen.

## Status table

| benchmark | verdict | headline status |
|---|---|---|
| `arc_dataset.rs:257` | **affected — worst case** | **retracted** |
| `arc_noise.rs:204` | **affected** | **provisional** |
| `arc_fewshot.rs:172` | **affected — construct void** | **retracted** |
| `arc_strict.rs:268` | not affected by this confound | provisional, for a *different* reason |
| `arc_staircase.rs:111` | not affected | supported, balanced **by accident** |
| `arc_scaling.rs:127` | not affected | supported, balanced **by accident** |
| `arc_chain.rs` | **was affected — FIXED** | superseded; new numbers valid |

---

## `arc_dataset` — retracted

- **Intended construct:** HDC encoding/retrieval fidelity on genuine ARC-AGI tasks.
- **Geometry:** unbalanced, and the imbalance splits the task set nearly in half.

Verified independently on **1000 real ARC-AGI-2 training tasks**, replicating
`evaluate_arc_tasks`'s scoring exactly and bucketing on the geometric property alone:

```
OVERALL                              n=1000  acc=0.6840
target NEARER input than distractor  n= 417 (41.7%)  acc=0.9976
target FARTHER from input            n= 583 (58.3%)  acc=0.4597
```

The headline is 41.7% structurally-guaranteed wins averaged with 58.3% below-chance losses. A
parallel audit reproduced the historical figure on ARC-AGI-1 (64.9%, n=399; 145 near at 1.0000,
255 far at 0.4510) — two datasets, same structure, so this is a property of the scoring.

- **Status: RETRACTED.** Not 99%, not 64.9%, not 68.4%. Under a distance-matched distractor the
  score should be expected to fall toward ~0.50, because the guaranteed-win bucket disappears.
- This is the third distinct number this benchmark has reported. The first (99.0%, "100% 2-AFC")
  was retracted 2026-07-18 for a random-`BinaryHV` distractor; `fair_distractor_grid` was the fix
  for *that*, and is the cause of this one.

## `arc_noise` — provisional

- **Intended construct:** accuracy degradation as input noise rises.
- **Geometry:** unbalanced for one of four transform families.

Reported decomposition of the published `accuracy_0pct` = 0.8187:

| transform | accuracy |
|---|---|
| ColorFill | 1.0000 |
| **Translation** | **0.2750** |
| ColorReplacement | 1.0000 |
| Reflection | 1.0000 |

One below-chance cell and three at ceiling, averaged into a plausible-looking 0.82. The
below-chance cell is invisible in every published metric because the benchmark only reports the
mean across transforms.

- **Status: PROVISIONAL.** The noise-degradation *shape* may survive — noise is applied
  identically across transforms — but the absolute values do not, and `noise_resilience` /
  `accuracy_drop` are computed from them.

## `arc_fewshot` — retracted, construct void

- **Intended construct:** does accuracy improve with more examples?
- **Geometry:** unbalanced for one of four transform families — the only one that moves.

Reported: three of four transform families sit at **1.0000 at every k**. Only Translation varies,
and it tracks *bundle parity* rather than shot count — even counts make the prediction more
identity-like, which makes the confound bite harder.

- **Status: RETRACTED, and the construct itself is void rather than merely noisy.** If three of
  four cells are pinned at ceiling regardless of k, the benchmark cannot answer its own question.
  `learning_rate`, `fewshot_gain` and `saturation_point` are artifacts of an oscillation in the
  one unpinned cell.
- Independent verification of this decomposition was in progress when this record was written and
  had not completed under host load; the per-transform figures above are from the audit sweep and
  are **not** ones I reproduced myself. The *structural* claim — that a single confounded cell
  carries the entire signal — follows from the mechanism regardless.

## `arc_strict` — not affected here, but do not read it as capability

- **Geometry:** biased the **opposite** way. The target moves 19.6% of cells; the distractor 71.5%.
- Measured `twoafc_accuracy` = 1.0000 with `strict_solve_rate` = 0.0.
- **Status: PROVISIONAL for a different reason.** A pure identity predictor also scores ~1.0 here,
  so a perfect 2-AFC alongside a zero strict-solve rate is evidence about *encoding fidelity*, not
  rule transfer. Not a distractor-geometry defect; a construct-validity one.

## `arc_staircase`, `arc_scaling` — supported, balanced by accident

- Both use `reflect_x` as their only task. `fair_distractor_grid` therefore skips its first
  candidate (it equals the true output) and returns `reflect_y`. Target and distractor are both
  single reflections — **equidistant by construction**.
- Measured geometry delta between −0.0086 and +0.0042 across 2×2 … 20×20.
- **Status: SUPPORTED**, with a caveat that matters: nothing in `fair_distractor_grid` *arranges*
  this balance. It falls out of these two benchmarks happening to use a single task type whose
  transform collides with the first candidate. Change the task set and the balance silently
  disappears. Record it as luck, not design.
- Separate non-confound issue for `arc_staircase`: accuracy never falls below the 0.75 staircase
  target even at 20×20, so the staircase only ever steps up and `capacity_threshold` pins at its
  20.0 ceiling.

## `arc_chain` — fixed, superseded

Distractor is now the output of a **different chain of the same length**, so both options are
equidistant from the input by construction. Per-chain effect:

```
c0 1.000 -> 1.000    c2 0.033 -> 0.700    c4 0.267 -> 0.867
c1 0.867 -> 1.000    c3 0.000 -> 0.533    c5 0.033 -> 0.567
```

Nothing below chance any more. Note this made the benchmark **honest, not good** — several chains
now sit near 0.50, the correct reading of "rule composition does not generalise here."

---

## Why `fair_distractor_grid` was not changed

Six benchmarks share it, and two of those (`arc_staircase`, `arc_scaling`) are balanced only by
accident of their task set. A single edit to the shared helper moves all six at once, including
the two that are currently fine. The fix has to be per-benchmark and distance-matched — as done
for `ArcChain` — not a change to the helper.

## What would restore each headline

A distance-matched distractor: same number of transforms from the input as the target, so
`sim(input, target) ≈ sim(input, distractor)`. For a chain benchmark that is a sibling chain of
equal length. For a single-transform benchmark it is a *different* transform of comparable
cell-change magnitude — noting that magnitudes differ enormously (`color_replace` ~15%,
`fill_region(2×2)` ~14%, `reflect` ~67%, `translate(±1)` ~83%), so "one transform" is not a
distance proxy.

**Do not regenerate any baseline before the distractor is fixed.** Regenerating first would bake
the artifact in as the expected value — the same error the psych-bench regression baseline already
embodies, where every `ArcChain` entry sits saturated at `mean=1.0, ci=[1.0,1.0]`.
