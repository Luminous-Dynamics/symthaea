# ARC 2-AFC scores measure distractor geometry, not rule transfer

**The 2026-07-18 fix for the ARC 2-AFC inflation introduced a second, subtler inflation of the
same class.** This is the third time this benchmark family has reported a number that turned out
to be an artifact of how the wrong answer was constructed.

## The lineage

| date | distractor | reported | why it was wrong |
|---|---|---|---|
| pre-2026-07-18 | a **random `BinaryHV`** | 99.0% (published as "100% 2-AFC") | every structured grid encoding beats random noise, regardless of whether the rule was right |
| 2026-07-18 fix | `fair_distractor_grid` — a generic wrong transform of the input | 64.9% | **this document** |
| 2026-07-31 | — | — | the 64.9% is a mixture of a geometrically guaranteed win and a below-chance loss |

The first retraction is recorded in `arc_dataset.rs`'s own header and in
`book/src/research/validation.md`. `fair_distractor_grid` was the fix for it. That fix is what is
now in question.

## Measured, independently, on 1000 real ARC-AGI-2 training tasks

Scoring is 2-AFC: a hit is `sim(pred, target) > sim(pred, distractor)`. Chance is 0.50.
Splitting the same run by a purely **geometric** property of each task — is the target nearer the
input than the distractor is? — gives:

```
OVERALL                              n=1000  acc=0.6840
target NEARER input than distractor  n= 417 (41.7%)  acc=0.9976
target FARTHER from input            n= 583 (58.3%)  acc=0.4597
```

The headline is not a capability measurement. It is **41.7% of tasks where the answer is
structurally guaranteed** (99.76%) averaged with **58.3% where scoring is below chance** (45.97%).
Which bucket a task lands in is decided before any rule is learned.

*(A parallel audit reproduced the historical figure on ARC-AGI-1: 64.9%, n=399, splitting
145/400 near at 1.0000 against 255/400 far at 0.4510. Two different datasets, same structure —
this is a property of the scoring, not of a particular task set.)*

## Mechanism

`apply_rule` is near-identity when the rule does not generalise, so the prediction stays close to
the **input**. `fair_distractor_grid` returns a generic transform *of the input*. So:

- When the target happens to sit **nearer** the input than the distractor does, a near-identity
  prediction wins automatically — 99.76%, essentially by construction.
- When the target sits **farther** — which is the common case, 58.3% — the same near-identity
  prediction matches the distractor instead, giving **below-chance** accuracy.

2-AFC cannot go meaningfully below chance without a biased comparison. Below-chance is therefore
the signature, and it is the same defect fixed in `ArcChain` on 2026-07-31 (see
`CI_FAILURE_CATALOGUE_2026-07-30.md`, Addendum 2), where it produced 0.0167 on 3-step chains.

## Two detection tests that do NOT work

Both were proposed for this audit, and each would have cleared an affected benchmark. Recording
them because they are the obvious things to try:

1. **"Affected if the target is more than one transform from the input."** Step count is a bad
   proxy for distance. Transforms move wildly different fractions of cells — `color_replace`
   ~15%, `fill_region(2×2)` ~14%, `reflect_x/y` ~67%, `translate(±1)` ~83%. A **single**
   translate-by-one lands the target *farther* from the input than the reflect_x distractor
   (measured: sim(in,target) 0.5498 vs sim(in,distractor) 0.6511, accuracy 0.1948 — below chance,
   at one step).

2. **"A recorded accuracy below 0.50 is the smoking gun."** `arc_noise` and `arc_fewshot` publish
   no metric below 0.50, because they average four transform types into every metric and never
   break out per type. The below-chance cell is **structurally invisible** — precisely the
   blindness that hid the ArcChain defect until per-chain metrics were added.

The test that does work is direct: compare `sim(input, target)` against `sim(input, distractor)`
per task and split on it.

## Per-benchmark status

| benchmark | verdict | evidence |
|---|---|---|
| `arc_dataset.rs:257` | **affected, worst case** | the 1000-task split above |
| `arc_noise.rs:204` | **affected** | published `accuracy_0pct` 0.8187 = ColorFill 1.0000 / **Translation 0.2750** / ColorReplacement 1.0000 / Reflection 1.0000 — one below-chance cell and three at ceiling, averaged into a plausible 0.82 |
| `arc_fewshot.rs:172` | **affected** | ColorFill, ColorReplacement, Reflection are 1.0000 at *every* k. Only Translation moves: 0.8250 / 0.3250 / 0.9250 / 0.5500 / 0.9250 — tracking bundle **parity**, not shot count |
| `arc_strict.rs:268` | not affected | biased the *opposite* way (target moves 19.6% of cells, distractor 71.5%) — but note a pure identity predictor also scores ~1.0 here |
| `arc_staircase.rs:111` | not affected | reflect_x is the only task, so the distractor falls through to reflect_y — both single reflections, equidistant by construction |
| `arc_scaling.rs:127` | not affected | same reason |

`arc_staircase` and `arc_scaling` are balanced **by accident**, not by design. Nothing in
`fair_distractor_grid` arranges it; their task set happens to make the first candidate collide
with the true output.

## Consequences worth stating plainly

**`arc_fewshot`'s construct is void, not merely noisy.** Its question is "does accuracy improve
with more examples?" Three of its four transform types are pinned at 1.0000 for every k, so the
entire signal comes from the Translation cell — which oscillates with bundle parity (even counts
make the prediction more identity-like, which makes the confound bite harder). `learning_rate`
(0.0106), `fewshot_gain` (0.0250) and `saturation_point` (1.4) are artifacts of that oscillation.

**The `arc_dataset` headline should not be quoted as an ARC result in any form.** Not 99%, not
64.9%, not 68.4%. Under a distance-matched distractor it should be expected to fall toward ~0.50,
because the guaranteed-win bucket disappears.

## What was deliberately not done

`fair_distractor_grid` was **not changed.** Six benchmarks share it, and altering it would move
all six at once, including the two that are currently balanced. The fix needs to be
distance-matched per benchmark — as was done for `ArcChain`, where the distractor became the
output of a *different chain of the same length* — rather than a single edit to the shared helper.

Nothing in this document changes a recorded score. It says which recorded scores mean something.

## Reproducing

The split above comes from loading real tasks via `load_arc_tasks`, replicating
`evaluate_arc_tasks`'s scoring exactly, and bucketing on
`sim(in, target) > sim(in, distractor)`. Data is at
`data/benchmarks/arc-agi-2/data/training` (1000 tasks). The per-transform decompositions for
`arc_noise` and `arc_fewshot` come from byte-faithful replications that reproduce their published
values exactly before decomposing them.
