# Recurrent-dimension masking — Step 1 characterization

**Date:** 2026-07-30
**Harness:** `examples/recurrent_mask_characterization.rs` (commit `59e12b235c`, enabled `89c2536e36`)
**Mechanism under test:** the fixed-suffix CfC recurrent-state lesion made reachable by Step 0 (`132b919be1`)
**Status:** measured, single run of each section, release build, on a shared host at load ~35–50

---

## Headline: the Step 0 "ordering" finding is RETRACTED

Step 0's test printed, and its commit message (`132b919be1`) asserted as a finding:

> State energy is heavily concentrated in the leading dimensions. Removing 75%
> of dimensions removes only 13.2% of the state's L2 energy… lesion severity is
> dominated by the dimension ordering rather than by the nominal fraction.

**That interpretation is wrong.** Section B below measures the energy carried by the
trailing subset the mask actually amputates, against the leading subset and against 16
random subsets of identical size, on state captured from an **unmasked** run. All four
agree with the flat expectation:

| frac | dims cut | trailing (production) | leading | random (median) | flat expectation |
|-----:|---------:|----------------------:|--------:|----------------:|-----------------:|
| 0.75 | 64 | 0.2728 / 0.2040 / 0.2514 | 0.2571 / 0.2603 / 0.2911 | 0.2553 / 0.2471 / 0.2494 | 0.2500 |
| 0.50 | 128 | 0.5871 / 0.4490 / 0.5188 | 0.4131 / 0.5510 / 0.4817 | 0.5071 / 0.4986 / 0.5062 | 0.5000 |
| 0.25 | 192 | 0.7431 / 0.7402 / 0.7092 | 0.7272 / 0.7961 / 0.7491 | 0.7518 / 0.7422 / 0.7506 | 0.7500 |

(three independent runs, `state dim = 256`, 40 snapshots each)

Trailing and leading **swap rank between runs** — at frac=0.50, run 0 has trailing
(0.5871) above leading (0.4131) and run 1 has the reverse (0.4490 vs 0.5510). Systematic
concentration cannot flip sign; noise can. Energy in the CfC hidden state is
**essentially uniform across dimensions**, and the fraction maps roughly linearly to
energy removed.

### Why Step 0 saw 13.2%

Step 0 read `pre_mask_norm` / `post_mask_norm` **at the mask site inside an
already-masked run**. The trailing dimensions had been zeroed on the previous cycle and
only partially repopulated by the step, so they held little energy — and removing little
energy is exactly what it measured.

Both numbers are correct measurements. The error was mine, in reading an *in-situ*
measurement of a lesion already in progress as evidence about the *state's dimension
ordering*. It is evidence about **history**, not ordering — which Section C confirms
directly.

---

## A. Reachability

| config | frac | source | masked? |
|---|---:|---|---|
| default (nothing enabled) | 1.000 | `Disabled` | no |
| encoding noise only | 1.000 | `Disabled` | no |
| masking on, silicon, no override | 1.000 | `SubstratePressure` | no |
| **masking on, quantum, speed-mod on** | **0.300** | `SubstratePressure` | **yes** |
| masking on, quantum, speed-mod OFF | 1.000 | `SubstratePressure` | no |
| masking on, override 1.00 | 1.000 | `FixedOverride` | no |
| masking on, override 0.50 | 0.500 | `FixedOverride` | yes |
| masking on, override 0.00 | 0.000 | `FixedOverride` | yes |

The three no-op routes for the **default** config are confirmed. But a second correction:
**the override is not the only entry point.** A scale-constrained substrate with speed
modulation enabled reaches a sub-unit fraction unaided (`QuantumComputer` → 0.300, and it
masks for real). Any later ladder must hold `substrate_type` *and*
`enable_substrate_speed_modulation` fixed, or a run that varies substrate for unrelated
reasons silently varies the lesion too.

## C. Redistribution — the lesion compounds, and does not recover

Sustained masking at frac=0.5 against an unmasked control:

| cycle | masked pre-mask L2 | unmasked L2 | ratio |
|------:|-------------------:|------------:|------:|
| 0 | 0.00010 | 0.00007 | 1.272 |
| 40 | 0.01570 | 0.02291 | 0.686 |
| 80 | 0.06258 | 0.11879 | 0.527 |
| 120 | 0.28149 | 1.42158 | 0.198 |
| 160 | 0.54414 | 9.29463 | 0.059 |
| 199 | 0.61524 | 7.37446 | 0.083 |

The ratio decays monotonically to ~6–8%. The surviving prefix does **not** compensate;
the lesion compounds. Holding frac=0.5 for 200 cycles is therefore not "running at 50%
capacity" — it is a progressive collapse toward roughly a tenth of the control's state
magnitude.

**Caveat on this section:** the control is not stationary. The unmasked norm grows about
five orders of magnitude over 200 cycles (0.00007 → ~9) and is non-monotonic at the tail
(9.29 → 7.37). The ratio therefore compares two moving quantities, and the *rate* of
decay should not be over-read. The direction and magnitude of the gap are the load-bearing
observation; the precise trajectory is not.

## E. Overhead — measured with interleaving

Three conditions cycled **round-robin in one process**, 200 rounds, so contention is
common-mode:

| condition | min_step (µs) | med_step (µs) | sum_step (µs) | sum overhead (µs) | masked | median paired delta (µs) |
|---|---:|---:|---:|---:|---:|---:|
| frac 1.00 (no-op) | 171 | 267.0 | 65658 | 0 | 0 | 0.0 |
| frac 0.50 | 163 | 265.0 | 63934 | 418 | 200 | −1.0 |
| frac 0.25 | 158 | 269.0 | 68566 | 457 | 200 | +4.0 |

**Mask overhead: 0.64%–0.70% of step time** (~2.1–2.3 µs per cycle). Real, strictly
additive, and small.

### Methodological result: which load-robustness technique actually worked

- **Minimum-of-N did not suffice.** The minima still show masked conditions as *faster*
  (−8 µs, −13 µs), which is impossible — the mask runs after the step, on its output.
  The minimum is an extreme order statistic and stays noisy even at N=200.
- **The interleaved paired difference did work.** −1.0 µs and +4.0 µs, straddling zero,
  roughly an order of magnitude tighter than the naive minima.
- **The median across interleaved conditions was cleanest of all**: 267 / 265 / 269 µs,
  agreeing within 1.5%, which is the correct answer — the step is called identically in
  every condition.

For "does this change the cost of an operation," on a loaded shared host: **interleave
and compare medians or paired differences; do not trust minima, and never compare
sequential runs.** The Step 0 gate's sequential single-shot comparison produced a masked
step apparently 2.6× faster than the unmasked one; the same question answered by
interleaved medians gives agreement within 1.5%.

---

## Implications for any later controller ladder

1. **Fraction is a usable severity dial instantaneously** — the ordering confound
   reported after Step 0 does not exist. This *simplifies* the ladder design.
2. **But sustained masking is not a steady state.** The effective lesion deepens with
   duration, so a fixed-fraction arm is not a fixed condition. Duration must be a
   controlled variable, not an incidental one.
3. **Expansion is not the inverse of contraction.** Restored dimensions re-enter at
   near-zero and must be repopulated by the dynamics over many cycles. Hysteresis in a
   controller interacts with this directly, and "contraction restores efficiency when
   demand passes" should not be assumed as a success criterion without measuring the
   recovery time constant first.
4. **Substrate and speed-modulation settings are part of the lesion**, per Section A.
5. **Cost is ~0.7% of step time** — cheap, but still strictly a cost. Nothing here
   supports a metabolic-efficiency claim; that still requires a genuinely narrowed
   computation path, which this mechanism is not.

---

# Step 1.5 — actuator bandwidth (2026-07-31)

**Harness:** `examples/recurrent_mask_recovery.rs` (commit `9e7eb2e89d`)
**Rule:** preregistered in that commit message *before* any result existed.

## Verdict: MARGINAL

| contraction | repeat | share at expansion | half-recovery | full-recovery | final share |
|---:|---:|---:|---:|---:|---:|
| 40 | 0 | 0.000000 | 7 | 58 | 0.4502 |
| 40 | 1 | 0.000000 | 7 | 29 | 0.4508 |
| 40 | 2 | 0.000000 | 7 | 28 | 0.4500 |
| 160 | 0 | 0.000000 | 7 | 16 | 0.4550 |
| 160 | 1 | 0.000000 | 10 | **>120** | 0.3529 |
| 160 | 2 | 0.000000 | 7 | 20 | 0.4550 |

**Median half-recovery = 7.0 cycles, 0 censored.** Between the preregistered 5 and 20,
so: **Step 2 proceeds, restricted to duty-cycle-matched arms only.** The mechanism is
not disqualified as an actuator.

Two things the data says beyond the verdict:

- **Bandwidth does not degrade with use.** Half-recovery is 7 cycles after a 40-cycle
  contraction and 7 cycles after a 160-cycle one. The worry that a deeper lesion would
  take proportionally longer to undo did not materialize.
- **Full recovery is slow and unreliable**, ranging 16 → 58 cycles with one run failing
  to reach 0.45 within the 120-cycle window at all (final share 0.3529). It does not
  correlate with contraction duration either.

### A limitation of my own rule design, disclosed

I preregistered on **half**-recovery. The data suggests **full**-recovery is the more
informative metric — it is where all the variance and the single failure live, and it is
what a controller actually needs if "expand when the signal demands capacity" is to mean
anything. Half-recovery is tight and boring; full-recovery is the interesting quantity.

The verdict above is reported against the metric as preregistered, **not** re-derived
from full-recovery, because switching metrics after seeing results is exactly the move
preregistration exists to prevent. But future rules in this area should key on full
recovery, and any Step 2 controller should assume an adaptation period of tens of cycles,
not the 7 the headline number suggests.

## F. The norm trajectory — resolved, and it corrects Section C

| cycle | unmasked state L2 | trailing share |
|---:|---:|---:|
| 0 | 0.000055 | 0.3147 |
| 50 | 0.028665 | 0.5039 |
| 100 | 0.185657 | 0.4919 |
| 150 | 4.102859 | 0.4836 |
| **200** | **42.623472** | 0.4678 |
| 250 | 14.534687 | 0.4697 |
| 300 | 7.573672 | 0.4990 |
| 400 | 5.024497 | 0.5529 |
| 500 | 4.218694 | 0.5607 |
| 599 | 4.106174 | 0.5574 |

**Not unbounded growth.** The norm overshoots to ~42.6 around cycle 200, then decays and
settles near ~4.1 by cycle ~500. The Step 1 caveat is retired: absolute magnitudes are
usable, but only after roughly 400–500 cycles of warmup.

`trailing_share` holds near 0.5 throughout, independently confirming Step 1's uniformity
finding and validating it as the recovery target.

### Correction to Section C

**Section C ran only 200 cycles — precisely the peak of that overshoot.** Its control was
not at steady state; it was at the top of a transient where the norm changes by an order
of magnitude within ~50 cycles (and run-to-run variance in that region is correspondingly
large: Section C's control read 9.29 at cycle 160 where this run reads ~4.1 at 150 and
~42.6 at 200).

So Section C's headline — masked state falling to "~6–8% of control" — **overstates the
compounding**. Part of that collapse was the *control* spiking, not the masked run
sinking. The direction is real and the gap is real; the magnitude is inflated by the
choice of window. A re-run of Section C past cycle 500, against a settled control, is
the honest version of that measurement and has not been done.

## Consequences for Step 2

1. **Duty-cycle-matched arms are mandatory** (per the rule). Without them, a controller
   that spends time unmasked beats a sustained fixed fraction for reasons unrelated to
   its signal.
2. **Controller adaptation period ≥ ~20–30 cycles.** A controller reacting to
   cycle-to-cycle prediction error is far faster than this actuator can follow.
3. **Warm up ≥ 500 cycles before measuring anything absolute**, per Section F.
4. **"Contraction restores efficiency when demand passes" remains unproven** as a
   success criterion — full recovery failed outright in 1 of 6 runs.

## Not measured

- Recovery time constant after expansion (implied by point 3; needs a runtime fraction
  setter, which does not exist).
- Whether the compounding in Section C depends on the fraction, or on the schedule.
- Any effect on task competency — this is a mechanism characterization only, and says
  nothing about whether masking helps or harms behavior.
- Single run per section. Section B/D has three repeats; A, C and E do not.
