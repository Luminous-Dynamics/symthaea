# Temporal Benchmark V2 — pre-registration

**Registered:** 2026-07-31, **before** any task generator has been run against any
model and before any credit-assignment mechanism exists.
**Binds:** `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md` §5–§11, and the Step 4 temporal
credit-assignment work designed in `TEMPORAL_CREDIT_ASSIGNMENT_STEP4_DESIGN_2026-07-31.md`.

Nothing in this document may be revised after results exist. If a threshold here
turns out to be badly chosen, the correct response is to say so, report against it
anyway, and register a *new* threshold for a *future* run — not to move this one.

## Why this exists in this form

This arc has produced, in sequence: three retracted findings, and two decision
thresholds set so loosely that anything cleared them (Step 1.5's half-recovery metric
where full-recovery held all the information; Step 3's `max_spread > 1e-6`, which
reported "responds" for an actuator that selected one action everywhere). Both were
mine, both were caught only by inspecting raw data after the verdict printed.

The lesson is not "try harder." It is that a threshold chosen while the result is
unknown is worth more than a careful judgment made once the numbers are visible.

## 1. Arms

| arm | mechanism | role |
|---|---|---|
| `Static` | no temporal state | **mechanically guaranteed** negative control — at chance at aliased points by construction, not by hoped-for weakness |
| `EmaBank` | multi-timescale exponential traces, tuned | the baseline that must be beaten. EMA already showed *stronger* regime separation than HDC-LTC in the predecessor arc |
| `PermutationVsa` | explicit positional binding | structurally favored on order-structured tasks |
| `HdcLtc` | current production coupling | the mechanism under evaluation |
| `HdcLtc+Traces` | + eligibility traces (Step 4) | the intervention being tested |

`EmaBank` must be genuinely tuned, not a strawman. If it is not tuned at least as hard
as `HdcLtc`, the comparison is void regardless of outcome.

## 2. Gates that must pass before any arm runs

All are hard-fail. A run that proceeds past a failed gate is void, not caveated.

1. **Task validation.** Every corpus passes `symthaea_evidence_plane::task_validator`
   at its intended depth — and, for §5.2, `validate_timed` with explicitly declared
   bin edges.
2. **Depth is real.** Each corpus *fails* validation below its required depth. A task
   that validates at every depth does not control difficulty.
3. **Negative control is mechanical.** `Static` reads chance at aliased points by
   construction. If it scores above chance, the task leaks and the run is void.
4. **Timing is load-bearing (§5.2 only).** The corpus fails token-only validation and
   passes timed validation. If both pass, timing is not being tested.

## 3. Primary metric

**Ambiguous-point accuracy**: accuracy restricted to decision points the validator
identifies as requiring history. Global accuracy is explicitly *not* primary — most
tokens in any corpus are solvable without memory, and averaging over them is how the
predecessor hid its flaw.

Secondary, reported always, never substituted for the primary: NLL, Brier/calibration,
retention vs. delay, §5.2 interpolation/extrapolation split, sample efficiency, and
compute per correct ambiguous prediction.

Representational probes (order/history-swap sensitivity) are reported in a **separate
section** and are never combined with predictive performance into a single verdict.
Conflating them was the predecessor's original sin.

## 4. Thresholds, fixed now

- **Minimum meaningful effect**: +0.10 absolute ambiguous-point accuracy over the best
  non-`Static` baseline. Chosen to match the validator's `DEFAULT_ORACLE_MARGIN`, so
  the effect a task is certified capable of separating is the same size as the effect
  required to claim one.
- **Significance**: paired bootstrap 95% CI on the per-seed difference excluding zero.
  A point estimate above threshold with a CI spanning zero is **not** a win.
- **Seeds**: ≥ 8 per arm per task family, fixed before running, disjoint from any seed
  used during development.
- **Compute ceiling**: an arm exceeding 3× the `EmaBank` compute per correct ambiguous
  prediction cannot be called a production win on that family regardless of accuracy —
  it may only be called an experimental capability result.

## 5. Declared outcomes — all legitimate, none preferred

Per plan §11. Registering these in advance so no outcome can be reframed after the
fact as the one that was expected:

- `HdcLtc` uniquely wins → promote for those workloads.
- `EmaBank` matches it → **use EMA in production, keep LTC experimental.**
- `PermutationVsa` wins on order-structured families → use explicit sequence structure there.
- Each wins a different family → route by task class.
- **No meaningful difference anywhere → prefer the simpler mechanism.**

The fifth outcome is the one prior evidence most supports: the coupling ablation found
no demonstrated predictive superiority for HdcLtc over EMA, at a consistent 4–5× compute
premium. That is the prior this benchmark is testing against, not a result to be
avoided.

## 6. What would falsify the Step 4 hypothesis

The hypothesis is that **training signal, not architecture, is the binding constraint** —
so adding temporal credit assignment should move ambiguous-point accuracy where
architecture changes did not.

It is falsified if `HdcLtc+Traces` fails to clear the +0.10 threshold over `HdcLtc` on
any task family while the gates all pass. That result would mean the Keystone
conclusion generalizes: the substrate is not the limit and neither is credit
assignment, and the next hypothesis must come from somewhere else entirely.

Registering this explicitly because the temptation on a null will be to conclude "the
traces were implemented wrong" rather than "the hypothesis was wrong." Implementation
doubt is only admissible if a *mechanical* integrity check fails — trace update counts
not matching declared call counts, via the evidence plane's existing contract. Absent
such a failure, a null is a null.

## 7. Not covered

- §5.3 hysteresis generator does not exist yet; its thresholds are inherited from §4
  above but its gates must be registered when it is built.
- No claim about the main cognitive loop. Step 3 established its actions are internal
  hyperparameter adjustments that cannot generate interventional data; nothing here
  changes that, and a win on this benchmark would not.
