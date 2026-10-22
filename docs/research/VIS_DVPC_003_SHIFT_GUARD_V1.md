# VIS-DVPC-003: Observable Shift-Guarded Perception Escalation v1

**Status:** Draft research protocol and synthetic reference fixture; not a deployed policy.  
**Date:** 2026-10-22  
**Issue:** [#8457](https://github.com/Luminous-Dynamics/symthaea/issues/8457)  
**Parent:** [VIS-DVPC-001 #8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156)  
**Companion:** [VIS-DVPC-002 #8440 / PR #8442](https://github.com/Luminous-Dynamics/symthaea/pull/8442)

## Research question

When the value estimator's sign can reverse on a shifted input, does gating a conservative lower-confidence-bound (LCB) escalation policy on a producer-visible shift alarm lower harmful escalations enough to justify the beneficial decisions it blocks when the alarm is false?

This tests a *decision admission guard*, not a new perception encoder, and not a claim that reported uncertainty is calibrated. The guard is eligible only if the alarm is an observable producer-side signal whose generation and cost can be measured independently of evaluator labels.

## Research motivation

The September 2026 DEEP preprint finds that improved perception can worsen downstream decisions and that charging an allocator for its own computation can change which methods outperform a random baseline. It evaluates real geometry/detection outputs on KITTI, nuScenes and nuPlan, so this synthetic fixture is not a substitute for its external evidence: https://arxiv.org/abs/2609.35910

Memory benchmarks point toward a related longer-term integration: MemoryAgentBench treats accurate retrieval, test-time learning, long-range understanding and selective forgetting as separate competencies (ICLR 2026): https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/. WorldMemArena evaluates memory through write, maintain, retrieve and use across 400 multi-session multimodal tasks: https://worldmemarena-mem.github.io/. The current slice intentionally isolates the shift-gating mechanism; joint perception-memory-action budgeting is a later integration, not claimed here.

## Policy comparison

One deterministic Rust program, `tools/vision-dvpc-shift-guard.rs`, compares five predeclared policies:

1. **No escalation** — lower-work reference.
2. **Point estimate** — escalate when estimated net decision benefit exceeds a fixed threshold.
3. **Conservative LCB** — escalate when estimate minus 1.25 times reported uncertainty exceeds the threshold.
4. **Shift-guarded LCB** — the same LCB rule, additionally blocked when the producer-visible shift alarm is on.
5. **Budget-constrained random** — deterministic seeded comparator under the same admission budget.

The LCB multiplier and threshold are fixed in source before the synthetic outcomes are scored. The hindsight oracle is an evaluator-only upper reference and is never a candidate online policy.

## Synthetic conditions

| Condition | Estimator behavior | Purpose |
|---|---|---|
| Stable calibrated | Estimate = true DVPC + bounded noise; reported uncertainty 0.75 | Check opportunity cost of unnecessary guard activations |
| Overconfident errors | Some negative values are reported as positive while uncertainty remains 0.05 | Demonstrate that a drift alarm cannot repair every form of model error |
| Sign reversal on shift | 35% hidden shifted examples reverse the estimate sign; reported uncertainty 0.20 | Test whether a separate noisy observable alarm blocks harmful escalations |

The generated alarm has 80% sensitivity to the synthetic shifted stratum and a 10% false-positive rate. This is a *test parameter*, not measured detector performance. The evaluator-only shift flag is used only after choices are committed to score false positives, false negatives and stratum outcomes.

Each regime uses 8 fixed seeds, 256 trials per seed, budget ceilings at 20%, 40% and 60% of a predeclared reference full-fidelity cost of four units per trial, and all five policies. Expected output: 360 per-seed run rows and 45 across-seed summary rows. Each run reports CVaR95 of per-trial incremental decision loss: sort trial losses descending (a harmful escalation contributes positive loss, a beneficial escalation negative loss, and a cheap-path decision zero) and average the worst ceil(5% of trials). Abstract integer cost units are not latency, FLOPs, joules or money.

## Information and authority boundaries

The online policy accepts only a `ProducerInput` with:
- trial identifier;
- estimated net benefit;
- reported uncertainty;
- declared cost, or an explicit unknown value;
- a noisy observable shift alarm.

It cannot receive the hidden shift stratum, true DVPC, actual cost, future observation or hindsight oracle. The evaluator record is a separate type. The architecture decision remains: Symthaea owns the cognitive policy; Symtropy is the world/execution substrate and can supply receipts and evaluator-private ground truth, but not leak that ground truth into online selection.

## Resource and failure accounting

- Reserve declared positive cost before escalation; unknown or invalid declared cost fails closed.
- Charge the evaluator's actual cost after execution.
- Preserve reservation overruns and run-budget overruns as explicit invalidating outcomes.
- Mark regret as `INVALID_BUDGET` after any reservation overrun or total-budget overrun; do not clip spend, exclude only inconvenient rows silently, or present invalid runs as successful budget-relative comparisons.
- Score the exact knapsack reference only after all policy decisions are committed.
- Keep task loss, total decision benefit, CVaR95 incremental decision loss, harmful escalations, missed beneficial escalations, shifted-stratum harm, alarm frequency, false alarms, beneficial decisions blocked by false alarms, cost, and valid regret visible separately. CVaR95 is computed over all trials in each regime/seed run, not only the hidden shifted subset. There is deliberately no single weighted score that can hide the safety/utility tradeoff.
- Deterministic replay is a qualification invariant, not evidence of external validity.

## Predeclared interpretation gates

These are candidate gates for this synthetic mechanism, not claims about current results:

1. In the sign-reversal condition, the shift-guarded policy should reduce shifted-stratum harmful escalations by at least 25% relative to conservative LCB in at least two of the three fixed budget levels, averaged across the seeded runs. If the comparator's denominator is zero, report the reduction as undefined, not infinite. A tail-risk improvement claim also requires CVaR95 of incremental decision loss to improve; counts and CVaR95 must both be shown.
2. In the stable-calibrated condition, total decision benefit should retain at least 90% of conservative LCB's benefit when that benefit is positive; otherwise report absolute paired differences. Report the exact count of beneficial opportunities blocked by false alarms.
3. The guard must not gain apparent utility by violating reservation or total budgets. Every invalidated run stays visible and cannot support a positive regret claim.
4. The overconfident-errors control is mandatory. A shift gate that only helps sign reversal must not be described as a general estimator-robustness solution.
5. These small seeded results are exploratory. Before a confirmatory statement, increase to at least 60 independent held-out world seeds, preserve matched policy traces, predeclare the primary contrast, and use seed-level paired uncertainty intervals. Do not treat trials within a seed as independent replicates.

Failure to meet a gate is informative: it may indicate alarm quality, false-positive opportunity cost, missing calibration, or a guard that refuses too much work. Do not retune thresholds against the held-out evaluation and then call it confirmatory.

## Benchmark ladder: synthetic mechanism to deployment

- **L0 — Mechanics:** compile, unit tests, schema checks, byte-identical replay, hidden-label separation and budget invariants.
- **L1 — Synthetic stress:** report all regimes and budget levels, not only the most favorable stratum; add alarm sensitivity/false-positive sweeps only in a separately preregistered follow-up.
- **L2 — Recorded outputs:** replace the synthetic alarm with a reproducible shift detector evaluated on frozen public perception traces; do not use ground-truth labels unavailable at decision time.
- **L3 — Closed-loop simulation:** replay matched tasks through Symtropy and measure downstream task loss, failures, latency, energy and action changes under the same task set and budget.
- **L4 — Hardware-qualified evaluation:** measure on pinned hardware/software with warm-up, repeated runs, energy-meter provenance and all allocator overhead included.
- **L5 — External comparison:** compare under matched datasets, downstream loss, compute/latency/energy ceilings and disclosed hardware. Do not claim SOTA from synthetic gains or compare raw scores across incompatible protocols.

## References

- Pham Cong & Luong, “The Decision Value of Perception Compute” (DEEP), arXiv:2609.35910, 2026-09-28: https://arxiv.org/abs/2609.35910
- Hu, Wang & McAuley, “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions” (MemoryAgentBench), ICLR 2026: https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/
- Liu et al., “WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction,” 2026: https://worldmemarena-mem.github.io/

## Claim boundary

This protocol and fixture test a synthetic mechanism. Even if all focused tests pass, the result will not demonstrate a working real-world shift detector, robust physical perception, superior downstream robotics performance, general cognitive competence or state of the art. Those claims require the later benchmark ladder with exact-head evidence and externally reproducible inputs.
