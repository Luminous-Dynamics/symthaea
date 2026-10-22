# VIS-DVPC-004: Shift-Guard Operating Envelope v1

**Status:** Draft protocol and synthetic reference harness.  
**Date:** 2026-10-22  
**Issue:** [#8470](https://github.com/Luminous-Dynamics/symthaea/issues/8470)  
**Mechanism predecessor:** [VIS-DVPC-003 PR #8458](https://github.com/Luminous-Dynamics/symthaea/pull/8458)  
**Parent program:** [VIS-DVPC-001 #8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156)

## Research question

Under what combinations of shift prevalence, detector sensitivity, false-positive rate and alarm temporal persistence does a shift-gated perception-compute policy outperform an ungated conservative lower-confidence-bound policy, once detector compute, delays, missed opportunities and tail loss are charged?

A single synthetic detector point cannot establish an operating envelope. This slice measures the sensitivity surface and compares an immediate alarm gate with a 2-of-3 temporal confirmation gate. It does not train or tune a detector.

## Why this follows the external evidence

The September 28, 2026 DEEP paper reports that perception escalation may make downstream decisions worse, and that lightweight allocators can lose their apparent advantage when their own computation is charged. Its evaluation uses real perception outputs from KITTI and nuScenes and open-loop planner outcomes on nuPlan: https://arxiv.org/abs/2609.35910

This benchmark is deliberately weaker in external validity: all values and alarms are synthetic. Its role is to identify which mechanisms and detector operating regions are worth evaluating on frozen real traces before integrating a guard into Symtropy.

A related lesson from interactive memory evaluation is that raw recall or memory-write scores are not enough; memory must be judged by later decisions and actions. MemoryArena tests interdependent multi-session memory-agent-environment tasks: https://arxiv.org/abs/2602.16313. WorldMemArena evaluates a write-maintain-retrieve-use lifecycle over 400 multi-session multimodal tasks: https://arxiv.org/abs/2605.29341. The future integrated benchmark should couple perception escalation with memory refresh/retrieval and actual action loss rather than report a standalone vision metric as cognition.

## Factorial design

The Rust harness is `tools/vision-dvpc-alarm-envelope.rs`. It includes 72 shifted conditions:

- Hidden shift prevalence: 5%, 15%, 35%, 60%.
- Nominal detector sensitivity: 50%, 70%, 90%.
- Nominal detector false-positive rate: 1%, 5%, 20%.
- Alarm temporal persistence: 0 (independent alarm samples) or 0.70 (bursty alarm process).

Two mandatory controls are included: a no-shift/calibrated-estimator control and a no-shift/overconfident-estimator control. The hidden shift itself is generated in persistent episodes with transition probabilities chosen to preserve the declared long-run prevalence in the stationary model.

For each scenario, 8 fixed seeds, 256 sequential observations per seed, 3 budget levels (20%, 40%, 60%), and 3 policies are evaluated. Expected output is 5,328 run rows (74 scenarios × 8 seeds × 3 budgets × 3 policies) and 666 summary rows (74 × 3 × 3).

## Compared policies

1. **LCB only:** no shift detector; escalate when estimated net value minus 1.25 times reported uncertainty exceeds a frozen threshold.
2. **Instant guard:** pay for the detector on each observed input and block a qualifying escalation when the current alarm is positive.
3. **2-of-3 guard:** pay for the detector and require at least two positive alarms in the three most recent computed observations before blocking a qualifying escalation. It fails closed during warm-up until three alarm outputs are available.

The temporal arm intentionally trades faster reaction for possible resistance to isolated false alarms. The observed-history parameter passed to the policy contains only boolean alarms, never shift labels or true value. The evaluator's episode identifier is not an input to the policy and is used only to attribute detection delay correctly across distinct shift episodes.

## Compute and accounting

Each full-fidelity reference pass costs 400 abstract units. Each detector invocation costs 20 units (5% of the reference pass) and is charged to the *same* budget as escalation. The 20/40/60% budget ceilings are fixed before scoring. Units are abstract and must not be described as FLOPs, milliseconds, joules, or dollars.

The action may have an unknown declared cost or a declared cost that is lower than the realized cost. Unknown, zero or unaffordable reservations fail closed. A realized cost above its declaration and a total budget exceedance are retained as explicit violations; no violated run should support a positive budget-efficiency claim.

## Producer/evaluator boundary

The producer sees:
- estimated net benefit and its reported uncertainty;
- declared escalation cost or unknown;
- only alarm results for which the 20-unit computation was paid.

The evaluator alone owns true decision value (DVPC), actual escalation cost, shift truth, shift-episode identity, and alarm labels used for post-commit scoring. No true value, future alarm, hidden episode identifier or oracle is passed to online policy selection.

Nominal sensitivity/FPR values are scenario generator settings, not empirical measurements. The report computes empirical sensitivity and false-positive rate only over detector outputs actually obtained inside each policy run's budget. If a denominator is zero, the rate is `NA`, not zero. Detection delay is measured in sequential trial steps from the onset of a hidden shift episode to the first gate trigger supported by alarms observed within that same episode; undetected episodes are counted separately. No policy can be called robust because of the intended synthetic generator alone.

## Reported outcomes

Per-run and across-seed output retains:
- escalation count, beneficial/harmful escalations and missed-benefit count;
- shifted-trial, shifted-escalation and shifted-harm counts;
- observed detector count, empirical sensitivity/FPR numerators and denominators;
- number of shift episodes, detected/undetected episodes and mean detection delay;
- false-alarm-blocked candidates and beneficial opportunities actually blocked by false alarms;
- total decision benefit, task loss and CVaR95 of per-trial incremental decision loss;
- total charged units, detector units, budget, reservation overruns and run-budget overruns;
- same-seed paired deltas in total benefit, CVaR95 and shifted-stratum harmful escalations relative to LCB-only, so each guard’s change is visible for the same condition and budget.

CVaR95 averages the worst ceil(5% of per-trial incremental losses), with positive loss for harmful escalation, negative loss for beneficial escalation, and zero for choosing the cheap path. It is a risk diagnostic, not a safety guarantee. No single weighted score is reported.

## Interpretation and statistical limits

Use paired seed-level contrasts against LCB only. The 8-seed report is exploratory and descriptive; observations within a seed are temporally dependent and are not independent replicates. Do not use raw trial counts to manufacture narrow confidence intervals. A confirmatory follow-on should freeze a real-trace evaluation, use at least 60 independent world seeds or appropriately grouped trace units, predeclare primary contrasts, and report paired uncertainty intervals plus every scenario/stratum outcome.

A guard should only advance if it reduces shifted-stratum harm in conditions it is intended to address, preserves useful decisions in stable controls, and remains within the same budget with its full compute overhead included. False-alarm cost and detector miss behavior must be shown explicitly. The overconfident control is important because a shift detector will not generally correct arbitrary estimator errors that are unrelated to distribution shift.

## Benchmark ladder

- **L0 — Mechanism:** exact-head compilation, tests, byte-identical replay, report-schema verification and hidden-label separation.
- **L1 — Operating envelope:** the frozen prevalence × sensitivity × false-positive × temporal-persistence sweep with cost accounting.
- **L2 — Real detector traces:** replace generator alarms with an observable detector output on frozen public traces; measure its own runtime/energy and ensure no offline ground truth enters the policy.
- **L3 — Symtropy closed-loop replay:** compare matched actions and downstream loss, with memory retrieval/update alternatives and the same total resource ceiling.
- **L4 — Hardware evaluation:** pinned hardware/software; report latency distributions, energy, thermal behavior, allocator overhead and failed/abstained runs.
- **L5 — External comparison:** compare only compatible task suites, budgets and evaluator definitions. Synthetic gains are not a SOTA claim.

## Claim boundary

This is a synthetic operating-envelope experiment. Even if every test passes, it does not establish real shift-detector quality, calibrated deployment risk, physical perception robustness, safer robotics, general cognition or state of the art.
