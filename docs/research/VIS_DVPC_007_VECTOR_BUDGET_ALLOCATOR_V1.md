# VIS-DVPC-007: Vector-Budget Online Allocator v1

**Status:** Draft reference policy and adversarial unit tests.  
**Issue:** [#8517](https://github.com/Luminous-Dynamics/symthaea/issues/8517)  
**Parent online-policy PR:** [#8431](https://github.com/Luminous-Dynamics/symthaea/pull/8431)  
**Post-commit evaluator:** [#8506](https://github.com/Luminous-Dynamics/symthaea/pull/8506)  
**Research comparator:** [DEEP, arXiv:2609.35910](https://arxiv.org/abs/2609.35910)

## Why this change

The earlier online reference policy represents escalation cost as one scalar. The paired-trace scorer records latency and energy as distinct resources. A scalar proxy can accept work that fits one constraint while violating the other, or conceal which constraint was binding.

This prototype treats escalation as a transaction over a resource vector:
- **reserve** an upper bound for both latency and energy before expensive work begins;
- **commit** the chosen action and input digest;
- **settle** against host-measured actual resource use;
- **block new escalation** after an overrun or unknown measurement.

Admission is componentwise. There is no weighted latency-plus-energy score that can trade a hard violation of one cap for slack in the other.

DEEP studies decision-value-of-perception under latency and energy budgets, charges allocator overhead, and warns that an escalated perception output may worsen downstream decisions. This reference aligns the budget mechanics with that evaluation framing; it does not reproduce DEEP or establish empirical superiority.

## Standalone qualification

No Cargo workspace integration or third-party dependencies are required:

```sh
rustc --edition=2024 --test tools/vision-decision-value-vector-budget.rs -o /tmp/vis-dvpc-007-tests
/tmp/vis-dvpc-007-tests
```

The dedicated workflow also checks rustfmt on the exact PR head.

## Decision contract

The online `decide` interface receives only:
- opaque trial ID;
- frozen loss-profile ID;
- digest of permitted pre-escalation inputs;
- whether the cheap path is admissible for the task;
- a calibrated pre-escalation lower-bound benefit estimate, or missing estimate;
- declared upper bounds for latency and energy.

It has no parameter for evaluator truth, full-mode outputs, or paired counterfactual labels. An estimate must name its estimator artifact, match the loss profile and input digest, and be finite. If the estimate is invalid or unavailable, the policy chooses cheap only when the cheap path is contractually safe; otherwise it abstains.

A resource vector with either zero bound is treated as unknown rather than a free escalation. This conservative restriction prevents incomplete measurements from looking affordable.

## Ledger invariants

1. **Atomic reserve:** either both resource reservations succeed or neither changes.
2. **Reservation-first accounting:** outstanding work consumes budget before execution.
3. **Componentwise admission:** each resource's available amount is capacity minus charged minus reserved; subtraction and addition are checked.
4. **Measured settlement:** actual latency and energy are independently charged. Costs are not clipped to the declared reservation.
5. **Unknown is not zero:** if either measurement is missing, known costs are still recorded, the pair reservation is released, and the run is blocked from further escalation.
6. **Fail closed on breach:** any per-resource reservation overrun, run-budget overrun, or unknown measurement makes availability return `None`.
7. **No double settlement:** only currently outstanding reservation IDs can settle.
8. **Stable receipt:** every decision can render a deterministic TSV record binding trial ID, choice, reason, estimate, both resource bounds, reservation ID, available pre-reservation resources, and the input digest.

This is an in-process reference ledger. It does not provide a durable crash-safe journal, trusted clock, energy sensor, process-isolation boundary, or proof that a host's reported measurements are honest. Those belong in a separately designed and tested execution adapter.

## Adversarial unit suite

The test source exercises:
- latency-only admission failure without partial energy reservation;
- energy-only admission failure without partial latency reservation;
- multiple outstanding reservations competing for both resource caps;
- unknown energy measurement with known latency, pair release, and blocked follow-on work;
- single-resource reservation overrun;
- total run-budget overrun;
- unknown and duplicate settlement;
- arithmetic overflow behavior;
- unsafe cheap-path abstention when the estimate is missing;
- zero cost bounds rejected as unknown;
- duplicate/zero trial IDs;
- deterministic receipts and input digest linkage;
- mismatched estimator/input digest rejection;
- strict benefit-threshold behavior.

Passing unit tests would establish only these reference-code mechanics. They would not establish that estimates are calibrated, costs are accurately measured, reservations match hardware, or decisions improve real task utility.

## Comparison plan and benchmark targets

Use the paired scorer from PR #8506 once a frozen, rights-cleared trace adapter exists. On exactly the same traces and loss profile, compare:
- all-cheap and all-full;
- budget-matched random escalation;
- a cheap/lightweight score threshold;
- a conservative lower-confidence-bound router;
- this vector-budget policy and subsequent candidates;
- the evaluator-only hindsight oracle as an upper reference, never as a deployable baseline.

Keep latency and energy ceilings simultaneously binding. Charge allocator/runtime overhead. Report utility mean and equal-group-weighted utility, harmful/beneficial escalations, safety changes, CVaR95 decision loss, reservation/run overruns, tail latency, actual energy, abstentions and oracle gap. Stratify by scenario and source group; use paired bootstrap intervals and a frozen untouched test split. Retain invalid and over-budget runs in the report.

For driving tasks, pair the perceptual decision with downstream planner outcomes and keep log-replay/open-loop and closed-loop/reactive results separate. nuPlan provides explicit open-loop and closed-loop evaluation modes and safety, comfort, progress and scenario-specific metrics: https://www.nuplan.org/. Dataset availability does not itself grant commercial rights; check the applicable license and data-usage agreement before building an adapter.

### Proposed targets (not results)

- **Hard invariant:** zero admissions that exceed either declared remaining resource bound.
- **Accounting:** every admitted action is settled once; unknown/overrun outcomes block future admissions and remain visible.
- **Determinism:** identical committed inputs and version produce byte-identical receipts.
- **Decision quality:** beat matched-budget random and lightweight score-threshold policies on the frozen test set without worsening the pre-registered safety/tail-loss constraints.
- **Robustness:** report worst-group and distribution-shift slices, not just average gain.
- **Efficiency:** report allocator p50/p95/p99 latency and measured energy separately from expensive-mode cost.
- **Generalization:** freeze thresholds on validation, then reproduce on a test set and at least one held-out environment before making external claims.

No numeric decision-quality target is claimed until task utility, resource ceilings, safety limits and the real trace protocol are frozen. A meaningful benchmark win requires a paired confidence interval and no pre-registered safety/tail-loss regression, not merely a higher aggregate score.

## Paradigm-shift roadmap

The longer-term step is **decision-value control under explicit constraints** rather than confidence-triggered escalation:
1. predict the distribution of downstream utility change, including probability and severity of harm;
2. select among cheap, expensive, delayed, alternate-sensor, and abstain actions with explicit hard resource caps;
3. use online predictions only from causal-prefix evidence, with costs calibrated on the target hardware and uncertainty propagated to admission;
4. score committed actions against paired task outcomes after execution; keep the hindsight oracle sealed from the allocator;
5. compare lightweight transparent policies to learned routers, and only add model complexity when the paired test demonstrates additional value after overhead.

This PR implements step zero/one of the *resource transaction contract*, not the full policy or a SOTA result.
