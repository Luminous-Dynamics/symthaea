# VIS-DVPC-015: All-in online budget ledger v1

**Status:** Draft reference accounting module.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8580  
**Related:** VIS-DVPC-007 (vector-budget allocator), VIS-DVPC-006 (post-commit scorer), DEEP (https://arxiv.org/abs/2609.35910)

## Problem

VIS-DVPC-007 atomically reserves incremental escalation latency and energy but intentionally does not reserve the allocator's own compute. VIS-DVPC-006 can charge measured allocator overhead in the evaluator, but that is only an after-the-fact score. It cannot prevent the online allocator from using more resources than the run can afford while deciding whether to save resources.

This tranche creates a separate outer ledger to compose around VIS-DVPC-007, not a second value estimator or router.

## Transaction

For each decision ID:

1. **Reserve allocator overhead first.** Before invoking feature extraction, shift assessment, decision-value prediction or routing bookkeeping, reserve the precommitted upper bounds for allocator latency and energy. If both don't fit the remaining run budget, reject the decision call. The system must use a separately accounted safe-cheap or abstain route; the allocator does not run for free.
2. **Reserve escalation separately.** Only after an allocator reservation exists may a full-mode escalation reserve its incremental latency and energy. Failure on either resource blocks escalation, but does not erase the allocator work already reserved.
3. **Settle actual cost.** After work, settle measured allocator resources and, when reserved, measured escalation resources. Actuals within bounds release unused reserve. Unknown/nonpositive measurements or work performed without a reservation contaminate the run.
4. **Fail closed after overrun.** If actual cost exceeds any per-action bound or all-in run ceiling, record the actual measured use, mark the run contaminated, and block further allocation/escalation admission. Existing open reservations remain visible until settled; unresolved reservations are not treated as zero cost.

The ledger uses componentwise latency and energy accounting. It never scalarizes milliseconds and microjoules or lets slack in one dimension compensate for violating the other. Outstanding reservations reduce capacity before new work starts. Reserve/release operations are atomic across the vector.

## Bounds and trust

Configuration is valid only when run budgets and per-action bounds are strictly positive in both resources, and the allocator bound fits the total budget. The escalation bound may exceed the remaining total budget; a particular escalation is then denied while the cheap decision can still be settled.

The ledger proves accounting mechanics relative to the configured bounds; it cannot prove those bounds are valid. A credible deployment bound needs target-hardware calibration for cold and warm starts, feature extraction, shift detection, the value estimator, bookkeeping, failed routes and measurement jitter. The energy meter and latency clock are part of the trusted measurement path and need independent qualification.

## Receipt semantics

The deterministic line-oriented receipt reports both resource dimensions, spent/reserved/remaining budgets, settled cheap/full decisions, budget rejections, contaminated decisions and per-decision reservation/actual amounts. The receipt explicitly carries performance_claim=NONE.

A contaminated run is not a valid budget-feasible comparison, even if a later aggregate score looks good. Preserve the receipt and report the violation. Do not delete the run or reinterpret missing measurements as zero.

## Qualification

The standalone Rust unit tests cover allocator reservation before decision, componentwise/atomic resource admission, all-in cost reservation, cheap-path settlement, separately reserved full path, energy-bound rejection, overrun contamination, missing measurements, unreserved work, duplicate/double settlement, outstanding-reservation accounting, deterministic receipts, invalid configurations and arithmetic overflow.

No hardware-specific bound, empirical decision-value gain, deployment safety or SOTA claim is made. Synthetic tests establish mechanics only.
