# AOBENCH-012: information accounting for sequential experiments

**Date:** 2026-11-14  
**Parent:** [AOBENCH-011 / #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173)  
**Initial evaluator:** [AOBENCH-011 / #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173) · [initial evaluator PR #12186](https://github.com/Luminous-Dynamics/symthaea/pull/12186) · [AOBENCH-012 / #12194](https://github.com/Luminous-Dynamics/symthaea/issues/12194) · [implementation PR #12195](https://github.com/Luminous-Dynamics/symthaea/pull/12195)

## Why this correction matters

A probe can be well chosen without producing qualified knowledge. Delivery can fail, an intervention can be contaminated, an observation can be unavailable, or a qualified observation can contradict the finite model class. A single cumulative "information gain" number risks hiding the difference between a good plan and evidence that actually changed the posterior.

The evaluator now distinguishes:

| Quantity | Meaning | Credit rule |
|---|---|---|
| Expected information gain | Prospective entropy reduction expected from the chosen probe under the frozen finite model | May be reported even when execution evidence fails |
| Observed information gain | Actual target-class posterior entropy reduction induced by an observation | Only credited after delivery, isolation and observation are verified and the observation is consistent with evaluator truth |
| Myopic efficiency regret | Difference between the chosen probe and the best currently eligible one-step probe, measured by expected information gain per cost | Not a globally optimal multi-step policy regret |
| Informative probe selections | Selected probes with positive expected target-class information gain | Kept separate from evidence-qualified trial count |

The evaluator preserves the distinction between identifying a target-equivalence class and identifying a unique mechanism implementation.

## Metamorphic / adversarial controls

- Unavailable delivery can have positive expected information value but zero observed information gain and cannot shrink the active hypothesis set.
- A qualified observation receives information credit only for its realized posterior entropy reduction.
- Multiplying every positive prior weight by a common positive factor preserves expected information gain and the decision outcome.
- Reversing hypothesis and probe input order preserves the evaluation result.
- Bijectively renaming every outcome label in both the frozen probe table and observed trace preserves qualification, posterior state and both information-gain metrics.

These are deterministic finite-model contract tests, not measurements of real-world causal discovery.

## Limitations that remain explicit

This pilot does not implement stochastic outcome distributions, empirical power, confidence-interval coverage, multiplicity correction, adaptive stopping under sampling noise, interference, hidden confounding, or external real-world adjudication. The next increment should introduce noisy repeated observations with preregistered stopping boundaries, calibration and family-disjoint holdouts. Classical active-design baselines (including appropriate PC/FCI/GES-family methods and greedy information-gain policies) remain required for a fair comparison.

No test, build, formatting, Clippy, CI, or SOTA result is claimed by this draft. Exact-head Runtime Qualification remains pending.
