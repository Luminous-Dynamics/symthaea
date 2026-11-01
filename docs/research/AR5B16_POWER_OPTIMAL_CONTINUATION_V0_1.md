# AR-5B16: Power-Optimal Sequential Continuation Under a Null-Size Constraint v0.1

**Status:** finite simple-null/simple-alternative reference draft; seven tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → #9854 → #9856 → this branch.  
**Tracking:** #9857.  
**Research anchor:** Koning & van Meer, “Anytime validity is free: inducing sequential tests,” JRSSB 88(4), 1366–1384, published 21 February 2026. https://doi.org/10.1093/jrsssb/qkag050

## 1. Why a separate alternative model is necessary

AR-5B15 audits the maximum null rejection expectation of an adaptive stopping/continuation policy. That is a validity-oriented objective. It should not be confused with power optimization.

This tranche explicitly separates:
- **P0 validity:** a [0,1]-valued test process must be a nonnegative supermartingale under the declared null transition kernel.
- **P1 utility:** the policy is selected to maximize expected rejection probability under a separately declared alternative transition kernel.

The alternative affects the selected policy and its power, but not the type-I proof. After selecting the policy under P1, the implementation evaluates that exact policy under P0 in a second pass and rejects any result above alpha.

This is a finite model-based analogue of optimizing utility subject to a hard safety constraint.

## 2. Paired finite-tree contract

Each node contains:
- a stable node and parent ID;
- a conditional P0 transition probability;
- a conditional P1 transition probability on the same observable child support;
- a [0,1] test/rejection probability if the policy stops at that node.

Every internal node must normalize to one under both models. Under P0, the conditional expected child test value must not exceed the parent test value. The first-stage root is bounded by alpha. Each conditional continuation root is bounded by the corresponding first-stage interim value.

The optimizer uses P1 only:
```
Power(h) = max(
    test_value(h),
    sum_child P1(child | h) * Power(child)
)
```

The outer controller chooses between launching the history-specific conditional continuation and gathering more first-stage evidence:
```
AdaptivePower(h) = max(
    Power(continuation at h),
    sum_child P1(child | h) * AdaptivePower(child)
)
```

After the actions are fixed, a separate dynamic program evaluates the selected policy under P0:
```
NullSize(h) = test_value(h)                  if policy stops
NullSize(h) = sum_child P0(child | h) * NullSize(child) otherwise
```

The final root size must be <= alpha. The plan/report digests bind model IDs, both transition tables, test values, and reconstructed policies.

## 3. Worked fixture (synthetic, exact arithmetic)

The first-stage tree uses:
- root test budget 0.05;
- early branch P0=0.4, P1=0.7, interim value 0.08;
- late branch P0=0.6, P1=0.3, interim value 0.03;
- from late, good/bad have P0=0.5/0.5, P1=0.8/0.2, interim values 0.03/0.02.

P0 checks:
- root: 0.4×0.08 + 0.6×0.03 = **0.05**;
- late: 0.5×0.03 + 0.5×0.02 = **0.025 <= 0.03**.

At the early history, the conditional continuation root is 0.06. Its conditional hit path has P0=0.5, P1=0.8 and test value 0.1, with zero-valued miss path. Its alternative power is 0.8×0.1=**0.08**, while its selected-policy null size is 0.5×0.1=**0.05**, within the 0.06 conditional budget.

At late-good, continuation power is 0.8×0.1=0.08 and selected-policy null size is 0.1×0.1=0.01. At late-bad, the P1-optimal policy stops at the continuation root 0.01, so its selected-policy null size is 0.01. (The optimizer does not incorrectly substitute the terminal-leaf expectation when it selected the root stop action.)

The outer alternative-power candidates are:
- launch at root: **0.016**;
- continue at root, launch at early and late: 0.7×0.08 + 0.3×0.025 = **0.0635**;
- continue at root, launch at early and continue through late: 0.7×0.08 + 0.3×(0.8×0.08 + 0.2×0.01) = **0.0758**.

The DP selects the third policy. Re-evaluating that policy under P0:
- early contribution: 0.4×0.05 = 0.02;
- late contribution: 0.6×(0.5×0.01 + 0.5×0.01) = 0.006;
- total selected-policy null size: **0.026 <= 0.05**.

The prior first-stage-only policy has power 0.065 and null size 0.05. In this synthetic fixture, continuation raises power to 0.0758 while the separately evaluated null size falls to 0.026. This is a constructed example of the mechanism, not empirical evidence of general power gains.

An alternative-misspecification sensitivity test halves the early continuation hit probability under P1 (0.8 → 0.5, with the miss probability adjusted to preserve mass). The policy then stops at the early continuation root rather than buying the low-yield continuation observation. The overall power falls to **0.0618** and selected-policy P0 size remains **0.03**. This illustrates that P1 misspecification changes both selected policy and power while the P0 certificate remains separately checked.

## 4. Fail-closed controls

The implementation rejects:
- Null or alternative transition mass that does not normalize to one.
- P0 conditional expected child test value above the parent budget.
- Duplicate IDs, missing continuation processes, invalid roots, broken parent topology and non-finite/out-of-range probabilities.
- Continuation roots above their interim budgets.
- A selected P1-optimal policy whose exact P0 size exceeds the relevant node budget or alpha.
- Report tampering or canonical-order inconsistency.

The report makes both quantities visible: maximum alternative power and selected-policy null size. Neither should be presented alone.

## 5. Claim boundaries and adversarial caveats

- Type-I validity is conditional on the declared P0 model and supermartingale contract.
- Power optimality is conditional on the declared P1 model.
- P1 misspecification changes power and may change which policy is selected; it does not itself break the P0 proof if P0 and the test process remain correct.
- P0 misspecification or omitted null distributions can invalidate the type-I guarantee.
- A finite model family, finite tree and [0,1] terminal tests are assumed. No arbitrary continuous filtration, composite-null, real evaluator, or universal SOTA claim is made.
- The policy optimizes rejection power only. Production decision utility should include asymmetric error costs, evaluator cost, wall time, energy and intervention risk in a future constrained objective.

## 6. Industry/SOTA benchmark protocol

Use matched frozen DGPs and compare:
1. Power-optimal finite policy with exact P0 size audit.
2. Fixed-horizon Neyman–Pearson.
3. O'Brien–Fleming and Pocock / Lan–DeMets group-sequential designs.
4. Exact sequential E-variables / test martingales.
5. Conditional-error and inverse-normal/Fisher combination tests.
6. Stop-and-discard and fixed alpha splits.

Sources for comparator families:
- Turner, Ly & Grünwald, exact sequential E-variables for k-sample tests: https://doi.org/10.1016/j.jspi.2023.106116
- rpact documentation (group-sequential, adaptive design, combination and conditional-error methods): https://docs.rpact.org/reference/rpact.html
- gsDesign technical manual (alpha/beta spending and boundary designs): https://keaven.github.io/gsd-tech-manual/
- Spotify Engineering's sequential framework comparison (operating-regime trade-offs): https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions

Required metrics:
- Exact P0 type-I size and empirical type-I across many independent seeds.
- P1 power and expected decision loss.
- Expected number of observations/evaluator calls.
- Median and tail time-to-decision, memory and energy.
- P0 and P1 misspecification stress tests, including held-out alternatives and omitted nulls.
- Reproducibility, exact-head evidence receipts and negative-control detection.

Use pre-registered alternative and utility grids. Select policies on a training/calibration split, then report held-out performance. No SOTA claim is justified by a synthetic fixture.

## 7. Proposed acceptance gates (targets, not results)

- Both conditional kernels normalize exactly within tolerance at every internal node.
- P0 supermartingale contract is checked at every history.
- Every continuation root is within its P0 interim budget.
- P1 dynamic program matches exhaustive enumeration on tiny trees.
- Selected policy's P0 size independently reconstructs <= alpha.
- Negative controls for P0 overrun, P1 mass mismatch, missing processes, topology and report tampering fail closed.
- Reordering does not change the semantic report/digest.
- Exact-head cargo fmt/test/clippy pass before promotion.

## 8. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering power_optimal_continuation -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, test, clippy, CI, benchmark or SOTA pass is claimed.
