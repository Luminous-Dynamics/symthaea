# AR-5B15: Finite-Filtration Optional-Stopping Continuation Oracle v0.1

**Status:** finite reference draft; six tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → #9854 → this branch.  
**Tracking:** #9855.  
**Research anchor:** Koning & van Meer, “Anytime validity is free: inducing sequential tests,” JRSSB 88(4), 1366–1384, published 21 February 2026. https://doi.org/10.1093/jrsssb/qkag050  
**Claim ceiling:** one explicitly supplied simple-null finite probability filtration and finite conditional continuation trees.

## 1. Research shift: certify the policy class, not one stopping schedule

A fixed two-stage tree only verifies the stopping rule already written into the input. Theorem 5 of Koning & van Meer permits an anytime-valid process to stop at an arbitrary stopping time and continue with a conditionally anytime-valid process whose conditional significance level is the stopped test value.

This reference makes the stopping decision itself part of the finite model:
- Every node is an observable first-stage history.
- A [0,1] test value is attached to every history.
- A history's children have a normalized conditional null distribution.
- The conditional expected child test value may not exceed the parent value.
- Every history has its own conditional continuation process.
- The continuation root value may not exceed the test value at that history.
- Every continuation process is itself a [0,1] nonnegative supermartingale.

The supermartingale conditions supply the finite conditional anytime-valid contract. The probability distributions and test values are explicit inputs; the code does not infer their validity from observed samples.

## 2. Exact dynamic program over all stop/continue policies

At a first-stage history h, a policy may:
1. Stop observing and launch that history's conditional continuation process; or
2. Continue observing, then make the same choice at the next history.

For a process node h with test value phi(h), the best optional-stopping value is computed bottom-up:

```
V(h) = max(
    phi(h),
    sum_{child c} P(c | h) * V(c)
)
```

The conditional continuation tree at h is evaluated by the same recurrence. Its root value is constrained by phi(h).

A second dynamic program combines both choices:

```
W(h) = max(
    max_optional_stopping_value(continuation_at_h),
    sum_{child c} P(c | h) * W(c)
)
```

The root W is the largest null rejection expectation obtainable by any adaptive policy that decides at each observed history whether to continue or launch the conditional continuation. The report reconstructs the maximizing action at every node, as well as the maximizing optional-stopping policy within each continuation tree.

The recurrence is an exact finite-tree oracle, not a Monte Carlo estimate or a heuristic policy search.

## 3. Worked fixture (synthetic, exact arithmetic)

First-stage tree:
- Root: test value 0.05.
- Early history: probability 0.4, test value 0.08; it is a leaf.
- Late history: probability 0.6, test value 0.03.
- From late: good with probability 0.5 and test value 0.03; bad with probability 0.5 and test value 0.02.

The supermartingale checks are:
- Root: 0.4 × 0.08 + 0.6 × 0.03 = **0.05**.
- Late: 0.5 × 0.03 + 0.5 × 0.02 = **0.025 <= 0.03**.

Conditional continuation root budgets:
- Root: 0.01.
- Early: 0.06 (terminal continuation size 0.05).
- Late: 0.025 (terminal continuation size 0.015).
- Late-good: 0.02 (terminal continuation size 0.01).
- Late-bad: 0.01 (terminal continuation size 0.005).

The outer policy choices reduce to:
- Stop at root: 0.01.
- Continue at root, then launch continuation at late: 0.4 × 0.06 + 0.6 × 0.025 = **0.039**.
- Continue through late and use the two child continuations: 0.4 × 0.06 + 0.6 × (0.5 × 0.02 + 0.5 × 0.01) = **0.033**.

The exact dynamic program and exhaustive enumeration both return **0.039**, below alpha = 0.05. The maximum optional-stopping expectation of the first-stage test process is 0.05. These are constructed finite arithmetic values, not empirical system results.

## 4. Fail-closed validation

The implementation rejects:
- Empty/oversized trees, duplicate IDs, missing parents, multiple roots, detached cycles and unreachable nodes.
- Root probability other than one.
- Child conditional probability mass that does not sum to one.
- A conditional expected child test value above its parent budget.
- Missing or duplicate continuation process for any history.
- A continuation root above its stopped interim budget.
- A continuation process that violates its own conditional supermartingale bound.
- Non-finite values and report reconstruction/tampering.

Canonical sorting makes plan and report hashes independent of input vector order.

## 5. What this does not establish

This is a finite discrete filtration under one supplied null probability model. It does not establish:
- Correctness or coverage of the supplied null transition probabilities.
- Robustness to dependence misspecification, distribution shift, omitted nulls or unobserved confounding.
- Continuous-time or unbounded stopping-time validity.
- A continuous null-family essential infimum or composite confidence sequence.
- Power or decision utility under alternatives.
- Authenticity of evaluator receipts or execution of an actual measurement.

AR-5B14 covers a finite family of pointwise null tests. A future synthesis would need to combine the per-null dynamic programs and take the pointwise minimum over a declared null family without accidentally averaging away conditional validity.

## 6. Benchmark programme against mature methods

Use the exact tree oracle as a ground truth for small state spaces, then compare on matched larger scenarios:
- Exhaustive enumeration versus dynamic programming (exactness and computational scaling).
- Group-sequential alpha-spending designs (O'Brien–Fleming, Pocock, Hwang–Shih–DeCani).
- E-process / e-variable test martingales with optional stopping.
- Conditional-error and combination-test adaptation (inverse-normal / Fisher).
- Stop-and-discard and predeclared alpha splits.

Report maximum type-I error over all represented stopping policies, conditional-budget overruns, power under alternatives, expected sample count, evaluator calls, wall time, and decision loss. Stratify by whether the sample horizon is known, whether historical variance reduction is available, and whether the evaluator can be changed after interim analysis. Do not claim one universal winner.

The rpact and gsDesign implementations are practical comparators for confirmatory group-sequential/adaptive designs; Turner, Ly & Grünwald's exact sequential E-variables provide an anytime-valid k-sample comparator; Spotify's engineering analysis discusses the operational trade-off between AVI and group-sequential power:
- https://docs.rpact.org/reference/rpact.html
- https://keaven.github.io/gsd-tech-manual/
- https://doi.org/10.1016/j.jspi.2023.106116
- https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions

## 7. Proposed acceptance gates (targets, not results)

- First-stage and every continuation tree pass the local probability and supermartingale contracts.
- The maximum first-stage optional-stopping value is no greater than alpha.
- Every continuation process is valid at its stopped interim budget.
- The combined adaptive stop/continue maximum is no greater than alpha.
- DP output equals exhaustive policy enumeration over all policies on tiny fixtures.
- Negative controls for transition overrun, conditional budget overrun, missing process, cycles and malformed mass fail closed.
- Reordered inputs reconstruct byte-identical semantic reports and report digests.
- Exact-head Rust formatting, test and clippy commands complete before qualification.

## 8. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering finite_stopping_continuation -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, test, clippy, CI, benchmark or SOTA pass is claimed.
