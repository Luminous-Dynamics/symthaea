# AR-5B18: Exact Two-Policy Randomization and Robust-Utility Convexification v0.1

**Status:** finite reference draft; six tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → #9854 → #9856 → #9860 → #9895 → this branch.  
**Tracking:** #9896.

## 1. Why deterministic maximin can leave utility unused

AR-5B17 enumerates deterministic policies and selects the one with the highest worst-case utility across a finite alternative family. That is a defensible deployable baseline, but it does not convexify the achievable utility set.

Suppose policy A is better under one alternative and policy B under another. A single random draw made **before observing any data** can select A with probability q and B with probability 1-q. For any fixed probability model, expected outcomes are linear in q:

```
size(q) = q * size(A) + (1-q) * size(B)
power_k(q) = q * power_k(A) + (1-q) * power_k(B)
cost_k(q) = q * cost_k(A) + (1-q) * cost_k(B)
utility_k(q) = q * utility_k(A) + (1-q) * utility_k(B)
```

Therefore, a mixture preserves each linear null-size limit by convexity. With a finite composite null family, it must satisfy the size limit separately for every declared null model; the feasible set is the intersection of all those linear constraints. Expected-cost constraints are preserved by the same argument.

This is a randomized policy over two **complete deterministic strategies**. The random choice is made before the experiment; it is not a post-outcome selection and must not depend on which alternative model is thought to be true.

## 2. Exact one-dimensional robust optimization

For a finite alternative family, each model's mixture utility is affine in q. Worst-case utility is the lower envelope of those lines, a piecewise-linear concave function.

The optimum on the feasible interval must occur at one of:
- q=0 or q=1;
- an intersection of two alternative-model utility lines;
- a size boundary for any declared null model;
- a per-model expected-cost feasibility boundary.

The implementation enumerates all these breakpoints, evaluates each feasible point and selects the highest worst-case utility. Ties are broken by lower maximum expected cost, lower null size, then lower q. This is exact for two-policy mixtures and a finite supplied model family.

The receipts must share the same alternative-model IDs and a SHA-256 utility-contract identifier. This binds the utility definition, but does not authenticate the truth of externally supplied policy metrics.

## 3. Worked crossing-utility fixture

Policy A is the middle-depth deterministic policy from AR-5B17; policy B is the deeper policy. Use two alternatives and the same utility contract.

| Policy | Alternative | Power | Expected cost | Utility |
|---|---|---:|---:|---:|
| A (middle) | alt-a | 0.660 | 3.0 | 5.620 |
| A (middle) | alt-b | 0.540 | 3.0 | 4.180 |
| B (deep) | alt-a | 0.576 | 3.7 | 4.542 |
| B (deep) | alt-b | 0.615 | 3.3 | 5.050 |

Composite-null size receipts:
- Policy A: null-a = 0.035; null-b = 0.020.
- Policy B: null-a = 0.0225; null-b = 0.030.

The optimizer enforces alpha separately under both null models.

With alpha=0.05 and a maximum expected cost of 3.6, the utility lines intersect at:

```
q = P(select A) = (5.05 - 4.542) / ((5.620 - 4.542) - (4.180 - 5.050))
  = 0.508 / 1.948
  = 0.2607802875...
```

The selected randomized strategy therefore chooses:
- A with probability **0.26078**;
- B with probability **0.73922**.

The independently calculated mixed metrics are:
- alt-a power: **0.59791**;
- alt-b power: **0.59544**;
- worst-case power: **0.59544**;
- utility under both alternatives: **4.82312**;
- null-a size: **0.02576**;
- null-b size: **0.02739**;
- worst-case null size across the family: **0.02739**;
- maximum expected cost: **3.51745**;
- binary mixture entropy: **0.57387 nats**.

The best deterministic policy has worst-case utility 4.542. The randomized mixture raises it to 4.82312 while respecting alpha and the 3.6 cost cap.

When the cost cap is tightened to 3.2, the utility intersection is infeasible. The cost boundary forces q >= 0.7142857; the exact optimizer selects that boundary, yielding worst-case utility 4.42857 and maximum expected cost 3.2. This is a useful negative case: randomization cannot preserve the unconstrained optimum after the resource set contracts.

The null-family intersection also matters: at alpha = 0.026, null-a requires q <= 0.28 while null-b requires q >= 0.40. No randomized mixture can meet both constraints, so the analyzer correctly reports infeasibility rather than averaging the two null sizes into a misleading single guarantee.

These values are exact constructed arithmetic, not compiled test output or empirical results.

## 4. Failure modes and claim boundary

The implementation rejects:
- different alternative families between policy receipts;
- duplicate/missing model IDs;
- incompatible utility-contract hashes;
- non-finite/out-of-range power, size, cost or utility;
- empty or identical policy IDs;
- infeasible alpha/cost constraints;
- report tampering or non-canonical receipt ordering.

The reference only mixes **two supplied policies**. It checks null size under each model in the declared finite null family and robust utility under each supplied alternative. It does not solve the full LP over every deterministic policy, nor randomized policies over a continuous action space. It does not protect against omitted null or alternative models. It assumes that each policy receipt accurately represents the complete policy's operating characteristics and that randomization is independent of experiment outcomes.

A future full policy-frontier LP could optimize a distribution over many policies. The current two-policy routine is a small exact convexification oracle and benchmark point, not a general robust optimizer.

## 5. Benchmark protocol

For every frozen AR-5B17 scenario:
1. Compare the best deterministic policy to the best two-policy randomized mixture.
2. Hold alpha, alternative family, utility weights and cost cap fixed.
3. Report worst-case utility, worst-case power, per-model power/cost, null size, mixture entropy and feasibility.
4. Add held-out alternative models and omitted-model negative controls.
5. Repeat the operational randomization with recorded, reproducible random seeds; the mathematical report describes the *ex ante* mixture, while each actual run should retain its realized policy ID.
6. Later compare with a full LP over the deterministic policy frontier to measure the gap left by two-policy mixing.

Proposed target: at equal null-size and cost limits, demonstrate at least a 5% improvement in worst-case utility in registered crossing-utility scenarios, with no type-I budget violation. This is a target, not a result beyond the synthetic fixture.

## 6. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering two_policy_convexification -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

Seven tests are authored. No compile, formatting, clippy, unit-test, CI, benchmark or SOTA pass is claimed.
