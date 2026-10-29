# VIS-DVPC-041 — Exact-path MOPE endpoint and coverage audit

**Status:** implementation under exact-head CI; no qualified run or coverage claim until the focused workflow passes.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8851

## Purpose

VIS-DVPC-040 audits the streaming coefficient identities and the inequality between the MOPE quadratic lower bound and actual log wealth. This increment tests the next mathematical layer: turning roots into running one-sided endpoints and combining two sides into an interval over repeated looks.

The implementation reuses the prior stream and optimizer modules. It does not reimplement bet optimization. It enumerates finite paths exactly, with no Monte Carlo, and checks direct wealth whenever a root is used to advance an endpoint.

## Registered DGPs

Both controls are single-context, two-action contextual-bandit models with target policy pi(A=1)=0.5, IID Bernoulli rewards, and true target value 0.5.

| Scenario | Logger P_h(A=1) | E[R|A=0] | E[R|A=1] | Maximum importance weight |
|---|---:|---:|---:|---:|
| healthy | 0.50 | 0.20 | 0.80 | 1 |
| weak overlap | 0.99 | 0.20 | 0.80 | 50 |

The importance ratio is w=pi(a)/h(a). Each observation has four possible action/reward outcomes, and every ordered path of length 8 is evaluated. There are 4^8 = 65,536 paths per scenario; their exact probabilities are products of the registered per-step outcome probabilities.

These are small finite controls, not public datasets and not an empirical industry comparison.

## Endpoint and alpha semantics

Use alpha_side=0.025 in each one-sided stream. Thus the quadratic lower-bound threshold is

ln(2 / alpha_side) = ln(80).

The factor 2 comes from the paper's half-wealth hedge. Allocating 0.025 to each one-sided confidence sequence gives a union-bound target of at most 0.05 simultaneous noncoverage, provided each side's martingale conditions and construction are correct. The code still measures joint exact-path coverage; it does not substitute the argument for evidence.

After each observation:
1. evaluate roots for the lower stream on (w,r) and the complement stream on (w,1-r);
2. consider the largest real root in [0,1];
3. before advancing an endpoint, independently recompute the polynomial and the direct log wealth for the recorded predictable bets;
4. require the root residual to equal ln(80), the direct log wealth to exceed the threshold, and each second-coordinate bet to be nonnegative;
5. update the lower endpoint monotonically, and update the upper endpoint as one minus the complement's lower endpoint;
6. choose next bets only after the current sample has been observed.

For fixed past bets with nonnegative second-coordinate values, every wealth factor is nonincreasing in candidate v. With positive factors this makes a certified crossing at root v reject every smaller candidate. The benchmark checks the direct product at every configured grid value below the root as a numerical control, while the positivity and bet-sign checks support the continuous monotonicity argument.

## Output and hard gates

The TSV contains per-look exact probability mass, pointwise coverage, weighted mean endpoints, reversed/empty-interval probability, endpoint monotonicity diagnostics, and root-certificate counts. Per scenario it reports sequence-wide coverage, false-exclusion probability, root-certificate failures, and mass-conservation error. The workflow also emits source and result SHA-256 receipts plus the exact commit and Rust compiler version.

Hard gates:
- each of the 16 scenario/look probability-mass rows sums to 1 within tolerance;
- sequence-wide coverage is at least 0.95 on both registered controls;
- no advancing endpoint root fails the direct polynomial/wealth certificate;
- both one-sided streams use predictable bets and endpoints never move outward;
- compiler/tests/results are tied to the exact source head.

A failure is a research result: retain it, diagnose it, and do not lower the coverage or root-certificate threshold to make the workflow green.

## Comparison protocol: research frontier, not a superiority claim

The closest conceptual baseline is the paper's original MOPE confidence-sequence construction. Industrial and academic OPE suites also compare Direct Method, trajectory/per-decision importance sampling, doubly robust and self-normalized estimators, confidence bounds, policy ranking, regret and type-I/type-II errors. A relevant newer comparator is Ryu et al. (COLT 2025), Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing, which studies variance-adaptive off-policy selection and freezing.

This PR does not implement those comparators. After this exact-path gate is qualified, the defensible next benchmark matrix is:

1. **Validity first:** exact path coverage and false exclusion at every look and over the whole sequence, with adaptive stopping and repeated peeking.
2. **Statistical efficiency:** interval width at fixed time and time-to-correct-decision at fixed thresholds; include weak overlap and rare target actions.
3. **Decision quality:** Type-I false promotion, Type-II miss rate, regret@1 / top-k regret, and rank correlation over many candidate policies.
4. **Shift and identification:** zero target support, near-zero overlap, misspecified or drifting logger propensity, and unmeasured support change. Return NOT_IDENTIFIED rather than fabricating a narrow bound where assumptions fail.
5. **Systems cost:** wall-clock time and allocation per observation, peak memory, stable throughput from 10^3 to 10^7 observations, and deterministic replay receipts.

Only compare width/power after all methods pass the same coverage contract. An asymptotic pointwise interval must not be reported as an anytime-valid baseline. A narrow interval with incorrect repeated-look coverage is a failed safety result, not an efficiency win.

## References

- Karampatziakis, Mineiro & Ramdas, Off-Policy Confidence Sequences, ICML 2021: https://proceedings.mlr.press/v139/karampatziakis21a.html
- Microsoft Research publication summary: https://www.microsoft.com/en-us/research/publication/off-policy-confidence-sequences/
- Ryu, Kwon, Koppe & Jun, Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing, COLT 2025: https://proceedings.mlr.press/v291/ryu25a.html
- SCOPE-RL documentation, which includes DM/TIS/PDIS/DR and HCOPE implementations and evaluates MSE, regret, rank correlation, type-I/type-II errors and safety violations: https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html

No SOTA, production-safety, or real-world effectiveness claim is made by this synthetic exact-path audit.
