# VIS-DVPC-049 — pCRP-star lower confidence bound for OPE selection

**Status:** standalone numerical comparator draft. Exact-head tests and the focused workflow must pass before treating outputs as Rust-qualified. This is not the complete COLT 2025 experimental reproduction and makes no SOTA or production claim.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8958

## Why this tranche

Ryu, Kwon, Koppe and Jun, *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing*, COLT 2025, Section 3 develops a variance-adaptive lower confidence bound for nonnegative random variables using universal portfolios. Their off-policy selection method applies that bound separately to each candidate policy's importance-weighted reward and selects the largest lower confidence bound.

Reference: https://proceedings.mlr.press/v291/ryu25a.html  
Full text: https://arxiv.org/abs/2502.10826

The implemented primitive is a conservative fixed-bet-grid approximation to the penalized best constant-rebalanced-portfolio lower confidence bound (pCRP-star-LCB), paper equations (3.3)–(3.5). The selection rule is PUB-style pessimistic selection over a finite policy set, with this pCRP-star variant supplying each policy's LCB; it is not the exact continuous-bet version. For nonnegative scores Y_1,...,Y_n, candidate mean nu>0, and a constant bet b in [0,1]:

log W_CRP(b; nu) = sum_t log(1 - b + b Y_t / nu)

log W_pCRP-star(nu) = sup_b log W_CRP(b; nu) - 0.5 log(pi(n+1))

The lower bound is the smallest nu whose penalized wealth is at most 1/delta. This implementation approximates the bet supremum by a fixed 129-point grid and numerically inverts that monotone grid-witness condition. Since the grid maximum never exceeds the continuous supremum, the resulting lower bound is conservative but can be looser than the exact pCRP-star bound. The confidence-bound argument relies on the paper's nonnegative-process assumptions; in the frozen synthetic evaluation, logger, policies and reward mechanism are stationary and actions/rewards are IID.

## Numerical-safety design

A confidence bound should not become optimistic because numerical optimization changes the direction of the guarantee. This tranche uses a fixed grid of 129 feasible bets on [0,1]. For each candidate mean, the maximum wealth over this fixed grid is no greater than the continuous supremum. Because each fixed-bet wealth is nonincreasing in the candidate mean, the maximum over the fixed grid is also nonincreasing; this monotonicity matters for the root inversion. Underestimating the continuous supremum can only move the resulting lower confidence bound down. A small candidate-independent multiplicative shrink is applied to the fixed-grid log-wealth witness. Because the shrink factor depends only on sample count, it preserves monotonicity in candidate mean, unlike a candidate-dependent additive pad. This is a numerical engineering precaution, not a machine-checked interval-arithmetic proof. The grid is a conservative approximation, not the paper's exact continuous-bet supremum.

The code includes endpoint cases, zero rewards, non-finite / negative inputs, probability validation, importance-weight support checks, the paper-specific reward-complement upper endpoint, and a distinct NOT_IDENTIFIED status for structural zero support. A winner certificate is suppressed if the candidate's own interval is reversed. It never creates a numeric estimate or bound for unsupported target-policy mass.

## Frozen comparison

- Three candidate policies with P(A=1) in {0.10, 0.50, 0.90}.
- Clear-margin controls: E[R|A=0]=0.25 and E[R|A=1]=0.75, giving exact values 0.30, 0.50 and 0.70.
- Small-margin control: E[R|A=0]=0.49 and E[R|A=1]=0.51, giving exact values 0.492, 0.500 and 0.508. This distinguishes weak decision margin from weak support.
- Policy-tie control: both action reward means are 0.50, so all three target policies have exact value 0.50. Top-1 accuracy treats all true-value maximizers as correct; a valid unique-winner certificate should not occur when policies are tied.
- Healthy fixed logger P(A=1)=0.50; weak-overlap fixed logger P(A=1)=0.99.
- Structural zero-support logger P(A=1)=1.00; target policies put positive mass on action zero, so all value, interval and paired-metric outputs must be suppressed.
- The same logged stream and seed schedule are shared by every candidate and both methods within each scenario.
- Horizons 64, 256 and 512; 1,024 deterministic paired replicates per identified scenario. This remains synthetic and cannot establish external validity.

Methods:
1. Fixed-grid pCRP-star-LCB witness with `delta_tail = 0.05 / (2M)` for each policy and each tail. The UCB follows Appendix B.6 of the paper: build the nonnegative complement score `w(a)(1-r)` with expectation `1 - V(pi)` and use `U(pi) = 1 - LCB(w(1-r))`. This is the pCRP-star variant of PUB-style pessimistic selection, with a conservative fixed-bet-grid approximation that may widen the bound.
2. A bounded-reward Hoeffding interval with the same two-sided familywise allocation and a known importance-score range (fixed-horizon comparator, not advertised as anytime-valid).

Metrics include familywise two-sided interval coverage, top-1 policy selection accuracy, winner-certificate rate, unconditional and conditional false-certificate probability, reversed-interval candidate rate, mean interval width, plausible-winner set size, simple regret, selected lower bound, lower-bound gap to the selected policy's known truth, Wilson 95% intervals for empirical rates, and paired pCRP-star-minus-Hoeffding differences with standard errors for accuracy, regret, certification and width. Exact score-frequency compression makes the pCRP bet-grid evaluation much cheaper on this finite-support DGP without changing the sample stream or objective. Timing rows are excluded from deterministic replay comparisons.

## Acceptance and limits

- Analytic constant-stream root and zero-stream tests pass.
- The fixed-grid wealth witness is checked against a dense grid of log wealth in a deterministic fixture, and its monotonicity in candidate mean is tested.
- The known expectation identity for importance-weighted reward matches the analytical policy value.
- The reward-complement identity `E[w(1-r)] = 1 - V(pi)` is checked for the registered logger and candidate policies.
- Nonnegative finite score validation and structural-zero-support fail-closed behavior pass.
- Repeated smoke runs produce identical non-timing output; exact source/output digests are published.
- The full workflow must compile and test the exact PR head. Queue status, skipped general benchmark jobs, or a passing syntax check alone do not qualify the Rust result.

This implements a conservative fixed-grid pCRP-star variant of PUB-style finite-policy selection. It is not the exact continuous-supremum pCRP-star bound, not the full Cover universal-portfolio integral, and not the separate freezing-based off-policy learning experiment. The fixed-grid approximation is intentionally conservative and may be looser than the paper's method. Do not claim superiority until a faithful comparison on matched workloads and an external confirmation dataset have been completed.

## Reference

Ryu, J. J., Kwon, J., Koppe, B., and Jun, K.-S. (2025). *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing.* Proceedings of Machine Learning Research 291, 5015–5053. https://proceedings.mlr.press/v291/ryu25a.html
