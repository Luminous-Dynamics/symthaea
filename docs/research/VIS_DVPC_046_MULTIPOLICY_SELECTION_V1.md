# VIS-DVPC-046 — Multiplicity-aware anytime-valid multi-policy selection

**Status:** deterministic Rust benchmark awaiting exact-head qualification. Simulation estimates are not a confidence-sequence proof.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8909

## Why this is a distinct frontier

A valid confidence sequence for one fixed target policy does not automatically justify selecting a winner from a portfolio. Reusing separate 95% intervals across dozens of policies and repeatedly inspecting them creates two distinct risks:

1. optional stopping over observation time;
2. multiplicity / winner's curse over the selected policy.

The benchmark separates these. It compares a reused pointwise Wald baseline, individually anytime-valid Hoeffding confidence sequences that each receive alpha=0.05, familywise-valid Hoeffding sequences that allocate alpha/M to each policy, and a **shared cell-mean Hoeffding sequence** that bounds the four primitive context/action reward means. All methods see the same logged actions and rewards. The shared method spends its error budget across four outcome cells rather than M policy estimates; if all four simultaneous mean bounds hold, every policy value formed as a nonnegative weighted sum is covered at once.

A policy is **certified best** only when its lower bound strictly exceeds every other policy's upper bound. An empty or reversed interval makes the portfolio ineligible for certification. This is a selection contract, not just an estimation metric.

## Frozen contextual bandit

The environment has two equally likely contexts and two actions:

| Context | E[R | A=0] | E[R | A=1] |
|---|---:|---:|
| 0 | 0.10 | 0.90 |
| 1 | 0.95 | 0.15 |

Each finite portfolio has M in {4, 8, 32} stochastic policies, parameterized by
`pi_j(A=1|x=0) = 0.1 + 0.8*j/(M-1)` and
`pi_j(A=1|x=1) = 0.1 + 0.8*((7j) mod M)/(M-1)`.

The permutation creates a nontrivial trade-off between contexts. Each portfolio has a unique optimal policy under the frozen reward means.

Two logging policies are evaluated:
- **healthy:** h(A=1|x)=(0.50,0.50);
- **weak support:** h(A=1|x)=(0.95,0.05).

The context distribution is the same for logging and target policies, so the per-observation importance ratio is pi(a|x)/h(a|x). For each candidate, the code computes its own maximum W_j across both contexts and actions. Every logged importance score is checked to lie in [0,W_j]. The exact true value of every policy is calculated analytically from the reward means.

## Confidence sequence construction

For policy j, the score X_{j,t} = w_j(X_t,A_t)R_t lies in [0,W_j]. At time t with empirical mean ̅X_{j,t}, the two-sided Hoeffding radius is

`r_{j,t} = W_j * sqrt(log(1/delta_{j,t})/(2t))`

where `delta_{j,t} = alpha_policy / [2 t(t+1)]`. Because ∑_{t>=1} 1/[t(t+1)] = 1, each tail spends alpha_policy/2 across all looks and the two tails spend alpha_policy total. Bounds are intersected over time so endpoints remain nested.

- **Independent CS:** alpha_policy=0.05 for every policy. This is time-valid for each fixed policy individually, but has no 0.05 guarantee for simultaneous coverage of the portfolio.
- **Familywise CS:** alpha_policy=0.05/M. A union bound across policies makes all M confidence sequences jointly valid at every time with total error budget 0.05, subject to the bounded-score and sampling assumptions.
- **Shared cell-mean CS:** builds four nested Hoeffding confidence sequences for E[R|context, action], with alpha_cell=0.05/4 per cell and per-tail local-count spending alpha_cell/[2 n(n+1)]. A policy interval is the target-context/action weighted sum of the cell lower and upper bounds. If the four cell sequences jointly cover their means at all times, every candidate policy (and any stochastic policy over the same finite support) is covered simultaneously; no extra M correction is needed. This holds for this tabular design with known context probabilities and does not automatically generalize to unknown context distributions or high-dimensional reward models.
- **Reused Wald:** pointwise 95% normal interval from sample variance, recomputed at each look. It is intentionally not anytime-valid; its job is to expose the benefit of a valid time-uniform comparison, not to compete on equal guarantees.

The portfolio certificate is conservative by design: even if one policy looks promising, the system abstains unless the entire interval family separates a unique best candidate.

## Paired experiment

Each logger/replicate has a fixed SplitMix64 seed. Candidate counts M=4,8,32 re-use the same logged context/action/reward prefix within a logger condition. At every look n in {256,1024,4096}, the benchmark records:

- familywise point coverage: every candidate's interval contains its true value at that look;
- familywise sequence coverage: every candidate's interval has contained its true value at every look so far;
- probability of any false best-policy certificate by that horizon;
- rate and conditional median/90th percentile time of a best-policy certificate;
- mean across-policy interval width and mean maximum interval width;
- IPS-argmax top-1 accuracy and simple regret.

This adds a policy-family uncertainty-sharing ablation: Bonferroni OPE pays a correction that grows with M, whereas the shared-cell method pays for the number of primitive outcome cells and maps those same intervals into any number of policies. It is a fair comparison for the registered tabular DGP because context probabilities and the outcome-cell definition are known.

A false certificate means some certified candidate is suboptimal relative to the known best policy. The false-certificate metric is cumulative over looks, so optional stopping remains visible.

## Hard gates

The exact-head workflow checks:
- expected output row counts, including 88 explicit policy definitions, and deterministic replay (wall-time rows excluded);
- every importance ratio stays below its computed policy-specific W_j;
- the familywise CS has a sequence-coverage Wilson lower bound above 0.90 for every portfolio;
- its false-certificate probability has a Wilson upper bound below 0.10;
- probabilities remain finite and in [0,1];
- source, output, summary and seed schedule receive SHA-256 receipts.

These are statistical regression controls for the synthetic DGPs, not proofs of the confidence-sequence theorem. The stated Hoeffding/union-bound construction provides the formal design; measured coverage is a sanity check.

## What counts as improvement

Do not rank methods only by how often they choose a policy. Report the trade-off between false-certification risk, familywise sequence coverage, regret and decision latency. If familywise validity makes the system defer on 32-policy weak-support portfolios, that is a meaningful limit of the current bounded nonparametric method—not permission to relax alpha.

The next research step is to add a modern variance-adaptive or betting-and-freezing comparator under the **same familywise contract**, then test support-misspecification, behavior-policy estimation error, policy-drift and context-distribution shift.

Useful baseline resource: SCOPE-RL's off-policy evaluation and high-confidence interval documentation, including Type-I/II error, regret and safety metrics:
https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html

Recent comparator for adaptive offline policy selection: Ryu, Kwon, Koppe & Jun, *Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing* (COLT 2025):
https://proceedings.mlr.press/v291/ryu25a.html

This benchmark does not claim state-of-the-art superiority. It adds a falsifiable, multiplicity-aware decision frontier to the portfolio selection problem.