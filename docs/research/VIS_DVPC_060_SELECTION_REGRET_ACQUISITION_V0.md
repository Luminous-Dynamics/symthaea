# VIS-DVPC-060 — Decision-regret-driven counterfactual annotation acquisition v0

**Status:** finite sample-average-approximation (SAA) oracle plus independent holdout evaluation. The SAA optimizer is exact for its frozen simulation bank, not for population expected regret.

## Motivation

VIS-DVPC-057/058 optimize prior-weighted or worst-policy value MSE and subsequently report best-policy selection accuracy and simple regret. Minimizing marginal MSE does not necessarily minimize the loss from selecting the wrong policy. The annotation budget should be aligned with the decision that follows OPE.

This direction complements multi-policy high-confidence evaluation (Dann et al., AISTATS 2023) and shared behavior-policy design for multiple target policies (Liu et al., AAAI 2025). The new objective is direct acquisition for portfolio selection regret, rather than marginal value precision alone.

## Objective

For target-policy portfolio Π, allocation n, profile s, and common-sample OPE estimate vector, let `j_hat(n)` be the policy with the highest estimated value and `j*` the policy with highest true value. Simple regret is:

\[
R(n,s)=\\mathbb E_s\\left[V_s(\\pi_{j^*})-V_s(\\pi_{\\widehat j(n)})\\right].
\]

V0 minimizes the worst expected simple regret across the four coherent profiles inherited from VIS-DVPC-058:

\[
J_{regret}(n)=\\max_{s\\in\\mathcal S}R(n,s).
\]

## Common-random-number sample-average approximation

The population expected regret is non-smooth and expensive to optimize directly. V0 freezes a finite simulation bank:

- annotation-model fit seeds are deterministic and derived independently from final holdout seeds;
- each profile uses a registered seed stream;
- every feasible allocation is evaluated with the same SAA seed bank (common random numbers), so pairwise allocation comparisons have reduced Monte Carlo noise;
- for each replicate, one q-hat and one factual logged evaluation sample are shared across all target policies;
- the optimizer exhaustively enumerates all feasible allocation combinations under the finite context/source budget and selects the minimum worst-profile sample-average regret.

This is an exact global minimizer of the frozen SAA objective only. It may overfit its finite random bank. The selected plan is therefore evaluated on a separate holdout seed bank with a larger replicate budget. The SAA result and the independent holdout result are reported separately.

## Baselines

- balanced single-policy exact MSE allocation;
- nominal prior-weighted portfolio exact MSE allocation;
- nominal worst-policy portfolio exact MSE allocation;
- coherent-profile robust worst-policy MSE exact allocation;
- greedy SAA simple-regret allocation;
- exhaustive SAA simple-regret oracle.

## Output and metrics

The executable emits one `REGRET_CONFIG` row and 24 `REGRET_RESULT` rows (six plans × four coherent profiles). Each result records:

- worst-profile SAA regret and its gap to the exact SAA oracle;
- profile-specific independent holdout simple regret;
- worst-profile holdout simple regret and selection accuracy;
- prior-weighted and worst-policy MSE, plus each policy's MSE;
- annotation cost/count, source allocation, and holdout regret gap to the SAA-oracle plan's holdout performance.

An SAA plan with excellent training-bank regret but poor independent holdout regret is evidence of optimizer overfitting, not a success.

## Acceptance gates

1. The exact SAA allocation matches independent exhaustive enumeration on a reduced finite instance.
2. Fixed allocation and fixed SAA seed bank produce deterministic regret values.
3. The SAA and holdout seed streams are independent and are explicitly recorded in configuration output and SHA-256 receipts.
4. Every target policy in one replicate shares the same fitted q-hat and logged evaluation sample.
5. Six baseline/optimizer plans are evaluated across all four coherent profiles.
6. SAA objective gaps are nonnegative relative to the exact finite-bank oracle; holdout regret gaps may be positive or negative and are reported without truncation.
7. All policy-specific MSEs, selection accuracy and simple regret are separately reported; no population optimality is inferred from SAA exactness.
8. Annotations never replace factual residual rewards/propensities or repair zero support.

## Research frontier

After this finite SAA baseline, compare robust simple-regret acquisition against Bayesian best-policy identification and anytime-valid top-policy confidence sequences. Replace the finite SAA bank with confidence bounds on expected regret, and optimize annotation actions that maximally separate the two or three policies most likely to occupy the top ranks. This is the decision-centric shift: collect evidence to avoid the wrong policy choice, not merely to reduce average prediction error.