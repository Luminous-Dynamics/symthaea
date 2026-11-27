# SDCB exact two-stage adaptive experiment-selection oracle v1

**Status:** exact finite categorical reference; eight Rust tests authored, not compiled or executed here.  
**Implementation:** \`scripts/research/sdcb_two_stage_adaptive_oracle.rs\`  
**Fixture:** \`SDCB_ADAPTIVE_ORACLE_V1.example.json\`  
**Expected result:** \`SDCB_ADAPTIVE_ORACLE_V1.expected.json\`

## 1. Motivation

A fixed subset planner chooses measurements before seeing any outcomes. A sequential scientific agent should be able to choose a second measurement based on the first result.

The Gaussian covariance oracle is not the right place to demonstrate this: under a fixed linear-Gaussian model with known covariance, posterior covariance is outcome-independent, so a variance-only objective does not benefit from outcome-dependent second actions. Adaptive value appears when the decision problem has categorical/mixture hypotheses, nonlinear likelihoods, state transitions, or outcome-dependent utility.

This reference extends paired-world reasoning from #13831 into an exact two-stage contingent policy.

## 2. Model

There are two competing worlds \(W_A,W_B\), a prior \(\pi=P(W_A)\), an abstention utility \(u_\bot\), a finite experiment set, and a hard budget.

Each experiment \(a\) has a cost \(c_a\) and outcome likelihoods:

\[
P_A(y\mid a),\quad P_B(y\mid a).
\]

The first experiment is selected before observing \(y_1\). For every possible \(y_1\), the policy either stops or selects a second experiment \(a_2(y_1)\). The second experiment's outcome is conditionally independent of the first given the paired world and selected experiment. A repeated experiment ID denotes a fresh independent replicate under the same declared likelihood.

At each terminal history, the agent chooses world A, world B, or abstains. Correct world choice has utility 1, incorrect choice 0, and abstention utility \(u_\bot\).

For unnormalized posterior masses \(m_A,m_B\), the optimal terminal contribution is:

\[
U(m_A,m_B)=\max\{m_A,m_B,u_\bot(m_A+m_B)\}.
\]

The policy's expected utility is the sum of these contributions over all categorical outcome paths.

The no-experiment baseline is:

\[
U_0=\max\{\pi,1-\pi,u_\bot\}.
\]

## 3. Hard-budget rule

The budget is pathwise, not merely expected:

\[
c_{a_1}+c_{a_2(y_1)}\le B
\]

for every possible first outcome where a second action is selected.

Expected cost is reported separately:

\[
E[C]=c_{a_1}+\sum_{y_1}P(y_1)c_{a_2(y_1)}.
\]

A policy with low expected cost but one over-budget branch is invalid. The reference therefore enforces both a hard maximum path cost and an expected-cost report.

## 4. Frozen adaptive fixture

The example has equal world priors, abstention utility 0.60, and budget two. The optimal policy is:

1. First, run \`triage-test\`.
2. On outcome 0, run \`confirm-b\`.
3. On outcome 1, take a fresh replicate of \`triage-test\`.

The exact expected utility is 0.864. The best policy constrained to use the same second experiment for both first-outcome branches has utility 0.860. Thus the contingent policy improves expected utility by 0.004 in this deliberately small fixture.

The selected policy's expected cost and maximum path cost are both two. Baseline abstention utility is 0.60; the policy's decision-value gain is 0.264.

This is an exact result for constructed likelihoods, not a physical intervention outcome or benchmark leaderboard score.

## 5. What this reference does and does not prove

It demonstrates:
- exact enumeration of first actions and every outcome-to-second-action map;
- second-stage action dependence on first-stage observations;
- explicit terminal abstention;
- deterministic tie-breaking;
- pathwise budget enforcement;
- a known fixture where adaptivity improves on the best fixed second action.

It does not yet include:
- more than two competing worlds;
- robust maximin across likelihood scenarios;
- Markov evolution of common-cause state between stages;
- action-dependent outcome alphabets;
- correlated observations conditional on world;
- continuous outcomes or posterior-mean utility;
- intervention feasibility, authorization, or physical execution;
- an arbitrary utility matrix or multi-step horizons.

The conditional-independence assumption is material. If both observations share a persistent or drifting nuisance state, that state must be represented explicitly in the world or transition model before this oracle's likelihood products are used.

## 6. Integration path

1. Use #13831 to identify competing worlds that remain plausible under current evidence.
2. Construct experiment likelihoods for each world, with common causes included in the state.
3. Use this exact two-stage planner as a small-instance evaluator.
4. Train an approximate HDC/CfC policy to reproduce or improve resource efficiency on held-out likelihood families.
5. Expand to multiple worlds and Markov common-cause transitions only after the exact two-stage reference passes its frozen controls.
6. Bind any physical action to authorized execution and independently verified receipts before #13805 or a future dynamic EGBT service changes belief.

## 7. Evidence status

Eight Rust tests are authored but not compiled or executed here. The expected fixture is independently enumerated using a separate finite calculation. No real-data benchmark or physical intervention is claimed.
