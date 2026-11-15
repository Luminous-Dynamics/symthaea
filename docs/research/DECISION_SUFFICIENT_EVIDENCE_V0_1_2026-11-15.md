# EGDDM D3 — decision-sufficient evidence acquisition v0.1

**Status:** Rust prototype, schema, synthetic fixture and hand-derived expected values authored; no compilation, test execution, calibration study or benchmark run is claimed.

## 1. Research objective

EGDDM should not maximize the amount of information it retrieves. It should acquire the lowest-cost observation that is expected to change a consequential decision enough to justify the acquisition.

D3 treats evidence acquisition as a decision problem. It separates:
- uncertainty about the state of the system;
- the actions available to the decision-maker;
- the loss associated with each action under each possible state;
- observations that could discriminate among states;
- the cost of acquiring each observation; and
- hard authority, scope, isolation, prerequisite and resource constraints.

The intended output is an acquisition recommendation, not authorization to execute the underlying engineering action.

## 2. Mathematical model

Let the mutually exclusive hypotheses be \(H_1,\ldots,H_n\), with prior probabilities \(p_i=P(H_i)\). Let candidate engineering actions be \(a\in\mathcal A\), and let \(L(a,H_i)\) be the decision loss.

The current Bayes risk is

\[
R_0=\min_{a\in\mathcal A}\sum_i p_i L(a,H_i).
\]

For an evidence-acquisition option \(e\), let its possible observations be \(o\in\mathcal O_e\), with likelihoods \(P(o\mid H_i,e)\). For each outcome,

\[
P(o\mid e)=\sum_i p_iP(o\mid H_i,e).
\]

The posterior is

\[
P(H_i\mid o,e)=
\frac{p_iP(o\mid H_i,e)}{P(o\mid e)}
\]

when the denominator is positive.

The expected post-observation Bayes risk is

\[
R_e=\sum_{o\in\mathcal O_e}P(o\mid e)
\min_{a\in\mathcal A}
\sum_iP(H_i\mid o,e)L(a,H_i).
\]

The expected value of sample information is

\[
EVSI(e)=R_0-R_e.
\]

The net value used to choose among feasible observations is

\[
NetEVSI(e)=EVSI(e)-C_e,
\]

where \(C_e\) is a caller-declared acquisition cost expressed in the same utility units as action loss. This conversion must be explicit. The implementation separately preserves money, latency, reviewer minutes and compute units and applies independent hard budgets to each.

The planner selects an observation only when its net EVSI is positive and all hard gates pass. Otherwise it reports no positive-net-value observation. It does not take the engineering action itself.

## 3. Hard gates are not scores

Each observation must explicitly satisfy:
- authority;
- exact source scope;
- execution-isolation requirements; and
- prerequisites.

It must also remain within every configured resource budget.

A candidate with high theoretical EVSI but insufficient authority is infeasible. Its expected information value cannot compensate for an authorization failure. This avoids a common failure of weighted-score decision systems in which safety constraints can be traded away by unrelated benefits.

The current booleans are supplied assertions. Production EGDDM must replace them with independently verifiable evidence references, signed policy decisions, exact revision/world pins and isolation receipts.

## 4. Prototype API and CLI

The standalone crate is tools/decision-sufficient-evidence.

Public Rust API:
- plan_evidence(&DecisionProblem)
- plan_evidence_json(input)

From the crate directory:

    cargo test
    cargo run -- ../../docs/development-map/fixtures/decision-sufficient-evidence-v0.1.json

The input schema is docs/research/decision-sufficient-evidence-v0.1.schema.json.

## 5. Hand-derived example

A contract has two possible states:

| Hypothesis | Prior |
|---|---:|
| Compatible | 0.5 |
| Breaking | 0.5 |

Available actions:

| Action | Compatible loss | Breaking loss | Expected loss |
|---|---:|---:|---:|
| Ship | 0 | 100 | 50 |
| Hold | 5 | 5 | 5 |

The best current action is Hold, with Bayes risk 5.

A perfectly discriminating pinned contract test has two outcomes:
- Pass has likelihoods [1, 0].
- Fail has likelihoods [0, 1].

After Pass, Ship has loss 0. After Fail, Hold has loss 5. Both outcomes have probability 0.5, so expected post-observation loss is 2.5. Gross EVSI is 2.5. With a declared acquisition utility cost of 1, net EVSI is 1.5.

An uninformative search has likelihoods [0.5, 0.5] for both outcomes under both hypotheses. Its posterior remains unchanged, gross EVSI is 0, and a positive acquisition cost makes net EVSI negative.

The hand-derived output is stored in docs/development-map/fixtures/decision-sufficient-evidence-v0.1.expected.json. It is not an observed execution result.

## 6. D3 benchmark design

The benchmark should contain independently adjudicated decision problems with:
- hidden system-state hypotheses;
- action-loss matrices established before the system sees the case;
- multiple feasible and infeasible evidence options;
- known outcome likelihoods for synthetic cases and measured likelihoods for real cases;
- hard authority, scope, isolation, prerequisite and budget controls;
- explicit cost units; and
- a trusted independent evaluator.

Baselines:
1. retrieve the largest amount of context within a fixed token/line budget;
2. retrieve the top-ranked evidence by relevance;
3. choose the cheapest observation;
4. choose the observation with maximum expected entropy reduction;
5. choose by EVSI but ignore hard gates (a deliberately unsafe ablation);
6. full D3 with hard constraints and separate resource budgets;
7. expert-human acquisition choices.

Metrics:
- EVSI ranking regret against exhaustive enumeration on synthetic problems;
- net-utility regret against the best feasible option;
- acquisition cost and each resource dimension;
- posterior calibration (Brier score and log loss on held-out outcomes);
- hard-gate false-acceptance rate;
- infeasible-option rejection recall;
- fraction of cases correctly reporting no positive-net-value observation;
- decision loss after the acquired observation;
- human review and acquisition latency.

Prospective qualification targets:
- exact agreement with exhaustive enumeration on 100% of frozen synthetic cases;
- zero selection of any option that fails a hard gate;
- zero resource-budget violations;
- ≥95% correct acquisition/no-acquisition decisions on independently adjudicated pilot cases;
- calibration better than a prior-only baseline on held-out outcomes;
- lower net-utility regret than cheapest-evidence and relevance-only baselines.

These are research goals, not achieved results. Real likelihoods and costs are often uncertain. D3 should eventually use likelihood intervals and robust EVSI bounds; it should defer rather than manufacture precision when plausible distributions reverse the acquisition ranking.

## 7. Next research increments

1. Probability calibration and uncertainty intervals for priors and observation likelihoods.
2. Robust EVSI under partially identified likelihoods and distribution shift.
3. Sequential acquisition: update beliefs after each observation, then stop when the marginal expected value is non-positive.
4. Correlated evidence: prevent double-counting two observations derived from the same upstream source.
5. Value of delay: account for decision deadlines and changing system state.
6. Multi-objective Pareto planning that exposes money, latency, human attention, compute and operational risk rather than collapsing them.
7. Independently verified gate inputs and evidence custody.

The scientific objective is not to make EGDDM appear decisive. It is to make the system acquire only evidence that can improve a decision, preserve hard boundaries, and recognize when no available observation is worth its cost.
