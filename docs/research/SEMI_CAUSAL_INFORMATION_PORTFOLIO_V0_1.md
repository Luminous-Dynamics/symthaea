# SEMI-CAUSAL-PORTFOLIO-001 — budgeted causal information selection v0.1

**Issue:** #13674  
**Implementation:** `symthaea-semiconductor::benchmark::causal_information_portfolio`  
**Status:** exact small-portfolio optimizer and independent reference tests authored; exact-head validation pending.

## Research objective

Select the next physical metrology measurement for the information it provides about competing drift hypotheses and downstream decisions, rather than selecting only the wafer with the largest virtual-metrology prediction uncertainty.

This is complementary to:
- adaptive sampling and EVSI in #13328;
- two-stage screen/confirm planning in #13387;
- delayed evidence qualification in #13467/#13461;
- falsifiable causal hypotheses in #13673;
- delayed conformal risk work in #13479.

## Exact V1 model

The input declares:
- a prior over at least two competing hypotheses;
- optional action/loss table for downstream decision value;
- candidate physical measurements;
- outcome likelihoods P(outcome | hypothesis);
- cost, queue delay, capacity units and randomized-audit status;
- calibration/source-path/safety eligibility;
- predictive-risk reduction as a separate model-supplied term;
- dependence-group identity;
- objective weights and hard resource limits.

For each candidate outcome, likelihood probabilities are provided for every hypothesis and must normalize to one per hypothesis. Model, schema and source digests are required.

The exact optimizer enumerates candidate subsets (at most 10 candidates) and enumerates the joint outcome space (at most 4,096 combinations per subset). Larger portfolios are explicitly outside V1.

## Information gain

For prior hypothesis distribution p(H), measurement outcome Y, and posterior p(H|Y), the expected information gain is:

`IG = H(H) - sum_y P(y) H(H | y)`

For multiple selected measurements, the joint outcome likelihood is the product of each measurement's likelihood conditional on H **only when** a conditional-independence assumption is explicitly declared and sealed by a protocol digest.

Candidates sharing a dependency group (e.g. same sensor, reference standard or calibration epoch) cannot be selected together in V1. This prevents straightforward duplicate evidence from being counted twice. It does not prove that different groups are independent; the independence protocol is a declared modeling assumption, not a physical fact.

## Decision-loss reduction

When an action/loss matrix is supplied, the optimizer calculates Bayes decision risk before measurement and expected Bayes risk after the observed outcome. It reports the expected reduction separately from hypothesis entropy reduction.

Predictive-risk reduction is another separate input term. V1 sums the candidate-declared predictive-risk reductions for selected candidates; that term is a proxy and must not be interpreted as a jointly calibrated physical benefit without an independently validated interaction/dependence model.

The objective weights choose the default portfolio, while the Pareto frontier retains nondominated alternatives across:
- information gain;
- expected decision-loss reduction;
- predictive-risk reduction;
- physical measurement cost;
- expected result delay;
- capacity use.

The system does not hide these trade-offs behind the scalar objective score.

## Qualification and audit rules

A candidate receives no information credit unless:
- its measurement calibration is qualified;
- the declared source path is qualified;
- safety preconditions are satisfied;
- the likelihood model and outcome schema digests are valid.

This is a pre-measurement plan. A future physical outcome is **not** pre-qualified by the plan: the report always records that it must pass the downstream K1 numeric-label qualification before model or conformal updates.

A randomized-audit candidate must provide:
- a sampling-design digest;
- an inclusion probability in (0, 1].

If the requested randomized-audit floor cannot be met within cost/capacity constraints, the output is AUDIT_FLOOR_UNSATISFIED. Population estimation is only marked available when the selected portfolio meets a positive audit floor. Selected-only measurements without valid audit inclusion probabilities cannot support production prevalence claims.

## Deterministic exact search

The optimizer:
- rejects invalid priors, incomplete likelihood maps, non-normalized probability rows, malformed digests, duplicate IDs and invalid loss matrices;
- excludes unqualified candidates with explicit reason codes;
- enforces hard cost and capacity limits;
- prohibits repeated dependence groups;
- refuses multi-measurement combinations without a declared conditional-independence protocol;
- exhaustively enumerates feasible subsets and bounded joint outcomes;
- returns the maximum weighted objective and Pareto frontier;
- binds input, portfolio and report digests.

The integration reference tests independently enumerate a two-candidate joint-outcome table and compare the expected information gain to the production calculation.

## Authored adversarial coverage

The source currently includes 12 Rust unit tests plus 2 integration reference tests covering:
1. known binary-test information gain;
2. hard cost and capacity;
3. same dependency-group exclusion;
4. missing conditional independence declaration;
5. unqualified measurement receives no information credit;
6. mandatory randomized-audit floor;
7. randomized-audit inclusion probability carried in output;
8. invalid audit design/inclusion probability rejected;
9. invalid likelihood normalization rejected;
10. deterministic input-order behavior;
11. positive decision-loss reduction under an informative test;
12. Pareto frontier retaining information/cost trade-offs;
13. independent joint-outcome entropy oracle;
14. no multi-candidate assumption when conditional independence is undeclared.

Tests are authored; exact-head rustfmt, strict Clippy, semiconductor crate tests, independent reference controls and SEMI-BENCH-001 are pending.

## Claim ceiling

This is an exact optimizer over declared likelihood models. It does not establish that those likelihoods are physically correct, prove cross-sensor independence, authenticate source records, execute measurements or interventions, authorize equipment actions, or demonstrate manufacturing savings. A selected candidate is a measurement plan, not permission to operate equipment.
