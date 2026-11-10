# Correlated Sequential Battery Diagnostic Planner v0.2

**Status:** Rust-first finite-horizon planner with explicit independent or joint test-likelihood semantics. Synthetic known-truth fixtures only; not a physical test-accuracy model, service authorizer or BMS.
**Owner:** [EV-SEQUENTIAL-DIAGNOSTIC-002 #11512](https://github.com/Luminous-Dynamics/symthaea/issues/11512).
**Base:** [Sequential diagnostic planner v0.1 PR #11325](https://github.com/Luminous-Dynamics/symthaea/pull/11325).

## Why v0.2 exists

v0.1 assumes outcomes are conditionally independent given the modeled battery hypothesis. Repeated measurements from shared sensors, calibration, excitation, environment or feature pipelines violate that assumption. Multiplying marginal likelihoods after every observation can count the same evidence more than once.

## Likelihood modes

- `conditional_independent`: explicit v0.1 semantics.
- `joint_assignment`: a finite joint distribution over a complete outcome assignment for every declared test, conditional on each hypothesis.

For a joint assignment distribution q(o_1,...,o_m | H), after history h the next test outcome likelihood is:

P(O_t=o | H,h) = Σ q(a|H) over assignments a compatible with h and O_t=o, divided by Σ q(a|H) over assignments compatible with h.

The planner updates the posterior using these history-conditioned likelihoods. A zero-probability history under a hypothesis contributes zero probability to all child outcomes; an outcome with zero marginal probability across the current posterior is omitted. No malformed joint table is silently downgraded to an independence assumption.

The joint table is capped at 4096 rows. Each row identifies one hypothesis and a complete test-outcome assignment. Per-hypothesis row probabilities must sum to one. The joint table's per-test marginals must match the reviewable marginal likelihood table within 1e-9. Duplicate, partial, unknown, over-cap or non-normalized assignments are rejected.

## Known-truth anti-double-counting experiment

Two tests, A and B, share the same underlying signal. Under H0, both are negative with probability 0.8 and both positive with probability 0.2. Under H1, both are negative with probability 0.2 and both positive with probability 0.8. Each test therefore has the same marginal likelihood, but after observing A, B is deterministic and adds no evidence.

The loss model gives baseline minimum expected loss 2.0. Either single test costs 0.10 total (0.05 test + 0.05 downtime) and reduces expected terminal decision loss to 1.4, for net value +0.5. With the joint table, the selected policy runs test A and stops in both branches: expected tests = 1.0. An independence assumption would incorrectly treat B as a fresh noisy signal and may pay for it again.

This is a synthetic model check. It does not establish that any particular real battery tests share this dependence.

## Frozen corpus

Fifteen cases: the eleven v0.1 cases plus a correlated duplicate-test case, a duplicate joint-assignment rejection, a joint/marginal disagreement rejection and an incomplete-assignment rejection. The correlated case requires root TEST=test-a, net value +0.5, expected tests 1.0, and NO_TEST after either outcome.

## Benchmark tournament

Compare no-test, fixed/random, entropy-greedy, one-step marginal EVSI, v0.1 independence planner and v0.2 joint-likelihood planner against an exact finite-horizon joint oracle. Report decision regret, net EVSI, cost/downtime, repeated-test count, false escalation/missed inspection, posterior calibration and likelihood misspecification sensitivity.

Physical joint likelihoods must be estimated on group-disjoint assets or through a justified hierarchical model including shared sensor/calibration/environment nuisance states. Report joint log score, marginal Brier/log score and dependence diagnostics by chemistry, temperature, SOC, age, protocol and physical asset.

## Deliberate limits

The joint table can represent correlation, but it cannot make unknown dependence disappear. If the table omits common-mode nuisance variables or fails domain calibration, the optimizer can still be confidently wrong. Synthetic fixtures validate algorithm semantics only; physical likelihoods and test costs require independent measurement and calibration.
