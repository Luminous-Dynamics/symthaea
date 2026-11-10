# Sequential Battery Diagnostic Planner v0.1

**Status:** Rust-first finite-horizon synthetic decision planner. Not a diagnostic accuracy model, service authorizer or BMS.
**Owner:** [EV-SEQUENTIAL-DIAGNOSTIC-001 #11325](https://github.com/Luminous-Dynamics/symthaea/issues/11325).
**Predecessor:** [Decision-value battery diagnostic planner #11286](https://github.com/Luminous-Dynamics/symthaea/pull/11286).

## Research hypothesis

Greedy one-test EVSI is myopic. Two tests can each have negative standalone net value because neither outcome changes the preferred terminal action, while a sequence can have positive value because the combined observations cross a decision threshold.

For policy π, the finite-horizon objective is the expected sum of test costs, downtime costs and terminal decision loss. Each outcome updates the hypothesis distribution using Bayes' rule. The planner considers STOP/NO_TEST at every node. It chooses the policy minimizing expected total loss. The terminal action is selected from currently admissible actions only; if none is admissible, the node returns ABSTAIN with the declared abstention loss.

## Hard eligibility

A test is selectable only if safety approval, calibration validity, domain support and equipment availability are all true. Every test binds protocol and equipment digests. Conditional tests can require a specific earlier test outcome. Test IDs cannot be reused; dependency cycles are rejected; the horizon is capped at eight.

Likelihoods must be finite, between zero and one, and sum to one over outcomes separately for every hypothesis. Invalid likelihoods are rejected, not renormalized.

## Known-truth synthetic sequence

Four latent states have priors A0B0=0.20, A1B0=0.30, A0B1=0.30 and A1B1=0.20.

Terminal actions are Hold (loss 12 only in A1B1) and Replace (loss 5 in every state). Test A reveals the A bit; test B reveals the B bit. Each test costs 0.10 synthetic loss units plus 0.05 downtime units.

Before testing, Hold has expected loss 2.4. After one test, the posterior probability of A1B1 on a positive branch is 0.4, below the replacement threshold 5/12. Thus either test alone costs 0.15 without changing the terminal decision. A two-test policy can isolate A1B1 and trigger replacement. The expected net value is +1.175, the expected number of tests is 1.5, and the maximum number of tests on a path is two. This is a synthetic decision-theory fixture, not a claim about physical test accuracy or real costs.

A separate fixture makes test B conditionally available only after A=a1. The selected policy must not place B on the other branch.

## Frozen corpus

Seven cases: complementary positive sequence, conditionally available second test, positive one-step test, perfect-information/zero-decision-value control, missing-safety-approval control, invalid-likelihood rejection, and no-admissible-terminal-action ABSTAIN.

## Benchmark tournament

Compare no-test/current evidence, fixed periodic scheduling, random eligible testing, maximum entropy/information gain, greedy one-step EVSI, sequential EVSI, and an exact finite-horizon oracle (upper bound only). Report expected terminal decision loss/regret, net value, expected tests and downtime per asset, maximum tests per path, time-to-decision, false escalation, missed inspection and sensitivity to likelihood misspecification.

Calibrate likelihoods on independent group-disjoint physical assets. Report Brier score, log score and reliability by chemistry, temperature, SOC, age and protocol before any physical diagnostic claims.

## Deliberate limitations

v0.1 assumes a static latent hypothesis during one diagnostic episode and conditionally independent test likelihoods given that hypothesis. It does not model degradation evolving during the sequence, queues, test failure, operator error or physical safety dynamics. It does not collect data, execute a test, authorize repair or control a vehicle. The optimizer is only as trustworthy as its frozen priors, likelihoods and losses.
