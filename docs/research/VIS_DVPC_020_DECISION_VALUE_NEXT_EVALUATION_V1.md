# VIS-DVPC-020 — Decision value of the next perception evaluation

**Status:** synthetic benchmark adapter/scorer; no real-data, calibration or SOTA claim.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8642  
**Shared decision-value contract:** https://github.com/Luminous-Dynamics/symthaea/issues/7803  
**Existing evidence-acquisition planner:** https://github.com/Luminous-Dynamics/symthaea/issues/5596  
**DVPC program:** https://github.com/Luminous-Dynamics/symthaea/issues/8156

## Why this work exists

The current DVPC frontier identifies supported, unresolved and infeasible operating points. It does not answer which *next independent evaluation* is most likely to change an evidence-supported decision. The benchmark adapts existing selector policies to perception-compute evidence selection; it does not introduce a competing universal optimizer.

The central negative control is deliberate: high uncertainty is not high decision value. A measurement that cannot alter feasibility or the chosen policy should not consume budget merely because its metric is noisy.

Relevant research:
- [DEEP: The Decision Value of Perception Compute (2026)](https://arxiv.org/abs/2609.35910)
- [Robust Bayesian Decision Making under Adversarial Uncertainty, UAI/PMLR 2026](https://proceedings.mlr.press/v337/harikumar26a.html)
- [Anytime validity is free: inducing sequential tests, JRSSB 2026](https://academic.oup.com/jrsssb/article/88/4/1366/8493290)

The robust-design paper warns that high nominal information gain can still produce a fragile decision under misspecification. The anytime-valid literature matters if future versions adapt data collection or stopping to interim results.

## First synthetic known-answer fixture

- Visible candidate actions: `tests/fixtures/vision-dvpc-020-actions.tsv`
- Evaluator-only outcomes: `tests/fixtures/vision-dvpc-020-oracle.tsv`
- Reference scorer/baseline adapters: `tools/vision-dvpc-acquisition-bench.rs`

The fixture includes decision-irrelevant uncertainty, a feasibility cliff, a ranking-reversal measurement, meter-calibration evidence, a shift-sign-reversal measurement, an unauthorized source, a sealed-test action and a no-information action. A failed calibration and a harmful realized outcome are not hidden.

The visible action file carries no realized outcome column. The selection function accepts only the visible candidate structs and resource budgets; it has no oracle parameter. The tool scores chosen action IDs against the evaluator-private oracle only after the selection function returns. This is an API-level separation check, not a sandbox or a proof that an external policy was unable to read the oracle file. Synthetic outcomes are for known-answer tests only.

## Build and run

```sh
rustc --edition=2024 --test tools/vision-dvpc-acquisition-bench.rs -o /tmp/vis-dvpc-020-tests
/tmp/vis-dvpc-020-tests
rustc --edition=2024 -O tools/vision-dvpc-acquisition-bench.rs -o /tmp/vis-dvpc-020

actions_sha="$(sha256sum tests/fixtures/vision-dvpc-020-actions.tsv | cut -d ' ' -f1)"
oracle_sha="$(sha256sum tests/fixtures/vision-dvpc-020-oracle.tsv | cut -d ' ' -f1)"
/tmp/vis-dvpc-020 tests/fixtures/vision-dvpc-020-actions.tsv "$actions_sha" \
  tests/fixtures/vision-dvpc-020-oracle.tsv "$oracle_sha" decision-value-lcb
```

Reference policies are:
- `decision-value-lcb`: highest declared distribution-free regret-reduction lower bound, subject to hard resource budgets;
- `highest-uncertainty`: uncertainty-only baseline;
- `cost-first`: lexicographic latency-then-energy baseline, not a composite cost score;
- `seeded-random`: deterministic seeded baseline;
- `abstain`: no-information control.

These are benchmark baselines, not a replacement for the selector in #5596. The decision-value field is explicitly `DISTRIBUTION_FREE_REGRET_REDUCTION_LCB`; the tool never turns interval width into a probability. A future distribution-backed EVSI adapter must name and hash its probability model and is a separate protocol.

## Accounting and result semantics

The tool selects from action IDs and visible metadata only. Both budgets are independent hard ceilings; no latency/energy weighted score is computed. At most one action for each decision ID can be selected, so alternative measurements for a single claim cannot spend twice.

Every action remains in the result with a disposition. Unauthorized, sealed-test, unknown-cost, dependent-source and untrusted-instrument candidates are excluded from selection with reasons. Selected failed/censored/missing outcomes remain charged; a missing actual cost is conservatively charged at its reserved upper bound and marked incomplete. An observed cost above its reservation is reported as an overrun. The realized regret reduction is descriptive for this synthetic oracle and conditional on the declared `DISJOINT_DECISION_CLAIMS_DECLARED` additivity assumption.

## Acceptance targets (proposed, not achieved)

- Zero selection access to oracle outcome values through the selector API.
- Zero unauthorized or sealed-test actions selected.
- 100% attempted/action/outcome disposition and cost accounting.
- Zero cross-budget violations caused by the *planned upper reservations* on the known-answer fixture.
- On the fixture, the decision-value baseline yields greater realized regret reduction than uncertainty-only at the same constraints; the no-information action remains unselected by the decision-value policy.
- Report realized decision-regret reduction, latency and energy separately; do not publish only a composite score.

The hosted workflow is the exact-source mechanical check. Passing synthetic tests would validate only these mechanics, not selector superiority on real traces.

## Next empirical milestone

Use an authorized, rights-cleared development pool for selector adaptation, freeze policy/config/utility/budget and its inference method before evaluation, and preserve a separate sealed confirmation domain. For adaptive repeated acquisition, use an appropriate anytime-valid or otherwise predeclared sequential-inference method. Compare random, uncertainty-only, cost-first, declared information-gain, decision-value and robust-feasibility adapters at equal all-in costs. Measure decision-regret reduction, false-feasible/missed-feasible outcomes, ranking stability, measurement failures, costs and abstentions separately.

No real-trace or SOTA claim until that experiment and a second-domain replication exist.
