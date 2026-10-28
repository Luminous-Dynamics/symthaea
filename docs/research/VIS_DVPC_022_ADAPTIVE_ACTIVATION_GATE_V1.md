# VIS-DVPC-022: Adaptive Perception Activation Gate v1

**Status:** benchmark protocol / synthetic known-answer implementation. Not a production authorization mechanism.

## Question

Should a DVPC adaptive branch be enabled at all, or should the system retain the predeclared safe fallback? This is upstream of VIS-DVPC-020's question of which evaluation to acquire next. The benchmark is an adapter/scorer, not a competing universal optimizer.

## Research context

The July 2026 preprint *Certifying when decision-time information justifies adaptive experimentation* identifies the upstream activation decision, including target calibration, finite-campaign risk, non-trivial use and executed-value accounting: https://academ.us/article/2607.27651/ . The Autonomous Science Bench 0.1 (published October 1, 2026) evaluates complete science agents with budgets and separate discovery/learning gates: https://www.autonomous-science-bench.ai/ . LLM-AutoSciLab / ActiveSciBench studies hypothesis-conditioned closed-loop experiment selection: https://scientific-discovery.github.io/llm-autoscilab-project/ . These methods are related, but their results are not directly comparable to this toy DVPC gate.

## Decision contract

The fixture evaluator authorizes only when all conditions hold:

1. Provenance and action digest are valid.
2. Target calibration exists and no target-shift veto is raised.
3. The sealed confirmation set has not been touched.
4. All-in campaign costs stay inside budget.
5. The declared campaign fits the available campaign capacity.
6. The supplied net decision-value lower bound clears its preregistered minimum.
7. The activation rate clears a minimum-use floor.
8. The one-sided Clopper-Pearson upper confidence bound for the failure rate is no greater than the risk cap.

Any hard contract violation is **DENIED**. Evidence that is structurally valid but too small to certify the risk cap is **INSUFFICIENT_EVIDENCE**, not a pass. Only a fully qualifying case is **AUTHORIZED**. The code treats target shift, provenance, and seal violations as hard denials, even when the point estimate is attractive.

The net-value lower bound is a fixture input, not a bound estimator. Its conversion from currency/resources to decision value must be specified by a real deployment protocol before use. The model does not infer transportability from source data.

## Exact risk arithmetic

For n independent Bernoulli target-region trials, k observed failures and one-sided error level alpha, the upper Clopper-Pearson bound solves P_p[X <= k] = alpha. The implementation uses the closed form for k=0 and binomial-tail bisection otherwise. At alpha=0.05, zero failures require at least 59 independent trials to put the upper bound below 0.05; 58 trials do not suffice. One failure requires more trials. This is valid only under the stated independent Bernoulli model and the declared target-region sampling design.

## Frozen fixture and baselines

The fixture includes positive known answers and negative controls for the 58/59 boundary, tiny samples, one observed failure, no-op opportunity, source/target shift, absent target calibration, trivial activation, negative net value, budget overrun, campaign truncation, sealed-data contamination, missing provenance and digest mismatch. The evaluator does not see outcomes for unselected decisions; this activation gate is based on preregistered aggregate evidence fields.

Compare at least: fixed safe fallback, forced adaptation, naive point-estimate threshold, and certificate gate. Report activation rate, false-activation rate, risk upper bound, net decision-value lower bound, all-in cost, abstention/insufficient-evidence rate, and risk-coverage/value-cost frontiers separately. Do not collapse them into an opaque scalar.

## Required research gates

- 100% correct known-answer classifications on the frozen fixture.
- Zero authorization on all fixture negative controls.
- Exact 58-versus-59 zero-failure boundary and correct monotonicity in sample size/failures.
- 100% fail-closed handling of provenance, digest, target calibration, shift and sealed-set controls.
- Explicit all-in costs and campaign capacity in every report.
- No SOTA, safety, calibration, or generalization claim from synthetic fixtures alone.

A real evaluation needs independent target clusters, preregistered criteria, appropriate multiple-testing/optional-stopping treatment, and one untouched confirmation set. A hash is evidence of byte identity, not a proof of scientific validity or non-leakage from a malicious external process.
