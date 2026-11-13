# Evidence-to-action chain: six-gate benchmark extension

This extension composes with Symthaea #11122 (decision-focused EVSI), #11338 (distributionally robust active sensing), and #11496 (robotics FEP/thermal benchmark). It does not create another diagnostic planner, posterior engine, model adequacy authority, action projector or safety controller.

## Why this matters

A sensor request is not a successful diagnostic. A physical observation may be invalid, fail to identify the mechanism, never be used by the policy, fail to change the downstream decision, lose to a cheaper fixed acquisition strategy, or lead to a proposed action rejected by independent physical assurance.

Use the following ordered gates:

1. **Meaningful reference:** instrument, calibration, clock, reference condition, and comparison interval are valid for the claim.
2. **Identifiability:** competing mechanisms produce sufficiently distinguishable observation distributions under the chosen experiment.
3. **Evidence use:** acquired evidence enters the declared inference/decision path; posterior or predictive outputs respond appropriately.
4. **Decision value:** evidence changes expected downstream Bayes risk or another frozen task-loss quantity enough to justify acquisition.
5. **Selection value:** adaptive acquisition outperforms a fixed/periodic/cheaper baseline under equal resource and action budgets.
6. **Safe realization:** the evidence-conditioned action is independently projected/admitted and physically realizable before any success credit is assigned.

A gate can be Fail or Unknown. An Unknown earlier gate blocks positive downstream qualification. Do not replace the chain with one aggregate accuracy or recovery score.

## Synthetic known-answer corpus

- `fixtures/evidence_use_v0_1.json`
- `fixtures/evidence_use_v0_1.schema.json`

Eight cases distinguish missing reference, non-identifiability, evidence acquired-but-unused, posterior change without EVSI, positive EVSI without selection value over a fixed baseline, independent realization rejection, full evidence-chain eligibility, and out-of-support input.

The expected final disposition `evidence_chain_qualified` means only that the six diagnostic/evidence gates pass in the declared synthetic fixture. It does not confer physical motor authority; HAL/RTA remains independent.

## Scoring

Report each gate rate and conditional denominator. In particular:
- reference validity by channel, calibration, clock and environment;
- identifiability by mechanism pair and action/experiment;
- evidence-use response, including calibration and OOD support;
- predicted versus realized EVSI and downstream regret;
- adaptive selection value against fixed/periodic/cheapest baselines at matched cost;
- projection/RTA rejection and successful physical realization separately.

Preserve failed, abstained, Unknown, OOD and rejected cases in the denominators. Report both end-to-end chain success and the first failing gate. A model must not gain a better score by refusing to classify difficult cases without reporting abstention coverage.

## External motivation

Zhang, *When Does Test-Time Physical Diagnosis Pay?* (arXiv:2609.22299v2, 26 October 2026) identifies these six ordered empirical conditions. Its transfer findings motivate gate-level rather than aggregate evaluation: a diagnostic trace can contain information while the frozen decision model fails to use it. This paper is a comparator and motivation, not a universal estimate of robotic diagnosis performance.

https://arxiv.org/abs/2609.22299

## Claim boundary

These fixtures test synthetic gate composition. They do not validate physical sensors, causal identifiability in an actual plant, the accuracy of a posterior model, downstream task utility, hardware safety, or deployed action authority.
