# VIS-DVPC-022: SOTA comparison and evaluation plan v1

**Purpose:** define how Symthaea should be compared to strong adaptive-experimentation and autonomous-science baselines without claiming results that the synthetic activation-gate fixture cannot establish.

## Research positioning

The closest conceptual match is the 2026 preprint *Certifying when decision-time information justifies adaptive experimentation*, which treats enabling an adaptive branch as a distinct certification decision, with target calibration, finite-campaign risk, non-trivial use, and executed-value accounting: https://academ.us/article/2607.27651/. This means the high-level research question is not uniquely new; Symthaea's engineering contribution should be a reproducible, fail-closed, cost-accounted evidence contract integrated with sequential perception acquisition.

The Autonomous Science Bench evaluates complete science-agent workflows with budgets and separate discovery/learning gates: https://www.autonomous-science-bench.ai/. LLM-AutoSciLab / ActiveSciBench studies hypothesis-conditioned, closed-loop experiment selection and sample efficiency: https://scientific-discovery.github.io/llm-autoscilab-project/. Value-driven adaptive-design work motivates comparing on expected net benefit of sampling, not uncertainty reduction alone: https://doi.org/10.1177/0272989X261423177. Sequential / optional-stopping claims need inference methods designed for repeated looks, such as anytime-valid procedures, rather than repeatedly reusing a fixed-sample interval: https://doi.org/10.1093/jrsssb/qkag050.

These sources are related but not directly comparable to the VIS-DVPC-022 synthetic gate. Cross-benchmark scores, prices, or model rankings must not be transplanted as if they measured the same task.

## Baselines to run on the same task set

| Baseline | What it tests | Main failure mode it exposes |
|---|---|---|
| Fixed safe fallback | No adaptation | Safe-but-useless behavior and opportunity cost |
| Forced adaptation | Always enable the adaptive branch | Safety / cost failures when value is absent or evidence is weak |
| Naive point-estimate threshold | Enable when estimated net value clears a threshold | Small-sample overconfidence and target-shift sensitivity |
| Highest uncertainty | Select the most uncertain measurement | Epistemic novelty with little or negative decision value |
| Cost-first | Select cheapest eligible measurements | Cheap evidence that does not change the decision |
| Value-of-information / expected-net-benefit sampler | Acquire evidence by expected downstream benefit net of cost | Whether the gate adds value over a strong acquisition policy |
| Robust certificate gate (DVPC-022) | Authorize only when risk, value, activation, provenance, split and budget gates pass | Whether certification suppresses false opportunities without collapsing into abstention |

The fixed fallback and forced-adaptation baselines should be included in every run. The acquisition-policy comparison belongs in VIS-DVPC-020 / the shared planner; DVPC-022 contributes the authorization layer, not a competing general-purpose selector.

## Metrics: never compress safety and utility into one score

1. **Authorization quality:** false-authorization rate, one-sided upper risk bound, joint confidence budget, denial / insufficient-evidence counts, selective risk by predeclared target slice.
2. **Decision quality:** realized downstream decision regret, net decision-value gain against the fixed fallback, positive-opportunity sensitivity, and regret relative to the known oracle in synthetic worlds.
3. **Calibration:** empirical coverage of risk upper bounds and value lower bounds across generator-defined truth cells; interval width; target-shift detection and calibration failure rates.
4. **Usefulness:** activation / coverage on eligible positive cases, abstention rate, time-to-certification, and value-versus-risk coverage frontier. A gate that always denies must not be called a success.
5. **All-in cost:** selection and verification overhead, calibration, inference, measurements, unsuccessful / censored / missing runs, reuse, compute time, energy, and latency. Report both decision value at equal total budget and total budget needed for equal decision quality.
6. **Integrity:** exact input/source/evidence digests, split isolation, sealed-set non-use, stale/mismatched receipt rejection, reproducible output, and negative controls.

Report value-cost and risk-coverage Pareto frontiers, not a colorful aggregate dashboard. Slice worst-group / target-domain risk separately; a favorable global mean cannot hide a failing activated region.

## Proposed internal target gates (aspirational, not industry standards)

- **Synthetic mechanism:** 100% known-answer verdicts; zero authorizations on every negative control; exact coverage-bound arithmetic; fail-closed on every malformed/mismatched receipt; all-in costs present for every campaign.
- **Statistical validity:** on preregistered simulation grids with known truth, the joint risk/value procedure's empirical coverage must be consistent with its declared 95% lower bound after accounting for Monte Carlo uncertainty. The grid varies sample size, failure rate, outcome range, alpha split, target shift, and optional-stopping pattern. The current CLI/fixture is fixed-sample; do not claim anytime-valid guarantees for it.
- **Real-trace confirmation:** compare on independent held-out clusters and predeclared target slices, with campaign-level alpha allocation, a power analysis and one untouched confirmation set. The evidence receipt must bind raw observations, aggregation, units, bounds, clusters, code/configuration, and split manifest. A boolean flag in the synthetic fixture is only a negative-control switch, not a cryptographic verification of real records.
- **Decision-value target:** on at least two preregistered task families, aim for at least **10% greater realized net decision value than the strongest predeclared baseline at equal all-in cost**, with a paired cluster-based confidence interval excluding zero, while maintaining the risk cap and minimum-use requirement. This is an internal go/no-go target, not a claim about existing results.
- **Cost target:** alternatively or additionally, demonstrate at least **20% lower all-in cost at non-inferior decision quality and risk**, using a preregistered non-inferiority margin. Do not pursue a cost reduction by excluding missing, failed, censored or selector-overhead costs.
- **Robustness target:** no unsafe authorization in the predeclared no-op, target-shift, low-sample, negative-net-value, under-budgeted, and sealed-leakage challenge families; report each family separately.

The internal 10% / 20% thresholds are proposed engineering targets, not published SOTA values. The current 16-case fixture is a scorer-mechanism check only and does not demonstrate any of the real-trace targets.

## Experimental protocol

1. Freeze task definitions, target slices, unit semantics, baseline versions, cost accounting, outcome bounds, and the family-wise alpha budget before running confirmation.
2. Split by independent cluster / site / scenario source, not random rows from the same trajectory. Reserve a sealed confirmation split that tuning code cannot read.
3. Estimate power through simulation before selecting the confirmatory sample count. For multiple claims, segments, or repeated looks, preallocate alpha across them or use a valid sequential method.
4. Pair methods on the same scenarios, seeds and total budgets. Tune only on development clusters; freeze hyperparameters and the acquisition policy before confirmation.
5. Publish every campaign's decision, risk UCB, value LCB, activation rate, all-in cost, provenance receipt and reason codes. A result with missing evidence is INSUFFICIENT_EVIDENCE, never implicitly successful.
6. Treat synthetic expected-answer performance, local tests, remote CI, and real-world effectiveness as four separate evidence levels.

## Next implementation milestones

- Extend the generator to sweep known risk, value, range, shift, sample-size and cost conditions rather than rely on one hand-authored table.
- Add calibration/coverage tests over thousands of deterministic synthetic draws with known truths and Monte Carlo intervals.
- Compose DVPC-022 authorization with the VIS-DVPC-020 sequential acquisition policy; evaluate the full loop and the fixed fallback on paired scenarios.
- Add an explicit per-target-slice alpha allocation and report worst-slice risk to prevent subgroup dilution.
- Replay real, permissioned traces only after the raw evidence receipt and split rules are implementable and reviewable.

No claim of SOTA, real safety, calibration, generalization, or production authority should be made until those gates are met.
