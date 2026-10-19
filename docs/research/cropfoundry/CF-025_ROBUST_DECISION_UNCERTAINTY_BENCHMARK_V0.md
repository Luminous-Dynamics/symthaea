# CF-025: robust multi-objective crop decisions under hybrid uncertainty

**Status:** benchmark contract proposal; not run. No real-world crop, water-saving, nutrition, economic or welfare outcome is claimed.

## Research motivation

A policy can win on its expected score and still fail badly under heat, drought, sensor failure, price shocks or uncertain biological response. CropFoundry therefore needs a robust-decision benchmark across the full crop-to-food chain, not just one nominal simulation or one scalar score.

A 2026 paper proposes chance-constrained multi-objective robust programming for multi-cropping irrigation, combining uncertainty and multiple goals, and explores how risk preference and robustness penalties change solutions ([Agronomy 2026, 16(18), 1786](https://www.mdpi.com/2073-4395/16/18/1786)). A 2026 site-specific irrigation study reports that the dominant source of soil-moisture forecast uncertainty shifts from model parameters toward weather uncertainty at longer lead times ([SWIM2 study](https://doi.org/10.1016/j.agwat.2026.110300)). These are methodological examples, not evidence of South African or CropFoundry performance.

## Paradigm shift

Do not ask only, "Which plan has the highest mean score?" Ask:
- Which hard constraints remain feasible across credible uncertainty?
- How bad is the upper-tail/worst-case outcome for each objective?
- Which assumptions or uncertainty sources cause a recommendation to reverse?
- Are results stable when reasonable risk preferences change?
- Does the plan improve one average by imposing concentrated losses on a crop, plot, community, nutrient or future season?

Treat uncertainty as a versioned part of the input contract. Do not hide it in one confidence field, and do not conflate measured uncertainty, literature-derived ranges, elicited expert judgement and invented scenario parameters.

## Required uncertainty model

Every uncertainty entry must identify:
- parameter/variable, unit, target stage and affected decision;
- evidence class: directly estimated from observations, literature-derived, expert-elicited, or synthetic-only;
- marginal distribution or interval, estimation/calibration dataset, time/site/lot applicability and uncertainty in the estimate itself;
- dependence/correlation with other variables, and the rationale for any independence assumption;
- uncertainty-set or scenario version, random-seed policy where used, scenario weights, and adversarial stress cases;
- whether the uncertainty is epistemic (limited knowledge), aleatory (variation), or not currently distinguished.

Weather, soil properties, plant response, sensors, harvest loss, prices, energy, labor, nutrient composition, storage/preparation retention and resource availability may interact. Independent one-at-a-time perturbations are not sufficient when joint dependence matters. Any invented variable must stay explicitly synthetic and must never be reported as an estimated real-world probability.

## Baselines

1. **Deterministic expected-value policy:** optimize expected inputs/outcomes using declared central estimates.
2. **Chance-constrained policy:** respect a declared probability constraint only when the probability distribution is empirically or otherwise defensibly calibrated.
3. **Minimax policy:** minimize worst-case objective loss over a declared plausible uncertainty set.
4. **Tail-aware policy:** minimize a declared conditional-tail risk measure such as CVaR, with the tail fraction and risk penalty predeclared.
5. **Pareto policy family:** expose non-dominated trade-offs across yield, quality, nutrient delivery, water, land, energy, affordability, labor and distributional/fairness metrics.

All policies must face the same scenario data and hard constraints. For synthetic scenarios, call chance constraints "scenario-frequency constraints" unless the scenario weights truly have probabilistic meaning.

## Acceptance criteria and proposed targets

- **Integrity:** deterministic replay for the same pinned inputs; every run records source/snapshot hashes, uncertainty-set version, model/code head, scenario IDs, objective/constraint definitions, preference sweep and all outputs.
- **Hard feasibility:** zero hard safety, daily-water-budget and per-plot actuator-limit violations across every scenario designated as hard-constrained. Do not average constraint violations away.
- **Tail reporting:** publish expected value, median, 90th/95th percentile, worst case and chosen tail-risk metric for each objective and for each materially affected group. If the sample is too small to estimate a percentile reliably, say so.
- **Preference robustness:** sweep predeclared weights/risk penalties; report recommendation reversals and the objective changes that cause them. Never claim preference-invariant decisions where the recommendation changes.
- **Calibration:** for probabilistic forecasts/constraints, report calibration/coverage against held-out empirical data. No calibrated probability claim from synthetic scenarios alone.
- **Provisional target:** after uncertainty data and domain review are pinned, seek at least 95% feasibility on the declared *non-adversarial* evaluation suite and keep 95th-percentile normalized shortfall below a preregistered, domain-approved cap. Keep adversarial scenarios separate and report all violations explicitly. These are proposed targets, not universal standards.
- **No scalar-only pass:** a result must disclose Pareto trade-offs and worst-group/zone outcomes. A single weighted mean cannot conceal a hard constraint, a severe tail outcome, or one group bearing the loss.

## Evaluation protocol

1. Pin the decision question, scope, crop/food lots, site-year split, objectives, hard constraints, uncertainty source classes, scenario weighting and risk preferences before looking at final test outcomes.
2. Separate calibration from evaluation. Fit distributions or intervals on training/calibration data only. Keep future site-year, crop-lot or environment blocks held out according to the intended transfer claim.
3. Generate matched baseline comparisons using identical scenario bundles, resource envelopes and decision horizons. Preserve correlated factors in the joint scenarios.
4. Add negative controls: remove one uncertainty source, shift its mean/variance, deliberately miscalibrate a forecast, drop a sensor stream, perturb prices/costs, and test whether the policy detects loss of support instead of continuing with false confidence.
5. Report per-scenario objectives, constraint slack, tail contribution, recommendation/ranking changes, per-zone/per-group outcomes and full receipts.
6. Recompute and preserve results if the score/objective changes. Do not compare scores across different proxy versions or uncertainty contracts as though they were the same benchmark.

## Current CropFoundry status

CF-022's five-policy irrigation harness is a useful starting point: it already includes water caps, per-plot actuator limits, per-zone stress, a dose-matched hindsight comparator, and a greedy minimax heuristic. Receipt v2 also corrected a starvation-scoring blind spot and exposes the count of scenarios contributing to its worst-to-best ratio. This is not yet a robust optimization benchmark: it uses a small deterministic synthetic suite, not a calibrated joint uncertainty distribution. CF-025 adds the next gate: scenario sensitivity and preference stability first; calibrated uncertainty and chance constraints only when defensible data exist.

## Integration and claim limits

- **Symthaea:** uncertainty contracts, deterministic policy evaluation, Pareto/tail metrics and reproducible receipts.
- **Symtropy:** scenario exploration; visualized scenarios remain simulations, not biological evidence.
- **Sol-Atlas:** weather, soil/water/resource context and their spatial-temporal uncertainty, with source/licence/provenance retained.
- **Mycelix:** permission and provenance status for community/farm records, parameter estimates and decision receipts.

A passing synthetic robust-optimization run can establish deterministic algorithm behavior over its declared scenarios. It does not establish calibrated risk, crop efficacy, field transfer, food safety, nutrition adequacy, affordability or community welfare.
