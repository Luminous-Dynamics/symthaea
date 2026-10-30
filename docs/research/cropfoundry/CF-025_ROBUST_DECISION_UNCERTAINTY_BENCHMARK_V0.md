# CF-025: robust multi-objective crop decisions under hybrid uncertainty

**Status:** benchmark contract proposal; not run. No real-world crop, water-saving, nutrition, economic or welfare outcome is claimed.

## Research motivation

A policy can win on its expected score and still fail badly under heat, drought, sensor failure, price shocks or uncertain biological response. CropFoundry therefore needs a robust-decision benchmark across the full crop-to-food chain, not just one nominal simulation or one scalar score.

A 2026 paper proposes chance-constrained multi-objective robust programming for multi-cropping irrigation, combining uncertainty and multiple goals, and explores how risk preference and robustness penalties change solutions ([Agronomy 2026, 16(18), 1786](https://www.mdpi.com/2073-4395/16/18/1786)). A 2026 site-specific irrigation study reports that the dominant source of soil-moisture forecast uncertainty shifts from model parameters toward weather uncertainty at longer lead times ([SWIM2 study](https://doi.org/10.1016/j.agwat.2026.110300)). These are methodological examples, not evidence of South African or CropFoundry performance.

## Paradigm shift

A useful agricultural decision-design reference is Burbano-Figueroa's 2025 *Addressing Uncertainty in Agriculture: An Introductory Guide to Robust Decision-Making in the Face of Climate Change*, catalogued by [FAO AGRIS](https://agris.fao.org/search/es/records/691daf5b49e41686f7929a0b). It contrasts point-forecast-driven “predict then act” with exploring multiple plausible futures before selecting strategies. FAO's climate-risk guidance also emphasizes combining hazard, exposure, vulnerability and adaptive capacity rather than collapsing resilience to one nominal result ([FAO climate and disaster risk management](https://www.fao.org/climate-change/cross-cutting-areas/climate-and-disaster-risk-management/en)). These support the method, not a claimed CropFoundry benefit.


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

## Executable synthetic diagnostic — CF-025 harness v0

The Rust command `cropfoundry-robust-policy-demo` reuses the existing CF-022 six-scenario/five-policy receipts. It does not introduce a second water simulator or consume external datasets:

```bash
cargo run -q -p xtask -- cropfoundry-robust-policy-demo \
  --out reports/cropfoundry/robust-policy-synthetic-v0.json
cargo test -p xtask cropfoundry_robust_policy
```

For each policy, the receipt reports mean and median synthetic stress, nearest-rank p95, upper-tail mean/CVaR95, maximum and mean worst-zone burden, water totals, allocation dispersion, and mean regret against the per-scenario best stress score. It emits a Pareto-front membership flag and dominated-policy set across mean stress, tail stress, worst-zone burden, water use and allocation dispersion. Four declared preference profiles then sweep weighted normalized scores and record the complete policy ranking and top-ranked policy for each profile.

**Critical sample-size caveat:** the base CF-022 suite has six scenarios. CF-025 now evaluates all five policy implementations across a predeclared 30-case deterministic sensitivity lattice that extends the same simulation over multiple water budgets, forecast biases, sensor dropouts, heat/drought timings, and compound failures. For a 5% tail fraction, `ceil(30 × 0.05) = 2`; the receipt therefore emits `tail_sample_count: 2` for CVaR95. This gives the designed stress-test summary more resolution, but the cases are not independent draws from a weather/climate distribution and do not estimate calibrated 95th-percentile field outcomes. Do not translate these diagnostics into a claim of “95% reliability.”

The preference sweep is deliberately transparent, but each weighted score encodes normative trade-offs. A lower water-use objective can inadvertently reward under-irrigation; therefore it is always paired with stress/tail/worst-zone burden and hard constraints, and even a top rank is **not** an agronomic recommendation. Pareto dominance means only that another declared policy is no worse on every one of these five synthetic objectives and strictly better on at least one; omitted objectives may reverse the comparison.

## Fixed robustness sensitivity lattice v0

The robust-policy executable creates 30 fully specified scenario records and executes all five irrigation policies on each record. The original six CF-022 base scenarios remain unchanged and the extra 24 cases expand the same simulator across one-factor and compound changes. Scenario IDs are unique; each policy must appear once for every scenario; the matched-uniform control must preserve the risk-aware policy's per-day dose trace. Water-budget and per-plot actuator violations are hard failures.

This makes the upper-tail metric more informative as a **designed stress-test summary**: for (N=30), the 5% tail includes two ordered scenario scores. The tail sample count is emitted per policy and a negative control should catch changed/missing scenario sets even when row counts happen to match. These 30 cases are not independent draws from weather/climate distributions, and giving each equal weight is a sensitivity convention, not a calibrated probability measure. A field-risk probability needs source-backed joint uncertainty, a defensible distribution/weighting method and held-out empirical validation.

## Current CropFoundry status

CF-022's five-policy irrigation harness includes water caps, per-plot actuator limits, per-zone stress, a dose-matched rotating-uniform counterfactual, and a greedy minimax heuristic. Receipt v2 corrected a starvation-scoring blind spot and exposes worst-zone and allocation-distribution metrics. CF-025 now uses the same policy implementations on a 30-case fixed sensitivity lattice rather than reusing the six base scenarios alone. A code-level feasibility pass means all 30 × 5 policy/scenario pairs meet the synthetic water and actuator bounds; it does not validate physical actuators, agronomy or risk probabilities. The next gate remains calibrated joint uncertainty and held-out outcomes before any chance-constraint or field-risk claim.

## Integration and claim limits

- **Symthaea:** uncertainty contracts, deterministic policy evaluation, Pareto/tail metrics and reproducible receipts.
- **Symtropy:** scenario exploration; visualized scenarios remain simulations, not biological evidence.
- **Sol-Atlas:** weather, soil/water/resource context and their spatial-temporal uncertainty, with source/licence/provenance retained.
- **Mycelix:** permission and provenance status for community/farm records, parameter estimates and decision receipts.

A passing synthetic robust-optimization run can establish deterministic algorithm behavior over its declared scenarios. It does not establish calibrated risk, crop efficacy, field transfer, food safety, nutrition adequacy, affordability or community welfare.
