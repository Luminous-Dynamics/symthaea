# ORE-VOI-001: Decision-aware measurement selection benchmark v0.1

Status: synthetic contract benchmark  
Owner: Symthaea decision support; reuse existing assay and blend owners  
Parent: [Symthaea #8060](https://github.com/Luminous-Dynamics/symthaea/issues/8060)  
Implementation: standalone Rust tool under tools/ore-voi-bench/

## Paradigm shift

Uncertainty is not the same thing as decision value. A test can shrink an interval substantially yet never change the admissible blend. A narrowly targeted contaminant, sample-identity, recovery-response, or comminution-energy test can be more valuable when it changes whether a candidate blend is feasible. Conversely, a model can overestimate decision gain, so the benchmark must expose estimation error instead of defining the policy as correct by construction.

The first increment intentionally evaluates a **one-step selection policy**, not a complete assay planner or geometallurgical simulator. It compares:

1. Fixed priority (simple operational baseline).
2. Highest declared uncertainty reduction (uncertainty-first baseline).
3. Highest estimated net decision gain, with hard budget and evidence gates (decision-aware candidate).
4. Evaluator-only realized-gain oracle, used as a ceiling on these finite synthetic cases.

The policy selector receives only action descriptions and the budget. Evaluator-only synthetic gains are passed to separate scoring functions, not to the selection functions. The oracle is not a deployable policy and must never be reported as a fair online competitor.

## Typed action contract

Each measurement proposal has an exact action ID and measurement kind, eligibility flag and reason, integer test cost, estimated uncertainty reduction, estimated gross decision gain, and stable fixed-order priority. An action is selectable only when it is marked available and its cost fits the scenario budget. An unavailable action must have a blocking reason such as unresolved sample/lot identity, missing source provenance, or unresolved analytical QC.

The evaluator corpus records realized synthetic gross decision gains for every action. Net value is gross gain minus cost. The oracle selects the best strictly positive eligible net value; if no eligible action has positive net value, it selects no test. This avoids rewarding tests merely because they are informative or because a policy always spends its budget.

## Frozen cases and adversarial controls

The eight scenarios cover:

- high uncertainty that is not the highest decision value;
- a high-scoring test that exceeds the available budget;
- measurement cost that dominates broad uncertainty reduction;
- miscalibrated decision-gain estimates where the decision-aware heuristic loses to the evaluator oracle;
- an unqualified sample-identity action that must be blocked;
- no positive net value, for which selecting no test is correct;
- no evidence-admissible measurement within budget;
- deterministic tie-breaking across all policies.

All values are invented, dimensionless synthetic units. The publicly checked-in evaluator values make v0.1 reproducible, but **do not make it a blinded holdout**. Future predictive evaluations need a separately governed hidden test split or licensed measured data with declared method, specimen identity, uncertainty and independent outcome records.

## Acceptance gates

For the frozen v0.1 corpus:

- 8/8 scenario selections match exact expected action IDs for all four policies.
- Zero blocked or out-of-budget action selections.
- Zero selection when the decision-aware estimate has no strictly positive net value.
- An explicit, nonzero decision regret in the miscalibration counterexample; do not tune it away.
- Exact-corpus SHA-256 and 40-character source commit are present in the receipt.
- Unknown JSON fields, duplicate scenario/action IDs, invalid gain ranges, missing evaluation keys, inconsistent availability reasons, and expected selections of blocked or over-budget actions fail closed.
- Stable lexical tie-breaking makes exact reruns deterministic.

Report separately: fixed-order regret, uncertainty-first regret, decision-aware regret, oracle net value, budget violations, blocked-action selections, control-match count, runtime and exact receipt identity. Units are synthetic; no monetary, recovery, energy, safety, environmental, or production benefit may be inferred.

## Research grounding

- Jeuken and Forbes, “The Value of Drilling—A Chance-Constrained Optimization Approach” (2024), studies the relationship between information-gathering and uncertain mine-planning decisions: https://doi.org/10.1007/s42461-024-01061-8
- Andrade and Dimitrakopoulos, “Integrated Stochastic Underground Mine Planning with Long-Term Stockpiling” (2024), connects uncertain resource models to integrated planning and stockpiling: https://doi.org/10.3390/min14020123
- The simultaneous stochastic optimization study of an open-pit copper mining complex integrates uncertain supply, recovery and market demand across the value chain (2024): https://doi.org/10.1080/17480930.2024.2381904
- Saavedra et al., “Blending Characterization for Effective Management in Mining Operations” (2025), motivates modeling blend heterogeneity and the difference between nominal and operationally useful feed: https://doi.org/10.3390/min15090891
- Adaptive open-pit planning research compares rolling-horizon decisions with updated information against plans made before the information arrives: https://doi.org/10.1016/j.resourpol.2021.102086

These papers motivate the research questions, not the synthetic fixture values or any claimed performance improvement.

## Next research increments

1. v0.2: separated policy-input and evaluator datasets, multiple outcome branches, and a hidden held-out synthetic seed set.
2. v0.3: sequential measurement decisions under a multi-test budget, including turnaround time, stale results, interactions between measurements, and opportunity cost of delay.
3. v0.4: consume the existing ORE-GEO assay contract and ORE-FEED-BLEND feasibility outputs rather than duplicating their schema or solver. Keep assay validation, blend optimization, acquisition policy, and independent verification as separate components.
4. Measured-data pilot: only with permissioned source records, exact sample/lot custody, laboratory QA/QC, method comparability and an independently defined validation split.

This benchmark has no physical side effects. It cannot authorize sampling, purchase testing, start equipment, certify an assay, qualify reserves, or recommend mine operation.