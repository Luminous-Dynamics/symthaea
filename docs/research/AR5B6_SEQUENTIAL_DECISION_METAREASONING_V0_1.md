# AR-5B6: Sequential Decision Metareasoning v0.1

**Status:** implementation draft; source and tests authored, not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → this branch.  
**Tracking:** #9696 (decision-value replication), #9659 (noisy/expensive MOO).  
**Claim ceiling:** a bounded finite-horizon dynamic-programming reference over the supplied finite posterior, likelihood, loss, action, and integer-cost model. It is not a scalable optimizer, a robust guarantee under model misspecification, or a measured performance improvement.

## 1. Answer: do we need something better than an oracle?

Do not discard the exact finite-state EVSI oracle. Change its role.

An oracle is not an optimizer family or a universal source of truth. It is a reference calculation conditional on a declared model. The current EVSI module is valuable because it computes the one-step expected reduction in terminal Bayes risk without Monte Carlo approximation. It can tell us whether an approximate *one-step* acquisition score is wrong on a small problem.

It does not solve the campaign-level problem. In particular, ranking each action by EVSI divided by its cost is a myopic heuristic. A ratio is not generally optimal under a fixed total budget: a cheap action can have the best ratio while a larger action produces a bigger absolute reduction in terminal risk and consumes the whole budget more effectively.

The research architecture should therefore have three layers:

1. **Exact finite reference models.** One-step EVSI for action-level calibration, then finite-horizon dynamic programming for adaptive policy-level calibration.
2. **Approximate decision controllers.** Knowledge gradient, information-directed sampling, posterior sampling, rollout/MCTS, model-predictive policies, and decision-aware Bayesian optimisation. These are what we scale and compete with.
3. **Robust decision qualification.** Detect model uncertainty, drift, dependence, missingness, and cost-model errors; use ambiguity-aware decisions or abstain when the model contract is not credible.

The reference calculation is an instrument. The controller is the product.

## 2. New implementation: finite-horizon adaptive policy reference

The new `pareto_sequential_policy.rs` reuses the existing `ExactEvsiProblem` schema. It enumerates possible observation branches, updates the finite posterior after each observation, removes the selected measurement action, and chooses between stopping and taking another action under an integer budget.

For posterior (b), remaining action set (S), budget (B), terminal decision (d), action (a), and outcome (y):

```
V(b, S, B) =
  min(
    min_d  E_b[L(d, w)],
    min_{a in S, cost(a) <= B}
      sum_y P_b(y | a) V(b_(a,y), S \ {a}, B - cost(a))
  )
```

where

```
P_b(y | a) = sum_w b(w) P(y | w, a)
b_(a,y)(w) = b(w) P(y | w, a) / P_b(y | a)
```

Stopping is always available. An explicit abstention action can be represented as a terminal decision with a declared loss. Model-discriminating or calibration experiments can be represented as actions. A model identity can be part of the latent joint world.

### Why this matters: a counterexample to greedy EVSI-per-cost

The new unit fixture uses a two-world decision problem with initial Bayes risk 5 and budget 2:

| Measurement | One-step EVSI | Cost | EVSI / cost |
|---|---:|---:|---:|
| Cheap signal | 2.0 | 1 | **2.00** |
| Expensive signal | 3.5 | 2 | 1.75 |

The one-step ratio ranks the cheap signal first. With a total budget of 2, taking it leaves only one cost unit and prevents the expensive measurement. The finite-horizon policy instead chooses the expensive signal and reduces expected terminal Bayes risk from 5.0 to 1.5. This is a design fixture and a mathematical consequence of the specified likelihood/loss tables, not an empirical benchmark result.

### Current v0.1 limits

- At most 16 distinct action IDs and 100,000 expanded policy nodes.
- Every action ID is usable at most once. Repeated replicates are represented as distinct action IDs/slots.
- Likelihood rows are conditional on the finite latent joint world. Dependence across measurements must be represented by enriching that joint world; otherwise conditional-independence assumptions are being made.
- The budget is one integer resource and is capped at 2^53 - 1 so integer budget/cost units remain exactly representable in f64 fields. Multi-resource Pareto budgets and nonlinear cost are not yet implemented.
- The current solver counts declared measurement costs. Controller compute, model fitting, acquisition optimisation, evidence retrieval, and verification must still be measured and charged in campaign-level comparisons.
- Calculations enumerate all finite branches but use IEEE-754 `f64`; “exact” means no sampling approximation for the supplied finite model, subject to floating-point roundoff. It does not mean symbolic rational arithmetic.
- A policy tree is digest-bound and reconstructed from the frozen problem. This proves deterministic reconstruction, not evaluator correctness or model truth.

## 3. A better-than-greedy controller stack

The next scalable controller should be hybrid rather than committed to one acquisition function.

**Decision layer:** define terminal actions and explicit losses before sampling. For multi-objective campaigns, terminal actions may be: ship candidate A, retain a set of feasible trade-offs, run another test, reject all candidates, or abstain. Keep safety/feasibility loss separate from quality preference.

**Measurement layer:** score an experiment by its effect on the terminal decision, not generic uncertainty reduction. Compare knowledge gradient (KG), information-directed sampling (IDS), myopic EVSI/cost, posterior sampling, and cost-aware rollout policies. Information gain is useful only when it is predictive of a better decision.

**Pareto layer:** retain qLogNEHVI/qNEHVI and qNParEGO as serious noisy/batched Bayesian MOO baselines. They optimise Pareto-front acquisition criteria; our comparison must translate their outcomes into the same predeclared terminal decision loss rather than pretending their acquisition scores are decision regret.

**Metareasoning layer:** let the controller choose whether to spend resources on a physical measurement, model validation, verifier audit, or more computation. The acquisition optimiser itself has a cost. A controller that spends 30 seconds optimising an acquisition function to save a 10-second measurement is not cost-efficient.

**Safety layer:** when likelihood calibration, observation independence, stationarity, or evidence authenticity is contradicted, downgrade the decision, expand its ambiguity set, request a diagnostic action, or abstain. Never silently keep using the nominal posterior.

## 4. Model uncertainty is the next frontier

Bayesian model averaging over a finite model index is useful when the plausible model set and its weights are defensible. It is not protection against omitted models.

The follow-on should compare at least three modes:

1. **Known-model Bayesian:** the current finite posterior/likelihood contract.
2. **Bayesian model uncertainty:** the model identity is part of the latent state; observations can update both candidate quality and model identity.
3. **Distributionally robust / ambiguity-aware:** maintain a declared set of plausible posteriors or predictive distributions and report worst-case decision risk, worst-case policy regret, and sensitivity to ambiguity-set radius.

Do not compress these into a single “confidence” score. Report nominal value, robust lower value, worst-case regret, and the ambiguity-set definition. If the robust set is so broad that every decision is unsafe, abstention is a valid output.

A useful current reference is DRO with Bayesian Ambiguity Sets (ICML 2025), which explicitly addresses posterior model uncertainty through posterior-informed ambiguity sets: https://proceedings.mlr.press/v267/dellaporta25a.html. This is a research baseline to study, not a result we have reproduced.

## 5. Benchmark programme

Use three linked scoreboards and keep their acceptance criteria separate.

### A. Terminal decision quality

- Expected terminal Bayes risk and realised decision loss.
- Regret versus the finite-horizon DP reference on overlapping small problems.
- Regret versus the clairvoyant perfect-information bound, labelled as a bound rather than an achievable policy.
- Cost to reach a preregistered risk/regret threshold.
- False-stop, false-feasibility, false-dominance, premature-ship, and unnecessary-abstention rates.
- Decision stability under reasonable perturbations of posterior, loss, and model assumptions.

### B. Pareto search quality

- Hypervolume with a locked reference point; IGD+; additive epsilon; attainment surfaces and target runtime.
- Fixed-budget quality curves and fixed-target empirical cumulative distributions.
- Feasibility probability and simultaneous coverage under noisy observations.
- Front-membership uncertainty and sensitivity to reference-point/normalisation choices.

### C. Evidence and economics

- EventTape/AnytimeHistory attempt and report parity.
- Exact raw-artifact resolution and receipt reconstruction rates.
- False acceptance under candidate, decision, seed, plan, vector, artifact, and verifier substitutions.
- All-in objective observations, failed attempts, model fitting, acquisition optimisation, CPU, wall time, peak memory, energy, accelerator use, verifier work, storage retrieval, and controller overhead.
- Report unknown resource values as unknown; do not impute zero.

### Benchmark matrix

| Regime | Public / in-repo comparators | What it tests |
|---|---|---|
| Small exact finite decision problems | One-step EVSI, sequential DP, greedy EVSI/cost, uniform, random | Exact action/policy regret and stopping |
| Classical deterministic MOO | ZDT, DTLZ, WFG; NSGA-II, MOEA/D-DE, SMS-EMOA | Pareto archive quality and fixed-budget performance |
| Noisy / expensive MOO | COCO bbob-biobj, bbob-noisy, constrained tracks; qLogNEHVI, qNParEGO, TS-TCH, JES, SPMO | Noisy acquisition, costed replication, calibration |
| Many-objective | DTLZ2 5/10 objectives, WFG many-objective | Degradation as objective count grows |
| Model failure | Misspecified likelihood, omitted latent state, drift, heteroskedasticity, correlated noise, heavy tails, informative missingness | Model checks, robust regret, safe stopping |
| Real engineering | Casting / manufacturing, energy design, software performance tasks | External validity and complete resource accounting |

COCO’s public data archive separates bi-objective, noisy, constrained, and large-scale regimes; keep these as distinct tracks rather than merging incomparable scores: https://coco-platform.org/data-archive/index.html. BoTorch’s maintained multi-objective documentation includes qLogNEHVI, qLogNParEGO, and noisy/batched tutorials, with qNEHVI recommended for noisy or batched settings: https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo. The current SOTA2 noisy scaled DTLZ2 five-objective page lists TS-TCH, NParEGO, JES, SPMO, Sobol, and NEHVI on 30 independent runs; use it for baseline discovery only and reproduce its exact setup before making a numerical comparison: https://www.sota2.com/research/sota/multi-objective-optimization-on-noisy-scaled-dtlz2-5-objectives.

### Required adversarial decision landscapes

Include paired cases where the same information has different decision value:

- Wide interval but no chance of changing the chosen terminal action.
- Narrow interval around a safety threshold where one observation can prevent a catastrophic false-feasible decision.
- Correlated objectives where marginal-width sampling double-counts redundant information.
- Heteroskedastic and heavy-tailed observations.
- Nonstationary noise and change points.
- Missing-not-at-random observations and timeout/censoring correlated with bad outcomes.
- Model misspecification with high nominal EVSI but negative realised value.
- Cheap, correlated measurements versus expensive, decision-disambiguating measurements.
- Expensive compute versus cheap evaluation, to test whether acquisition optimisation is itself worth running.
- Calibration experiments that do not directly improve the Pareto front but prevent model-induced wrong decisions.

## 6. Proposed acceptance targets (not achieved)

1. **Reference correctness:** 100% deterministic reconstruction of finite-horizon policy trees; 0 budget overruns; 0 report/hash divergence; impossible outcomes never create fabricated posterior states.
2. **Greedy counterexample suite:** the sequential DP must outperform greedy EVSI/cost on preregistered budget-allocation traps, while matching the greedy choice on cases where one-step selection is actually optimal.
3. **Approximation quality:** on held-out finite problems, preregister maximum normalised regret against DP; a reasonable first target is median ≤1% and 95th percentile ≤5% of initial Bayes risk, with absolute and tail values also reported.
4. **Decision safety:** report false-stop and false-feasibility rates with confidence bounds; require no statistically supported safety regression versus the strongest baseline.
5. **Cost efficiency:** compare at equal all-in resource cost, including controller compute, failed attempts, fitting, verifier and retrieval.
6. **Generalisation:** at least 100 independent task/landscape clusters for confirmatory claims; seeds nested within a landscape are not independent clusters.
7. **Robustness:** each declared misspecification regime is either detected and handled conservatively or explicitly excluded from the claim.

The numerical regret targets are proposed gates to preregister and validate, not known achievable results. A global SOTA claim requires held-out independent task families and comparison to strong external baselines.

## 7. Next engineering sequence

1. Compile, format, lint, and run this finite DP on the exact branch head; cross-check every small fixture against a separate reference calculation.
2. Add multi-resource budget constraints and an explicit cost receipt for controller computation.
3. Add a policy runner that consumes EventTape prefixes and produces the next admissible measurement/stop action; bind every recommendation to the exact visible-prefix digest.
4. Add model-check actions and posterior-predictive diagnostics; measure detection delay and false alarms under drift and dependence.
5. Add a distributionally robust policy reference for a bounded ambiguity set.
6. Build the matched-campaign benchmark harness against the baseline matrix above.
7. Only after these stages, test rollout/MCTS or learned policies on larger campaigns and compare them back to exact DP on tractable problems.

## 8. Exact-head validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_sequential_policy -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_evsi_oracle -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No Rust toolchain execution, CI pass, benchmark result, or SOTA improvement is claimed in this research tranche.
