# SDCB integration map: existing Symthaea experiment-selection capabilities (26 November 2026)

**Status:** source-tree audit and integration plan; no production code was changed by this document.
**Audited repository snapshot:** DSSI parent commit `d5b6ce1474d64ecbb4aeb054f50243f60494a5a9`.

## 1. Main conclusion

Do not add another production experiment planner yet. Symthaea already contains several domain-specific selectors and a decision-focused benchmark constitution. SDCB/DSSI/RPSD should first supply exact oracle contracts and adapters, then compare the existing methods under matched observations, action sets, utility schedules and budgets.

The best first integration target is the existing synthetic ISRU experiment-selection example, followed by the decision-relevant predictability/chaos benchmark. The core selector and Spark Engine then provide direct baseline adapters.

## 2. Existing selector inventory

| Existing module | What the code currently does | Proper SDCB/DSSI role | Boundary / missing capability |
|---|---|---|---|
| `crates/core/symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs` | Generic selector computes population variance across candidate-hypothesis point predictions; missing predictions are excluded; selects the highest disagreement experiment | Prediction-disagreement / query-by-committee baseline | No likelihood calibration, measurement noise model, explicit cost, utility, robust scenario, or downstream decision value |
| `crates/domains/spark-engine/src/optimal_experiment.rs` | EIG in bits over discrete outcome classes; ranks by EIG per dollar and creates a greedy budget sequence | Bayesian EIG baseline for LCF anomaly designs | The documented outcome model is signature-level and approximate; unpredicted hypotheses are assigned uniform outcome probability; greedy sequence advances belief using the current MAP-hypothesis simulated outcome, not an actual observation |
| `crates/domains/symthaea-cell-foundry/src/experiment_planner.rs` | HDC encoder combines uncertainty, cost, time remaining and information-gain features; temporal predictor; FEP-style threshold action selection | HDC/CfC contextual policy baseline and future deployment integration | Similarity to a reference state and fixed thresholds are not calibrated EVSI, model-robust inference, or a multi-objective safety case |
| `crates/domains/symthaea-nuclear/src/active_learning.rs` | Test-only heuristic ranks unmeasured nuclei by predictor uncertainty, unmeasured-neighbour count and shell-proximity bonus, then applies a diversity constraint | Domain-specific heuristic acquisition baseline | The inspected implementation is inside `#[cfg(test)]`; it is not a public production planner and has no exact decision-value objective |
| `crates/core/symthaea-core/examples/isru_material_experiment_selection.rs` | Synthetic response surfaces; compares fixed, OFAT, space-filling, seeded random and Symthaea prediction-disagreement policies; equal candidate cost/noise, hidden synthetic world is used only after selection | First integration fixture for disagreement vs EIG vs EVSI vs exact adaptive policy | The file explicitly says the response curves are synthetic and not material physics; it is not a validated ISRU process or a physical-performance result |

## 3. Existing benchmark constitution to reuse

`docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md` already defines:

- information-set Bayes risk, candidate policy risk, a separately labelled perfect-information lower bound, and decision regret;
- EVSI from expected posterior Bayes risk with acquisition cost in task-loss units;
- resource-vector Pareto dominance over sensing, acquisition, compute, p99 latency, memory, training/update/preprocessing, measured energy and communication;
- paired passive-observation-equivalent worlds where a genuinely discriminating probe can resolve the decision;
- an explicit high-information / zero-decision-value nuisance observation negative control;
- fixed-cadence, event-triggered, variance/uncertainty, greedy entropy/EIG, EVSI and adaptive-model baselines;
- paired family-level splits, hidden truth separation, and claims that require external reproduction.

This is a much better host for DSSI than another independent benchmark. Add robust model scenarios, utility uncertainty and hard safety constraints as versioned extensions to this constitution.

## 4. Repository consistency issue

The inspected tree at the audited parent commit contains only `research/decision-predictability-benchmark/README.md`. The README describes an isolated Cargo crate and gives commands using `research/decision-predictability-benchmark/Cargo.toml`, but that manifest and the described Rust source files were not present in the recursive Git tree. Before using those commands or calling W1 an executable crate, reconcile this mismatch: restore the intended isolated crate or correct the documentation to the actual source location.

This is a structural tree observation at the pinned commit, not proof that no other branch or untracked local workspace has the files.

## 5. Reuse-first integration sequence

### I0 — Restore one canonical evaluator package

- Resolve the missing manifest/source discrepancy in `research/decision-predictability-benchmark/`.
- Keep evaluator-only truth separate from candidate-visible observations using the existing type split.
- Keep exact finite reference fixtures dependency-free and outside the production inference modules.

### I1 — Adapter around the existing selector

Use the existing `select_most_informative_experiment` function as the prediction-disagreement baseline. On the same synthetic ISRU candidate pool, compare:

1. fixed canonical;
2. OFAT;
3. space filling;
4. seeded random;
5. prediction disagreement;
6. Spark EIG-per-cost (where its hypothesis/outcome contract can be faithfully represented);
7. DSSI one-step EVSI;
8. DSSI exact non-myopic policy;
9. RPSD finite-scenario robust decision design.

Do not change the existing selector's score and then compare it to the old score. Add a separate adapter and preserve the original output as the baseline.

### I2 — Use the decision-predictability benchmark as the evaluator

Use its paired-world controls and information-set Bayes risk. The exact oracle should receive the evaluator's declared finite outcome model and objective; the candidate receives only authorized observations. Compare decisions at equal observation budget and compute/resource budgets.

### I3 — Add model-misspecification controls

Generate a true hidden world outside the candidate's nominal likelihood family: omitted variable, wrong noise family, heteroskedastic noise, covariate shift, regime change and sensor drift. Freeze the finite ambiguity set before evaluation. Score worst-case decision regret and evidence integrity, not only nominal posterior entropy.

### I4 — Add multi-objective and safety tests

Keep benefit, energy, cost, latency, environmental impact and safety margins as separate axes. Hard safety floors must be checked against all plausible states/scenarios, not the posterior mean alone. Publish the full Pareto set and hypervolume with a frozen reference point.

### I5 — Connect HDC/CfC without confusing representation with decision quality

Use HDC/CfC to produce compressed hypothesis/likelihood/value representations or amortized policy proposals. Evaluate those proposals against the exact finite oracle on unseen families. The exact oracle remains an evaluator, not a production network. Report model quality, decision regret, calibration and cost separately.

## 6. Initial acceptance gates

- Existing prediction-disagreement selector reproduces its current baseline output on pinned inputs.
- EIG implementation reproduces hand-computable mutual information in bits.
- One-step EVSI chooses the decision-boundary probe over a higher-EIG irrelevant probe.
- Non-myopic policy matches the exact adaptive two-step reference on small worlds.
- Robust policy shares one outcome-to-action mapping across all hidden ambiguity scenarios.
- Pareto front preserves non-dominated options and never trades away a hard safety floor.
- A no-information measurement is not credited for randomizing an action; compare with a cost-free lottery baseline.
- All model/algorithm variants share observation access, candidate pool, budgets and task seeds.
- Every invalid evidence receipt causes zero belief-state or decision-state mutation.
- No claim of model superiority until paired, held-out, independently reproduced evaluation exists.

## 7. Source paths

- Prediction-disagreement selector: `crates/core/symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs`
- Spark EIG and greedy sequence: `crates/domains/spark-engine/src/optimal_experiment.rs`
- Cell Foundry experiment planner: `crates/domains/symthaea-cell-foundry/src/experiment_planner.rs`
- Nuclear active-learning heuristic: `crates/domains/symthaea-nuclear/src/active_learning.rs`
- Existing synthetic ISRU experiment selection bake-off: `crates/core/symthaea-core/examples/isru_material_experiment_selection.rs`
- Decision-relevant predictability benchmark constitution: `docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md`