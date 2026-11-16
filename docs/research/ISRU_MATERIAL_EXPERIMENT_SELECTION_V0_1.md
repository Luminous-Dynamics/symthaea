# ISRU Material Experiment Selection Benchmark v0.1

**Status:** synthetic evaluator / research example  
**Issue:** [ISRU-MATERIAL-DOE-001 #12476](https://github.com/Luminous-Dynamics/symthaea/issues/12476)  
**Symtropy observation input layer:** [ASTEROID-OPTICAL-ISRU-003 #5577](https://github.com/Luminous-Dynamics/symtropy/issues/5577), [draft data PR #5578](https://github.com/Luminous-Dynamics/symtropy/pull/5578)

## Purpose

This is the first executable application of Symthaea's currently exported hypothesis-disagreement experiment selector to a material-response problem.

It is deliberately not a model of Nectar A/B, asteroid simulant chemistry, thermal spallation, volatile release, ablation, or real laser processing. The response functions are dimensionless synthetic curves used to test selection and abstention semantics.

## Actual Symthaea API reused

The example uses:
- `select_most_informative_experiment`;
- `epistemic_value` (in the independent test for score reproduction).

The implementation in `src/hdc/conjecture_engine/experiment_selection.rs` scores population variance of the surviving hypotheses' predictions at each candidate experiment. This is **prediction disagreement**, not Bayesian mutual information, not a calibrated posterior expected information gain, and not cost/noise-weighted acquisition.

The benchmark intentionally equalizes declared cost and measurement noise across all eligible experiments, so selection compares only the existing disagreement score. A noise-aware or cost-aware acquisition function is a future research increment, not an implied feature of this API.

## Candidate space and worlds

The synthetic catalogue contains:
- seven irradiance labels: 100, 200, 300, 400, 500, 600 and 700 W/cm²;
- three normalized morphology indices: 0.1, 0.5 and 0.9;
- 21 candidate combinations, with one intentionally ineligible negative control removed before ranking;
- one declared experiment cost of 1 chamber-hour and one common measurement sigma of 0.01 for every eligible candidate;
- a deterministic small observation perturbation and a fixed consistency tolerance.

The morphology index is an abstract synthetic coordinate, not a particle-size or mineral-morphology measurement.

Three in-domain worlds correspond exactly to three candidate response-family hypotheses:
1. low-flux peak;
2. high-flux peak;
3. morphology-shifted peak.

A fourth world is a broad out-of-domain response not represented by the candidate hypothesis set. The hidden world is **not** passed to the selector. After selection, the synthetic world generates an observation; the benchmark then retains hypotheses within the frozen tolerance. No surviving hypothesis means `ABSTAIN_OUT_OF_DOMAIN`; one survivor means identified; multiple survivors after the maximum budget means unresolved.

This is a deterministic set-consistency filter, not Bayesian posterior updating.

## Policies

Five policies use the same candidate catalogue and maximum six-experiment budget:

| Policy | Selection rule |
|---|---|
| Fixed canonical | Predeclared 2 × 3 factorial subset |
| One-factor-at-a-time | Vary irradiance at the central morphology index |
| Space-filling | Predeclared spread across both input axes |
| Seeded random | Deterministic shuffled feasible candidates |
| Symthaea prediction disagreement | Repeatedly select the candidate maximizing surviving-hypothesis prediction variance |

The selector sees only remaining eligible experiments and surviving hypotheses. The hidden world and outcome evaluator are only used after each selected experiment.

## Reproducible toy-world result

The example's deterministic run produces this decision count table:

| Synthetic world | Fixed | OFAT | Space-filling | Seeded random | Symthaea disagreement |
|---|---:|---:|---:|---:|---:|
| Low-flux family | 1 / correct | 4 / correct | 2 / correct | 1 / correct | 1 / correct |
| High-flux family | 1 / correct | 4 / correct | 1 / correct | 1 / correct | 1 / correct |
| Morphology-shifted family | 1 / correct | 2 / correct | 2 / correct | 1 / correct | 1 / correct |
| Out-of-domain shift | 1 / **false confident** | 2 / abstained | 1 / abstained | 1 / abstained | 1 / abstained |

The fixed canonical negative control is intentional: it demonstrates that a design can identify one of its registered candidate hypotheses confidently while being wrong because the true response is outside the model family. The selector's out-of-domain case is a correct abstention on this single synthetic world.

These outcomes are deterministic results from four authored toy worlds and one random seed. They are **not** a statistical claim that Symthaea beats SOTA, a real-world experiment count reduction, or a physical result. The seeded-random arm has not been evaluated over a distribution of seeds, and no confidence intervals are claimed.

## What this v0.1 does and does not test

It tests:
- use of the current public Symthaea selector API;
- deterministic candidate choice;
- repeated selection from the remaining feasible pool;
- simple elimination of inconsistent hypotheses;
- an explicit OOD abstention disposition;
- a counterexample where a fixed design false-commits;
- equal nominal experiment-time and sensor-noise budgets.

It does not test:
- a calibrated measurement likelihood or posterior;
- Bayesian mutual information or expected decision utility;
- heterogeneous observation noise or experiment costs;
- sensor drift, sample carry-over, batch effects or common-cause errors;
- experiment safety envelope qualification or real laboratory execution;
- material properties, thermal/chemical response, or product recovery;
- source-data reproduction or physical predictive performance.

## Next benchmark upgrades

1. Add calibrated observation likelihoods and compare disagreement variance with noise-aware expected information gain, expected decision-loss reduction and cost-normalized policies.
2. Add replicate counts, censoring, measurement drift, calibration revocation and batch-level random effects.
3. Freeze entire material lots/morphology families for sealed tests; do not random-split observations from a shared batch.
4. Compare at equal beam/chamber hours, sample preparation, sensor/assay cost and compute budgets over many preregistered seeds.
5. Report family-specific accuracy, OOD precision/recall, false-confidence rate, abstention, decision regret, experiments-to-decision and interval calibration separately.
6. Promote published Nectar observations from the Symtropy register only with their evidence classes and measurement caveats intact; source-derived estimates are not calibrated product measurements.

Candidate research target for a later preregistered held-out study: at least 20% fewer experimental hours to reach a predeclared decision-accuracy and false-confidence bound than the strongest eligible baseline, with no degradation in OOD abstention or product-quality hard gates. This is a hypothesis for future testing, not a current result.

## Run

From the Symthaea repository root, after the branch is available:

```bash
cargo test -p symthaea-core --example isru_material_experiment_selection
cargo run -p symthaea-core --example isru_material_experiment_selection
```

Tests are authored but not executed in the current environment because Cargo/rustc are not installed. The code's presence on a draft branch is not a compile pass.

## Claim ceiling

This benchmark demonstrates only a deterministic synthetic experiment-selection example using the existing prediction-disagreement API. It establishes no physical material-response accuracy, safe laser operating point, captured product, asteroid reserve, economic viability, laboratory repeatability or SOTA superiority.
