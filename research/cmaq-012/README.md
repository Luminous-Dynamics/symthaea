# CMAQ-012: Joint Decision-Risk Calibration Under Candidate-Set Expansion

**Status:** protocol v0.1 + dependency-free Rust synthetic harness; draft  
**Tracking issue:** [#9299](https://github.com/Luminous-Dynamics/symthaea/issues/9299)  
**Claim ceiling:** synthetic decision-calibration diagnostic; no safety guarantee, production result or SOTA claim

## Paradigm shift

Calibrating individual predictions is not sufficient when a planner searches over many candidate actions and selects the maximum. The selected action is subject to a winner's-curse effect: as the candidate set expands, the planner has more opportunities to select an over-optimistic prediction error.

CMAQ-012 changes the calibration unit from a scalar prediction to the **entire action set for one decision**. It asks whether uncertainty bounds hold simultaneously over all candidates and whether those bounds lower actual decision regret at equal candidate budgets.

It also separates two different uncertainty sources:

- **Epistemic uncertainty:** error in the predicted expected value. This should influence how much the planner trusts a counterfactual value estimate.
- **Aleatoric uncertainty:** irreducible outcome variation. This affects downside risk, but is not automatically evidence that the model's expected value is wrong.

A single undifferentiated variance penalty can therefore be too optimistic under epistemic shift and too conservative under a pure increase in outcome noise.

## Experimental protocol

The dependency-free Rust CLI uses disjoint calibration and qualification seed families. It evaluates 2,048 in-distribution calibration episodes per candidate count K (reused across evaluation lanes) and 2,048 qualification episodes for each scenario/candidate-count cell.

Candidate-set sizes are **8, 16, 32 and 64**. Four lanes are evaluated:

1. **In distribution:** calibration and qualification epistemic errors share the same generator.
2. **Epistemic scale shift:** the actual expected-value error scale increases while the model's reported epistemic standard deviation stays fixed.
3. **Aleatoric noise shift:** the expected-value prediction-error process remains stable, but the realized outcome noise increases.
4. **Optimistic failure candidates:** 12% of candidates are genuinely low value but receive a deliberately over-optimistic prediction bias while retaining the same reported epistemic uncertainty.

The calibration split is generated only from the in-distribution mechanism. For every candidate count K, it computes:

- Marginal split-conformal quantile over individual action-value residuals.
- Joint split-conformal quantile over the maximum standardized residual across all K candidates in a decision.

The fixed-scope joint policy calibrates at K=8 and reuses that quantile at larger search budgets. The matched-scope joint policy calibrates at the same candidate count used during qualification. This separates the effect of candidate-set multiplicity from the cost of recalibration.

## Policies

- **Direct fallback:** uses a fixed baseline value of 0.42 and does not select a modeled action.
- **Greedy predicted mean:** chooses the candidate with the largest predicted mean; no uncertainty penalty.
- **Total-variance LCB:** subtracts 1.645 times the square root of reported epistemic plus aleatoric variance.
- **Marginal conformal LCB:** subtracts the marginal split-conformal quantile times epistemic standard deviation.
- **Fixed-K joint conformal LCB:** subtracts the K=8 joint quantile at all candidate counts.
- **Matched-K joint conformal LCB:** uses a separately calibrated joint quantile for each candidate count.
- **Oracle action:** selects the candidate with the highest true expected value; diagnostic ceiling only.

The lower-confidence-bound policies fall back when their best lower bound is below the fixed fallback value. That cost is counted as reduced model-use coverage and decision regret—not hidden.

## Required metrics

Each scenario × candidate count × policy row reports:

- Mean and p95 expected decision regret versus the true best candidate.
- Mean realized reward and lower-tail CVaR05 realized reward.
- Model-use rate / abstention rate.
- Simultaneous candidate-set interval coverage.
- Selected-action interval coverage.
- Paired regret difference versus greedy predicted-mean selection with a normal-approximation 95% interval.

The episode is the unit of analysis. The fixed synthetic outcomes are paired across policies. The normal interval is exploratory; confirmatory claims require a larger independently generated qualification family and robust bootstrap or randomization inference.

## Preregistered hypotheses

1. Greedy selection's optimism exposure increases as candidate count grows.
2. Marginal per-action coverage does not imply simultaneous candidate-set coverage.
3. A K=8 joint quantile performs better than a marginal quantile at K=8, but its coverage deteriorates when deployed at larger K without recalibration.
4. Matched-K joint calibration recovers more simultaneous coverage at larger candidate counts, at the cost of wider lower bounds and potentially more fallback.
5. Aleatoric inflation increases realized-reward downside while expected-value calibration remains comparatively stable. Total-variance LCB may over-abstain relative to an epistemic-aware policy.
6. Epistemic scaling and optimism-biased failure candidates expose calibration failures that a single aggregate prediction error may conceal.

A green focused workflow validates Rust compilation, tests, report dimensions and deterministic contracts. It does not mean the performance hypotheses pass. The workflow prints the observed metrics; preregistered target misses are retained.

## Interpretation limits

Split-conformal marginal and joint coverage relies on exchangeability between calibration and qualification samples. The epistemic-scale and optimism-biased lanes deliberately violate that assumption; under those shifts, coverage loss is a result, not a reason to retune on qualification data.

The generator is a simple synthetic action-value environment. It is not a robot, a physical simulator, an integrated Symthaea planner or proof that an uncertainty bound is safe for real actions. The `centered_normal_approx` generator is a difference of uniforms scaled to unit variance, not a standard Gaussian.

## Run

```sh
cargo test --locked
cargo run --quiet --locked > cmaq-012-report.json
```

## Next research step

Use development data to fit a decision-regret calibrator that predicts the regret of the *selected action*, then test it on a separately sealed environment-family split. Compare it against joint split conformal, ensemble-min value estimates, Bayesian posterior intervals, and MRS.Q-style pessimistic value aggregation under identical search and inference budgets. Extend the same protocol to MiraBench action-conditioned failures and WorldArena 2.0 visuo-tactile or online-RL tasks only after the adapter has an exact reproducible environment contract.
