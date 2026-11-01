# AR-5B7: Finite Total-Variation Robust Decision Reference v0.1

**Status:** implementation draft; tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → this branch.  
**Claim ceiling:** exact finite mass-transfer solution for worst-case expected loss over a declared total-variation ball, subject to f64 roundoff. No learned ambiguity set, radius-calibration guarantee, sequential robust policy, external benchmark result, or SOTA claim.

## 1. Why this tranche exists

The exact EVSI reference and finite-horizon sequential policy assume that the supplied posterior and likelihood model are credible. That is a useful conditional reference, but posterior certainty does not protect against omitted worlds or a misspecified data-generating process.

This tranche adds a transparent robust terminal-decision baseline. Given a nominal posterior p and a declared TV radius rho, the ambiguity set is:

```
Q(p, rho) = { q in simplex(W) : TV(q,p) <= rho }
TV(q,p) = 0.5 * sum_w |q(w) - p(w)|
```

For each terminal decision d, compute:

```
R_robust(d) = max_{q in Q(p,rho)} sum_w q(w) L(d,w)
d_robust = argmin_d R_robust(d)
```

The implementation returns the maximizing distribution for every decision, not just the worst-case scalar loss. A reviewer can independently recompute the loss from the exact distribution in the receipt.

## 2. Finite LP solution

For a fixed decision, maximizing a linear loss over a TV ball can be solved by moving probability mass from the lowest-loss worlds with available mass to the highest-loss worlds with available capacity, up to rho. Each unit of transferred mass consumes one unit of TV distance.

The implementation sorts donor worlds by ascending loss and receiver worlds by descending loss. It then transfers the minimum of donor mass, receiver capacity, and remaining radius. This is a finite linear-program optimum up to floating-point roundoff; no optimizer library or stochastic approximation is used.

The result includes:
- nominal expected loss for each terminal decision;
- worst-case expected loss over the declared TV set;
- actual TV mass moved;
- the explicit worst-case posterior distribution;
- nominal Bayes-optimal and robust minimax decision IDs;
- canonical problem, ambiguity-set and report digests;
- independent report reconstruction.

## 3. Regression fixture

The design fixture has nominal posterior p=(0.9,0.1), two uncertain worlds, and three terminal decisions:

- choose-a: losses (0,40), nominal risk 4;
- choose-b: losses (10,0), nominal risk 9;
- abstain: losses (5,5), nominal risk 5.

At TV radius 0.2:
- choose-a worst-case posterior is (0.7,0.3), risk 12;
- choose-b worst-case risk is 10;
- abstain risk remains 5.

The nominal decision is choose-a; the robust decision is abstain. This is an analytically transparent design fixture, not a measured result.

## 4. What this does not solve

A fixed TV ball is not automatically calibrated to the data-generating process. An arbitrary radius can be too small (false confidence) or too large (unnecessary conservatism). This tranche does not learn an ambiguity set, infer the radius from held-out data, or adapt the radius to the decision loss.

It is also a terminal minimax decision, not a robust sequential experiment policy. Sequential robust optimisation has a subtle issue: a single non-rectangular ambiguity set over the initial latent world is not generally interchangeable with independently choosing a worst-case posterior at every observation node. Future robust policy work must state whether nature chooses one distribution once per campaign, whether ambiguity is rectangular across stages, or whether the model represents a distribution over complete trajectories. These assumptions must never be hidden inside a Bellman recursion.

## 5. Relationship to current research

- **DRO-BAS (ICML 2025):** posterior-informed ambiguity sets and tractable strong dual formulations for specified exponential-family settings. https://proceedings.mlr.press/v267/dellaporta25a.html
- **DRO-RoBAS (Entropy, 11 April 2026):** robust nonparametric Bayesian predictive distributions and MMD ambiguity sets designed to address model misspecification; the paper notes that overly broad ambiguity can be conservative. https://doi.org/10.3390/e28040430
- **Loss-Aware OT-DRO (ICML 2026):** trainable optimal-transport ambiguity sets shaped by downstream loss through a bilevel objective and hypergradients. This is the important next comparator; this PR does not implement that method. https://proceedings.mlr.press/v306/ohnemus26a.html

The immediate research ladder should be:
1. TV-ball robust Bayes action as the transparent finite baseline.
2. Posterior-informed ambiguity sets (DRO-BAS / RoBAS) where the underlying model assumptions fit.
3. Loss-aware trainable OT ambiguity sets, calibrated only on training/validation campaigns.
4. Held-out misspecification tests comparing nominal Bayesian, fixed-TV, posterior-informed DRO, and loss-aware OT-DRO.
5. Sequential robust experiment design with a clearly declared ambiguity-set timing/rectangularity contract.

## 6. A second important current result: correlated objective noise

A paper published in Neurocomputing on 1 November 2026 studies the effect of correlated observation noise in multi-objective Bayesian optimization with intrinsic coregionalization models. Its key reported result is conditional: multi-output GPs with correlated noise can improve noisy MOO when objectives and noise are correlated; noiseless gains over independent GPs are small, and real-world effects are mixed. https://doi.org/10.1016/j.neucom.2026.134400

This is immediately relevant to our replicate data model. EventTape already treats one noisy observation as a complete objective vector, preserving the possibility of cross-objective dependence in the raw evidence. However, the current bounded-Hoeffding coordinate intervals do not estimate or exploit that covariance.

Create a separate matched benchmark comparing:
- independent per-objective surrogates with independent noise;
- multi-output ICM with independent observation noise;
- multi-output ICM with correlated observation noise;
- low-rank covariance variants;
- robust/decision-aware acquisition policies.

The benchmark must pair identical candidates, objective contracts, sampling seeds/common random numbers, noise processes, evaluator calls and all-in model-fitting/acquisition cost. Include both positively and negatively correlated objective signals, correlated noise independent of signal correlation, heteroskedasticity, and model-bias controls. A richer covariance model should only be enabled when its incremental decision value exceeds its fitting and inference cost.

## 7. Benchmark scoreboards

### Robust decision quality
- Nominal expected loss and worst-case expected loss.
- Realized held-out decision regret under shifts drawn outside the nominal posterior.
- Worst-case regret over the declared ambiguity set.
- False-safe and false-feasible decisions.
- Abstention rate and loss conditional on abstention.
- Sensitivity curves across a preregistered grid of ambiguity radii.

### Calibration and misspecification
- Coverage of the true distribution by the ambiguity set across independent campaigns.
- Coverage as a function of sample count and shift magnitude.
- Frequency of missed shift, false alarm, and unnecessarily conservative action.
- Worst-case risk bound tightness.
- Proper scoring of posterior predictive distributions, separated from decision loss.

### Resource economics
- Model-fit CPU/wall time and peak memory.
- Acquisition optimization and robust inner-solver cost.
- Measurement/evaluator/verifier and artifact-resolver cost.
- All-in expected and realized resource cost.
- Decision-risk reduction per unit of total resource, with unknown resource values left unknown.

## 8. Proposed gates (not results)

- Every reported adversarial posterior lies in the declared simplex and satisfies the TV radius.
- The report reconstructs exactly from the frozen problem and ambiguity-set identity.
- Zero-radius robust decisions reproduce nominal Bayes decisions.
- The finite TV result agrees with an independent LP solver on randomized small instances before promotion.
- Radius sweeps are reported, not a single cherry-picked radius.
- Under held-out shifts within the declared TV set, robust risk is never worse than its declared worst-case bound; outside-set results are reported honestly.
- Robustification should not be promoted as an improvement unless it reduces realized held-out decision loss or materially improves safety without unjustified conservatism at equal all-in cost.

## 9. Validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_robust_decision -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_sequential_policy -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_evsi_oracle -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No Rust compile, test, clippy, CI, benchmark, or SOTA pass is claimed.
